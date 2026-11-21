"""ISRC → MusicBrainz recording MBID resolution.

The MBID is the join key into the open recommendation world (ListenBrainz
similar-recordings, acoustic datasets). Resolution order per recording:
exact ISRC lookup first (we already hold ISRCs for ~90% of the universe),
then a scored artist+title search for the rest (accepted at score ≥ 90).

MusicBrainz allows 1 req/s with a proper identifying User-Agent; the backfill
is resumable (recording_mbid is the ledger) and takes ~2.5h for the full
universe — run it in the background, like the Last.fm tagger.
"""

from __future__ import annotations

import os
import sqlite3
import time

import requests

from . import db
from .audio import _universe

MB_API = "https://musicbrainz.org/ws/2"
DEFAULT_USER_AGENT = "acetate/0.2 (+https://github.com/charlesbine/acetate)"
USER_AGENT = {"User-Agent": os.environ.get("ACETATE_USER_AGENT") or DEFAULT_USER_AGENT}
MIN_INTERVAL = 1.1  # MusicBrainz policy: 1 req/s; leave a margin
SEARCH_MIN_SCORE = 90
MAX_ATTEMPTS = 3

_last_call = 0.0


def _headers() -> dict[str, str]:
    ua = os.environ.get("ACETATE_USER_AGENT") or DEFAULT_USER_AGENT
    return {"User-Agent": ua}


def _get(path: str, **params) -> dict | None:
    global _last_call
    wait = MIN_INTERVAL - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()
    resp = requests.get(
        f"{MB_API}/{path}", params={**params, "fmt": "json"}, timeout=20, headers=_headers()
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def _by_isrc(isrc: str) -> str | None:
    data = _get(f"isrc/{isrc}")
    recs = (data or {}).get("recordings") or []
    return recs[0]["id"] if recs else None


def _by_search(artist: str, title: str) -> str | None:
    q = f'recording:"{title.replace(chr(34), "")}" AND artist:"{artist.replace(chr(34), "")}"'
    data = _get("recording", query=q, limit=3)
    for rec in (data or {}).get("recordings") or []:
        if int(rec.get("score", 0)) >= SEARCH_MIN_SCORE:
            return rec["id"]
    return None


def enqueue(conn: sqlite3.Connection, min_plays: int = 5) -> int:
    """Queue universe recordings (with their best ISRC) that aren't in the ledger."""
    now = db.utcnow()
    rows = [(r["recording_key"], r["isrc"], now) for r in _universe(conn, min_plays)]
    before = conn.execute("SELECT COUNT(*) FROM recording_mbid").fetchone()[0]
    conn.executemany(
        "INSERT OR IGNORE INTO recording_mbid(recording_key, isrc, status, updated_at) "
        "VALUES(?,?,'pending',?)",
        rows,
    )
    conn.commit()
    return conn.execute("SELECT COUNT(*) FROM recording_mbid").fetchone()[0] - before


def resolve_pending(
    conn: sqlite3.Connection, batch: int | None = None, min_plays: int = 5, progress=None
) -> dict:
    """Work the queue most-played-first. Resumable; re-run to continue."""
    added = enqueue(conn, min_plays)
    todo = conn.execute(
        """
        SELECT m.recording_key, m.isrc, rs.artist, rs.title
        FROM recording_mbid m JOIN recording_stats rs USING(recording_key)
        WHERE m.status = 'pending' AND m.attempts < ?
        ORDER BY rs.plays DESC
    """,
        (MAX_ATTEMPTS,),
    ).fetchall()
    if batch:
        todo = todo[:batch]

    counts = {"queued_new": added, "resolved": 0, "miss": 0, "error": 0}
    for i, rec in enumerate(todo):
        try:
            mbid, method = None, None
            if rec["isrc"]:
                mbid, method = _by_isrc(rec["isrc"]), "isrc"
            if not mbid and rec["artist"] and rec["title"]:
                mbid, method = _by_search(rec["artist"], rec["title"]), "search"
            status = "resolved" if mbid else "miss"
            conn.execute(
                "UPDATE recording_mbid SET mbid=?, method=?, status=?, attempts=attempts+1, "
                "updated_at=? WHERE recording_key=?",
                (mbid, method if mbid else None, status, db.utcnow(), rec["recording_key"]),
            )
            counts[status] += 1
        except requests.RequestException as e:
            conn.execute(
                "UPDATE recording_mbid SET attempts=attempts+1, "
                "status=CASE WHEN attempts+1 >= ? THEN 'error' ELSE 'pending' END, "
                "updated_at=? WHERE recording_key=?",
                (MAX_ATTEMPTS, db.utcnow(), rec["recording_key"]),
            )
            counts["error"] += 1
            _ = e
        # per-item commit: keep the WAL write lock at ~ms (fetch/embed run concurrently)
        conn.commit()
        if i % 25 == 0 and progress:
            progress(i, len(todo), counts)
    conn.commit()
    counts["remaining"] = conn.execute(
        "SELECT COUNT(*) FROM recording_mbid WHERE status='pending' AND attempts < ?",
        (MAX_ATTEMPTS,),
    ).fetchone()[0]
    return counts
