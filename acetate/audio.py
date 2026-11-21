"""Preview-audio cache for the embedding layer (Hyperspace).

Fetches a ~30s preview clip per recording: Deezer by exact ISRC first (the
ISRCs already sitting in `tracks`), then Deezer search, then iTunes search —
search hits must pass a duration check against Spotify's duration_ms (or an
exact normalized artist+title match when we have no duration). Previews are
plenty for the 16 kHz-mono embedding models and double as click-to-play
audition clips in the viz; a full-track source (NAS archive) can slot in
later as another `source`.

Read-only against the outside world, no accounts needed. Resumable: the
queue/ledger lives in audio_cache, ordered most-played-first.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from . import db
from .canon import _norm

DEEZER_API = "https://api.deezer.com"
ITUNES_API = "https://itunes.apple.com/search"
USER_AGENT = {"User-Agent": "acetate/0.2 (personal)"}

# Deezer allows 50 requests / 5s per IP. The throttle paces *API* calls
# globally across worker threads (CDN clip downloads aren't API-rate-limited).
_MIN_INTERVAL = 0.25
_last_call = 0.0
_throttle_lock = threading.Lock()
WORKERS = 4

# iTunes is far stricter (~20 req/min per IP; it 403/429-banned a fast pass).
# Separate, much slower global pacing for the fallback only.
_ITUNES_MIN_INTERVAL = 3.5
_itunes_last = 0.0
_itunes_lock = threading.Lock()

# A search hit counts as "the same recording" if its duration is within this
# many seconds of Spotify's (radio edits / remasters differ by much more).
DURATION_TOLERANCE_S = 5

MAX_ATTEMPTS = 3  # transient-error retries before a row is parked as 'error'
EMBED_MODELS = ("discogs-effnet", "discogs400", "clap-music")


def audio_root() -> Path:
    """data/audio — sibling of the DB so it deploys/backs up with it."""
    return db.db_path().parent / "audio"


def _throttle() -> None:
    """Global API pacing: sleeping inside the lock serializes callers onto the
    shared interval, so N threads still sum to ≤1/_MIN_INTERVAL calls per second."""
    global _last_call
    with _throttle_lock:
        wait = _MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def _get_json(url: str, **params) -> dict | list | None:
    _throttle()
    resp = requests.get(url, params=params or None, timeout=20, headers=USER_AGENT)
    resp.raise_for_status()
    return resp.json()


class _DeezerProviderMiss(ValueError):
    pass


def _deezer_payload(data: dict | list | None) -> dict:
    if not isinstance(data, dict):
        raise ValueError(f"unexpected Deezer response: {str(data)[:120]}")
    provider_error = data.get("error")
    if provider_error:
        code = provider_error.get("code") if isinstance(provider_error, dict) else None
        message = str(provider_error)[:240]
        if code == 800:
            raise _DeezerProviderMiss(message)
        raise ValueError(message)
    return data


def _deezer_entity(data: dict | list | None, surface: str) -> dict | None:
    try:
        return _deezer_payload(data)
    except _DeezerProviderMiss:
        return None
    except ValueError as exc:
        raise requests.RequestException(f"unexpected Deezer {surface} response: {exc}") from exc


def _deezer_by_isrc(isrc: str) -> dict | None:
    return _deezer_entity(_get_json(f"{DEEZER_API}/track/isrc:{isrc}"), "ISRC")


def deezer_track(track_id: str) -> dict | None:
    """Fetch current Deezer track metadata, including a fresh signed preview URL."""
    return _deezer_entity(_get_json(f"{DEEZER_API}/track/{track_id}"), "track")


def deezer_search(artist: str, title: str) -> list[dict]:
    try:
        data = _deezer_payload(
            _get_json(f"{DEEZER_API}/search", q=f'artist:"{artist}" track:"{title}"')
        )
    except ValueError as exc:
        raise requests.RequestException(f"unexpected Deezer search response: {exc}") from exc
    results = data.get("data")
    if not isinstance(results, list):
        raise requests.RequestException("unexpected Deezer search response: missing data list")
    return results


def _itunes_search(artist: str, title: str) -> list[dict]:
    """Last-resort fallback. Failure here means 'no preview', never a row retry —
    and it gets its own slow pacing so iTunes doesn't ban the whole run."""
    global _itunes_last
    with _itunes_lock:
        wait = _ITUNES_MIN_INTERVAL - (time.monotonic() - _itunes_last)
        if wait > 0:
            time.sleep(wait)
        _itunes_last = time.monotonic()
    try:
        data = _get_json(ITUNES_API, term=f"{artist} {title}", entity="song", limit=5)
    except requests.RequestException:
        return []
    out = []
    for r in data.get("results", []) if isinstance(data, dict) else []:
        if r.get("previewUrl"):
            out.append(
                {
                    "id": r.get("trackId"),
                    "preview": r["previewUrl"],
                    "duration": (r.get("trackTimeMillis") or 0) / 1000,
                    "artist": {"name": r.get("artistName", "")},
                    "title": r.get("trackName", ""),
                }
            )
    return out


def _acceptable(cand: dict, artist: str, title: str, duration_ms: int | None) -> bool:
    """A search hit is trusted if the duration matches Spotify's; with no known
    duration, require the normalized artist+title to match exactly."""
    if not cand.get("preview"):
        return False
    if duration_ms:
        return abs(cand.get("duration", 0) - duration_ms / 1000) <= DURATION_TOLERANCE_S
    return _norm(cand.get("artist", {}).get("name")) == _norm(artist) and _norm(
        cand.get("title")
    ) == _norm(title)


def _download(preview_url: str, recording_key: str) -> tuple[str, Path]:
    """Fetch the clip to data/audio/previews/<h2>/<sha1>.mp3; returns (rel, abs)."""
    h = hashlib.sha1(recording_key.encode()).hexdigest()
    rel = f"audio/previews/{h[:2]}/{h}.mp3"
    dest = db.db_path().parent / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(preview_url, timeout=30, headers=USER_AGENT)
    resp.raise_for_status()
    if len(resp.content) < 10_000:  # a real 30s MP3 is ~500 KB; tiny = stub/error page
        raise ValueError(f"preview too small ({len(resp.content)} bytes)")
    dest.write_bytes(resp.content)
    return rel, dest


def _universe(conn: sqlite3.Connection, min_plays: int):
    """The embedding universe: liked ∪ ≥min_plays recordings, with the best
    ISRC/duration available among each recording's member track ids."""
    return conn.execute(
        """
        SELECT rs.recording_key, rs.artist, rs.title, rs.plays,
               COALESCE(MAX(CASE WHEN t.track_id = rs.rep_track_id THEN t.isrc END),
                        MAX(t.isrc)) AS isrc,
               COALESCE(MAX(CASE WHEN t.track_id = rs.rep_track_id THEN t.duration_ms END),
                        MAX(t.duration_ms)) AS duration_ms
        FROM recording_stats rs
        JOIN track_recording tr ON tr.recording_key = rs.recording_key
        LEFT JOIN tracks t ON t.track_id = tr.track_id AND t.isrc IS NOT NULL AND t.isrc != ''
        WHERE rs.liked = 1 OR rs.plays >= ?
        GROUP BY rs.recording_key
    """,
        (min_plays,),
    ).fetchall()


def enqueue(conn: sqlite3.Connection, min_plays: int = 5) -> int:
    """Add universe recordings missing from audio_cache as 'pending'. Idempotent;
    re-run any time the universe grows (new likes / plays crossing the bar)."""
    now = db.utcnow()
    rows = [(r["recording_key"], now) for r in _universe(conn, min_plays)]
    before = conn.execute("SELECT COUNT(*) FROM audio_cache").fetchone()[0]
    conn.executemany(
        "INSERT OR IGNORE INTO audio_cache(recording_key, status, updated_at) "
        "VALUES(?, 'pending', ?)",
        rows,
    )
    conn.commit()
    return conn.execute("SELECT COUNT(*) FROM audio_cache").fetchone()[0] - before


def _fetch_one(
    rec: sqlite3.Row, *, require_identity: bool = False
) -> tuple[str, str, str | None, float | None, str | None]:
    """Try the source chain for one recording.
    Returns (status, source, source_id, duration_s, rel_path).
    require_identity keeps artist/title checks even when a duration is known.
    """
    artist, title = rec["artist"] or "", rec["title"] or ""
    duration_ms = rec["duration_ms"]

    def acceptable(cand: dict) -> bool:
        return _acceptable(cand, artist, title, duration_ms) and (
            not require_identity or _acceptable(cand, artist, title, None)
        )

    cands: list[tuple[str, dict]] = []
    if rec["isrc"]:
        hit = _deezer_by_isrc(rec["isrc"])
        if hit and hit.get("preview"):
            cands.append(("deezer_isrc", hit))
    if not cands and artist and title:
        cands += [("deezer_search", c) for c in deezer_search(artist, title)]
        if not any(acceptable(c) for _, c in cands):
            cands += [("itunes_search", c) for c in _itunes_search(artist, title)]

    for source, cand in cands:
        # ISRC hits are exact by construction; search hits must pass the check.
        if source != "deezer_isrc" and not acceptable(cand):
            continue
        rel, _ = _download(cand["preview"], rec["recording_key"])
        return "fetched", source, str(cand.get("id")), float(cand.get("duration") or 0), rel
    return "miss", "", None, None, None


def fetch_pending(
    conn: sqlite3.Connection, batch: int | None = None, min_plays: int = 5, progress=None
) -> dict:
    """Work the queue most-played-first: enqueue new universe rows, then resolve
    each pending row through Deezer-ISRC → Deezer-search → iTunes-search."""
    added = enqueue(conn, min_plays)
    todo = conn.execute(
        """
        SELECT a.recording_key, u.artist, u.title, u.isrc, u.duration_ms, u.plays
        FROM audio_cache a
        JOIN (SELECT rs.recording_key, rs.artist, rs.title, rs.plays,
                     COALESCE(MAX(CASE WHEN t.track_id = rs.rep_track_id THEN t.isrc END),
                              MAX(t.isrc)) AS isrc,
                     COALESCE(MAX(CASE WHEN t.track_id = rs.rep_track_id THEN t.duration_ms END),
                              MAX(t.duration_ms)) AS duration_ms
              FROM recording_stats rs
              JOIN track_recording tr ON tr.recording_key = rs.recording_key
              LEFT JOIN tracks t ON t.track_id = tr.track_id AND t.isrc IS NOT NULL AND t.isrc != ''
              GROUP BY rs.recording_key) u ON u.recording_key = a.recording_key
        WHERE a.status = 'pending' AND a.attempts < ?
        ORDER BY u.plays DESC
    """,
        (MAX_ATTEMPTS,),
    ).fetchall()
    if batch:
        todo = todo[:batch]

    def _job(rec):
        """Network-only (threads); DB writes stay on the main thread below."""
        try:
            return rec, _fetch_one(rec), None
        except (requests.RequestException, ValueError, OSError) as e:
            return rec, None, e

    counts = {"queued_new": added, "fetched": 0, "miss": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, (rec, result, exc) in enumerate(ex.map(_job, todo)):
            if exc is None:
                status, source, source_id, duration_s, rel = result
                conn.execute(
                    "UPDATE audio_cache SET status=?, source=?, source_id=?, duration_s=?, "
                    "path=?, error=NULL, attempts=attempts+1, updated_at=? WHERE recording_key=?",
                    (
                        status,
                        source or None,
                        source_id,
                        duration_s,
                        rel,
                        db.utcnow(),
                        rec["recording_key"],
                    ),
                )
                if status == "fetched":
                    now = db.utcnow()
                    conn.executemany(
                        """
                        INSERT OR IGNORE INTO embedding_runs(
                            recording_key, model, status, attempts, updated_at
                        )
                        VALUES(?, ?, 'pending', 0, ?)
                        """,
                        [(rec["recording_key"], model, now) for model in EMBED_MODELS],
                    )
                counts[status] += 1
            else:
                # transient (network/CDN) — retried next run until MAX_ATTEMPTS parks it
                conn.execute(
                    "UPDATE audio_cache SET attempts=attempts+1, error=?, "
                    "status=CASE WHEN attempts+1 >= ? THEN 'error' ELSE 'pending' END, "
                    "updated_at=? WHERE recording_key=?",
                    (str(exc)[:300], MAX_ATTEMPTS, db.utcnow(), rec["recording_key"]),
                )
                counts["error"] += 1
            # Commit per item: three backfills (fetch/embed/mbid) share this DB, and
            # in WAL mode an open write txn holds THE write lock — keep holds at ~ms.
            conn.commit()
            if i % 25 == 0 and progress:
                progress(i, len(todo), counts)
    conn.commit()
    counts["remaining"] = conn.execute(
        "SELECT COUNT(*) FROM audio_cache WHERE status='pending' AND attempts < ?", (MAX_ATTEMPTS,)
    ).fetchone()[0]
    return counts


def _deezer_begin(conn: sqlite3.Connection, entity_type: str, entity_id: str | int) -> None:
    conn.execute(
        """
        INSERT INTO deezer_metadata_runs(
            entity_type, entity_id, status, attempts, error, updated_at
        )
        VALUES(?, ?, 'pending', 1, NULL, ?)
        ON CONFLICT(entity_type, entity_id) DO UPDATE SET
            status='pending',
            attempts=deezer_metadata_runs.attempts + 1,
            error=NULL,
            updated_at=excluded.updated_at
        """,
        (entity_type, str(entity_id), db.utcnow()),
    )
    conn.commit()


def _deezer_finish(
    conn: sqlite3.Connection,
    entity_type: str,
    entity_id: str | int,
    status: str,
    error: Exception | str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE deezer_metadata_runs
        SET status=?, error=?, updated_at=?
        WHERE entity_type=? AND entity_id=?
        """,
        (
            status,
            str(error)[:300] if error else None,
            db.utcnow(),
            entity_type,
            str(entity_id),
        ),
    )


def _deezer_fail(
    conn: sqlite3.Connection,
    entity_type: str,
    entity_id: str | int,
    error: Exception,
) -> str:
    attempts = conn.execute(
        "SELECT attempts FROM deezer_metadata_runs WHERE entity_type=? AND entity_id=?",
        (entity_type, str(entity_id)),
    ).fetchone()[0]
    status = (
        "provider_miss"
        if isinstance(error, _DeezerProviderMiss)
        else ("error" if attempts >= MAX_ATTEMPTS else "pending")
    )
    _deezer_finish(conn, entity_type, entity_id, status, error)
    return status


def _deezer_should_try(conn: sqlite3.Connection, entity_type: str, entity_id: str | int) -> bool:
    row = conn.execute(
        "SELECT status, attempts FROM deezer_metadata_runs WHERE entity_type=? AND entity_id=?",
        (entity_type, str(entity_id)),
    ).fetchone()
    return (
        row is None
        or row["status"] == "pending"
        or (row["status"] == "error" and row["attempts"] < MAX_ATTEMPTS)
    )


def enrich(conn: sqlite3.Connection, batch: int | None = None, progress=None) -> dict:
    """Harvest the Deezer metadata we originally discarded, for every recording
    whose preview came from Deezer (source_id in hand): track bpm/gain/rank →
    album label/genres → artist fans + related. Resumable at each of the three
    levels; run AFTER a fetch pass (same per-IP rate budget)."""
    todo = conn.execute("""
        SELECT a.recording_key, a.source_id FROM audio_cache a
        WHERE a.status='fetched' AND a.source LIKE 'deezer%' AND a.source_id IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM deezer_track d WHERE d.recording_key = a.recording_key)
        ORDER BY (SELECT plays FROM recording_stats rs WHERE rs.recording_key = a.recording_key) DESC
    """).fetchall()
    todo = [row for row in todo if _deezer_should_try(conn, "track", row["source_id"])]
    if batch:
        todo = todo[:batch]
    counts = {
        "tracks": 0,
        "albums": 0,
        "artists": 0,
        "provider_miss": 0,
        "errors": 0,
    }

    for i, rec in enumerate(todo):
        entity_id = rec["source_id"]
        _deezer_begin(conn, "track", entity_id)
        try:
            d = _deezer_payload(_get_json(f"{DEEZER_API}/track/{entity_id}"))
            conn.execute(
                "INSERT OR REPLACE INTO deezer_track(recording_key, deezer_id, bpm, gain, rank, "
                "release_date, artist_id, album_id, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    rec["recording_key"],
                    d.get("id"),
                    d.get("bpm") or None,
                    d.get("gain"),
                    d.get("rank"),
                    d.get("release_date"),
                    (d.get("artist") or {}).get("id"),
                    (d.get("album") or {}).get("id"),
                    db.utcnow(),
                ),
            )
            _deezer_finish(conn, "track", entity_id, "available")
            counts["tracks"] += 1
        except (requests.RequestException, ValueError) as exc:
            state = _deezer_fail(conn, "track", entity_id, exc)
            counts["provider_miss" if state == "provider_miss" else "errors"] += 1
        conn.commit()
        if i % 100 == 0 and progress:
            progress(i, len(todo), counts)

    import json as _json

    album_ids = [
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT album_id FROM deezer_track WHERE album_id IS NOT NULL "
            "AND album_id NOT IN (SELECT album_id FROM deezer_album)"
        ).fetchall()
    ]
    for album_id in (item for item in album_ids if _deezer_should_try(conn, "album", item)):
        _deezer_begin(conn, "album", album_id)
        try:
            d = _deezer_payload(_get_json(f"{DEEZER_API}/album/{album_id}"))
            genres = [g["name"] for g in (d.get("genres") or {}).get("data", [])]
            conn.execute(
                "INSERT OR REPLACE INTO deezer_album(album_id, title, label, genres, "
                "release_date, updated_at) VALUES(?,?,?,?,?,?)",
                (
                    album_id,
                    d.get("title"),
                    d.get("label"),
                    _json.dumps(genres),
                    d.get("release_date"),
                    db.utcnow(),
                ),
            )
            _deezer_finish(conn, "album", album_id, "available")
            counts["albums"] += 1
        except (requests.RequestException, ValueError) as exc:
            state = _deezer_fail(conn, "album", album_id, exc)
            counts["provider_miss" if state == "provider_miss" else "errors"] += 1
        conn.commit()

    artist_ids = [
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT artist_id FROM deezer_track WHERE artist_id IS NOT NULL "
            "AND artist_id NOT IN (SELECT artist_id FROM deezer_artist)"
        ).fetchall()
    ]
    for artist_id in (item for item in artist_ids if _deezer_should_try(conn, "artist", item)):
        _deezer_begin(conn, "artist", artist_id)
        try:
            d = _deezer_payload(_get_json(f"{DEEZER_API}/artist/{artist_id}"))
            rel = _get_json(f"{DEEZER_API}/artist/{artist_id}/related?limit=20")
            related = [
                {"id": a["id"], "name": a["name"]}
                for a in (rel.get("data", []) if isinstance(rel, dict) else [])
            ]
            conn.execute(
                "INSERT OR REPLACE INTO deezer_artist(artist_id, name, nb_fan, related_json, "
                "updated_at) VALUES(?,?,?,?,?)",
                (artist_id, d.get("name"), d.get("nb_fan"), _json.dumps(related), db.utcnow()),
            )
            _deezer_finish(conn, "artist", artist_id, "available")
            counts["artists"] += 1
        except (requests.RequestException, ValueError) as exc:
            state = _deezer_fail(conn, "artist", artist_id, exc)
            counts["provider_miss" if state == "provider_miss" else "errors"] += 1
        conn.commit()
    return counts


def status(conn: sqlite3.Connection) -> dict:
    """Queue + embedding coverage counts for CLI/dashboard status."""
    out = {
        r["status"]: r["n"]
        for r in conn.execute("SELECT status, COUNT(*) AS n FROM audio_cache GROUP BY status")
    }
    for model, n in conn.execute(
        "SELECT e.model, COUNT(*) FROM embeddings e "
        "JOIN recording_stats rs USING(recording_key) GROUP BY e.model"
    ):
        out[f"embedded:{model}"] = n
    out["hyperspace_points"] = conn.execute("SELECT COUNT(*) FROM hyperspace_points").fetchone()[0]
    for st, n in conn.execute("SELECT status, COUNT(*) FROM recording_mbid GROUP BY status"):
        out[f"mbid:{st}"] = n
    for entity_type, state, count in conn.execute(
        "SELECT entity_type, status, COUNT(*) FROM deezer_metadata_runs "
        "GROUP BY entity_type, status"
    ):
        out[f"deezer_metadata:{entity_type}:{state}"] = count
    return out
