"""The proposals ledger — one queue every downstream consumer writes into.

Health scans, discovery, and the genre classifier all produce the same shape
of suggestion: (target playlist, track, add/remove, score, reasons). They land
here as status='proposed'; the /review page (or CLI) drains the queue. An
accept/reject is never discarded — decided rows are the labeled training data
for the taste model, and re-proposing something already decided is a no-op so
a rejected suggestion stays rejected across re-scans.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from . import db

VALID_ACTIONS = ("add", "remove")
VALID_STATUS = ("proposed", "accepted", "rejected")


def propose(
    conn: sqlite3.Connection,
    *,
    playlist_id: str,
    playlist_name: str | None,
    action: str,
    kind: str,
    recording_key: str,
    source: str,
    score: float | None = None,
    reasons: list[str] | None = None,
    track_id: str | None = None,
    artist: str | None = None,
    title: str | None = None,
) -> bool:
    """Insert one proposal. Returns False when an identical proposal already
    exists (any status) — a decided row is a durable answer, not a retry slot.
    Fresh proposals refresh score/reasons so re-scans keep rankings current."""
    if action not in VALID_ACTIONS:
        raise ValueError(f"action must be one of {VALID_ACTIONS}")
    cur = conn.execute(
        """
        INSERT INTO playlist_proposals(playlist_id, playlist_name, action, kind,
            recording_key, track_id, artist, title, score, reasons, source,
            status, created_at)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,'proposed',?)
        ON CONFLICT(playlist_id, recording_key, action, source) DO UPDATE SET
            score=excluded.score, reasons=excluded.reasons,
            artist=COALESCE(excluded.artist, artist),
            title=COALESCE(excluded.title, title),
            track_id=COALESCE(excluded.track_id, track_id)
        WHERE status='proposed'""",
        (
            playlist_id,
            playlist_name,
            action,
            kind,
            recording_key,
            track_id,
            artist,
            title,
            score,
            json.dumps(reasons or []),
            source,
            db.utcnow(),
        ),
    )
    return cur.rowcount > 0


def decide(conn: sqlite3.Connection, proposal_id: int, status: str) -> dict | None:
    """Accept or reject one proposal; returns the updated row."""
    if status not in ("accepted", "rejected"):
        raise ValueError("status must be 'accepted' or 'rejected'")
    conn.execute(
        "UPDATE playlist_proposals SET status=?, decided_at=? WHERE id=? AND status='proposed'",
        (status, db.utcnow(), proposal_id),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM playlist_proposals WHERE id=?", (proposal_id,)).fetchone()
    return _decode(row) if row else None


def pending(
    conn: sqlite3.Connection,
    playlist_id: str | None = None,
    source: str | None = None,
    limit: int = 200,
) -> list[dict]:
    where, params = ["status='proposed'"], []
    if playlist_id:
        where.append("playlist_id=?")
        params.append(playlist_id)
    if source:
        where.append("source=?")
        params.append(source)
    rows = conn.execute(
        f"SELECT * FROM playlist_proposals WHERE {' AND '.join(where)} "
        f"ORDER BY playlist_name, score DESC LIMIT ?",
        (*params, limit),
    ).fetchall()
    return [_decode(r) for r in rows]


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        r["status"]: r["n"]
        for r in conn.execute("SELECT status, COUNT(*) n FROM playlist_proposals GROUP BY status")
    }


def labels(conn: sqlite3.Connection) -> list[dict]:
    """Decided proposals as (playlist, recording, action, verdict) training rows."""
    return [
        _decode(r)
        for r in conn.execute(
            "SELECT * FROM playlist_proposals WHERE status IN ('accepted','rejected') "
            "ORDER BY decided_at"
        )
    ]


def log_event(
    conn: sqlite3.Connection,
    *,
    recording_key: str,
    surface: str,
    target: str | None = None,
    track_id: str | None = None,
    score: float | None = None,
) -> None:
    """Record that acetate surfaced a track — the recommender's self-grading
    hook. Later plays/likes of track_id measure whether the surface worked."""
    conn.execute(
        "INSERT INTO recommendation_events(recording_key, track_id, surface, target, "
        "score, recommended_at) VALUES(?,?,?,?,?,?)",
        (recording_key, track_id, surface, target, score, db.utcnow()),
    )


def feedback(conn: sqlite3.Connection) -> list[dict]:
    """Per-surface report: of everything surfaced, what got played afterwards?
    A recommended track counts as a hit when a real play lands after its first
    recommendation timestamp."""
    rows = conn.execute("""
        WITH first_rec AS (
            SELECT surface, track_id, recording_key, MIN(recommended_at) at
            FROM recommendation_events WHERE track_id IS NOT NULL
            GROUP BY surface, track_id)
        SELECT fr.surface,
               COUNT(*) AS recommended,
               SUM(EXISTS (
                   SELECT 1 FROM play_events p
                   JOIN track_recording tr ON tr.track_id = p.track_id
                   JOIN track_recording tr2 ON tr2.recording_key = tr.recording_key
                   WHERE tr2.track_id = fr.track_id AND p.played_at > fr.at
                     AND (p.ms_played IS NULL OR p.ms_played >= 30000))) AS played_after
        FROM first_rec fr GROUP BY fr.surface ORDER BY recommended DESC""").fetchall()
    out = []
    for r in rows:
        played = r["played_after"] or 0
        out.append(
            {
                "surface": r["surface"],
                "recommended": r["recommended"],
                "played_after": played,
                "hit_rate": round(played / r["recommended"], 3) if r["recommended"] else 0.0,
            }
        )
    return out


def _decode(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    try:
        d["reasons"] = json.loads(d.get("reasons") or "[]")
    except (TypeError, ValueError):
        d["reasons"] = []
    return d
