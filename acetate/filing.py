"""Filing — the liked-but-unplaylisted inbox.

Surfaces liked recordings that live in no playlist (recording_stats.liked=1,
homed=0), most-played first, each with its top-N best-fit authored playlists
(taste centroid cosine, judged against that playlist's own member p10). Filing
appends the track to the chosen playlist and records the action in
`filing_actions`.

This module is the ONE sanctioned path by which acetate writes to a playlist it
did not create. Every add is:
  * append-only — the track goes to the end; nothing else is touched;
  * ledgered — recorded in filing_actions with before/after Spotify snapshots;
  * reversible — undo() removes EXACTLY the track the ledger says acetate added,
    identified from our own row, so it can never remove anything you added by hand.
"""

from __future__ import annotations

import sqlite3

import numpy as np

from . import db, taste

SOURCE = "filing"


def _uri(track_id: str) -> str:
    return track_id if track_id.startswith("spotify:") else f"spotify:track:{track_id}"


def _snapshot_id(sp, playlist_id: str) -> str | None:
    """Best-effort Spotify snapshot id for the audit trail (never fatal)."""
    try:
        return sp.playlist(playlist_id, fields="snapshot_id").get("snapshot_id")
    except Exception:
        return None


def unfiled_count(conn: sqlite3.Connection, min_plays: int = 1) -> int:
    """How many liked recordings are in no playlist (incl. ones we can't target)."""
    return conn.execute(
        "SELECT COUNT(*) FROM recording_stats WHERE liked=1 AND homed=0 AND plays >= ?",
        (min_plays,),
    ).fetchone()[0]


def _unfiled_rows(conn: sqlite3.Connection, min_plays: int, limit: int) -> list[sqlite3.Row]:
    """Liked, unplaylisted, not already filed this session, most-played first."""
    return conn.execute(
        "SELECT recording_key, rep_track_id, artist, title, plays, liked_at "
        "FROM recording_stats "
        "WHERE liked=1 AND homed=0 AND rep_track_id IS NOT NULL AND plays >= ? "
        "  AND recording_key NOT IN "
        "      (SELECT recording_key FROM filing_actions WHERE status='added') "
        "ORDER BY plays DESC, liked_at DESC LIMIT ?",
        (min_plays, limit),
    ).fetchall()


def _region_matrix(conn: sqlite3.Connection):
    """(ids, names, p10s[R], centroids[R,d]) for every cached taste region."""
    ids: list[str] = []
    names: list[str] = []
    p10s: list[float] = []
    vecs: list[np.ndarray] = []
    for r in conn.execute("SELECT playlist_id, name, member_p10, centroid FROM taste_regions"):
        ids.append(r["playlist_id"])
        names.append(r["name"])
        p10s.append(float(r["member_p10"]) if r["member_p10"] is not None else 0.0)
        vecs.append(np.frombuffer(r["centroid"], dtype=np.float32))
    if not vecs:
        return [], [], np.empty(0, dtype=np.float32), np.empty((0, 0), dtype=np.float32)
    return ids, names, np.array(p10s, dtype=np.float32), np.vstack(vecs)


def queue(
    conn: sqlite3.Connection,
    *,
    min_plays: int = 1,
    limit: int = 100,
    top_n: int = 3,
    model: str = taste.EMBED_MODEL,
) -> list[dict]:
    """Actionable filing inbox: liked-unplaylisted recordings that have an
    embedding, most-played first, each with its top-N best-fit playlists.
    A target is 'strong' when its centroid cosine clears that playlist's own
    member p10 (i.e. the track fits at least as well as the playlist's weakest
    real members)."""
    rows = _unfiled_rows(conn, min_plays, limit)
    if not rows:
        return []
    found, mat = taste._vectors(conn, [r["recording_key"] for r in rows], model)
    vec_of = {k: mat[i] for i, k in enumerate(found)}
    ids, names, p10s, centroids = _region_matrix(conn)
    out: list[dict] = []
    for r in rows:
        v = vec_of.get(r["recording_key"])
        if v is None or not centroids.shape[0]:
            continue  # can't recommend → not actionable
        sims = centroids @ v  # both normalized → cosine
        order = np.argsort(-sims)[:top_n]
        targets = [
            {
                "playlist_id": ids[j],
                "name": names[j],
                "score": round(float(sims[j]), 3),
                "strong": bool(sims[j] >= p10s[j]),
            }
            for j in order
        ]
        out.append(
            {
                "recording_key": r["recording_key"],
                "track_id": r["rep_track_id"],
                "artist": r["artist"],
                "title": r["title"],
                "plays": r["plays"],
                "targets": targets,
            }
        )
    return out


def file_track(
    conn: sqlite3.Connection,
    sp,
    *,
    recording_key: str,
    track_id: str,
    playlist_id: str,
    playlist_name: str | None = None,
    artist: str | None = None,
    title: str | None = None,
    score: float | None = None,
) -> int:
    """Append one track to a (user-authored) playlist and log the action. This is
    the only sanctioned acetate write to a playlist it did not create. Returns the
    filing_actions row id."""
    snap_before = _snapshot_id(sp, playlist_id)
    sp.playlist_add_items(playlist_id, [_uri(track_id)])  # append-only
    snap_after = _snapshot_id(sp, playlist_id)
    now = db.utcnow()
    cur = conn.execute(
        "INSERT INTO filing_actions(recording_key, track_id, playlist_id, playlist_name,"
        " artist, title, score, source, snapshot_before, snapshot_after, status, added_at)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,'added',?)",
        (
            recording_key,
            track_id,
            playlist_id,
            playlist_name,
            artist,
            title,
            score,
            SOURCE,
            snap_before,
            snap_after,
            now,
        ),
    )
    conn.commit()
    from . import proposals

    proposals.log_event(
        conn,
        recording_key=recording_key,
        track_id=track_id,
        surface="filing",
        target=playlist_name,
        score=score,
    )
    conn.commit()
    return cur.lastrowid


def undo(conn: sqlite3.Connection, sp, action_id: int) -> dict | None:
    """Reverse one filing: remove exactly the track this ledger recorded acetate
    adding, from exactly that playlist. Refuses anything not logged as ours and
    still 'added' — so it can never remove a track you filed by hand. Returns the
    reversed row, or None if the id is unknown / already undone."""
    row = conn.execute(
        "SELECT * FROM filing_actions WHERE id=? AND status='added'", (action_id,)
    ).fetchone()
    if not row:
        return None
    sp.playlist_remove_all_occurrences_of_items(row["playlist_id"], [_uri(row["track_id"])])
    conn.execute(
        "UPDATE filing_actions SET status='undone', undone_at=? WHERE id=?",
        (db.utcnow(), action_id),
    )
    conn.commit()
    return dict(row)


def history(
    conn: sqlite3.Connection, *, limit: int = 50, include_undone: bool = True
) -> list[dict]:
    """Recent filing actions, newest first — the ledger behind the undo UI."""
    where = "" if include_undone else "WHERE status='added'"
    return [
        dict(r)
        for r in conn.execute(
            f"SELECT * FROM filing_actions {where} ORDER BY added_at DESC, id DESC LIMIT ?",
            (limit,),
        )
    ]
