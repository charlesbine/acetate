"""Similar-track retrieval — the seed of the autoplaylist engine.

Two complementary notions of "similar", both surfaced side by side:
  sonic         cosine neighbors in the discogs-effnet embedding space
                (sounds alike; full coverage of whatever we've embedded)
  collaborative Last.fm track.getSimilar (people who play X play Y; carries
                the "recommendation feel" content embeddings can't)

At library scale (~10^4 vectors) a brute-force numpy dot IS the ANN index —
no FAISS needed. The retrieve→rerank engine will combine both spaces.
"""

from __future__ import annotations

import sqlite3
import time
import json

import numpy as np
import requests

from . import db, lastfm
from .canon import recording_key as _rkey

EMBED_MODEL = "discogs-effnet"

# Last.fm etiquette: ≤5 req/s per IP averaged; we make 2 calls per track and
# pace the loop so a full-library cache run stays well under it.
_CACHE_INTERVAL_S = 0.5
MAX_ATTEMPTS = 3


def _start_run(conn: sqlite3.Connection, recording_key: str, surface: str) -> int:
    conn.execute(
        """
        INSERT INTO lastfm_track_runs(recording_key, surface, status, attempts, updated_at)
        VALUES(?, ?, 'pending', 1, ?)
        ON CONFLICT(recording_key, surface) DO UPDATE SET
            status='pending', attempts=lastfm_track_runs.attempts + 1,
            error=NULL, detail_json=NULL, updated_at=excluded.updated_at
    """,
        (recording_key, surface, db.utcnow()),
    )
    conn.commit()
    return conn.execute(
        "SELECT attempts FROM lastfm_track_runs WHERE recording_key=? AND surface=?",
        (recording_key, surface),
    ).fetchone()[0]


def _finish_run(
    conn: sqlite3.Connection,
    recording_key: str,
    surface: str,
    status: str,
    *,
    detail: dict | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE lastfm_track_runs SET status=?, error=?, detail_json=?, updated_at=?
        WHERE recording_key=? AND surface=?
    """,
        (
            status,
            error[:300] if error else None,
            json.dumps(detail, sort_keys=True) if detail else None,
            db.utcnow(),
            recording_key,
            surface,
        ),
    )


def _surface_done(conn: sqlite3.Connection, recording_key: str, surface: str) -> bool:
    row = conn.execute(
        "SELECT status, attempts FROM lastfm_track_runs WHERE recording_key=? AND surface=?",
        (recording_key, surface),
    ).fetchone()
    return bool(
        row
        and (
            row["status"] in ("available", "provider_miss", "unknown_legacy")
            or (row["status"] == "error" and row["attempts"] >= MAX_ATTEMPTS)
        )
    )


def find_recording(conn: sqlite3.Connection, query: str) -> sqlite3.Row | None:
    """Resolve a free-text query ('artist - title' or a fragment of either)
    to the most-played matching recording."""
    if " - " in query:
        artist, title = (p.strip() for p in query.split(" - ", 1))
        row = conn.execute(
            "SELECT * FROM recording_stats WHERE artist LIKE ? AND title LIKE ? "
            "ORDER BY plays DESC LIMIT 1",
            (f"%{artist}%", f"%{title}%"),
        ).fetchone()
        if row:
            return row
    return conn.execute(
        "SELECT * FROM recording_stats WHERE artist LIKE ? OR title LIKE ? "
        "ORDER BY plays DESC LIMIT 1",
        (f"%{query}%", f"%{query}%"),
    ).fetchone()


# (row_count, keys, matrix) per model — reload only when embeddings grow.
_MAT_CACHE: dict[str, tuple[int, list[str], np.ndarray]] = {}


def _matrix(conn: sqlite3.Connection, model: str = EMBED_MODEL) -> tuple[list[str], np.ndarray]:
    n = conn.execute(
        "SELECT COUNT(*) FROM embeddings e JOIN recording_stats rs USING(recording_key) "
        "WHERE e.model = ?",
        (model,),
    ).fetchone()[0]
    cached = _MAT_CACHE.get(model)
    if cached and cached[0] == n:
        return cached[1], cached[2]
    rows = conn.execute(
        "SELECT e.recording_key, e.dim, e.vector FROM embeddings e "
        "JOIN recording_stats rs USING(recording_key) WHERE e.model = ?",
        (model,),
    ).fetchall()
    keys = [r["recording_key"] for r in rows]
    mat = np.vstack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    _MAT_CACHE[model] = (n, keys, mat)
    return keys, mat


def sonic_neighbors(
    conn: sqlite3.Connection, recording_key: str, k: int = 20, model: str = EMBED_MODEL
) -> list[dict]:
    """Top-k cosine neighbors of a recording in an embedding space."""
    keys, mat = _matrix(conn, model)
    try:
        idx = keys.index(recording_key)
    except ValueError:
        return []  # seed not embedded (yet)
    sims = mat @ mat[idx]
    order = np.argsort(-sims)
    out = []
    for j in order[: k + 1]:
        if keys[j] == recording_key:
            continue
        r = conn.execute(
            "SELECT rs.artist, rs.title, rs.plays, rs.liked, rs.rep_track_id, "
            "(SELECT style FROM recording_styles s WHERE s.recording_key = rs.recording_key "
            " AND s.rank = 1) AS style "
            "FROM recording_stats rs WHERE rs.recording_key = ?",
            (keys[j],),
        ).fetchone()
        if r:
            out.append(
                {
                    "artist": r["artist"],
                    "title": r["title"],
                    "plays": r["plays"],
                    "liked": bool(r["liked"]),
                    "style": r["style"],
                    "track_id": r["rep_track_id"],
                    "score": float(sims[j]),
                }
            )
        if len(out) >= k:
            break
    return out


def cache_lastfm(
    conn: sqlite3.Connection, batch: int | None = None, min_plays: int = 5, progress=None
) -> dict:
    """Persist the Last.fm co-listening graph + global stats over the universe
    (liked ∪ ≥min_plays), most-played first. Resumable: a lastfm_info row marks
    a recording as fetched. ~2 API calls/track, paced ~4 req/s → full run ≈ 1.5h."""
    candidates = conn.execute(
        """
        SELECT rs.recording_key, rs.artist, rs.title
        FROM recording_stats rs
        WHERE (rs.liked = 1 OR rs.plays >= ?) AND rs.artist IS NOT NULL
        ORDER BY rs.plays DESC
    """,
        (min_plays,),
    ).fetchall()
    todo = [
        r
        for r in candidates
        if not (
            _surface_done(conn, r["recording_key"], "info")
            and _surface_done(conn, r["recording_key"], "similar")
        )
    ]
    if batch:
        todo = todo[:batch]

    counts = {"fetched": 0, "miss": 0, "error": 0, "edges": 0, "in_library": 0}
    for i, rec in enumerate(todo):
        t0 = time.monotonic()
        now = db.utcnow()
        for surface in ("similar", "info"):
            if _surface_done(conn, rec["recording_key"], surface):
                continue
            attempts = _start_run(conn, rec["recording_key"], surface)
            try:
                if surface == "similar":
                    edges = lastfm.track_similar(rec["artist"], rec["title"], limit=50)
                    conn.execute(
                        "DELETE FROM lastfm_similar WHERE recording_key=?", (rec["recording_key"],)
                    )
                    for sim_artist, sim_title, match in edges:
                        target = conn.execute(
                            "SELECT recording_key FROM recording_stats WHERE recording_key = ?",
                            (_rkey(sim_artist, sim_title),),
                        ).fetchone()
                        conn.execute(
                            "INSERT INTO lastfm_similar(recording_key, sim_artist, sim_title, "
                            "sim_recording_key, match, updated_at) VALUES(?,?,?,?,?,?)",
                            (
                                rec["recording_key"],
                                sim_artist,
                                sim_title,
                                target["recording_key"] if target else None,
                                match,
                                now,
                            ),
                        )
                        counts["edges"] += 1
                        counts["in_library"] += int(target is not None)
                    _finish_run(
                        conn,
                        rec["recording_key"],
                        surface,
                        "available",
                        detail={"result_count": len(edges)},
                    )
                else:
                    info = lastfm.track_info(rec["artist"], rec["title"])
                    conn.execute(
                        "INSERT OR REPLACE INTO lastfm_info(recording_key, listeners, playcount, "
                        "updated_at) VALUES(?,?,?,?)",
                        (
                            rec["recording_key"],
                            (info or {}).get("listeners"),
                            (info or {}).get("playcount"),
                            now,
                        ),
                    )
                    _finish_run(
                        conn,
                        rec["recording_key"],
                        surface,
                        "available",
                        detail={"has_stats": info is not None},
                    )
                counts["fetched"] += 1
            except lastfm.LastfmAPIError as exc:
                if exc.code == 6:
                    _finish_run(
                        conn,
                        rec["recording_key"],
                        surface,
                        "provider_miss",
                        detail={"provider_code": 6},
                    )
                    counts["miss"] += 1
                else:
                    status = "error" if attempts >= MAX_ATTEMPTS else "pending"
                    _finish_run(conn, rec["recording_key"], surface, status, error=str(exc))
                    counts["error"] += 1
            except requests.RequestException as exc:
                status = "error" if attempts >= MAX_ATTEMPTS else "pending"
                _finish_run(conn, rec["recording_key"], surface, status, error=str(exc))
                counts["error"] += 1
        conn.commit()  # per-item: never hold the WAL write lock across API calls
        wait = _CACHE_INTERVAL_S - (time.monotonic() - t0)
        if wait > 0:
            time.sleep(wait)
        if i % 50 == 0 and progress:
            progress(i, len(todo), counts)
    counts["remaining"] = max(0, len(todo) - counts["fetched"]) if batch else 0
    return counts


def collaborative(conn: sqlite3.Connection, artist: str, title: str, limit: int = 20) -> list[dict]:
    """Last.fm co-listening neighbors, marked with in-library membership."""
    out = []
    for sim_artist, sim_title, match in lastfm.track_similar(artist, title, limit=limit):
        row = conn.execute(
            "SELECT plays, liked, rep_track_id FROM recording_stats WHERE recording_key = ?",
            (_rkey(sim_artist, sim_title),),
        ).fetchone()
        out.append(
            {
                "artist": sim_artist,
                "title": sim_title,
                "score": match,
                "in_library": row is not None,
                "plays": row["plays"] if row else 0,
                "liked": bool(row["liked"]) if row else False,
                "track_id": row["rep_track_id"] if row else None,
            }
        )
    return out
