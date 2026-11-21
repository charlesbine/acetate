"""Small, durable requests to fill missing audio embeddings for known tracks.

Chat only enqueues; the embedding sidecar fetches previews and runs its existing
isolated workers. No shell commands, URLs, model choices, or concurrency knobs
come from model arguments. This module stays compatible with sidecar Python 3.11.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone

from . import canon, db

MODELS = ("discogs-effnet", "discogs400", "clap-music")
MAX_TRACKS = 5
MAX_PENDING = 20
MAX_ATTEMPTS = 3
STAGE_TIMEOUT = 600


class PreviewUnavailable(LookupError):
    """The source chain completed without a matching preview."""


def resolve(conn: sqlite3.Connection, query: str) -> dict:
    """Resolve exact known identities only: writes must never use a fuzzy guess."""
    key = query.strip()
    spotify = re.fullmatch(
        r"(?:https://open\.spotify\.com/track/|spotify:track:)?([A-Za-z0-9]{22})(?:\?[^\s]*)?",
        key,
    )
    if spotify:
        rows = conn.execute(
            "SELECT recording_key FROM track_recording WHERE track_id=? UNION "
            "SELECT candidate_key FROM candidate_external_ids "
            "WHERE provider='spotify' AND external_id=?",
            (spotify[1], spotify[1]),
        ).fetchall()
        if len(rows) != 1:
            raise ValueError("Spotify track must identify one known recording")
        key = rows[0][0]
    elif "\t" not in key:
        artist, sep, title = key.replace(" — ", " - ").partition(" - ")
        if not sep or not artist.strip() or not title.strip():
            raise ValueError(
                "use an exact Artist - Title, recording key, or known Spotify track URL"
            )
        key = canon.recording_key(artist, title)
    row = conn.execute(
        "SELECT recording_key, artist, title, 'library' AS kind "
        "FROM recording_stats WHERE recording_key=?",
        (key,),
    ).fetchone()
    if row is None:
        row = conn.execute(
            "SELECT candidate_key AS recording_key, artist, title, 'candidate' AS kind "
            "FROM candidate_recordings WHERE candidate_key=? "
            "AND status NOT IN ('duplicate','dismissed')",
            (key,),
        ).fetchone()
    if row is None:
        raise ValueError(
            "no exact known recording; inspect the track identity before requesting work"
        )
    return dict(row)


def missing_models(conn: sqlite3.Connection, key: str) -> list[str]:
    available = {
        r[0] for r in conn.execute("SELECT model FROM embeddings WHERE recording_key=?", (key,))
    }
    return [model for model in MODELS if model not in available]


def describe(conn: sqlite3.Connection, row: sqlite3.Row | dict) -> dict:
    result = dict(row)
    result["missing_models"] = missing_models(conn, row["recording_key"])
    # A background job may have finished this track ahead of the request worker.
    if not result["missing_models"]:
        result["status"] = "available"
        result["error"] = None
    elif result["status"] == "available":
        result["status"] = "error"
        result["error"] = "previously available vectors are missing; request an explicit retry"
    return result


def enqueue(tracks: list[str], *, requested_by: str) -> dict:
    if (
        not isinstance(tracks, list)
        or not 1 <= len(tracks) <= MAX_TRACKS
        or any(not isinstance(q, str) or not q.strip() or len(q) > 500 for q in tracks)
    ):
        raise ValueError(
            f"supply 1–{MAX_TRACKS} exact track identities, each at most 500 characters"
        )
    conn = db.connect()
    try:
        # Resolve the whole batch before writing anything. Capacity and dedupe
        # are then checked under a SQLite write reservation across web processes.
        resolved = {}
        for query in tracks:
            track = resolve(conn, query)
            resolved[track["recording_key"]] = track
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE embedding_requests SET status='available', error=NULL, updated_at=? "
            "WHERE status IN ('queued','running') AND "
            "(SELECT COUNT(*) FROM embeddings e WHERE e.recording_key=embedding_requests.recording_key "
            "AND e.model IN ('discogs-effnet','discogs400','clap-music'))=3",
            (db.utcnow(),),
        )
        pending = conn.execute(
            "SELECT COUNT(*) FROM embedding_requests WHERE status IN ('queued','running')"
        ).fetchone()[0]
        results = []
        for key, track in resolved.items():
            row = conn.execute(
                "SELECT * FROM embedding_requests WHERE recording_key=?", (key,)
            ).fetchone()
            missing = missing_models(conn, key)
            if not missing:
                results.append({**track, "status": "available", "missing_models": []})
                continue
            if row and row["status"] in ("queued", "running"):
                results.append(describe(conn, row))
                continue
            if row and row["attempts"] >= MAX_ATTEMPTS:
                results.append({**describe(conn, row), "status": "attempt_limit"})
                continue
            if pending >= MAX_PENDING:
                results.append({**track, "status": "queue_full", "missing_models": missing})
                continue
            now = db.utcnow()
            conn.execute(
                """
                INSERT INTO embedding_requests(
                    recording_key, kind, artist, title, requested_by, status, requested_at, updated_at
                ) VALUES(?,?,?,?,?,'queued',?,?)
                ON CONFLICT(recording_key) DO UPDATE SET
                    kind=excluded.kind, requested_by=excluded.requested_by, status='queued',
                    requested_at=excluded.requested_at, updated_at=excluded.updated_at, error=NULL
                """,
                (key, track["kind"], track["artist"], track["title"], requested_by, now, now),
            )
            pending += 1
            results.append(
                describe(
                    conn,
                    conn.execute(
                        "SELECT * FROM embedding_requests WHERE recording_key=?", (key,)
                    ).fetchone(),
                )
            )
        conn.commit()
        return {
            "requests": results,
            "outstanding": pending,
            "limits": limits(),
            "worker": worker_status(conn),
            "note": "Queued work is asynchronous. Check embedding_request_status; queued is not completed.",
        }
    finally:
        conn.close()


def limits() -> dict:
    return {
        "tracks_per_turn": MAX_TRACKS,
        "outstanding_tracks": MAX_PENDING,
        "active_requests": 1,
        "inference_processes": 1,
        "attempts_per_track": MAX_ATTEMPTS,
        "stage_timeout_seconds": STAGE_TIMEOUT,
    }


def worker_status(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        "SELECT value, updated_at FROM kv_state WHERE key='embedding_worker'"
    ).fetchone()
    if row is None:
        return {"state": "unknown", "updated_at": None}
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(row["updated_at"])).total_seconds()
    stale_after = STAGE_TIMEOUT * 2 + 60 if row["value"] in ("running", "backfill") else 30
    return {
        "state": "stale" if age > stale_after else row["value"],
        "updated_at": row["updated_at"],
        "age_seconds": max(0, int(age)),
    }


def status(conn: sqlite3.Connection, request_ids: list[int] | None = None) -> dict:
    if request_ids is not None:
        if (
            not isinstance(request_ids, list)
            or not 1 <= len(request_ids) <= MAX_PENDING
            or any(type(i) is not int or i < 1 for i in request_ids)
        ):
            raise ValueError(f"request_ids must contain 1–{MAX_PENDING} positive integer ids")
        placeholders = ",".join("?" * len(request_ids))
        rows = conn.execute(
            f"SELECT * FROM embedding_requests WHERE id IN ({placeholders}) ORDER BY id",
            request_ids,
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM embedding_requests ORDER BY requested_at DESC, id DESC LIMIT ?",
            (MAX_PENDING,),
        ).fetchall()
    return {
        "requests": [describe(conn, row) for row in rows],
        "limits": limits(),
        "worker": worker_status(conn),
    }


def require_eligible_candidate(conn: sqlite3.Connection, key: str) -> None:
    row = conn.execute(
        "SELECT status FROM candidate_recordings WHERE candidate_key=?", (key,)
    ).fetchone()
    if row is None or row["status"] in ("duplicate", "dismissed"):
        state = row["status"] if row else "missing"
        raise ValueError(f"candidate is {state}; embedding request stopped")


def prepare_audio(conn: sqlite3.Connection, row: sqlite3.Row) -> None:
    """Use the existing preview source chain for exactly one requested recording."""
    from . import audio

    candidate = row["kind"] == "candidate"
    table, key_column = (
        ("candidate_audio", "candidate_key") if candidate else ("audio_cache", "recording_key")
    )
    key = row["recording_key"]
    if candidate:
        require_eligible_candidate(conn, key)
    cached = conn.execute(
        f"SELECT status, path FROM {table} WHERE {key_column}=?", (key,)
    ).fetchone()
    if cached and cached["status"] == "fetched" and cached["path"]:
        if (db.db_path().parent / cached["path"]).is_file():
            if candidate:
                conn.execute(
                    "UPDATE candidate_recordings SET status='preview',updated_at=? "
                    "WHERE candidate_key=? AND status IN ('metadata','resolved')",
                    (db.utcnow(), key),
                )
                conn.commit()
            return
    if candidate:
        metadata = conn.execute(
            "SELECT isrc, COALESCE(expected_duration_s, matched_duration_s)*1000 AS duration_ms "
            "FROM candidate_resolutions WHERE candidate_key=? AND provider='deezer'",
            (key,),
        ).fetchone()
    else:
        metadata = conn.execute(
            "SELECT t.isrc, t.duration_ms FROM track_recording tr "
            "JOIN tracks t USING(track_id) WHERE tr.recording_key=? "
            "ORDER BY (t.isrc IS NOT NULL AND t.isrc!='') DESC LIMIT 1",
            (key,),
        ).fetchone()
    rec = {
        **dict(row),
        "isrc": metadata["isrc"] if metadata else None,
        "duration_ms": metadata["duration_ms"] if metadata else None,
    }
    state, source, source_id, duration_s, path = audio._fetch_one(rec, require_identity=candidate)
    if candidate:
        # A resolver or review may exclude this candidate during download.
        # Reserve the writer before rechecking and keep promotion atomic.
        conn.execute("BEGIN IMMEDIATE")
        try:
            require_eligible_candidate(conn, key)
        except ValueError:
            conn.rollback()
            raise
    now = db.utcnow()
    conn.execute(
        f"INSERT INTO {table}({key_column},status,source,source_id,duration_s,path,attempts,updated_at) "
        f"VALUES(?,?,?,?,?,?,1,?) ON CONFLICT({key_column}) DO UPDATE SET "
        "status=excluded.status, source=excluded.source, source_id=excluded.source_id, "
        f"duration_s=excluded.duration_s,path=excluded.path,attempts={table}.attempts+1, "
        "error=NULL,updated_at=excluded.updated_at",
        (
            key,
            "source_not_covered" if candidate and state == "miss" else state,
            source or None,
            source_id,
            duration_s,
            path,
            now,
        ),
    )
    if state == "fetched":
        ledger = "candidate_embedding_runs" if candidate else "embedding_runs"
        for model in missing_models(conn, key):
            conn.execute(
                f"INSERT INTO {ledger}({key_column},model,status,attempts,updated_at) "
                f"VALUES(?,?,'pending',0,?) ON CONFLICT({key_column},model) DO UPDATE SET "
                "status='pending',error=NULL,updated_at=excluded.updated_at",
                (key, model, now),
            )
        if candidate:
            conn.execute(
                "UPDATE candidate_recordings SET status='preview',updated_at=? "
                "WHERE candidate_key=? AND status IN ('metadata','resolved')",
                (now, key),
            )
    conn.commit()
    if state != "fetched":
        raise PreviewUnavailable("source_not_covered: no matching preview is available")
