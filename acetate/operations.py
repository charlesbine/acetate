"""Read-only operational visibility for the chat agent.

The scheduler runs in a separate process from the web app, so it publishes a
small snapshot to ``kv_state``. Everything returned by ``system_status`` is
then read from SQLite through the chat agent's query-only connection.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from . import db

SCHEDULER_STATE_KEY = "scheduler_snapshot"


def _as_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _json_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _bounded_result(value: str | None, limit: int = 600) -> dict[str, Any]:
    parsed = _json_dict(value)
    if len(json.dumps(parsed, default=str)) <= limit:
        return parsed
    return {"summary": (value or "")[:limit], "truncated": True}


def record_scheduler_snapshot(
    scheduler,
    *,
    state: str,
    active_jobs: list[str] | None = None,
    poll_minutes: int = 15,
) -> None:
    """Persist the scheduler's actual next-fire times for other processes."""
    recorded_at = db.utcnow()
    scheduled_jobs = []
    for job in sorted(scheduler.get_jobs(), key=lambda item: item.id):
        next_run = getattr(job, "next_run_time", None)
        scheduled_jobs.append(
            {
                "job": job.id,
                "next_run_at": next_run.astimezone(timezone.utc).isoformat()
                if next_run is not None
                else None,
                "schedule": str(job.trigger),
            }
        )
    payload = {
        "state": state,
        "recorded_at": recorded_at,
        "active_jobs": sorted(active_jobs or []),
        "poll_minutes": poll_minutes,
        "jobs": scheduled_jobs,
    }
    conn = db.connect(cache_kib=2048)
    try:
        # A status write needs only a few pages. Full schema initialization
        # can scan the candidate ledgers and belongs on the bootstrap path.
        status_tables = conn.execute(
            "SELECT COUNT(*) FROM sqlite_schema "
            "WHERE type='table' AND name IN ('kv_state', 'job_run')"
        ).fetchone()[0]
        if status_tables != 2:
            db.init_db(conn)
        conn.execute(
            "INSERT INTO kv_state(key, value, updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (SCHEDULER_STATE_KEY, json.dumps(payload, sort_keys=True), recorded_at),
        )
        conn.commit()
    finally:
        conn.close()


def scheduler_status(conn, now: datetime | None = None) -> dict[str, Any]:
    """Read the scheduler's persisted timing without running the broader health report."""
    now = now or datetime.now(timezone.utc)
    last_poll = conn.execute(
        "SELECT started_at, finished_at, status FROM job_run "
        "WHERE job='poll' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    row = conn.execute(
        "SELECT value, updated_at FROM kv_state WHERE key=?",
        (SCHEDULER_STATE_KEY,),
    ).fetchone()
    if row is None:
        return {
            "state": "unknown",
            "reason": "no scheduler snapshot has been recorded",
            "recorded_at": None,
            "active_jobs": [],
            "next_poll_at": None,
            "last_poll_started_at": last_poll["started_at"] if last_poll else None,
            "last_poll_finished_at": last_poll["finished_at"] if last_poll else None,
            "last_poll_status": last_poll["status"] if last_poll else None,
            "jobs": [],
        }

    snapshot = _json_dict(row["value"])
    recorded_at = snapshot.get("recorded_at") or row["updated_at"]
    recorded = _as_utc(recorded_at)
    age_seconds = max(0, int((now - recorded).total_seconds())) if recorded else None
    poll_minutes = int(snapshot.get("poll_minutes") or 15)
    stale_after_seconds = max(300, poll_minutes * 60 * 2 + 60)
    reported_state = str(snapshot.get("state") or "unknown")
    state = (
        "stale"
        if reported_state == "running"
        and (age_seconds is None or age_seconds > stale_after_seconds)
        else reported_state
    )
    jobs = snapshot.get("jobs") if isinstance(snapshot.get("jobs"), list) else []
    next_poll = next(
        (
            item.get("next_run_at")
            for item in jobs
            if isinstance(item, dict) and item.get("job") == "poll"
        ),
        None,
    )
    return {
        "state": state,
        "reported_state": reported_state,
        "recorded_at": recorded_at,
        "snapshot_age_seconds": age_seconds,
        "stale_after_seconds": stale_after_seconds,
        "active_jobs": snapshot.get("active_jobs") or [],
        "next_poll_at": next_poll,
        "last_poll_started_at": last_poll["started_at"] if last_poll else None,
        "last_poll_finished_at": last_poll["finished_at"] if last_poll else None,
        "last_poll_status": last_poll["status"] if last_poll else None,
        "jobs": jobs,
    }


def _job_status(conn, recent_limit: int) -> dict[str, Any]:
    latest = conn.execute(
        """
        SELECT jr.job, jr.started_at, jr.finished_at, jr.status, jr.result, jr.error
        FROM job_run jr
        JOIN (
            SELECT job, MAX(id) AS id FROM job_run GROUP BY job
        ) last ON last.id = jr.id
        ORDER BY jr.job
        """
    ).fetchall()
    recent = conn.execute(
        "SELECT job, started_at, finished_at, status, result, error "
        "FROM job_run ORDER BY id DESC LIMIT ?",
        (recent_limit,),
    ).fetchall()
    counts = conn.execute(
        """
        SELECT job,
               SUM(status = 'ok') AS ok,
               SUM(status = 'error') AS error,
               COUNT(*) AS total
        FROM job_run
        WHERE julianday(started_at) >= julianday('now', '-7 days')
        GROUP BY job
        ORDER BY job
        """
    ).fetchall()

    def shape(row) -> dict[str, Any]:
        return {
            "job": row["job"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "status": row["status"],
            "result": _bounded_result(row["result"]),
            "error": row["error"][:600] if row["error"] else None,
        }

    return {
        "latest_by_job": [shape(row) for row in latest],
        "recent": [shape(row) for row in recent],
        "counts_7d": [dict(row) for row in counts],
    }


def _group_counts(conn, table: str, prefix: str = "") -> dict[str, int]:
    rows = conn.execute(
        f"SELECT status, COUNT(*) AS n FROM {table} GROUP BY status"  # noqa: S608
    )
    return {f"{prefix}{row['status']}": row["n"] for row in rows}


def _embedding_status(conn) -> dict[str, Any]:
    from . import embedding_requests

    required_models = ("discogs-effnet", "discogs400", "clap-music")
    universe = conn.execute(
        "SELECT COUNT(*) FROM recording_stats WHERE liked=1 OR plays>=5"
    ).fetchone()[0]
    model_rows = conn.execute(
        """
        SELECT e.model, COUNT(DISTINCT e.recording_key) AS n, MAX(e.updated_at) AS updated_at
        FROM embeddings e
        JOIN recording_stats rs USING(recording_key)
        WHERE rs.liked=1 OR rs.plays>=5
        GROUP BY e.model
        ORDER BY e.model
        """
    ).fetchall()
    observed_models = {row["model"]: dict(row) for row in model_rows}
    library_models = []
    for model in required_models:
        model_status = observed_models.pop(model, {"model": model, "n": 0, "updated_at": None})
        model_status["missing"] = max(0, universe - model_status["n"])
        model_status["percent"] = round(100 * model_status["n"] / universe, 1) if universe else 0.0
        library_models.append(model_status)
    library_models.extend(observed_models.values())

    points = conn.execute(
        "SELECT space, COUNT(*) AS n, MAX(updated_at) AS updated_at "
        "FROM hyperspace_points GROUP BY space ORDER BY space"
    ).fetchall()
    candidate_models = conn.execute(
        "SELECT model, status, COUNT(*) AS n, MAX(updated_at) AS updated_at "
        "FROM candidate_embedding_runs GROUP BY model, status ORDER BY model, status"
    ).fetchall()
    candidate_queue = []
    for model in required_models:
        eligible, embedded = conn.execute(
            """
            SELECT COUNT(*) AS eligible,
                   SUM(EXISTS(
                       SELECT 1 FROM embeddings e
                       WHERE e.recording_key=ca.candidate_key AND e.model=?
                   )) AS embedded
            FROM candidate_audio ca
            JOIN candidate_recordings cr USING(candidate_key)
            WHERE ca.status='fetched' AND cr.status IN ('preview', 'embedded')
            """,
            (model,),
        ).fetchone()
        candidate_queue.append(
            {
                "model": model,
                "eligible": eligible,
                "embedded": embedded or 0,
                "remaining": max(0, eligible - (embedded or 0)),
            }
        )
    latest_vector = conn.execute("SELECT MAX(updated_at) FROM embeddings").fetchone()[0]
    audio_counts = _group_counts(conn, "audio_cache")
    queued_audio = sum(audio_counts.values())
    return {
        "ad_hoc": embedding_requests.status(conn),
        "library_universe": universe,
        "library_audio": {
            "queued": queued_audio,
            "unenqueued": max(0, universe - queued_audio),
            "statuses": audio_counts,
        },
        "library_models": library_models,
        "hyperspace": [dict(row) for row in points],
        "latest_embedding_at": latest_vector,
        "candidate_audio": _group_counts(conn, "candidate_audio"),
        "candidate_queue": candidate_queue,
        "candidate_models": [dict(row) for row in candidate_models],
        "candidate_reranked": conn.execute("SELECT COUNT(*) FROM candidate_rerank").fetchone()[0],
    }


def _backfill_status(conn) -> dict[str, Any]:
    from . import genre

    genre_universe = genre.universe_count(conn)
    genre_tagged = len(genre._already_tagged(conn))
    ytmusic_universe = conn.execute(
        "SELECT COUNT(*) FROM recording_stats "
        "WHERE artist IS NOT NULL AND title IS NOT NULL AND (liked=1 OR plays>=5)"
    ).fetchone()[0]
    ytmusic = _group_counts(conn, "ytmusic_radio_seeds")
    ytmusic_attempted = sum(ytmusic.values())
    return {
        "genres": {
            "universe": genre_universe,
            "tagged": genre_tagged,
            "remaining": max(0, genre_universe - genre_tagged),
            "percent": round(100 * genre_tagged / genre_universe, 1) if genre_universe else 0.0,
        },
        "ytmusic_radio": {
            "universe": ytmusic_universe,
            "attempted": ytmusic_attempted,
            "remaining_unattempted": max(0, ytmusic_universe - ytmusic_attempted),
            "statuses": ytmusic,
        },
        "musicbrainz_ids": _group_counts(conn, "recording_mbid"),
        "candidate_resolution": {
            f"{row['provider']}:{row['status']}": row["n"]
            for row in conn.execute(
                "SELECT provider, status, COUNT(*) AS n FROM candidate_resolutions "
                "GROUP BY provider, status ORDER BY provider, status"
            )
        },
        "candidate_frontier": _group_counts(conn, "candidate_recordings"),
    }


def system_status(conn, *, recent_limit: int = 12) -> dict[str, Any]:
    """Return a bounded, read-only snapshot of Acetate's operational state."""
    recent_limit = max(1, min(int(recent_limit), 25))
    now = datetime.now(timezone.utc)
    return {
        "observed_at": now.isoformat(),
        "scheduler": scheduler_status(conn, now),
        "jobs": _job_status(conn, recent_limit),
        "embeddings": _embedding_status(conn),
        "backfills": _backfill_status(conn),
    }
