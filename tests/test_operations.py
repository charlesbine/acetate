from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from acetate import agent, db, operations


def test_scheduler_snapshot_records_actual_next_runs(conn, monkeypatch) -> None:
    db_path = conn.execute("PRAGMA database_list").fetchone()["file"]
    monkeypatch.setenv("ACETATE_DB", db_path)

    def reject_initialization(_conn):
        raise AssertionError("An existing database does not need to be initialized for a snapshot")

    monkeypatch.setattr(db, "init_db", reject_initialization)
    next_poll = datetime.now(timezone.utc) + timedelta(minutes=12)
    scheduler = SimpleNamespace(
        get_jobs=lambda: [
            SimpleNamespace(
                id="poll",
                next_run_time=next_poll,
                trigger="interval[0:15:00]",
            ),
            SimpleNamespace(
                id="backup",
                next_run_time=next_poll + timedelta(hours=5),
                trigger="cron[hour='4', minute='0']",
            ),
        ]
    )

    operations.record_scheduler_snapshot(
        scheduler,
        state="running",
        active_jobs=["backup"],
        poll_minutes=15,
    )

    row = conn.execute(
        "SELECT value FROM kv_state WHERE key=?",
        (operations.SCHEDULER_STATE_KEY,),
    ).fetchone()
    snapshot = json.loads(row["value"])
    assert snapshot["state"] == "running"
    assert snapshot["active_jobs"] == ["backup"]
    assert snapshot["jobs"][1]["job"] == "poll"
    assert snapshot["jobs"][1]["next_run_at"] == next_poll.isoformat()


def test_scheduler_snapshot_bootstraps_an_empty_database(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ACETATE_DB", str(tmp_path / "acetate.db"))
    scheduler = SimpleNamespace(get_jobs=lambda: [])
    operations.record_scheduler_snapshot(scheduler, state="running")

    conn = db.connect()
    try:
        assert operations.scheduler_status(conn)["state"] == "running"
    finally:
        conn.close()


def test_system_status_combines_jobs_embeddings_and_backfills(conn) -> None:
    now = datetime.now(timezone.utc)
    scheduler_snapshot = {
        "state": "running",
        "recorded_at": now.isoformat(),
        "active_jobs": [],
        "poll_minutes": 15,
        "jobs": [
            {
                "job": "poll",
                "next_run_at": (now + timedelta(minutes=10)).isoformat(),
                "schedule": "interval[0:15:00]",
            }
        ],
    }
    conn.execute(
        "INSERT INTO kv_state(key, value, updated_at) VALUES(?,?,?)",
        (operations.SCHEDULER_STATE_KEY, json.dumps(scheduler_snapshot), now.isoformat()),
    )
    conn.execute(
        "INSERT INTO job_run(job, started_at, finished_at, status, result) VALUES(?,?,?,?,?)",
        ("poll", now.isoformat(), now.isoformat(), "ok", '{"fetched": 10, "inserted": 2}'),
    )
    conn.execute(
        "INSERT INTO recording_stats(recording_key, artist, title, plays, liked, homed) "
        "VALUES('library-track', 'Artist', 'Track', 7, 1, 0)"
    )
    conn.execute(
        "INSERT INTO embeddings(recording_key, model, dim, vector, updated_at) "
        "VALUES('library-track', 'discogs-effnet', 1, X'00000000', ?)",
        (now.isoformat(),),
    )
    conn.execute(
        "INSERT INTO audio_cache(recording_key, status, updated_at) "
        "VALUES('library-track', 'fetched', ?)",
        (now.isoformat(),),
    )
    conn.execute(
        "INSERT INTO ytmusic_radio_seeds(recording_key, status, updated_at) "
        "VALUES('library-track', 'fetched', ?)",
        (now.isoformat(),),
    )
    conn.execute(
        "INSERT INTO recording_mbid(recording_key, status, updated_at) "
        "VALUES('library-track', 'resolved', ?)",
        (now.isoformat(),),
    )
    conn.commit()

    status = operations.system_status(conn)

    assert status["scheduler"]["state"] == "running"
    assert status["scheduler"]["next_poll_at"] == scheduler_snapshot["jobs"][0]["next_run_at"]
    assert status["scheduler"]["last_poll_status"] == "ok"
    assert status["jobs"]["latest_by_job"][0]["result"] == {"fetched": 10, "inserted": 2}
    assert status["embeddings"]["library_universe"] == 1
    assert status["embeddings"]["library_audio"]["statuses"] == {"fetched": 1}
    assert status["embeddings"]["library_models"][0]["model"] == "discogs-effnet"
    assert status["embeddings"]["library_models"][0]["percent"] == 100.0
    assert status["backfills"]["ytmusic_radio"]["remaining_unattempted"] == 0
    assert status["backfills"]["musicbrainz_ids"] == {"resolved": 1}


def test_scheduler_snapshot_becomes_stale(conn) -> None:
    old = datetime.now(timezone.utc) - timedelta(hours=1)
    conn.execute(
        "INSERT INTO kv_state(key, value, updated_at) VALUES(?,?,?)",
        (
            operations.SCHEDULER_STATE_KEY,
            json.dumps(
                {
                    "state": "running",
                    "recorded_at": old.isoformat(),
                    "poll_minutes": 15,
                    "jobs": [],
                }
            ),
            old.isoformat(),
        ),
    )
    conn.commit()

    status = operations.system_status(conn)

    assert status["scheduler"]["state"] == "stale"
    assert status["scheduler"]["reported_state"] == "running"


def test_chat_system_status_tool_returns_valid_read_only_json(conn) -> None:
    result, proposal = agent.exec_tool(
        conn,
        "system_status",
        {"recent_jobs": 5},
        {},
    )

    payload = json.loads(result)
    assert proposal is None
    assert payload["scheduler"]["state"] == "unknown"
    assert "embeddings" in payload
    assert "backfills" in payload
