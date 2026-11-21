import sqlite3
from contextlib import closing

import pytest

from acetate import db, jobs, scheduler


def test_scheduled_job_runs_in_child_and_records_result(tmp_path, monkeypatch):
    path = tmp_path / "scheduler.sqlite"
    monkeypatch.setenv("ACETATE_DB", str(path))
    with closing(db.connect()) as conn:
        db.init_db(conn)

    scheduler._run_job("canon")

    with closing(db.connect()) as conn:
        row = conn.execute("SELECT job, status FROM job_run").fetchone()
        assert tuple(row) == ("canon", "ok")


def test_backfill_closes_connection_on_failure(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "jobs.sqlite")
    monkeypatch.setattr(db, "connect", lambda: conn)

    def fail(*_args, **_kwargs):
        raise RuntimeError("test job failed")

    monkeypatch.setattr(jobs.genre, "tag_backfill", fail)
    with pytest.raises(RuntimeError, match="test job failed"):
        jobs.run_backfill()
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        conn.execute("SELECT 1")
