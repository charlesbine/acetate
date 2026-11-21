"""The "refresh my plays right now" path: the job itself, its run guard, the
admin-only chat tool, and the timezone conversion that happens above the model.
"""

from __future__ import annotations

import json
import threading

import pytest

from acetate import agent, db, jobs, timefmt, workflows


def _seed_play(conn, played_at: str, track_id: str, artist: str, title: str) -> None:
    conn.execute(
        "INSERT INTO play_events(played_at, track_id, source) VALUES(?,?,'recently_played')",
        (played_at, track_id),
    )
    conn.execute(
        "INSERT OR REPLACE INTO track_labels(track_id, track_name, artist_name, source, updated_at) "
        "VALUES(?,?,?,'test','2026-07-30T00:00:00+00:00')",
        (track_id, title, artist),
    )
    conn.commit()


@pytest.fixture
def live_db(conn, monkeypatch):
    """Point db.connect() at the test database so the job writes where we look."""
    path = conn.execute("PRAGMA database_list").fetchone()["file"]
    monkeypatch.setenv("ACETATE_DB", path)
    return conn


# --------------------------------------------------------------------------- #
#  The job
# --------------------------------------------------------------------------- #


def test_refresh_plays_reports_only_the_plays_it_just_learned(live_db, monkeypatch) -> None:
    _seed_play(live_db, "2026-07-30T01:00:00Z", "old1", "Prior Artist", "Prior Track")

    def fake_poll() -> dict:
        target = db.connect()
        _seed_play(target, "2026-07-30T02:00:00Z", "new1", "Fresh Artist", "Fresh Track")
        return {"fetched": 50, "inserted": 1}

    monkeypatch.setattr(jobs, "run_poll", fake_poll)

    res = jobs.refresh_plays()

    assert res["status"] == "ok"
    assert (res["fetched"], res["inserted"]) == (50, 1)
    assert [p["title"] for p in res["new_plays"]] == ["Fresh Track"]  # not the pre-existing play
    assert res["last_play_at"] == "2026-07-30T02:00:00Z"
    assert jobs.running_jobs() == []  # claim always released


def test_refresh_plays_records_the_run_in_the_audit_trail(live_db, monkeypatch) -> None:
    monkeypatch.setattr(jobs, "run_poll", lambda: {"fetched": 3, "inserted": 0})

    jobs.refresh_plays()

    row = live_db.execute("SELECT job, status FROM job_run ORDER BY id DESC LIMIT 1").fetchone()
    assert (row["job"], row["status"]) == ("poll", "ok")


def test_refresh_plays_surfaces_failure_without_stranding_the_claim(live_db, monkeypatch) -> None:
    def boom() -> dict:
        raise RuntimeError("spotify is down")

    monkeypatch.setattr(jobs, "run_poll", boom)

    res = jobs.refresh_plays()

    assert res["status"] == "error"
    assert "spotify is down" in res["error"]
    assert jobs.running_jobs() == []


def test_refresh_plays_refuses_to_overlap_a_poll_already_running(live_db, monkeypatch) -> None:
    started, may_finish = threading.Event(), threading.Event()

    def slow_poll() -> dict:
        started.set()
        may_finish.wait(timeout=5)
        return {"fetched": 1, "inserted": 0}

    monkeypatch.setattr(jobs, "run_poll", slow_poll)
    worker = threading.Thread(target=jobs.refresh_plays)
    worker.start()
    try:
        assert started.wait(timeout=5)
        busy = jobs.refresh_plays()
    finally:
        may_finish.set()
        worker.join(timeout=5)

    assert busy["status"] == "busy"
    assert jobs.running_jobs() == []


def test_run_guard_is_shared_with_the_web_trigger() -> None:
    assert jobs.try_claim("backup") is True
    try:
        assert jobs.try_claim("backup") is False  # second caller is turned away
        assert jobs.running_jobs() == ["backup"]  # and the status bar can see it
    finally:
        jobs.release("backup")
    assert jobs.try_claim("backup") is True
    jobs.release("backup")


# --------------------------------------------------------------------------- #
#  Admin gating
# --------------------------------------------------------------------------- #


def test_refresh_plays_is_registered_as_an_admin_only_workflow() -> None:
    workflow = workflows.BY_TOOL["refresh_plays"]

    assert workflow.admin_only is True
    assert {w.tool_name for w in workflows.WORKFLOWS if w.admin_only} == {
        "refresh_plays",
        "update_library_memory",
        "request_embeddings",
        "embedding_request_status",
    }


def test_non_admin_chat_turns_are_never_offered_the_tool() -> None:
    viewer = [tool["function"]["name"] for tool in agent.tools_for(can_admin=False)]
    admin = [tool["function"]["name"] for tool in agent.tools_for(can_admin=True)]

    assert "refresh_plays" not in viewer
    assert "refresh_plays" in admin
    assert viewer == [name for name in admin if name not in agent.ADMIN_TOOLS]


def test_non_admin_tool_call_is_refused_without_running_the_job(monkeypatch) -> None:
    called = False

    def spy() -> dict:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(jobs, "refresh_plays", spy)

    out, proposal = agent.exec_tool(None, "refresh_plays", {}, {}, can_admin=False)

    assert out.startswith("ERROR:")
    assert "administrator" in out
    assert proposal is None
    assert called is False


def test_admin_tool_call_runs_the_job_and_is_not_cached(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        jobs, "refresh_plays", lambda: calls.append(1) or {"status": "ok", "inserted": len(calls)}
    )
    seen: dict[str, str] = {}

    first, _ = agent.exec_tool(None, "refresh_plays", {}, seen, can_admin=True)
    second, _ = agent.exec_tool(None, "refresh_plays", {}, seen, can_admin=True)

    assert json.loads(first)["inserted"] == 1
    assert json.loads(second)["inserted"] == 2  # a re-check must actually re-check


def test_admin_only_tools_are_absent_from_a_viewer_system_prompt(conn) -> None:
    assert "refresh_plays" in agent.system_prompt(conn, can_admin=True)
    assert "refresh_plays" not in agent.system_prompt(conn, can_admin=False)


# --------------------------------------------------------------------------- #
#  Time, one step above the model
# --------------------------------------------------------------------------- #


def test_tool_results_reach_the_model_in_the_users_timezone(monkeypatch) -> None:
    monkeypatch.setenv("ACETATE_TZ", "America/New_York")
    monkeypatch.setattr(
        jobs,
        "refresh_plays",
        lambda: {
            "status": "ok",
            "new_plays": [{"played_at": "2026-07-30T02:15:00Z", "title": "Night Drive"}],
        },
    )

    out, _ = agent.exec_tool(None, "refresh_plays", {}, {}, can_admin=True)

    played_at = json.loads(out)["new_plays"][0]["played_at"]
    assert played_at == "2026-07-29T22:15:00-04:00"  # same instant, the user's clock


def test_localize_leaves_non_timestamps_alone() -> None:
    payload = {
        "month": "2026-07",
        "recording_key": "artist\x00title",
        "schedule": "cron[hour='*/2', minute='45']",
        "title": "Live at 2026-07-30T02:15:00Z",  # a timestamp inside prose, not a field
        "count": 12,
        "nested": [{"updated_at": "2026-07-30T02:15:00+00:00"}],
    }

    localized = timefmt.localize(payload)

    assert localized["month"] == "2026-07"
    assert localized["recording_key"] == payload["recording_key"]
    assert localized["schedule"] == payload["schedule"]
    assert localized["title"] == payload["title"]
    assert localized["count"] == 12
    assert localized["nested"][0]["updated_at"] == "2026-07-29T22:15:00-04:00"


def test_display_timezone_defaults_and_survives_a_bad_value(monkeypatch) -> None:
    monkeypatch.delenv("ACETATE_TZ", raising=False)
    assert timefmt.tz_name() == "America/New_York"

    monkeypatch.setenv("ACETATE_TZ", "Europe/Paris")
    assert timefmt.local_iso("2026-07-30T02:15:00Z") == "2026-07-30T04:15:00+02:00"

    monkeypatch.setenv("ACETATE_TZ", "Not/AZone")
    assert timefmt.local_iso("2026-07-30T02:15:00Z") == "2026-07-29T22:15:00-04:00"


def test_error_payloads_pass_through_localization_untouched() -> None:
    assert timefmt.localize_json_text("ERROR: no such track") == "ERROR: no such track"
