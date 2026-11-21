from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from acetate import auth, chatstore, db, operations, playlist_activity, smart, timefmt
from tests.conftest import add_playlist, add_recording


def _proposal(name="Night drive", spec=None):
    return {
        "name": name,
        "spec": spec if spec is not None else {"track_ids": ["a", "b"]},
        "summary": "A playlist proposal",
        "matches": 2,
    }


def _message(conn, proposals, title="A listening conversation"):
    thread = chatstore.create_thread(conn)
    chatstore.add_message(conn, thread["id"], "user", title)
    mid = chatstore.add_message(
        conn, thread["id"], "assistant", "Try these.", meta={"proposals": proposals}
    )
    return thread["id"], mid


def _scheduler(conn, *, state="running", minutes_old=0, next_minutes=20, active=False):
    now = datetime.now(timezone.utc)
    next_at = (now + timedelta(minutes=next_minutes)).isoformat()
    recorded = (now - timedelta(minutes=minutes_old)).isoformat()
    payload = {
        "state": state,
        "recorded_at": recorded,
        "poll_minutes": 15,
        "active_jobs": ["smart"] if active else [],
        "jobs": [
            {"job": "smart", "next_run_at": next_at},
            {"job": "monthly", "next_run_at": next_at},
            {"job": "poll", "next_run_at": next_at},
        ],
    }
    conn.execute(
        "INSERT OR REPLACE INTO kv_state(key, value, updated_at) VALUES(?,?,?)",
        (operations.SCHEDULER_STATE_KEY, json.dumps(payload), recorded),
    )
    conn.commit()
    return next_at


@pytest.fixture
def client(conn, monkeypatch):
    monkeypatch.setenv("ACETATE_DB", conn.execute("PRAGMA database_list").fetchone()["file"])
    monkeypatch.setenv("ACETATE_AUTH_MODE", "local")
    monkeypatch.setattr(auth, "load_credentials", lambda: None)
    from acetate import web

    return TestClient(web.app)


def test_drafts_are_distinct_unapproved_proposals_with_original_indices(conn):
    saved = _proposal("Already saved")
    _message(conn, [_proposal("Same proposal", {"liked": True, "limit": 20})])
    thread, mid = _message(
        conn, [saved, _proposal(), _proposal("Same proposal", {"limit": 20, "liked": True})]
    )
    chatstore.approve_playlist_proposal(conn, mid, 0)
    conn.execute("UPDATE chat_threads SET archived=1 WHERE id=?", (thread,))
    conn.execute(
        "INSERT INTO chat_messages(thread_id, role, meta) VALUES(?, 'assistant', '{')", (thread,)
    )
    conn.commit()
    changes = conn.total_changes

    drafts = chatstore.playlist_drafts(conn)

    assert [(draft["name"], draft["message_id"], draft["index"]) for draft in drafts] == [
        ("Night drive", mid, 1),
        ("Same proposal", mid, 2),
    ]
    assert conn.total_changes == changes
    assert chatstore.get_thread(conn, thread)["archived"] == 1


def test_legacy_saved_and_removed_approved_selections_do_not_return_as_drafts(conn):
    smart.create(conn, "Legacy", {"track_ids": ["a", "b"]})
    _, mid = _message(conn, [_proposal("Legacy"), _proposal("Removed")])
    sid = chatstore.approve_playlist_proposal(conn, mid, 1)
    smart.delete(conn, sid)
    assert chatstore.playlist_drafts(conn) == []


def test_drafts_approve_in_place_and_keep_the_conversation_link(client, conn):
    _, mid = _message(
        conn, [_proposal("Saved"), _proposal("Draft <img src=x>")], title="Source <img src=x>"
    )
    chatstore.approve_playlist_proposal(conn, mid, 0)
    page = client.get("/curated?view=drafts")
    assert page.status_code == 200
    assert "Playlist Drafts" in page.text
    assert f'action="/chat/messages/{mid}/proposals/1/approve" data-approve' in page.text
    assert "Draft &lt;img src=x&gt;" in page.text
    assert "Source &lt;img src=x&gt;" in page.text
    assert "<img src=x>" not in page.text
    assert "bindPlaylistApprovals(draftRoot)" in page.text
    assert client.post(f"/chat/messages/{mid}/proposals/1/approve").status_code == 200
    assert chatstore.playlist_drafts(conn) == []
    assert "No playlist drafts awaiting approval" in client.get("/curated?view=drafts").text


def test_viewers_can_read_drafts_but_cannot_approve(client, conn, monkeypatch):
    _, mid = _message(conn, [_proposal()])
    monkeypatch.setenv("ACETATE_AUTH_MODE", "authelia")
    headers = {"Remote-User": "friend", "Remote-Groups": "friends"}
    page = client.get("/curated?view=drafts", headers=headers)
    assert page.status_code == 200
    assert "Night drive" in page.text
    assert '<form method=post action="/chat/messages/' not in page.text
    assert (
        client.post(f"/chat/messages/{mid}/proposals/0/approve", headers=headers).status_code == 403
    )


def test_schedule_uses_reported_times_and_excludes_finished_curations(conn):
    rule = smart.create(conn, "Daily", {"liked": True})
    paused = smart.create(conn, "Paused", {"played": True})
    smart.update(conn, paused, enabled=False)
    smart.create(conn, "First publish", {"track_ids": ["a"]})
    published = smart.create(conn, "Published", {"track_ids": ["b"]})
    conn.execute("UPDATE smart_playlists SET published_at=? WHERE id=?", (db.utcnow(), published))
    conn.commit()
    next_at = _scheduler(conn)

    result = playlist_activity.scheduled_updates(conn)

    assert {job["job"] for job in result["jobs"]} == {"smart", "monthly"}
    items = {item["name"]: item for item in result["items"]}
    assert set(items) == {"Daily", "Paused", "First publish"}
    assert items["Daily"]["next_run_at"] == next_at
    assert items["First publish"]["next_run_at"] == next_at
    assert items["First publish"]["status"] == "First publication"
    assert items["Daily"]["url"] == f"/playlists/saved/{rule}"
    assert items["Paused"]["next_run_at"] is None
    assert items["Paused"]["status"] == "Paused"


@pytest.mark.parametrize(
    "state, minutes_old, next_minutes",
    [("running", 60, 20), ("stopped", 0, 20), ("running", 0, -5)],
)
def test_unconfirmed_schedule_never_presents_a_next_run(conn, state, minutes_old, next_minutes):
    smart.create(conn, "Daily", {"liked": True})
    _scheduler(conn, state=state, minutes_old=minutes_old, next_minutes=next_minutes)
    result = playlist_activity.scheduled_updates(conn)
    assert all(job["next_run_at"] is None for job in result["jobs"])
    assert all(item["next_run_at"] is None for item in result["items"])


def test_schedule_renders_unknown_and_actual_running_state(client, conn):
    smart.create(conn, "Daily", {"liked": True})
    unknown = client.get("/curated?view=scheduled")
    assert unknown.status_code == 200
    assert "Scheduler timing is unknown" in unknown.text
    assert "Next run not confirmed" in unknown.text

    next_at = _scheduler(conn)
    page = client.get("/curated?view=scheduled")
    assert timefmt.to_local(next_at, "%Y-%m-%d %H:%M") in page.text
    _scheduler(conn, active=True)
    assert "Update batch running" in client.get("/curated?view=scheduled").text


def test_history_counts_duplicate_occurrences_without_inventing_baseline_additions(conn):
    for tid in ("a", "b", "c"):
        add_recording(conn, f"artist\t{tid}", "Artist", tid.upper(), tid)
    add_playlist(conn, "P", "Selection", ["a", "a", "b"], backup_time="2026-10-20T04:00:00Z")
    add_playlist(conn, "P", "Selection", ["b", "a", "c", "c"], backup_time="2026-10-21T04:00:00Z")
    conn.commit()

    latest, baseline = playlist_activity.snapshot_history(conn)["items"]

    assert latest["added_count"] == 2
    assert latest["removed_count"] == 1
    assert latest["added"] == [{"track_id": "c", "title": "C", "artist": "Artist", "count": 2}]
    assert latest["removed"][0]["track_id"] == "a"
    assert baseline["baseline"] is True
    assert baseline["added_count"] == baseline["removed_count"] == 0


def test_history_distinguishes_reordering_and_unavailable_items(conn):
    add_playlist(conn, "P", "Order", ["a", "b"], backup_time="2026-10-20T04:00:00Z")
    add_playlist(conn, "P", "Order", ["b", "a"], backup_time="2026-10-21T04:00:00Z")
    add_playlist(conn, "Q", "Unavailable", ["a", None], backup_time="2026-10-20T04:00:00Z")
    add_playlist(conn, "Q", "Unavailable", ["a", None, None], backup_time="2026-10-21T04:00:00Z")
    conn.commit()
    changes = {
        item["playlist_id"]: item
        for item in playlist_activity.snapshot_history(conn)["items"]
        if not item["baseline"]
    }
    assert changes["P"]["reordered"] is True
    assert changes["P"]["added_count"] == changes["P"]["removed_count"] == 0
    assert changes["Q"]["unknown_before"] == 1
    assert changes["Q"]["unknown_after"] == 2
    assert changes["Q"]["added_count"] == 0
    assert changes["Q"]["reordered"] is False


def test_incomplete_membership_is_not_interpreted_as_removing_every_track(conn):
    add_playlist(conn, "P", "Incomplete", ["a", "b"], backup_time="2026-10-20T04:00:00Z")
    conn.execute(
        "INSERT INTO playlist_snapshots(playlist_id, name, backup_time, total_tracks) "
        "VALUES('P','Incomplete','2026-10-21T04:00:00Z',2)"
    )
    conn.commit()
    latest = playlist_activity.snapshot_history(conn)["items"][0]
    assert latest["complete"] is False
    assert latest["added_count"] == latest["removed_count"] == 0


def test_history_pages_keep_the_previous_comparison_and_escape_metadata(client, conn):
    add_playlist(conn, "P", "Name <img src=x>", ["a"], backup_time="2026-10-20T04:00:00Z")
    add_playlist(conn, "P", "Name <img src=x>", ["a", "b"], backup_time="2026-10-21T04:00:00Z")
    conn.commit()
    first = playlist_activity.snapshot_history(conn, page_size=1)
    second = playlist_activity.snapshot_history(conn, page=2, page_size=1)
    assert first["has_more"] is True
    assert first["items"][0]["previous_at"] == "2026-10-20T04:00:00Z"
    assert first["items"][0]["added_count"] == 1
    assert second["items"][0]["baseline"] is True
    assert second["has_more"] is False
    page = client.get("/curated?view=history")
    assert page.status_code == 200
    assert "Name &lt;img src=x&gt;" in page.text
    assert "<img src=x>" not in page.text
    assert "+1 added" in page.text
    assert "not exact edit times" in page.text


def test_workspace_read_models_work_on_a_query_only_connection(conn):
    _message(conn, [_proposal()])
    conn.execute("PRAGMA query_only=ON")
    assert len(chatstore.playlist_drafts(conn)) == 1
    assert playlist_activity.scheduled_updates(conn)["scheduler"]["state"] == "unknown"
    assert playlist_activity.snapshot_history(conn)["items"] == []


@pytest.mark.parametrize("path", ["/curated?view=missing", "/smart?view=history&page=0"])
def test_invalid_workspace_views_are_rejected(client, path):
    assert client.get(path).status_code == 404
