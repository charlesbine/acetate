from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from fastapi.testclient import TestClient

from acetate import agent, auth, chatstore, db, smart


@pytest.fixture
def client(conn, monkeypatch):
    monkeypatch.setenv("ACETATE_DB", conn.execute("PRAGMA database_list").fetchone()["file"])
    monkeypatch.setenv("ACETATE_AUTH_MODE", "local")
    monkeypatch.setattr(auth, "load_credentials", lambda: None)
    from acetate import web

    return TestClient(web.app)


def _proposal(**overrides):
    return {
        "name": "Dark Club",
        "spec": {"track_ids": ["one", "two"], "limit": 20},
        "summary": "2 hand-picked tracks",
        "matches": 2,
        **overrides,
    }


def _message(conn, *proposals):
    thread = chatstore.create_thread(conn)
    mid = chatstore.add_message(
        conn,
        thread["id"],
        "assistant",
        "Here is your playlist.",
        meta={"proposals": list(proposals), "model": "example"},
    )
    return thread["id"], mid


def _approve_url(mid, index=0):
    return f"/chat/messages/{mid}/proposals/{index}/approve"


def test_approval_saves_in_place_and_survives_reopening(client, conn):
    thread, mid = _message(conn, _proposal())
    before = client.get(f"/chat?thread={thread}")
    assert f'action="{_approve_url(mid)}" data-approve' in before.text

    result = client.post(_approve_url(mid), follow_redirects=False)

    assert result.status_code == 200
    sid = smart.list_all(conn)[0]["id"]
    assert result.json() == {"url": f"/playlists/saved/{sid}"}
    assert "location" not in result.headers
    stored = chatstore.messages(conn, thread)[0]["meta"]
    assert stored["proposals"][0]["saved_playlist_id"] == sid
    assert stored["model"] == "example"

    # A later rename or edit must not make an approved card forget its rule.
    smart.update(conn, sid, name="Renamed", spec={"track_ids": ["three"]})
    reopened = client.get(f"/chat?thread={thread}")
    assert 'class="proposal saved"' in reopened.text
    assert "role=status>✓ Saved</span>" in reopened.text
    assert f'href="/playlists/saved/{sid}" target=_blank rel=noopener>Open ↗</a>' in reopened.text
    assert "<button type=submit class=approve>" not in reopened.text
    assert f'id="curated-{sid}"' in client.get("/curated").text
    assert client.post(_approve_url(mid)).json() == result.json()
    assert len(smart.list_all(conn)) == 1


def test_concurrent_approvals_create_one_rule(client, conn):
    _, mid = _message(conn, _proposal())
    path = db.db_path()
    barrier = Barrier(2)

    def approve():
        connection = db.connect(path)
        try:
            barrier.wait(timeout=5)
            return chatstore.approve_playlist_proposal(connection, mid, 0)
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        ids = list(executor.map(lambda _: approve(), range(2)))

    assert ids[0] == ids[1]
    assert len(smart.list_all(conn)) == 1


def test_legacy_approval_matches_name_and_spec_not_just_name(client, conn):
    sid = smart.create(conn, "Dark Club", {"limit": 20, "track_ids": ["one", "two"]})
    thread, mid = _message(conn, _proposal(), _proposal(spec={"track_ids": ["other"]}))

    page = client.get(f"/chat?thread={thread}")

    assert page.text.count('class="proposal saved"') == 1
    assert f'action="{_approve_url(mid, 1)}" data-approve' in page.text
    assert client.post(_approve_url(mid)).json() == {"url": f"/playlists/saved/{sid}"}
    assert len(smart.list_all(conn)) == 1
    assert client.post(_approve_url(mid, 1)).status_code == 200
    assert len(smart.list_all(conn)) == 2


def test_open_stays_in_app_after_sync(client, conn):
    thread, mid = _message(conn, _proposal())
    client.post(_approve_url(mid))
    sid = smart.list_all(conn)[0]["id"]
    conn.execute(
        "INSERT INTO managed_playlists(kind, key, playlist_id, name) VALUES('smart',?,?,?)",
        (str(sid), "synced-playlist", "Dark Club"),
    )
    conn.execute("UPDATE smart_playlists SET published_at=? WHERE id=?", (db.utcnow(), sid))
    conn.commit()

    expected = f"/playlists/saved/{sid}"
    assert client.post(_approve_url(mid)).json() == {"url": expected}
    assert (
        f'href="{expected}" target=_blank rel=noopener>Open ↗</a>'
        in client.get(f"/chat?thread={thread}").text
    )


def test_incomplete_curated_publish_keeps_open_pointing_to_the_saved_selection(client, conn):
    _, mid = _message(conn, _proposal())
    client.post(_approve_url(mid))
    sid = smart.list_all(conn)[0]["id"]
    conn.execute(
        "INSERT INTO managed_playlists(kind, key, playlist_id, name) VALUES('smart',?,?,?)",
        (str(sid), "incomplete-playlist", "Dark Club"),
    )
    conn.commit()
    assert client.post(_approve_url(mid)).json() == {"url": f"/playlists/saved/{sid}"}


def test_deleted_rule_can_be_saved_again_explicitly(client, conn):
    thread, mid = _message(conn, _proposal())
    client.post(_approve_url(mid))
    smart.delete(conn, smart.list_all(conn)[0]["id"])

    page = client.get(f"/chat?thread={thread}")
    assert f'action="{_approve_url(mid)}" data-approve' in page.text
    assert smart.list_all(conn) == []
    assert client.post(_approve_url(mid)).status_code == 200
    assert len(smart.list_all(conn)) == 1


def test_deleted_rule_id_is_not_confused_with_an_unrelated_rule(client, conn):
    thread, mid = _message(conn, _proposal())
    client.post(_approve_url(mid))
    original_id = smart.list_all(conn)[0]["id"]
    smart.delete(conn, original_id)
    reused_id = smart.create(conn, "Unrelated", {"track_ids": ["other"]})
    assert reused_id == original_id

    page = client.get(f"/chat?thread={thread}")
    assert f'action="{_approve_url(mid)}" data-approve' in page.text
    assert client.post(_approve_url(mid)).json() != {"url": f"/playlists/saved/{reused_id}"}
    assert len(smart.list_all(conn)) == 2


@pytest.mark.parametrize("index", [-1, 1, 99])
def test_unknown_proposal_does_not_create_a_rule(client, conn, index):
    _, mid = _message(conn, _proposal())
    assert client.post(_approve_url(mid, index)).status_code == 404
    assert client.post(_approve_url(mid + 999)).status_code == 404
    assert smart.list_all(conn) == []


@pytest.mark.parametrize("overrides", [{"name": " "}, {"spec": []}])
def test_invalid_stored_proposal_is_rejected(client, conn, overrides):
    _, mid = _message(conn, _proposal(**overrides))
    assert client.post(_approve_url(mid)).status_code == 400
    assert smart.list_all(conn) == []


@pytest.mark.parametrize("groups", ["friends", "acetate-query"])
def test_approval_requires_admin(client, conn, monkeypatch, groups):
    thread, mid = _message(conn, _proposal())
    monkeypatch.setenv("ACETATE_AUTH_MODE", "authelia")
    headers = {"Remote-User": "friend", "Remote-Groups": groups}
    assert client.post(_approve_url(mid), headers=headers).status_code == 403
    assert (
        "<button type=submit class=approve>"
        not in client.get(f"/chat?thread={thread}", headers=headers).text
    )
    assert smart.list_all(conn) == []


def test_failed_save_rolls_back_and_allows_retry(client, conn, monkeypatch):
    thread, mid = _message(conn, _proposal())
    create = smart.create

    def fail_after_insert(connection, name, spec, **kwargs):
        create(connection, name, spec, **kwargs)
        raise RuntimeError("Save interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(smart, "create", fail_after_insert)
        with pytest.raises(RuntimeError, match="Save interrupted"):
            chatstore.approve_playlist_proposal(conn, mid, 0)

    assert smart.list_all(conn) == []
    assert "saved_playlist_id" not in chatstore.messages(conn, thread)[0]["meta"]["proposals"][0]
    assert client.post(_approve_url(mid)).status_code == 200


def test_streamed_proposal_has_a_persisted_approval_target(client, conn, monkeypatch):
    from acetate import web

    def fake_stream(messages, **kwargs):
        yield {"t": "token", "v": "Here is your playlist."}
        yield {"t": "proposal", "v": _proposal()}
        yield {"t": "done", "v": {"model": "example", "context": {"summarized": True}}}

    monkeypatch.setattr(web.agent, "chat_stream", fake_stream)
    response = client.post("/chat/stream", data={"q": "Make me a playlist"})
    events = [json.loads(line) for line in response.text.splitlines()]
    done = [event["v"] for event in events if event["t"] == "done"]
    assert len(done) == 1
    assert done[0]["context"] == {"summarized": True}
    proposal = done[0]["proposals"][0]
    assert proposal["kind"] == "curated"
    assert proposal["url"] is None
    assert client.post(proposal["save_url"]).status_code == 200
    assert len(smart.list_all(conn)) == 1
    assistant = conn.execute("SELECT meta FROM chat_messages WHERE role='assistant'").fetchone()
    assert json.loads(assistant["meta"])["proposals"][0]["saved_playlist_id"] is not None
    assert json.loads(assistant["meta"])["context"] == {"summarized": True}
    assert "earlier messages summarized" in client.get("/chat").text


def test_rule_approval_stays_in_the_smart_collection(client, conn):
    thread, mid = _message(conn, _proposal(spec={"liked": True, "limit": 20}))
    response = client.post(_approve_url(mid))
    sid = smart.list_all(conn)[0]["id"]
    assert response.json() == {"url": f"/playlists/saved/{sid}"}
    page = client.get(f"/chat?thread={thread}")
    assert "Smart rule · refreshes daily" in page.text
    assert f'id="smart-{sid}"' in client.get("/smart").text


def test_errored_stream_does_not_offer_an_unsaved_proposal(client, conn, monkeypatch):
    from acetate import web

    def fake_stream(messages, **kwargs):
        yield {"t": "proposal", "v": _proposal()}
        yield {"t": "error", "v": "Try again"}

    monkeypatch.setattr(web.agent, "chat_stream", fake_stream)
    response = client.post("/chat/stream", data={"q": "Make me a playlist"})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert not any(event["t"] == "done" for event in events)
    assert conn.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0] == 0


@pytest.mark.parametrize(
    "spec",
    [
        None,
        {"track_ids": None},
        {"track_ids": "one"},
        {"track_ids": [["one"]]},
        {"min_plays": "many"},
    ],
)
def test_invalid_specs_are_not_proposed_or_saved(client, conn, spec):
    out, proposal = agent.exec_tool(conn, "propose_playlist", {"name": "Invalid", "spec": spec}, {})
    assert out.startswith("ERROR:")
    assert proposal is None
    thread, mid = _message(conn, _proposal(spec=spec))
    response = client.post(_approve_url(mid))
    assert response.status_code == 400
    manual = client.post("/smart/create", data={"name": "Invalid", "spec": json.dumps(spec)})
    assert manual.status_code == 400
    assert smart.list_all(conn) == []
    stored = chatstore.messages(conn, thread)[0]["meta"]["proposals"][0]
    assert "saved_playlist_id" not in stored
