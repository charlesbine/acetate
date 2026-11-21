from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from acetate import agent, auth, db, jobs, playlists, smart, workflows
from tests.conftest import FakeSpotify, add_recording


@pytest.fixture
def collection(conn, monkeypatch):
    monkeypatch.setenv("ACETATE_DB", conn.execute("PRAGMA database_list").fetchone()["file"])
    monkeypatch.setenv("ACETATE_AUTH_MODE", "local")
    monkeypatch.setattr(auth, "load_credentials", lambda: None)
    from acetate import web

    add_recording(conn, "artist\tone", "Artist", "One", "one")
    add_recording(conn, "artist\ttwo", "Artist", "Two", "two")
    conn.commit()
    rule = smart.create(conn, "Daily rule", {"liked": True, "limit": 2})
    curated = smart.create(conn, "Night drive selection", {"track_ids": ["two", "one"]})
    sp = FakeSpotify()
    monkeypatch.setattr(jobs, "_client", lambda: sp)
    return TestClient(web.app), sp, rule, curated


def test_collections_have_distinct_cards_and_controls(collection, conn):
    client, _, rule, curated = collection
    rules = client.get("/smart")
    selections = client.get("/curated")

    assert rules.status_code == selections.status_code == 200
    assert "Daily rule" in rules.text
    assert "Night drive selection" not in rules.text
    assert f'id="smart-{rule}"' in rules.text
    assert "Refreshes daily" in rules.text
    assert ">pause</button>" in rules.text
    assert "new rule" in rules.text

    assert "Night drive selection" in selections.text
    assert "Daily rule" not in selections.text
    assert f'id="curated-{curated}"' in selections.text
    assert "Fixed selection · ready to publish" in selections.text
    assert ">publish to Spotify</button>" in selections.text
    assert ">pause</button>" not in selections.text
    assert ">refresh now</button>" not in selections.text
    assert "new rule" not in selections.text


def test_scheduled_job_refreshes_rules_but_publishes_curations_once(collection, conn):
    _, sp, rule, curated = collection
    assert jobs.run_smart()["built"] == 2
    mapping = {
        row["key"]: row["playlist_id"]
        for row in conn.execute("SELECT key, playlist_id FROM managed_playlists WHERE kind='smart'")
    }
    curated_pid, rule_pid = mapping[str(curated)], mapping[str(rule)]
    assert sp.replaced[curated_pid] == ["spotify:track:two", "spotify:track:one"]
    published = smart.get(conn, curated)["published_at"]
    assert published

    # Spotify edits survive both later daily jobs and stale manual refresh links.
    sp.replaced[curated_pid] = ["spotify:track:edited-on-spotify"]
    sp.replaced.pop(rule_pid)
    assert jobs.run_smart()["built"] == 1
    assert rule_pid in sp.replaced
    assert sp.replaced[curated_pid] == ["spotify:track:edited-on-spotify"]
    assert smart.get(conn, curated)["published_at"] == published
    assert jobs.run_smart(curated)["built"] == 0
    assert sp.replaced[curated_pid] == ["spotify:track:edited-on-spotify"]
    assert len(sp.created) == 2


def test_published_curations_show_open_without_republishing(collection, conn, monkeypatch):
    client, _, _, curated = collection
    jobs.run_smart(curated)

    def unexpected_publish(*args, **kwargs):
        pytest.fail("a published fixed selection must not be submitted for refresh")

    monkeypatch.setattr(jobs, "run_smart", unexpected_publish)
    page = client.get("/curated")
    assert "Fixed selection · published" in page.text
    assert f'href="/playlists/saved/{curated}">view playlist →</a>' in page.text
    assert ">publish to Spotify</button>" not in page.text
    assert "already published" in client.post(f"/smart/{curated}/run").text
    assert client.post(f"/smart/{curated}/toggle").status_code == 400


def test_publish_endpoint_returns_open_link_and_is_idempotent(collection, conn):
    client, sp, rule, curated = collection
    result = client.post(f"/curated/{curated}/publish", follow_redirects=False)
    assert result.status_code == 200
    assert result.json() == {"url": f"/playlists/saved/{curated}"}
    assert "location" not in result.headers
    assert smart.get(conn, curated)["published_at"]
    sp.replaced.clear()
    assert client.post(f"/curated/{curated}/publish").json() == result.json()
    assert not sp.replaced
    assert len(sp.created) == 1
    assert client.post(f"/curated/{rule}/publish").status_code == 404


def test_publish_endpoint_can_retry_a_provider_failure(collection, conn, monkeypatch):
    client, sp, _, curated = collection
    replace = sp.playlist_replace_items

    def fail(*args, **kwargs):
        raise RuntimeError("Provider unavailable")

    monkeypatch.setattr(sp, "playlist_replace_items", fail)
    failed = client.post(f"/curated/{curated}/publish")
    assert failed.status_code == 502
    assert smart.get(conn, curated)["published_at"] is None
    page = client.get("/curated")
    assert "data-publish=" in page.text
    assert ">open ↗</a>" not in page.text

    monkeypatch.setattr(sp, "playlist_replace_items", replace)
    assert client.post(f"/curated/{curated}/publish").status_code == 200
    assert smart.get(conn, curated)["published_at"]
    assert len(sp.created) == 1


@pytest.mark.parametrize("groups", ["friends", "acetate-query"])
def test_curated_publication_requires_admin(collection, monkeypatch, groups):
    client, sp, _, curated = collection
    monkeypatch.setenv("ACETATE_AUTH_MODE", "authelia")
    headers = {"Remote-User": "friend", "Remote-Groups": groups}
    assert client.post(f"/curated/{curated}/publish", headers=headers).status_code == 403
    assert "data-publish=" not in client.get("/curated", headers=headers).text
    assert not sp.created


def test_failed_first_publish_retries_same_playlist_then_stops(collection, conn):
    _, _, _, curated = collection
    sp = FakeSpotify()
    stale = smart.get(conn, curated)
    replace = sp.playlist_replace_items

    def fail(playlist_id, uris):
        raise RuntimeError("Spotify unavailable")

    sp.playlist_replace_items = fail
    with pytest.raises(RuntimeError, match="Spotify unavailable"):
        playlists.build_smart(conn, sp, stale)
    assert len(sp.created) == 1
    assert smart.get(conn, curated)["published_at"] is None

    # Startup must not mistake a partially created playlist for a completed publish.
    db.init_db(conn)
    assert smart.get(conn, curated)["published_at"] is None
    sp.playlist_replace_items = replace
    result = playlists.build_smart(conn, sp, stale)
    assert result["playlist_id"] == sp.created[0]["id"]
    assert smart.get(conn, curated)["published_at"]
    assert len(sp.created) == 1
    sp.replaced.clear()
    assert playlists.build_smart(conn, sp, stale)["skipped"] is True
    assert not sp.replaced


def test_upgrade_preserves_existing_curated_spotify_links(collection, conn):
    client, _, rule, curated = collection
    conn.execute("ALTER TABLE smart_playlists DROP COLUMN published_at")
    conn.execute(
        "INSERT INTO managed_playlists(kind, key, playlist_id, name) VALUES('smart',?,?,?)",
        (str(curated), "existing-curation", "Night drive selection"),
    )
    conn.commit()

    db.init_db(conn)

    assert smart.get(conn, curated)["published_at"]
    assert smart.get(conn, rule)["published_at"] is None
    page = client.get("/curated")
    assert "https://open.spotify.com/playlist/existing-curation" in page.text
    assert ">publish to Spotify</button>" not in page.text


def test_create_and_remove_return_to_the_correct_collection(collection):
    client, _, _, _ = collection
    for spec, destination in [({"track_ids": ["one"]}, "/curated"), ({"liked": True}, "/smart")]:
        response = client.post(
            "/smart/create", data={"name": "New", "spec": json.dumps(spec)}, follow_redirects=False
        )
        assert response.headers["location"] == destination

    # Removing a saved selection is local; it never invokes the Spotify client.
    _, _, _, curated = collection
    removed = client.post(f"/smart/{curated}/delete", follow_redirects=False)
    assert removed.headers["location"] == "/curated"


def test_paused_rule_stays_paused(collection, conn):
    _, sp, rule, curated = collection
    smart.update(conn, rule, enabled=False)
    jobs.run_smart()
    assert len(sp.created) == 1
    assert sp.created[0]["name"] == smart.get(conn, curated)["name"]


def test_chat_contract_describes_both_playlist_lifecycles(collection, conn):
    tool = next(
        tool["function"]
        for tool in workflows.agent_tools(can_admin=False)
        if tool["function"]["name"] == "propose_playlist"
    )
    assert tool["parameters"]["required"] == ["name", "spec"]
    assert "fixed curated selection" in tool["description"]
    assert "refreshes daily" in tool["description"]
    assert "published once without automatic updates" in agent.system_prompt(conn)


def test_collection_cards_escape_saved_names_and_track_metadata(collection, conn):
    client, _, _, _ = collection
    payload = '<img src=x onerror="alert(1)">'
    add_recording(conn, "unsafe\tname", payload, payload, "unsafe")
    smart.create(conn, payload, {"track_ids": ["unsafe"]})
    smart.create(conn, payload, {"genres": [payload]})
    for path in ("/curated", "/smart"):
        page = client.get(path)
        assert page.status_code == 200
        assert payload not in page.text
        assert "&lt;img src=x onerror=&#34;alert(1)&#34;&gt;" in page.text


def test_concurrent_first_publications_create_and_write_only_one_playlist(collection, conn):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Lock

    _, sp, _, curated = collection
    entered, release, second_create = Event(), Event(), Event()
    count_lock = Lock()
    calls = 0
    post = sp._post

    def slow_create(*args, **kwargs):
        nonlocal calls
        with count_lock:
            calls += 1
            first = calls == 1
        if first:
            entered.set()
            assert release.wait(5)
        else:
            second_create.set()
        return post(*args, **kwargs)

    sp._post = slow_create

    def publish():
        writer = db.connect()
        try:
            return playlists.build_smart(writer, sp, smart.get(writer, curated))
        finally:
            writer.close()

    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(publish)
        assert entered.wait(5)
        second = workers.submit(publish)
        try:
            second_create.wait(0.5)
        finally:
            release.set()
        results = [first.result(timeout=5), second.result(timeout=5)]
    assert len(sp.created) == 1
    assert results[0]["playlist_id"] == results[1]["playlist_id"]
    assert sorted(bool(result.get("skipped")) for result in results) == [False, True]
    assert len(sp.replaced) == 1


def test_empty_curated_selection_never_falls_back_to_library_matches(collection, conn):
    client, sp, _, _ = collection
    spec = {"track_ids": []}
    assert smart.evaluate(conn, {})  # The library has tracks a default rule would select.
    assert smart.evaluate(conn, spec) == []
    assert smart.describe(spec) == "0 hand-picked tracks"
    sid = smart.create(conn, "Empty selection", spec)
    result = client.post(f"/curated/{sid}/publish")
    assert result.status_code == 400
    assert sp.created == []
    assert not smart.get(conn, sid)["published_at"]
