from __future__ import annotations

import json
import re
from html import unescape
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from acetate import auth, chatstore, playlist_activity, smart
from tests.conftest import add_playlist, add_recording


@pytest.fixture
def client(conn, monkeypatch):
    monkeypatch.setenv("ACETATE_DB", conn.execute("PRAGMA database_list").fetchone()["file"])
    monkeypatch.setenv("ACETATE_AUTH_MODE", "local")
    monkeypatch.setattr(auth, "load_credentials", lambda: None)
    from acetate import web

    return TestClient(web.app)


def _managed(conn, pid, *, sid=None, kind="monthly", key="2026-10"):
    conn.execute(
        "INSERT INTO managed_playlists(kind,key,playlist_id,name) VALUES(?,?,?,?)",
        ("smart" if sid else kind, str(sid) if sid else key, pid, "Managed selection"),
    )
    conn.commit()


def test_all_playlists_includes_followed_managed_and_unpublished_without_duplicates(client, conn):
    add_playlist(conn, "owned", "My playlist", ["a"])
    add_playlist(conn, "followed", "Followed playlist", ["a"], owner="someone-else")
    add_playlist(conn, "linked", "Spotify name", ["a"])
    sid = smart.create(conn, "Saved name", {"track_ids": ["a"]})
    pending = smart.create(conn, "Unpublished rule", {"liked": True})
    _managed(conn, "linked", sid=sid)
    _managed(conn, "new-managed")
    page = client.get("/playlists")
    assert page.status_code == 200
    assert "5 playlists" in page.text
    assert 'href="/playlists/owned"' in page.text
    assert 'href="/playlists/followed"' in page.text
    assert 'href="/playlists/new-managed"' in page.text
    assert f'href="/playlists/saved/{sid}"' in page.text
    assert f'href="/playlists/saved/{pending}"' in page.text
    assert "Spotify name" not in page.text


def test_catalog_search_and_pagination_keep_every_playlist_reachable(client, conn):
    for i in range(62):
        add_playlist(conn, f"p{i}", f"Night {i:02}", [])
    conn.commit()
    page = client.get("/playlists?q=Night&page=2")
    assert "62 playlists" in page.text
    assert 'href="/playlists/p61"' in page.text
    assert 'href="/playlists/p0"' not in page.text
    assert "/playlists?q=Night&amp;page=1" in page.text
    assert "1 playlist matching" in client.get("/playlists?q=Night%2061").text


def test_detail_preserves_order_duplicates_and_local_snapshot_labels(client, conn):
    add_recording(conn, "artist\ta", "Artist", "A", "a")
    add_playlist(conn, "P", "Name <img src=x>", ["a", None, "a", "raw"])
    conn.execute(
        "UPDATE playlist_track_snapshots SET raw_json=? WHERE track_id IS NULL",
        (json.dumps({"track": {"name": "Local <song>", "artists": [{"name": "Local artist"}]}}),),
    )
    conn.execute(
        "UPDATE playlist_track_snapshots SET raw_json=? WHERE track_id='raw'",
        (json.dumps({"track": {"name": "From backup", "artists": [{"name": "New artist"}]}}),),
    )
    conn.commit()
    page = client.get("/playlists/P")
    assert page.status_code == 200
    assert "Name &lt;img src=x&gt;" in page.text
    assert "<img src=x>" not in page.text
    assert "Local &lt;song&gt;" in page.text
    assert "Local artist" in page.text
    assert "From backup" in page.text
    assert "unavailable or local" in page.text
    assert 'href="/curated?view=history&amp;playlist_id=P"' in page.text
    detail = playlist_activity.detail_tracks(conn, playlist_activity.playlist_detail(conn, "P"))
    assert [track["track_id"] for track in detail["tracks"]] == ["a", None, "a", "raw"]
    assert [track["rank"] for track in detail["tracks"]] == [1, 2, 3, 4]


def test_detail_uses_latest_empty_snapshot_instead_of_stale_membership(client, conn):
    add_playlist(conn, "P", "Old name", ["a"], backup_time="2026-10-20")
    add_playlist(conn, "P", "Now empty", [], backup_time="2026-10-21")
    conn.commit()
    page = client.get("/playlists/P")
    assert "Now empty" in page.text
    assert "tracks · 0" in page.text
    assert "This playlist has no tracks right now" in page.text
    assert "open.spotify.com/track/a" not in page.text


def test_detail_pages_full_track_list_beyond_preview_and_keeps_positions(client, conn):
    add_playlist(conn, "P", "Long playlist", [f"track-{i}" for i in range(101)])
    conn.commit()
    first = client.get("/playlists/P")
    second = client.get("/playlists/P?page=2")
    assert 'href="?page=2"' in first.text
    assert "track-99" in first.text and "track-100" not in first.text
    assert "track-100" in second.text
    assert "class=rank>101</span>" in second.text
    assert 'href="?page=1"' in second.text


def test_missing_snapshot_details_are_explicit(client, conn):
    add_playlist(conn, "P", "Incomplete", ["a", "b", "c"])
    conn.execute("DELETE FROM playlist_track_snapshots WHERE track_id='b'")
    conn.commit()
    page = client.get("/playlists/P")
    assert "2 of 3 track details were recorded" in page.text
    assert "class=rank>3</span>" in page.text
    assert "This playlist has no tracks" not in page.text


def test_saved_selections_and_current_rule_matches_are_viewable_before_publication(client, conn):
    add_recording(conn, "artist\ta", "Artist", "Known title", "a")
    sid = smart.create(conn, "Fixed", {"track_ids": ["a", "unknown"]})
    rule = smart.create(conn, "Daily", {"liked": True})
    page = client.get(f"/playlists/saved/{sid}")
    assert page.status_code == 200
    assert "Saved fixed selection" in page.text
    assert "Known title" in page.text and "unknown" in page.text
    assert "History starts when" in page.text
    assert "Open in Spotify" not in page.text
    assert "Current rule matches" in client.get(f"/playlists/saved/{rule}").text


def test_published_saved_page_uses_observed_spotify_edits_not_original_selection(
    client, conn, monkeypatch
):
    sid = smart.create(conn, "Saved selection", {"track_ids": ["original"]})
    _managed(conn, "P", sid=sid)
    add_playlist(conn, "P", "Spotify name", ["edited", "edited"])
    conn.commit()

    def no_evaluation(*args, **kwargs):
        pytest.fail("an observed playlist must not be replaced by its original rule")

    monkeypatch.setattr(smart, "evaluate", no_evaluation)
    page = client.get(f"/playlists/saved/{sid}")
    assert "Last recorded Spotify track list" in page.text
    assert page.text.count('href="https://open.spotify.com/track/edited"') == 2
    assert "open.spotify.com/track/original" not in page.text
    assert "View history" in page.text and "Open in Spotify" in page.text
    redirect = client.get("/playlists/P?page=2", follow_redirects=False)
    assert redirect.headers["location"] == f"/playlists/saved/{sid}?page=2"


def test_managed_page_distinguishes_missing_track_list_from_recorded_membership(client, conn):
    _managed(conn, "P")
    assert "No track list has been recorded yet" in client.get("/playlists/P").text
    conn.execute(
        "INSERT INTO managed_playlist_tracks(kind,key,track_id,currently_in) VALUES('monthly','2026-10','a',1)"
    )
    conn.commit()
    page = client.get("/playlists/P")
    assert "Spotify order is not recorded" in page.text
    assert "open.spotify.com/track/a" in page.text


def test_playlist_history_is_filtered_and_keeps_filter_across_pages(client, conn):
    for i in range(22):
        add_playlist(conn, "P", "Selected playlist", ["a"], backup_time=f"2026-10-{i + 1:02}")
    add_playlist(conn, "Q", "Unrelated playlist", ["b"])
    conn.commit()
    history = playlist_activity.snapshot_history(conn, playlist_id="P", page_size=1)
    assert history["items"][0]["previous_at"] == "2026-10-21"
    page = client.get("/curated?view=history&playlist_id=P")
    assert "History for" in page.text and "Selected playlist" in page.text
    assert "Unrelated playlist" not in page.text
    assert "playlist_id=P&amp;page=2" in page.text
    assert 'href="/playlists/P"' in page.text
    second = client.get("/curated?view=history&playlist_id=P&page=2")
    assert "First recorded snapshot" in second.text
    assert "playlist_id=P&amp;page=1" in second.text


def test_draft_full_view_is_read_only_and_approval_still_saves_in_place(client, conn):
    thread = chatstore.create_thread(conn)
    mid = chatstore.add_message(
        conn,
        thread["id"],
        "assistant",
        "Try this",
        meta={
            "proposals": [
                {
                    "name": "Unapproved",
                    "spec": {"track_ids": [f"track-{i}" for i in range(12)]},
                }
            ]
        },
    )
    url = f"/chat/messages/{mid}/proposals/0"
    before = client.get(url)
    assert before.status_code == 200
    assert "Draft · fixed selection" in before.text
    assert "track-11" in before.text
    assert f'action="{url}/approve" data-approve' in before.text
    assert "bindPlaylistApprovals(document.querySelector('.playlist-detail'))" in before.text
    assert smart.list_all(conn) == []
    approved = client.post(f"{url}/approve").json()["url"]
    assert approved.startswith("/playlists/saved/")
    assert client.get(url, follow_redirects=False).headers["location"] == approved


def test_viewer_can_read_every_detail_but_cannot_approve_drafts(client, conn, monkeypatch):
    add_playlist(conn, "P", "Readable", ["a"])
    thread = chatstore.create_thread(conn)
    mid = chatstore.add_message(
        conn,
        thread["id"],
        "assistant",
        "Draft",
        meta={
            "proposals": [
                {
                    "name": "Draft",
                    "spec": {"track_ids": ["a"]},
                }
            ]
        },
    )
    monkeypatch.setenv("ACETATE_AUTH_MODE", "authelia")
    headers = {"Remote-User": "friend", "Remote-Groups": "friends"}
    assert client.get("/playlists/P", headers=headers).status_code == 200
    page = client.get(f"/chat/messages/{mid}/proposals/0", headers=headers)
    assert page.status_code == 200
    assert "track/a" in page.text and "<form method=post" not in page.text


def test_track_labels_fall_back_to_normalized_catalog_before_canonicalization(conn):
    conn.execute("INSERT INTO tracks(track_id,name) VALUES('a','Newly backed up')")
    conn.execute(
        "INSERT INTO artists(artist_id,name) VALUES('artist','First artist'),('other','Second artist')"
    )
    conn.execute(
        "INSERT INTO track_artists(track_id,artist_id,position) VALUES('a','other',1),('a','artist',0)"
    )
    add_playlist(conn, "P", "New playlist", ["a"])
    conn.execute("PRAGMA query_only=ON")
    detail = playlist_activity.detail_tracks(conn, playlist_activity.playlist_detail(conn, "P"))
    assert detail["tracks"][0]["title"] == "Newly backed up"
    assert detail["tracks"][0]["artist"] == "First artist, Second artist"
    assert playlist_activity.playlist_catalog(conn)["total"] == 1


@pytest.mark.parametrize(
    "path",
    [
        "/playlists/missing",
        "/playlists/saved/999",
        "/playlists?page=0",
        "/playlists/P?page=0",
        "/chat/messages/999/proposals/0",
    ],
)
def test_invalid_playlist_pages_return_not_found(client, path):
    assert client.get(path).status_code == 404


def _shortcuts(page, label):
    return [
        unescape(url)
        for url in re.findall(r'<a href="([^"]+)"[^>]*>' + re.escape(label) + r"</a>", page.text)
    ]


def _question(url):
    assert urlsplit(url).path == "/chat"
    return parse_qs(urlsplit(url).query)["q"][0]


def test_track_shortcuts_keep_exact_identity_and_skip_unavailable_items(client, conn):
    add_recording(conn, "artist\ta", "Artist", "Same title", "a")
    add_recording(conn, "artist\tb", "Artist", "Same title", "b")
    add_playlist(conn, "P", "Selection", ["a", "b", None])
    conn.commit()
    page = client.get("/playlists/P")
    more = _shortcuts(page, "More like this ↗")
    about = _shortcuts(page, "About this track ↗")
    assert len(more) == 3  # playlist plus each available track
    assert len(about) == 2
    assert "https://open.spotify.com/track/a" in _question(more[1])
    assert "https://open.spotify.com/track/b" not in _question(more[1])
    assert "https://open.spotify.com/track/b" in _question(more[2])
    assert "Same title" in _question(about[0])
    assert 'target=_blank rel=noopener aria-label="More like Same title"' in page.text
    assert "open.spotify.com/track/None" not in page.text


def test_playlist_shortcuts_include_exact_playlist_and_bounded_examples(client, conn):
    sid = smart.create(conn, "Saved selection", {"track_ids": ["original"]})
    _managed(conn, "P", sid=sid)
    add_playlist(conn, "P", "Spotify name", [f"observed-{i}" for i in range(9)])
    conn.commit()
    page = client.get(f"/playlists/saved/{sid}")
    more = _question(_shortcuts(page, "More like this ↗")[0])
    about = _question(_shortcuts(page, "Ask about playlist ↗")[0])
    assert "Spotify playlist: https://open.spotify.com/playlist/P" in more
    assert f"Saved playlist ID: {sid}" in more
    assert "Example tracks from this page" in more
    assert "observed-5" in more and "observed-6" not in more
    assert "open.spotify.com/track/original" not in more
    assert "tracks that are not already in it" in more
    assert "tracks that stand out" in about


def test_draft_and_unpublished_shortcuts_use_available_selection_context(client, conn):
    sid = smart.create(conn, "Unpublished", {"track_ids": ["seed"]})
    page = client.get(f"/playlists/saved/{sid}")
    assert "open.spotify.com/track/seed" in _question(_shortcuts(page, "More like this ↗")[0])
    thread = chatstore.create_thread(conn)
    mid = chatstore.add_message(
        conn,
        thread["id"],
        "assistant",
        "Draft",
        meta={
            "proposals": [
                {
                    "name": "Draft",
                    "spec": {"track_ids": ["draft-seed"]},
                }
            ]
        },
    )
    page = client.get(f"/chat/messages/{mid}/proposals/0")
    assert "open.spotify.com/track/draft-seed" in _question(_shortcuts(page, "More like this ↗")[0])
    assert len(smart.list_all(conn)) == 1


def test_shortcuts_only_prepare_a_question_and_do_not_send_or_replace_history(
    client, conn, monkeypatch
):
    from acetate import web

    monkeypatch.setattr(
        web.agent, "chat_stream", lambda *a, **kw: pytest.fail("link must not run a model")
    )
    thread = chatstore.create_thread(conn)
    chatstore.add_message(conn, thread["id"], "user", "Keep this conversation")
    add_playlist(conn, "P", "Selection", ["seed"])
    conn.commit()
    page = client.get("/playlists/P")
    link = _shortcuts(page, "More like this ↗")[1]
    question = _question(link)
    chat = client.get(link)
    assert chat.status_code == 200
    assert (
        unescape(re.search(r"<textarea name=q[^>]*>(.*?)</textarea>", chat.text, re.S)[1])
        == question
    )
    assert "Keep this conversation" in chat.text
    assert len(chatstore.messages(conn, thread["id"])) == 1
    assert smart.list_all(conn) == []


def test_chat_prefill_escapes_textarea_content_and_defaults_to_blank(client):
    text = '</textarea><script>alert("x")</script>\nArtist & title'
    page = client.get("/chat", params={"q": text})
    assert "&lt;/textarea&gt;&lt;script&gt;" in page.text
    assert text not in page.text
    assert (
        unescape(re.search(r"<textarea name=q[^>]*>(.*?)</textarea>", page.text, re.S)[1]) == text
    )
    assert re.search(r"<textarea name=q[^>]*></textarea>", client.get("/chat").text)


@pytest.mark.parametrize("groups,can_query", [("friends", False), ("acetate-query", True)])
def test_chat_shortcuts_follow_query_access(client, conn, monkeypatch, groups, can_query):
    add_playlist(conn, "P", "Selection", ["seed"])
    conn.commit()
    monkeypatch.setenv("ACETATE_AUTH_MODE", "authelia")
    page = client.get("/playlists/P", headers={"Remote-User": "friend", "Remote-Groups": groups})
    assert bool(_shortcuts(page, "More like this ↗")) == can_query
    assert bool(_shortcuts(page, "About this track ↗")) == can_query
    assert "Spotify ↗" in page.text


def test_empty_playlist_offers_inspection_without_similarity_to_missing_tracks(client, conn):
    add_playlist(conn, "P", "Empty", [])
    conn.commit()
    page = client.get("/playlists/P")
    assert not _shortcuts(page, "More like this ↗")
    assert _shortcuts(page, "Ask about playlist ↗")


def test_chat_reload_escapes_shortcut_content_and_trace_metadata(client, conn):
    payload = '<img src=x onerror="alert(1)">'
    thread = chatstore.create_thread(conn)
    chatstore.add_message(conn, thread["id"], "user", payload)
    chatstore.add_message(
        conn,
        thread["id"],
        "assistant",
        'Artist "Quoted" ' + payload,
        meta={
            "steps": [{"tool": "track_report", "query": "</pre>" + payload}],
        },
    )
    page = client.get(f"/chat?thread={thread['id']}")
    assert page.status_code == 200
    assert payload not in page.text
    assert "&lt;img src=x onerror=&#34;alert(1)&#34;&gt;" in page.text
    assert 'data-md="Artist &#34;Quoted&#34; &lt;img' in page.text
    assert "&lt;/pre&gt;&lt;img" in page.text
