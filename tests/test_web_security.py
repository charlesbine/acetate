"""Security hardening tests: Jinja autoescaping / XSS, CSRF middleware, container defaults, CDN pinning."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from acetate import access, auth, chatstore, jobs, proposals, web

XSS = "<img src=x onerror=alert(1)>"
XSS_ESCAPED = "&lt;img src=x onerror=alert(1)&gt;"


@pytest.fixture
def client(conn, monkeypatch) -> TestClient:
    path = conn.execute("PRAGMA database_list").fetchone()["file"]
    monkeypatch.setenv("ACETATE_DB", path)
    monkeypatch.setenv("ACETATE_AUTH_MODE", "local")
    monkeypatch.setenv("ACETATE_LOCAL_USER", "local-dev")
    monkeypatch.setattr(auth, "load_credentials", lambda: None)
    return TestClient(web.app)


def test_jinja_autoescape_is_enabled_for_extensionless_templates() -> None:
    ae = web.env.autoescape
    enabled = ae("review") if callable(ae) else ae
    assert web.env.autoescape is True
    assert enabled is True


def test_xss_escaped_in_review_tracks_artists_and_decision(conn, client: TestClient) -> None:
    proposals.propose(
        conn,
        playlist_id="pl-xss",
        playlist_name=f"Mix {XSS}",
        action="add",
        kind="library",
        recording_key="xss-key",
        source="health",
        score=0.95,
        reasons=[f"reason {XSS}"],
        track_id="tid-xss",
        artist=f"Artist {XSS}",
        title=f"Title {XSS}",
    )
    conn.commit()
    pid = conn.execute("SELECT id FROM playlist_proposals LIMIT 1").fetchone()["id"]

    page = client.get("/review")
    assert page.status_code == 200
    assert XSS not in page.text
    assert f"Artist {XSS_ESCAPED}" in page.text
    assert f"Title {XSS_ESCAPED}" in page.text
    assert f"Mix {XSS_ESCAPED}" in page.text
    assert f"reason {XSS_ESCAPED}" in page.text

    decision = client.post(f"/review/{pid}/accept")
    assert decision.status_code == 200
    assert XSS not in decision.text
    assert f"Artist {XSS_ESCAPED}" in decision.text
    assert f"Title {XSS_ESCAPED}" in decision.text


def test_xss_escaped_in_home_tracks_artists_and_identity(
    conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = conn.execute("PRAGMA database_list").fetchone()["file"]
    monkeypatch.setenv("ACETATE_DB", path)
    monkeypatch.setenv("ACETATE_AUTH_MODE", "authelia")
    monkeypatch.setattr(auth, "load_credentials", lambda: None)

    conn.execute(
        "INSERT INTO play_events(played_at, track_id, ms_played, source) "
        "VALUES('2026-07-01T12:00:00Z', 'tid-1', 200000, 'recently_played')"
    )
    conn.execute("INSERT INTO tracks(track_id, name) VALUES('tid-1', ?)", (f"Track {XSS}",))
    conn.execute(
        "INSERT INTO track_labels(track_id, track_name, artist_name) VALUES('tid-1', ?, ?)",
        (f"Track {XSS}", f"Band {XSS}"),
    )
    conn.execute(
        "INSERT INTO track_recording(track_id, recording_key, rep_track_id, artist, title) "
        "VALUES('tid-1', 'rk-1', 'tid-1', ?, ?)",
        (f"Band {XSS}", f"Track {XSS}"),
    )
    conn.execute(
        "INSERT INTO recording_stats(recording_key, rep_track_id, artist, title, plays, liked, homed) "
        "VALUES('rk-1', 'tid-1', ?, ?, 5, 1, 0)",
        (f"Band {XSS}", f"Track {XSS}"),
    )
    conn.commit()

    client = TestClient(web.app)
    res = client.get(
        "/",
        headers={
            "Remote-User": "alice",
            "Remote-Name": f"Alice {XSS}",
            "Remote-Groups": "acetate-admin",
        },
    )
    assert res.status_code == 200
    assert XSS not in res.text
    assert f"Alice {XSS_ESCAPED}" in res.text
    assert f"Track {XSS_ESCAPED}" in res.text
    assert f"Band {XSS_ESCAPED}" in res.text
    # Macros still render real HTML elements rather than being double-escaped.
    assert '<li class="row rise"' in res.text


def test_xss_escaped_in_runs_error_and_unknown_job(
    conn, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        jobs,
        "recent_runs",
        lambda: [
            {
                "job": "poll",
                "started_at": "2026-07-01T12:00:00Z",
                "status": "error",
                "error": f"boom {XSS}",
                "playlists": [],
                "summary": "",
            }
        ],
    )
    runs_res = client.get("/runs")
    assert runs_res.status_code == 200
    assert XSS not in runs_res.text
    assert f"boom {XSS_ESCAPED}" in runs_res.text

    unknown = client.post("/run/%3Cimg%20src=x%20onerror=alert(1)%3E")
    assert unknown.status_code == 404
    assert XSS not in unknown.text
    assert XSS_ESCAPED in unknown.text


def test_xss_escaped_in_playlist_profile_name() -> None:
    html = web._render(
        "playlist_profile_row",
        identity=access.Identity(
            "a", "a", None, frozenset({"acetate-admin"}), access.Role.ADMIN, "local"
        ),
        profile={
            "playlist_id": "pl-1",
            "name": f"Crate {XSS}",
            "n_tracks": 10,
            "n_liked": 5,
            "add_days": 3,
            "member_plays_180d": 40,
            "cohesion": 0.85,
            "user_label": None,
            "suggestion": {"label": "rotation", "confidence": "high", "reasons": ["steady"]},
            "has_region": True,
        },
    )
    assert XSS not in html
    assert f"Crate {XSS_ESCAPED}" in html


def test_intentional_json_and_explicit_escape_not_double_escaped(conn, client: TestClient) -> None:
    smart_page = client.get("/smart")
    assert smart_page.status_code == 200
    assert 'const SMART_EX = {"Forgotten Techno":' in smart_page.text

    th = chatstore.create_thread(conn)
    conn.execute("UPDATE chat_threads SET title=? WHERE id=?", ("Rock & Roll", th["id"]))
    chatstore.add_message(conn, th["id"], "assistant", "A & B <tag>")
    conn.commit()
    chat_page = client.get(f"/chat?thread={th['id']}")
    assert chat_page.status_code == 200
    assert "Rock &amp; Roll" in chat_page.text
    assert "Rock &amp;amp; Roll" not in chat_page.text
    assert 'data-md="A &amp; B &lt;tag&gt;"' in chat_page.text


def test_csrf_blocks_cross_site_sec_fetch_site(client: TestClient) -> None:
    res = client.post("/chat/new", headers={"Sec-Fetch-Site": "cross-site"}, follow_redirects=False)
    assert res.status_code == 403

    same_site = client.post(
        "/chat/new", headers={"Sec-Fetch-Site": "same-site"}, follow_redirects=False
    )
    assert same_site.status_code == 403


def test_csrf_blocks_mismatched_and_null_origin(client: TestClient) -> None:
    evil = client.post(
        "/chat/new",
        headers={"Origin": "https://evil.example"},
        follow_redirects=False,
    )
    assert evil.status_code == 403

    null_origin = client.post(
        "/chat/new",
        headers={"Origin": "null"},
        follow_redirects=False,
    )
    assert null_origin.status_code == 403

    mismatched_combo = client.post(
        "/chat/new",
        headers={"Sec-Fetch-Site": "same-origin", "Origin": "https://evil.example"},
        follow_redirects=False,
    )
    assert mismatched_combo.status_code == 403


def test_csrf_allows_same_origin_matching_host_and_non_browser(client: TestClient) -> None:
    same_origin = client.post(
        "/chat/new",
        headers={"Sec-Fetch-Site": "same-origin", "Origin": "http://testserver"},
        follow_redirects=False,
    )
    assert same_origin.status_code == 303

    case_insensitive = client.post(
        "/chat/new",
        headers={"Sec-Fetch-Site": "same-origin", "Origin": "http://TESTSERVER"},
        follow_redirects=False,
    )
    assert case_insensitive.status_code == 303

    direct_nav = client.post(
        "/chat/new",
        headers={"Sec-Fetch-Site": "none"},
        follow_redirects=False,
    )
    assert direct_nav.status_code == 303

    non_browser = client.post("/chat/new", follow_redirects=False)
    assert non_browser.status_code == 303


def test_csrf_allows_cross_site_get_and_blocks_other_mutating_methods(client: TestClient) -> None:
    get_res = client.get("/", headers={"Sec-Fetch-Site": "cross-site"})
    assert get_res.status_code == 200

    for method in ("PUT", "PATCH", "DELETE"):
        res = client.request(method, "/chat/memory", headers={"Sec-Fetch-Site": "cross-site"})
        assert res.status_code == 403


def test_dockerfile_defaults_to_fail_closed_authelia_mode() -> None:
    dockerfile = (Path(__file__).parent.parent / "Dockerfile").read_text()
    assert "ACETATE_AUTH_MODE=authelia" in dockerfile


def test_cdn_scripts_are_version_pinned_with_sri() -> None:
    assert "marked@12/" not in web.CHAT
    assert "dompurify@3/" not in web.CHAT
    assert "marked@12.0.2/marked.min.js" in web.CHAT
    assert "dompurify@3.4.18/dist/purify.min.js" in web.CHAT
    assert "htmx.org@1.9.12/dist/htmx.min.js" in web.BASE
    assert 'integrity="sha384-' in web.BASE
    assert 'integrity="sha384-' in web.CHAT
    assert 'crossorigin="anonymous"' in web.BASE
    assert 'crossorigin="anonymous"' in web.CHAT
