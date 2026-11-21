"""The refresh button: who may press it, and what it says back."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from acetate import jobs


@pytest.fixture
def client(conn, monkeypatch) -> TestClient:
    path = conn.execute("PRAGMA database_list").fetchone()["file"]
    monkeypatch.setenv("ACETATE_DB", path)
    monkeypatch.setenv("ACETATE_AUTH_MODE", "local")  # local dev is admin
    monkeypatch.setenv("ACETATE_TZ", "America/New_York")
    from acetate import auth, web

    monkeypatch.setattr(auth, "load_credentials", lambda: None)
    return TestClient(web.app)


def test_refresh_reports_what_landed_in_local_time(client, monkeypatch) -> None:
    monkeypatch.setattr(
        jobs,
        "refresh_plays",
        lambda: {
            "status": "ok",
            "fetched": 50,
            "inserted": 2,
            "new_plays": [
                {
                    "played_at": "2026-07-30T02:15:00Z",
                    "artist": "Fresh Artist",
                    "title": "Night Drive",
                },
                {"played_at": "2026-07-30T02:05:00Z", "artist": "Fresh Artist", "title": "Earlier"},
            ],
        },
    )

    res = client.post("/refresh-plays")

    assert res.status_code == 200
    assert "+2 plays" in res.text
    assert "Night Drive — Fresh Artist" in res.text
    assert "at 22:15" in res.text  # 02:15Z rendered in New York
    assert res.headers["HX-Trigger"] == "plays-refreshed"


def test_refresh_says_so_when_nothing_is_new(client, monkeypatch) -> None:
    monkeypatch.setattr(
        jobs,
        "refresh_plays",
        lambda: {"status": "ok", "fetched": 50, "inserted": 0, "new_plays": []},
    )

    res = client.post("/refresh-plays")

    assert "up to date" in res.text
    assert "HX-Trigger" not in res.headers  # nothing to swap in


def test_refresh_reports_busy_and_failure_plainly(client, monkeypatch) -> None:
    monkeypatch.setattr(jobs, "refresh_plays", lambda: {"status": "busy"})
    assert "already running" in client.post("/refresh-plays").text

    monkeypatch.setattr(
        jobs, "refresh_plays", lambda: {"status": "error", "error": "token expired"}
    )
    failed = client.post("/refresh-plays")
    assert "refresh failed" in failed.text
    assert "token expired" in failed.text


def test_refresh_escapes_track_text(client, monkeypatch) -> None:
    monkeypatch.setattr(
        jobs,
        "refresh_plays",
        lambda: {
            "status": "ok",
            "fetched": 1,
            "inserted": 1,
            "new_plays": [
                {
                    "played_at": "2026-07-30T02:15:00Z",
                    "artist": "A",
                    "title": "<script>alert(1)</script>",
                }
            ],
        },
    )

    res = client.post("/refresh-plays")

    assert "<script>alert(1)</script>" not in res.text
    assert "&lt;script&gt;" in res.text


def test_home_offers_the_button_to_admins(client) -> None:
    page = client.get("/")

    assert page.status_code == 200
    assert "hx-post=/refresh-plays" in page.text
    assert "hx-get=/recent" in page.text


def test_non_admins_get_neither_the_button_nor_the_endpoint(conn, monkeypatch) -> None:
    path = conn.execute("PRAGMA database_list").fetchone()["file"]
    monkeypatch.setenv("ACETATE_DB", path)
    monkeypatch.setenv("ACETATE_AUTH_MODE", "authelia")
    from acetate import auth, web

    monkeypatch.setattr(auth, "load_credentials", lambda: None)
    viewer = {"remote-user": "guest", "remote-groups": "acetate-query"}
    client = TestClient(web.app)

    assert "hx-post=/refresh-plays" not in client.get("/", headers=viewer).text
    assert client.post("/refresh-plays", headers=viewer).status_code == 403
