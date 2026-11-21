from __future__ import annotations

from fastapi.testclient import TestClient

from acetate import playlist_profile, taste


def _client_for(conn, monkeypatch) -> TestClient:
    path = conn.execute("PRAGMA database_list").fetchone()["file"]
    monkeypatch.setenv("ACETATE_DB", path)
    from acetate import web

    return TestClient(web.app)


def test_playlist_page_profiles_and_labels_without_cli(library, monkeypatch) -> None:
    playlist_profile.rebuild(library)
    taste.build(library)
    client = _client_for(library, monkeypatch)

    page = client.get("/playlists?show=unlabeled")
    assert page.status_code == 200
    assert "Your playlists" in page.text
    assert "Warehouse Techno" in page.text
    assert "Find liked matches" in page.text

    labeled = client.post("/playlists/P1/label", data={"label": "fave"})
    assert labeled.status_code == 200
    assert "manual · fave" in labeled.text
    row = library.execute(
        "SELECT user_label, persona FROM playlist_stats WHERE playlist_id='P1'"
    ).fetchone()
    assert (row["user_label"], row["persona"]) == ("fave", "rotation")

    cleared = client.post("/playlists/P1/label", data={"label": ""})
    assert cleared.status_code == 200
    assert (
        library.execute("SELECT user_label FROM playlist_stats WHERE playlist_id='P1'").fetchone()[
            0
        ]
        is None
    )


def test_web_can_queue_liked_matches_without_removals(library, monkeypatch) -> None:
    playlist_profile.rebuild(library)
    taste.build(library)
    client = _client_for(library, monkeypatch)

    response = client.post("/health/P1/matches")
    assert response.status_code == 200
    assert "liked matches queued" in response.text
    actions = {
        r[0]
        for r in library.execute(
            "SELECT DISTINCT action FROM playlist_proposals WHERE playlist_id='P1'"
        )
    }
    assert actions == {"add"}


def test_tools_catalog_exposes_registered_workflows(library, monkeypatch) -> None:
    client = _client_for(library, monkeypatch)

    page = client.get("/tools")

    assert page.status_code == 200
    assert "Discover artists from a playlist" in page.text
    assert "agent: discover_artists" in page.text
    assert "Warehouse Techno" in page.text
