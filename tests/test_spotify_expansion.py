from __future__ import annotations

import sqlite3

import requests
import spotipy

from acetate import db, spotify_expansion


def _conn(tmp_path) -> sqlite3.Connection:
    conn = db.connect(tmp_path / "acetate-test.db")
    db.init_db(conn)
    conn.execute(
        "INSERT INTO artists(artist_id, name, updated_at) VALUES('artist-1', 'Seed Artist', 'now')"
    )
    conn.execute(
        "INSERT INTO albums(album_id, name, updated_at) VALUES('album-1', 'Seed Album', 'now')"
    )
    conn.execute(
        "INSERT INTO tracks(track_id, name, album_id, updated_at) "
        "VALUES('seed-track', 'Known Track', 'album-1', 'now')"
    )
    conn.execute(
        "INSERT INTO track_artists(track_id, artist_id, position) "
        "VALUES('seed-track', 'artist-1', 0)"
    )
    conn.execute(
        """
        INSERT INTO recording_stats(
            recording_key, rep_track_id, artist, title, plays, liked, homed
        ) VALUES('seed artist\tknown track', 'seed-track', 'Seed Artist', 'Known Track', 50, 1, 1)
        """
    )
    conn.commit()
    return conn


def _album(group: str, album_id: str, name: str) -> dict:
    return {
        "id": album_id,
        "name": name,
        "album_type": "album" if group != "appears_on" else "compilation",
        "release_date": "2025-01-01",
        "total_tracks": 2,
    }


def _track(track_id: str, title: str, artists: list[dict]) -> dict:
    return {
        "id": track_id,
        "name": title,
        "artists": artists,
        "disc_number": 1,
        "track_number": 1,
        "duration_ms": 200_000,
    }


class FakeSpotify:
    def __init__(self, *, top_unavailable: bool = False):
        self.top_unavailable = top_unavailable
        self.album_calls: list[str] = []
        self.track_calls: list[str] = []
        self.fail_album_group_once = False
        self.fail_album_tracks_once = False

    def artist_albums(self, _artist_id: str, *, album_type: str, limit: int, offset: int) -> dict:
        self.album_calls.append(album_type)
        if album_type == "album" and self.fail_album_group_once:
            self.fail_album_group_once = False
            raise requests.Timeout("slow")
        items = {
            "album": [_album("album", "album-1", "Seed Album")],
            "single": [],
            "appears_on": [_album("appears_on", "album-appear", "Guest Compilation")],
            "compilation": [],
        }[album_type]
        return {"items": items[offset : offset + limit], "total": len(items), "next": None}

    def artist_top_tracks(self, _artist_id: str, *, country: str) -> dict:
        assert country == "US"
        if self.top_unavailable:
            raise spotipy.SpotifyException(403, -1, "forbidden")
        return {
            "tracks": [
                _track("top-track", "Top Track", [{"id": "artist-1", "name": "Seed Artist"}])
            ]
        }

    def album_tracks(self, album_id: str, *, limit: int, offset: int) -> dict:
        self.track_calls.append(album_id)
        if self.fail_album_tracks_once:
            self.fail_album_tracks_once = False
            raise requests.Timeout("slow")
        tracks = {
            "album-1": [
                _track("seed-track", "Known Track", [{"id": "artist-1", "name": "Seed Artist"}]),
                _track("new-track", "New Track", [{"id": "artist-1", "name": "Seed Artist"}]),
            ],
            "album-appear": [
                _track(
                    "collab-track",
                    "Collab Track",
                    [
                        {"id": "guest-1", "name": "Guest Artist"},
                        {"id": "artist-1", "name": "Seed Artist"},
                    ],
                ),
                _track("guest-track", "Guest Track", [{"id": "guest-1", "name": "Guest Artist"}]),
            ],
        }[album_id]
        return {"items": tracks[offset : offset + limit], "total": len(tracks), "next": None}


def test_spotify_pacing_honors_retry_after_without_consuming_harvest_attempt(monkeypatch) -> None:
    calls = 0
    sleeps = []

    def call() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            error = spotipy.SpotifyException(429, -1, "slow down")
            error.headers = {"Retry-After": "2"}
            raise error
        return "ok"

    monkeypatch.setattr(spotify_expansion, "_last_request_at", 0.0)
    assert (
        spotify_expansion._spotify_call(
            call,
            min_interval_s=0,
            sleep=sleeps.append,
            monotonic=lambda: 0.0,
        )
        == "ok"
    )
    assert calls == 2
    assert sleeps == [2.0]

    paced_sleeps = []
    monkeypatch.setattr(spotify_expansion, "_last_request_at", 10.0)
    assert (
        spotify_expansion._spotify_call(
            lambda: "paced",
            min_interval_s=0.35,
            sleep=paced_sleeps.append,
            monotonic=lambda: 10.0,
        )
        == "paced"
    )
    assert paced_sleeps == [0.35]


def test_spotify_expansion_promotes_group_specific_provenance(tmp_path) -> None:
    conn = _conn(tmp_path)
    sp = FakeSpotify(top_unavailable=True)

    first = spotify_expansion.harvest(
        conn,
        sp=sp,
        artist_batch=1,
        album_batch=2,
    )
    second = spotify_expansion.harvest(
        conn,
        sp=sp,
        artist_batch=1,
        album_batch=2,
    )

    assert first["artist_fetched"] == 1
    assert first["top_unavailable"] == 1
    assert first["artist_albums"] == 2
    assert first["album_fetched"] == 2
    assert first["album_tracks"] == 4
    assert first["candidates"] == 3
    assert second["artist_seen"] == 0
    assert second["album_seen"] == 0
    assert (
        conn.execute(
            "SELECT status FROM spotify_expansion_capabilities WHERE surface='artist_top_tracks'"
        ).fetchone()[0]
        == "unavailable"
    )
    assert conn.execute("SELECT COUNT(*) FROM spotify_artist_albums").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM spotify_album_tracks").fetchone()[0] == 4

    new_sources = {
        row["source"]
        for row in conn.execute(
            "SELECT source FROM candidate_sources WHERE candidate_key='seed artist\tnew track'"
        )
    }
    assert new_sources == {
        spotify_expansion.SAME_ALBUM_SOURCE,
        spotify_expansion.SAME_ARTIST_SOURCE,
    }
    collab_sources = conn.execute(
        "SELECT source FROM candidate_sources WHERE candidate_key='guest artist\tcollab track'"
    ).fetchall()
    assert [row["source"] for row in collab_sources] == [spotify_expansion.APPEARS_ON_SOURCE]
    guest_sources = conn.execute(
        "SELECT source FROM candidate_sources WHERE candidate_key='guest artist\tguest track'"
    ).fetchall()
    assert [row["source"] for row in guest_sources] == [spotify_expansion.COPRESENCE_SOURCE]
    external = conn.execute(
        "SELECT provider, external_id FROM candidate_external_ids "
        "WHERE candidate_key='seed artist\tnew track'"
    ).fetchone()
    assert tuple(external) == ("spotify", "new-track")


def test_spotify_top_tracks_promote_when_the_capability_is_available(tmp_path) -> None:
    conn = _conn(tmp_path)
    sp = FakeSpotify()

    result = spotify_expansion.harvest(conn, sp=sp, artist_batch=1, album_batch=0)

    assert result["top_tracks"] == 1
    source = conn.execute(
        "SELECT source FROM candidate_sources WHERE candidate_key='seed artist\ttop track'"
    ).fetchone()
    assert source[0] == spotify_expansion.TOP_SOURCE
    assert (
        conn.execute(
            "SELECT status FROM spotify_expansion_capabilities WHERE surface='artist_top_tracks'"
        ).fetchone()[0]
        == "available"
    )


def test_spotify_artist_retry_keeps_album_groups_atomic(tmp_path) -> None:
    conn = _conn(tmp_path)
    sp = FakeSpotify()
    sp.fail_album_group_once = True

    first = spotify_expansion.harvest(conn, sp=sp, artist_batch=1, album_batch=0)
    second = spotify_expansion.harvest(conn, sp=sp, artist_batch=1, album_batch=0)

    assert first["artist_error"] == 1
    assert second["artist_fetched"] == 1
    assert sp.album_calls.count("album") == 2
    assert sp.album_calls.count("single") == 1
    row = conn.execute(
        "SELECT status, attempts FROM spotify_artist_expansion WHERE artist_id='artist-1'"
    ).fetchone()
    assert tuple(row) == ("fetched", 2)


def test_spotify_album_errors_retry_independently(tmp_path) -> None:
    conn = _conn(tmp_path)
    sp = FakeSpotify()
    sp.fail_album_tracks_once = True

    first = spotify_expansion.harvest(conn, sp=sp, artist_batch=0, album_batch=1)
    second = spotify_expansion.harvest(conn, sp=sp, artist_batch=0, album_batch=1)

    assert first["album_error"] == 1
    assert second["album_fetched"] == 1
    row = conn.execute(
        "SELECT status, attempts FROM spotify_album_expansion WHERE album_id='album-1'"
    ).fetchone()
    assert tuple(row) == ("fetched", 2)
