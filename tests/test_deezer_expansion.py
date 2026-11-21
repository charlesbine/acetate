from __future__ import annotations

import sqlite3

import requests

from acetate import db, deezer_expansion


def _conn(tmp_path) -> sqlite3.Connection:
    conn = db.connect(tmp_path / "acetate-test.db")
    db.init_db(conn)
    conn.execute(
        "INSERT INTO recording_stats(recording_key, artist, title, plays, liked, homed) "
        "VALUES('seed artist\tknown track', 'Seed Artist', 'Known Track', 50, 1, 1)"
    )
    conn.execute(
        "INSERT INTO deezer_track(recording_key, deezer_id, artist_id, album_id, updated_at) "
        "VALUES('seed artist\tknown track', 100, 10, 20, 'now')"
    )
    conn.execute(
        "INSERT INTO deezer_artist(artist_id, name, updated_at) VALUES(10, 'Seed Artist', 'now')"
    )
    conn.execute(
        "INSERT INTO deezer_album(album_id, title, updated_at) VALUES(20, 'Seed Album', 'now')"
    )
    conn.commit()
    return conn


def _top_payload() -> dict:
    return {
        "data": [
            {
                "id": 100,
                "title": "Known Track",
                "artist": {"id": 10, "name": "Seed Artist"},
                "album": {"id": 20, "title": "Seed Album"},
                "duration": 200,
            },
            {
                "id": 101,
                "title": "New Track",
                "artist": {"id": 10, "name": "Seed Artist"},
                "album": {"id": 20, "title": "Seed Album"},
                "duration": 210,
                "preview": "https://example.test/101.mp3",
            },
        ],
        "total": 2,
    }


def _albums_payload() -> dict:
    return {
        "data": [
            {
                "id": 20,
                "title": "Seed Album",
                "record_type": "album",
                "release_date": "2025-01-01",
                "fans": 50,
                "tracklist": "https://api.deezer.test/album/20/tracks",
            }
        ],
        "total": 1,
    }


def _tracks_payload() -> dict:
    return {
        "data": [
            {
                "id": 100,
                "title": "Known Track",
                "artist": {"id": 10, "name": "Seed Artist"},
                "track_position": 1,
                "disk_number": 1,
                "duration": 200,
                "isrc": "KNOWN0000001",
            },
            {
                "id": 101,
                "title": "New Track",
                "artist": {"id": 10, "name": "Seed Artist"},
                "track_position": 2,
                "disk_number": 1,
                "duration": 210,
                "isrc": "NEW000000001",
                "preview": "https://example.test/101.mp3",
            },
            {
                "id": 102,
                "title": "Guest Track",
                "artist": {"id": 11, "name": "Guest Artist"},
                "track_position": 3,
                "disk_number": 1,
                "duration": 220,
                "isrc": "GUEST0000001",
            },
        ],
        "total": 3,
    }


def test_deezer_artist_and_album_expansion_promotes_provenance(tmp_path) -> None:
    conn = _conn(tmp_path)
    calls = []

    def get(path: str, **_params) -> dict:
        calls.append(path)
        if path == "/artist/10/top":
            return _top_payload()
        if path == "/artist/10/albums":
            return _albums_payload()
        if path == "/album/20/tracks":
            return _tracks_payload()
        raise AssertionError(path)

    first = deezer_expansion.harvest(
        conn,
        artist_batch=1,
        album_batch=1,
        get=get,
    )
    second = deezer_expansion.harvest(
        conn,
        artist_batch=1,
        album_batch=1,
        get=get,
    )

    assert first["artist_fetched"] == 1
    assert first["album_fetched"] == 1
    assert first["top_tracks"] == 2
    assert first["artist_albums"] == 1
    assert first["album_tracks"] == 3
    assert first["candidates"] == 2
    assert second["artist_seen"] == 0
    assert second["album_seen"] == 0
    assert calls == ["/artist/10/top", "/artist/10/albums", "/album/20/tracks"]

    assert conn.execute("SELECT COUNT(*) FROM deezer_artist_top_tracks").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM deezer_artist_albums").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM deezer_album_tracks").fetchone()[0] == 3
    new_sources = {
        row["source"]
        for row in conn.execute(
            "SELECT source FROM candidate_sources WHERE candidate_key='seed artist\tnew track'"
        )
    }
    assert new_sources == {
        deezer_expansion.ARTIST_TOP_SOURCE,
        deezer_expansion.SAME_ARTIST_SOURCE,
        deezer_expansion.SAME_ALBUM_SOURCE,
    }
    assert (
        conn.execute(
            "SELECT score FROM candidate_recordings WHERE candidate_key='seed artist\tnew track'"
        ).fetchone()[0]
        > 0
    )
    guest_sources = conn.execute(
        "SELECT source FROM candidate_sources WHERE candidate_key='guest artist\tguest track'"
    ).fetchall()
    assert [row["source"] for row in guest_sources] == [deezer_expansion.SAME_ALBUM_SOURCE]
    external_ids = {
        (row["provider"], row["external_id"])
        for row in conn.execute(
            "SELECT provider, external_id FROM candidate_external_ids "
            "WHERE candidate_key='seed artist\tnew track'"
        )
    }
    assert external_ids == {("deezer", "101"), ("isrc", "NEW000000001")}


def test_deezer_artist_retry_keeps_successful_top_surface(tmp_path) -> None:
    conn = _conn(tmp_path)
    calls = []
    failed = False

    def get(path: str, **_params) -> dict:
        nonlocal failed
        calls.append(path)
        if path == "/artist/10/top":
            return _top_payload()
        if path == "/artist/10/albums" and not failed:
            failed = True
            raise requests.Timeout("slow")
        if path == "/artist/10/albums":
            return _albums_payload()
        raise AssertionError(path)

    first = deezer_expansion.harvest(
        conn,
        artist_batch=1,
        album_batch=0,
        get=get,
    )
    second = deezer_expansion.harvest(
        conn,
        artist_batch=1,
        album_batch=0,
        get=get,
    )

    assert first["artist_error"] == 1
    assert second["artist_fetched"] == 1
    assert calls.count("/artist/10/top") == 1
    assert calls.count("/artist/10/albums") == 2
    row = conn.execute(
        "SELECT status, attempts, top_json IS NOT NULL AS has_top FROM deezer_artist_expansion"
    ).fetchone()
    assert tuple(row) == ("fetched", 2, 1)


def test_deezer_album_errors_retry_independently(tmp_path) -> None:
    conn = _conn(tmp_path)
    calls = 0

    def get(path: str, **_params) -> dict:
        nonlocal calls
        assert path == "/album/20/tracks"
        calls += 1
        if calls == 1:
            raise requests.Timeout("slow")
        return _tracks_payload()

    first = deezer_expansion.harvest(
        conn,
        artist_batch=0,
        album_batch=1,
        get=get,
    )
    second = deezer_expansion.harvest(
        conn,
        artist_batch=0,
        album_batch=1,
        get=get,
    )

    assert first["album_error"] == 1
    assert second["album_fetched"] == 1
    row = conn.execute(
        "SELECT status, attempts FROM deezer_album_expansion WHERE album_id=20"
    ).fetchone()
    assert tuple(row) == ("fetched", 2)


def test_deezer_missing_artist_is_a_durable_miss(tmp_path) -> None:
    conn = _conn(tmp_path)
    calls = 0

    def get(_path: str, **_params) -> dict:
        nonlocal calls
        calls += 1
        return {"error": {"type": "DataException", "message": "no data", "code": 800}}

    first = deezer_expansion.harvest(
        conn,
        artist_batch=1,
        album_batch=0,
        get=get,
    )
    second = deezer_expansion.harvest(
        conn,
        artist_batch=1,
        album_batch=0,
        get=get,
    )

    assert first["artist_miss"] == 1
    assert second["artist_seen"] == 0
    assert calls == 1
    assert (
        conn.execute("SELECT status FROM deezer_artist_expansion WHERE artist_id=10").fetchone()[0]
        == "miss"
    )
