from __future__ import annotations

import sqlite3

import requests

from acetate import db, lastfm, lastfm_artists


def _conn(tmp_path) -> sqlite3.Connection:
    conn = db.connect(tmp_path / "acetate-test.db")
    db.init_db(conn)
    conn.execute(
        "INSERT INTO recording_stats(recording_key, artist, title, plays, liked, homed) "
        "VALUES('seed artist\tknown track', 'Seed Artist', 'Known Track', 50, 1, 1)"
    )
    conn.commit()
    return conn


def _payload(method: str) -> dict:
    if method == "artist.getinfo":
        return {
            "artist": {
                "name": "Seed Artist",
                "mbid": "artist-mbid",
                "stats": {"listeners": "100", "playcount": "1000"},
                "tags": {"tag": [{"name": "House"}]},
                "bio": {"summary": "Bio"},
            }
        }
    if method == "artist.getsimilar":
        return {
            "similarartists": {
                "artist": [
                    {"name": "Similar Artist", "match": "0.8", "mbid": "similar-mbid"},
                ]
            }
        }
    if method == "artist.gettoptracks":
        return {
            "toptracks": {
                "track": [
                    {
                        "name": "Known Track",
                        "playcount": "500",
                        "listeners": "50",
                        "artist": {"name": "Seed Artist"},
                    },
                    {
                        "name": "Outside Track",
                        "playcount": "400",
                        "listeners": "40",
                        "artist": {"name": "Seed Artist"},
                    },
                ]
            }
        }
    if method == "artist.gettopalbums":
        return {
            "topalbums": {
                "album": [
                    {"name": "Top Album", "playcount": "300", "artist": {"name": "Seed Artist"}},
                ]
            }
        }
    raise AssertionError(method)


def test_harvest_lastfm_artist_cache_and_promote_top_tracks(tmp_path) -> None:
    conn = _conn(tmp_path)
    calls = []

    def call(method: str, **_params) -> dict:
        calls.append(method)
        return _payload(method)

    first = lastfm_artists.harvest(conn, batch=1, call=call, pause_s=0)
    second = lastfm_artists.harvest(conn, batch=1, call=call, pause_s=0)

    assert first["fetched"] == 1
    assert first["candidates"] == 1
    assert first["similar"] == 1
    assert first["tracks"] == 2
    assert first["albums"] == 1
    assert second["seen"] == 0
    assert calls == [method for method, _column in lastfm_artists.SURFACES]
    cache = conn.execute("SELECT * FROM lastfm_artist_cache").fetchone()
    assert cache["status"] == "fetched"
    assert cache["listeners"] == 100
    assert cache["playcount"] == 1000
    assert conn.execute("SELECT COUNT(*) FROM lastfm_artist_similar").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM lastfm_artist_top_tracks").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM lastfm_artist_top_albums").fetchone()[0] == 1
    source = conn.execute("SELECT * FROM candidate_sources").fetchone()
    assert source["source"] == lastfm_artists.SOURCE
    assert source["source_rank"] == 2


def test_lastfm_artist_retry_keeps_successful_surfaces(tmp_path) -> None:
    conn = _conn(tmp_path)
    calls = []
    failed = False

    def call(method: str, **_params) -> dict:
        nonlocal failed
        calls.append(method)
        if method == "artist.gettoptracks" and not failed:
            failed = True
            raise requests.Timeout("slow")
        return _payload(method)

    first = lastfm_artists.harvest(conn, batch=1, call=call, pause_s=0)
    second = lastfm_artists.harvest(conn, batch=1, call=call, pause_s=0)

    assert first["error"] == 1
    assert second["fetched"] == 1
    assert calls.count("artist.getinfo") == 1
    assert calls.count("artist.getsimilar") == 1
    assert calls.count("artist.gettoptracks") == 2
    assert calls.count("artist.gettopalbums") == 1
    row = conn.execute("SELECT status, attempts FROM lastfm_artist_cache").fetchone()
    assert tuple(row) == ("fetched", 2)


def test_lastfm_missing_artist_is_cached_as_miss(tmp_path) -> None:
    conn = _conn(tmp_path)
    calls = 0

    def call(_method: str, **_params) -> dict:
        nonlocal calls
        calls += 1
        raise lastfm.LastfmAPIError("not found", 6)

    first = lastfm_artists.harvest(conn, batch=1, call=call, pause_s=0)
    second = lastfm_artists.harvest(conn, batch=1, call=call, pause_s=0)

    assert first["miss"] == 1
    assert second["seen"] == 0
    assert calls == 1
    assert conn.execute("SELECT status FROM lastfm_artist_cache").fetchone()[0] == "miss"
