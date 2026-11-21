from __future__ import annotations

import sqlite3

from acetate import db, ytmusic


class FakeYTMusic:
    def search(self, query, filter=None, limit=20, ignore_spelling=False):
        assert filter == "songs"
        return [
            {
                "title": "Seed Track",
                "videoId": "seed-video",
                "duration_seconds": 180,
                "artists": [{"name": "Seed Artist", "id": "seed-channel"}],
            }
        ]

    def get_watch_playlist(
        self, videoId=None, playlistId=None, limit=25, radio=False, shuffle=False
    ):
        assert videoId == "seed-video"
        assert radio is True
        return {
            "playlistId": "RDAMVMseed-video",
            "tracks": [
                {
                    "title": "Seed Track",
                    "videoId": "seed-video",
                    "artists": [{"name": "Seed Artist", "id": "seed-channel"}],
                },
                {
                    "title": "Outside Track",
                    "videoId": "outside-video",
                    "artists": [{"name": "Outside Artist", "id": "outside-channel"}],
                    "album": {"name": "Outside Album", "id": "outside-album"},
                    "year": "2026",
                },
                {
                    "title": "Known Track",
                    "videoId": "known-video",
                    "artists": [{"name": "Known Artist", "id": "known-channel"}],
                },
            ],
        }


def _conn(tmp_path) -> sqlite3.Connection:
    conn = db.connect(tmp_path / "acetate-test.db")
    db.init_db(conn)
    return conn


def _recording(
    conn: sqlite3.Connection, key: str, artist: str, title: str, plays: int = 0, liked: int = 0
) -> None:
    conn.execute(
        """
        INSERT INTO recording_stats(recording_key, artist, title, plays, liked, homed)
        VALUES(?, ?, ?, ?, ?, 0)
        """,
        (key, artist, title, plays, liked),
    )


def test_harvest_radio_records_seed_and_candidate(monkeypatch, tmp_path) -> None:
    conn = _conn(tmp_path)
    _recording(conn, "seed artist\tseed track", "Seed Artist", "Seed Track", plays=50, liked=1)
    _recording(conn, "known artist\tknown track", "Known Artist", "Known Track", plays=2)
    conn.commit()
    monkeypatch.setattr(ytmusic, "YTMusic", FakeYTMusic)

    result = ytmusic.harvest_radio(conn, batch=1, radio_limit=3, pause_s=0)

    assert result["fetched"] == 1
    assert result["candidates"] == 1
    seed = conn.execute("SELECT * FROM ytmusic_radio_seeds").fetchone()
    assert seed["recording_key"] == "seed artist\tseed track"
    assert seed["video_id"] == "seed-video"
    assert seed["playlist_id"] == "RDAMVMseed-video"
    assert seed["status"] == "fetched"
    candidate = conn.execute("SELECT * FROM candidate_recordings").fetchone()
    assert candidate["candidate_key"] == "outside artist\toutside track"
    source = conn.execute("SELECT * FROM candidate_sources").fetchone()
    assert source["source"] == ytmusic.SOURCE
    assert source["reason"] == ytmusic.REASON
    assert source["seed_external_id"] == "seed-video"
    external = conn.execute("SELECT * FROM candidate_external_ids").fetchone()
    assert external["provider"] == "ytmusic"
    assert external["external_id"] == "outside-video"


def test_harvest_radio_prefers_cached_laion_seed(monkeypatch, tmp_path) -> None:
    class SearchFailsYTMusic(FakeYTMusic):
        def search(self, query, filter=None, limit=20, ignore_spelling=False):
            raise AssertionError("search should not be called when a cached seed id exists")

        def get_watch_playlist(
            self, videoId=None, playlistId=None, limit=25, radio=False, shuffle=False
        ):
            assert videoId == "laion-video"
            return {
                "playlistId": "RDAMVMlaion-video",
                "tracks": [
                    {
                        "title": "Outside Track",
                        "videoId": "outside-video",
                        "artists": [{"name": "Outside Artist", "id": "outside-channel"}],
                    }
                ],
            }

    conn = _conn(tmp_path)
    _recording(conn, "seed artist\tseed track", "Seed Artist", "Seed Track", plays=50, liked=1)
    conn.execute(
        """
        INSERT INTO ytmusic_seed_matches(
            recording_key, song_id, source, confidence, status, updated_at
        )
        VALUES('seed artist\tseed track', 'laion-video', 'laion_disco_12m', 1.0, 'matched',
               '2026-07-10T00:00:00+00:00')
        """
    )
    conn.commit()
    monkeypatch.setattr(ytmusic, "YTMusic", SearchFailsYTMusic)

    result = ytmusic.harvest_radio(conn, batch=1, radio_limit=1, pause_s=0)

    assert result["fetched"] == 1
    seed = conn.execute("SELECT * FROM ytmusic_radio_seeds").fetchone()
    assert seed["video_id"] == "laion-video"
    assert seed["playlist_id"] == "RDAMVMlaion-video"


def test_resolve_laion_records_no_row_miss(monkeypatch, tmp_path) -> None:
    class EmptyDuckDB:
        def execute(self, query, params):
            return self

        def fetchall(self):
            return []

    conn = _conn(tmp_path)
    _recording(
        conn, "missing artist\tmissing track", "Missing Artist", "Missing Track", plays=50, liked=1
    )
    conn.commit()
    monkeypatch.setattr(ytmusic, "_catalog_exists", lambda: True)
    monkeypatch.setattr(ytmusic, "_catalog_glob", lambda: "catalog/*.parquet")
    monkeypatch.setattr(ytmusic.duckdb, "connect", lambda *_args, **_kwargs: EmptyDuckDB())

    result = ytmusic.resolve_laion_seeds(conn, batch=1, min_plays=20)

    assert result == {"seen": 1, "matched": 0, "miss": 1}
    row = conn.execute("SELECT * FROM ytmusic_seed_matches").fetchone()
    assert row["recording_key"] == "missing artist\tmissing track"
    assert row["source"] == ytmusic.LAION_SOURCE
    assert row["status"] == "miss"
    assert row["confidence"] == 0.0
    assert "no exact title rows" in row["raw_json"]


def test_title_variants_only_strip_version_like_suffixes() -> None:
    assert ytmusic._title_variants("Track (feat. Someone)") == ["track (feat. someone)"]
    assert ytmusic._title_variants("Track (Radio Edit)") == [
        "track (radio edit)",
        "track",
    ]
    assert ytmusic._title_variants("Track - Extended Mix") == [
        "track - extended mix",
        "track",
    ]


def test_resolve_laion_uses_variant_without_overwriting_seed_title(monkeypatch, tmp_path) -> None:
    class VariantDuckDB:
        def execute(self, query, params):
            self.params = params
            return self

        def fetchall(self):
            _path, title = self.params
            if title == "track - radio edit":
                return []
            if title == "track":
                return [
                    (
                        "yt-song-id",
                        "Track",
                        ["Seed Artist"],
                        "Seed Album",
                        "album-id",
                        180,
                    )
                ]
            raise AssertionError(f"unexpected title lookup: {title}")

    conn = _conn(tmp_path)
    _recording(
        conn,
        "seed artist\ttrack - radio edit",
        "Seed Artist",
        "Track - Radio Edit",
        plays=50,
        liked=1,
    )
    conn.commit()
    monkeypatch.setattr(ytmusic, "_catalog_exists", lambda: True)
    monkeypatch.setattr(ytmusic, "_catalog_glob", lambda: "catalog/*.parquet")
    monkeypatch.setattr(ytmusic.duckdb, "connect", lambda *_args, **_kwargs: VariantDuckDB())

    result = ytmusic.resolve_laion_seeds(conn, batch=1, min_plays=20)

    assert result == {"seen": 1, "matched": 1, "miss": 0}
    row = conn.execute("SELECT * FROM ytmusic_seed_matches").fetchone()
    assert row["recording_key"] == "seed artist\ttrack - radio edit"
    assert row["song_id"] == "yt-song-id"
    assert row["title"] == "Track"
    assert '"original_title": "Track - Radio Edit"' in row["raw_json"]
    assert '"matched_title_variant": "track"' in row["raw_json"]


def _track_rec(conn, track_id, key):
    conn.execute(
        "INSERT INTO track_recording(track_id, recording_key, rep_track_id) VALUES(?,?,?)",
        (track_id, key, track_id),
    )


def test_priority_seeds_ranks_techno_and_fave_members_excludes_rest(tmp_path) -> None:
    conn = _conn(tmp_path)
    # techno-genre tracks (in the pool via genre), a fave-playlist member (in via
    # playlist), and a high-play track that is NEITHER (must be excluded).
    _recording(conn, "th", "A", "Techno Hi", plays=50)
    _recording(conn, "tl", "B", "Techno Lo", plays=5)
    _recording(conn, "fm", "C", "Fave Member", plays=30)
    _recording(conn, "nz", "D", "Loud Noise", plays=100)  # most-played but off-topic
    for tid, key in [("th-t", "th"), ("tl-t", "tl"), ("fm-t", "fm"), ("nz-t", "nz")]:
        _track_rec(conn, tid, key)
    for tid in ("th-t", "tl-t"):
        conn.execute(
            "INSERT INTO track_genres(track_id, genre, source, confidence, assigned_at) "
            "VALUES(?,?, 'lastfm', 90, '2026-07-14')",
            (tid, "melodic techno"),
        )
    # a fave playlist whose member is NOT in the techno genre pool
    conn.execute(
        "INSERT INTO playlist_labels(playlist_id, label, updated_at) "
        "VALUES('PF','fave','2026-07-14')"
    )
    conn.execute(
        "INSERT INTO playlist_members(playlist_id, track_id, recording_key, position) "
        "VALUES('PF','fm-t','fm',0)"
    )
    conn.commit()

    seeds = ytmusic.priority_seeds(conn, min_plays=3)
    assert seeds == ["th", "fm", "tl"]  # by plays desc, off-topic 'nz' excluded
    assert "nz" not in seeds


def test_background_seeds_priority_first_then_covers_rest(tmp_path) -> None:
    conn = _conn(tmp_path)
    _recording(conn, "th", "A", "Techno Hi", plays=50)  # priority (genre)
    _recording(conn, "tl", "B", "Techno Lo", plays=5)  # priority (genre)
    _recording(conn, "fm", "C", "Fave Member", plays=30)  # priority (fave playlist)
    _recording(conn, "nz", "D", "Loud Noise", plays=100)  # NOT priority, but most-played
    for tid, key in [("th-t", "th"), ("tl-t", "tl"), ("fm-t", "fm"), ("nz-t", "nz")]:
        _track_rec(conn, tid, key)
    for tid in ("th-t", "tl-t"):
        conn.execute(
            "INSERT INTO track_genres(track_id, genre, source, confidence, assigned_at) "
            "VALUES(?,?, 'lastfm', 90, '2026-07-15')",
            (tid, "melodic techno"),
        )
    conn.execute(
        "INSERT INTO playlist_labels(playlist_id, label, updated_at) VALUES('PF','fave','x')"
    )
    conn.execute(
        "INSERT INTO playlist_members(playlist_id, track_id, recording_key, position) "
        "VALUES('PF','fm-t','fm',0)"
    )
    conn.commit()

    seeds = ytmusic.background_seeds(conn, min_plays=5, priority_min_plays=3)
    # priority order first (by plays), THEN the off-topic high-play track last
    assert seeds == ["th", "fm", "tl", "nz"]


def test_missing_optional_ytmusicapi_fails_cleanly(monkeypatch, tmp_path) -> None:
    import pytest

    from acetate import jobs

    conn = _conn(tmp_path)
    monkeypatch.setattr(ytmusic, "YTMusic", None)

    assert ytmusic.is_available() is False
    with pytest.raises(RuntimeError, match="ytmusic"):
        ytmusic.harvest_radio(conn, batch=1, pause_s=0)
    assert jobs.run_yt_radio(batch=1) == {"skipped": "ytmusicapi_not_installed"}
