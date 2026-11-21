from __future__ import annotations

import requests

from acetate import audio, db


def _conn(tmp_path):
    conn = db.connect(tmp_path / "acetate-test.db")
    db.init_db(conn)
    conn.execute(
        """
        INSERT INTO recording_stats(recording_key, artist, title, plays, liked, homed)
        VALUES('artist\ttrack', 'Artist', 'Track', 10, 1, 0)
        """
    )
    conn.execute(
        """
        INSERT INTO audio_cache(
            recording_key, status, source, source_id, attempts, updated_at
        )
        VALUES('artist\ttrack', 'fetched', 'deezer_search', '42', 1, 'now')
        """
    )
    conn.commit()
    return conn


def test_deezer_metadata_enrichment_records_each_entity_surface(tmp_path, monkeypatch) -> None:
    conn = _conn(tmp_path)

    def get_json(url: str, **_params):
        if url.endswith("/track/42"):
            return {
                "id": 42,
                "bpm": 123,
                "gain": -8.5,
                "rank": 100,
                "release_date": "2026-01-02",
                "artist": {"id": 10},
                "album": {"id": 20},
            }
        if url.endswith("/album/20"):
            return {
                "id": 20,
                "title": "Album",
                "label": "Label",
                "genres": {"data": [{"name": "Techno"}]},
                "release_date": "2026-01-02",
            }
        if url.endswith("/artist/10"):
            return {"id": 10, "name": "Artist", "nb_fan": 99}
        if "/artist/10/related" in url:
            return {"data": [{"id": 11, "name": "Neighbor"}]}
        raise AssertionError(url)

    monkeypatch.setattr(audio, "_get_json", get_json)

    result = audio.enrich(conn)

    assert result == {
        "tracks": 1,
        "albums": 1,
        "artists": 1,
        "provider_miss": 0,
        "errors": 0,
    }
    runs = {
        (row["entity_type"], row["entity_id"]): row["status"]
        for row in conn.execute("SELECT * FROM deezer_metadata_runs")
    }
    assert runs == {
        ("track", "42"): "available",
        ("album", "20"): "available",
        ("artist", "10"): "available",
    }


def test_deezer_metadata_provider_miss_is_not_pending(tmp_path, monkeypatch) -> None:
    conn = _conn(tmp_path)
    monkeypatch.setattr(
        audio,
        "_get_json",
        lambda *_args, **_kwargs: {
            "error": {"type": "DataException", "code": 800, "message": "no data"}
        },
    )

    first = audio.enrich(conn)
    second = audio.enrich(conn)

    assert first["provider_miss"] == 1
    assert second["provider_miss"] == 0
    run = conn.execute(
        "SELECT status, attempts FROM deezer_metadata_runs "
        "WHERE entity_type='track' AND entity_id='42'"
    ).fetchone()
    assert tuple(run) == ("provider_miss", 1)
    assert conn.execute("SELECT COUNT(*) FROM deezer_track").fetchone()[0] == 0


def test_deezer_metadata_transient_error_retries_then_parks(tmp_path, monkeypatch) -> None:
    conn = _conn(tmp_path)

    def timeout(*_args, **_kwargs):
        raise requests.Timeout("slow")

    monkeypatch.setattr(audio, "_get_json", timeout)

    results = [audio.enrich(conn) for _ in range(4)]

    assert [result["errors"] for result in results] == [1, 1, 1, 0]
    run = conn.execute(
        "SELECT status, attempts, error FROM deezer_metadata_runs "
        "WHERE entity_type='track' AND entity_id='42'"
    ).fetchone()
    assert run["status"] == "error"
    assert run["attempts"] == audio.MAX_ATTEMPTS
    assert "slow" in run["error"]


def test_audio_fetch_creates_explicit_pending_rows_for_every_embedding(
    tmp_path, monkeypatch
) -> None:
    conn = _conn(tmp_path)
    conn.execute("UPDATE audio_cache SET status='pending', attempts=0")
    conn.execute(
        "INSERT INTO tracks(track_id, isrc, duration_ms) VALUES('track-id', 'USABC1234567', 200000)"
    )
    conn.execute(
        """
        INSERT INTO track_recording(
            track_id, recording_key, rep_track_id, artist, title
        )
        VALUES('track-id', 'artist\ttrack', 'track-id', 'Artist', 'Track')
        """
    )
    conn.commit()
    monkeypatch.setattr(audio, "enqueue", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        audio,
        "_fetch_one",
        lambda _row: (
            "fetched",
            "deezer_isrc",
            "42",
            30.0,
            "audio/previews/aa/track.mp3",
        ),
    )

    result = audio.fetch_pending(conn, batch=1)

    assert result["fetched"] == 1
    assert dict(
        conn.execute(
            "SELECT model, status FROM embedding_runs "
            "WHERE recording_key='artist\ttrack' ORDER BY model"
        )
    ) == {
        "clap-music": "pending",
        "discogs-effnet": "pending",
        "discogs400": "pending",
    }
