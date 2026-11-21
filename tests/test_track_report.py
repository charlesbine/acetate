from __future__ import annotations

import json

import pytest
import requests

from acetate import agent, genre, ledger, lastfm, track_report
from tests.conftest import add_recording


def _fedorex(conn) -> None:
    add_recording(
        conn, "fedorex\tsorry", "FedoreX", "SORRY", "6wRHPTzhNXJyuowV3PTQwM", plays=6, liked=0
    )
    conn.execute(
        "UPDATE recording_stats SET first_at='2026-07-12T00:13:07Z', "
        "last_at='2026-07-21T04:10:00Z' WHERE recording_key='fedorex\tsorry'"
    )
    conn.execute(
        "INSERT INTO albums(album_id,name,release_date,raw_json,updated_at) VALUES(?,?,?,?,?)",
        (
            "album",
            "SORRY",
            "2025-12-02",
            json.dumps({"album_type": "single", "total_tracks": 1}),
            "2026-07-21T04:21:16Z",
        ),
    )
    conn.execute(
        "INSERT INTO tracks(track_id,name,album_id,duration_ms,popularity,raw_json,updated_at) "
        "VALUES(?,?,?,?,?,?,?)",
        (
            "6wRHPTzhNXJyuowV3PTQwM",
            "SORRY",
            "album",
            237165,
            None,
            json.dumps({"id": "6wRHPTzhNXJyuowV3PTQwM", "name": "SORRY"}),
            "2026-07-21T04:21:16Z",
        ),
    )
    conn.execute(
        "INSERT INTO track_genres(track_id,genre,source,confidence,assigned_at) "
        "VALUES(?, '_none_', 'lastfm_none', 0, ?)",
        ("6wRHPTzhNXJyuowV3PTQwM", "2026-07-12T00:30:00Z"),
    )
    conn.commit()


def test_track_report_distinguishes_missing_states(conn) -> None:
    _fedorex(conn)

    result = track_report.report(conn, "https://open.spotify.com/track/6wRHPTzhNXJyuowV3PTQwM?si=x")

    assert result["track"]["artist"] == "FedoreX"
    assert result["personal"]["plays"] == 6
    assert result["surfaces"]["spotify_catalog_popularity"] == {
        "state": "source_not_covered",
        "reason": "spotify_track_payload_omitted_popularity",
        "updated_at": "2026-07-21T04:21:16Z",
    }
    assert result["surfaces"]["lastfm_tags"]["state"] == "unknown_legacy"
    assert result["surfaces"]["classic_spotify_features"]["state"] == "not_scheduled"
    assert result["surfaces"]["musicbrainz_recording"]["state"] == "not_scheduled"
    assert result["surfaces"]["ytmusic_seed_resolution"]["state"] == "not_scheduled"
    assert result["surfaces"]["ytmusic_radio"]["state"] == "not_scheduled"
    assert result["surfaces"]["audio_preview"]["state"] == "not_scheduled"
    assert result["surfaces"]["embeddings"]["discogs-effnet"]["state"] == "blocked"
    assert result["surfaces"]["deezer_metadata"]["track"]["state"] == "blocked"


def test_track_report_normalizes_provider_misses_and_pending_work(conn) -> None:
    _fedorex(conn)
    recording_key = "fedorex\tsorry"
    conn.execute(
        """
        INSERT INTO audio_cache(
            recording_key, status, source, source_id, attempts, updated_at
        )
        VALUES(?, 'fetched', 'deezer_search', '42', 1, 'now')
        """,
        (recording_key,),
    )
    conn.execute(
        """
        INSERT INTO deezer_metadata_runs(
            entity_type, entity_id, status, attempts, error, updated_at
        )
        VALUES('track', '42', 'provider_miss', 1, 'no data', 'now')
        """
    )
    conn.execute(
        """
        INSERT INTO recording_mbid(
            recording_key, isrc, status, attempts, updated_at
        )
        VALUES(?, 'USABC1234567', 'miss', 1, 'now')
        """,
        (recording_key,),
    )
    conn.execute(
        """
        INSERT INTO ytmusic_seed_matches(
            recording_key, source, confidence, status, updated_at
        )
        VALUES(?, 'laion_disco_12m', 0, 'miss', 'now')
        """,
        (recording_key,),
    )
    conn.execute(
        """
        INSERT INTO ytmusic_radio_seeds(
            recording_key, status, attempts, result_count, updated_at
        )
        VALUES(?, 'pending', 0, 0, 'now')
        """,
        (recording_key,),
    )
    conn.commit()

    result = track_report.report(conn, "https://open.spotify.com/track/6wRHPTzhNXJyuowV3PTQwM")
    surfaces = result["surfaces"]

    assert surfaces["audio_preview"]["state"] == "available"
    assert surfaces["deezer_metadata"]["track"]["state"] == "provider_miss"
    assert surfaces["musicbrainz_recording"]["state"] == "provider_miss"
    assert surfaces["ytmusic_seed_resolution"]["state"] == "provider_miss"
    assert surfaces["ytmusic_radio"]["state"] == "pending"


def test_legacy_repair_keeps_empty_lastfm_ambiguous(conn) -> None:
    _fedorex(conn)
    conn.execute(
        "INSERT INTO track_genres(track_id,genre,source,confidence,assigned_at) "
        "VALUES('tagged', 'techno', 'lastfm', 80, '2026-07-01')"
    )
    conn.commit()

    dry = ledger.repair_legacy(conn)
    assert dry["would_repair"]["lastfm_none_ambiguous_without_ledger"] == 1

    applied = ledger.repair_legacy(conn, apply=True)
    assert applied["mode"] == "applied"
    states = dict(
        conn.execute(
            "SELECT track_id,status FROM lastfm_tag_runs WHERE track_id IN "
            "('6wRHPTzhNXJyuowV3PTQwM','tagged')"
        )
    )
    assert states["6wRHPTzhNXJyuowV3PTQwM"] == "unknown_legacy"
    assert states["tagged"] == "available"


def test_legacy_repair_materializes_pending_embedding_work(conn) -> None:
    conn.execute(
        """
        INSERT INTO audio_cache(recording_key, status, path, updated_at)
        VALUES('library', 'fetched', 'audio/library.mp3', 'now')
        """
    )
    conn.execute(
        """
        INSERT INTO candidate_recordings(candidate_key, status, updated_at)
        VALUES('candidate', 'preview', 'now')
        """
    )
    conn.execute(
        """
        INSERT INTO candidate_audio(candidate_key, status, path, updated_at)
        VALUES('candidate', 'fetched', 'audio/candidate.mp3', 'now')
        """
    )
    conn.commit()

    dry = ledger.repair_legacy(conn)
    assert dry["would_repair"]["library_embedding_work_without_ledger"] == 3
    assert dry["would_repair"]["candidate_embedding_work_without_ledger"] == 3

    ledger.repair_legacy(conn, apply=True)

    assert (
        conn.execute(
            "SELECT COUNT(*) FROM embedding_runs WHERE recording_key='library' AND status='pending'"
        ).fetchone()[0]
        == 3
    )
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM candidate_embedding_runs "
            "WHERE candidate_key='candidate' AND status='pending'"
        ).fetchone()[0]
        == 3
    )


def test_lastfm_request_errors_are_not_empty_tags(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise requests.Timeout("provider timed out")

    monkeypatch.setattr(lastfm, "_call", fail)
    with pytest.raises(requests.Timeout):
        lastfm.best_tags_result("FedoreX", "SORRY")


def test_raw_sql_budget_is_enforced(conn, monkeypatch) -> None:
    monkeypatch.setattr(agent.query, "run_query", lambda sql: {"rows": [[sql]], "row_count": 1})
    seen: dict[str, str] = {}
    usage: dict[str, int] = {}

    for i in range(agent.SQL_BUDGET):
        out, _ = agent.exec_tool(conn, "query_db", {"sql": f"SELECT {i}"}, seen, usage)
        assert "row_count" in out

    spent, _ = agent.exec_tool(conn, "query_db", {"sql": "SELECT over"}, seen, usage)
    assert spent.startswith("ERROR: raw SQL budget exhausted")


def test_genre_backfill_does_not_persist_request_error_as_none(conn, monkeypatch) -> None:
    monkeypatch.setenv("LASTFM_API_KEY", "test")

    def fail(*args, **kwargs):
        raise requests.Timeout("provider timed out")

    monkeypatch.setattr(lastfm, "best_tags_result", fail)
    result = genre._run_tagging(
        conn, [{"track_id": "failed", "artist_name": "Artist", "track_name": "Track"}]
    )

    assert result["error"] == 1
    assert (
        conn.execute("SELECT COUNT(*) FROM track_genres WHERE track_id='failed'").fetchone()[0] == 0
    )
    assert (
        conn.execute("SELECT status FROM lastfm_tag_runs WHERE track_id='failed'").fetchone()[0]
        == "pending"
    )


def test_genre_backfill_records_authoritative_provider_miss(conn, monkeypatch) -> None:
    monkeypatch.setenv("LASTFM_API_KEY", "test")
    monkeypatch.setattr(
        lastfm,
        "best_tags_result",
        lambda *a, **k: lastfm.TagLookup([], "provider_miss", "track_and_artist_missing"),
    )

    genre._run_tagging(
        conn, [{"track_id": "missing", "artist_name": "Missing", "track_name": "Track"}]
    )

    assert conn.execute("SELECT status FROM lastfm_tag_runs WHERE track_id='missing'").fetchone()[
        0
    ] == ("provider_miss")
    assert (
        conn.execute("SELECT source FROM track_genres WHERE track_id='missing'").fetchone()[0]
        == "lastfm_none"
    )
