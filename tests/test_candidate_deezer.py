from __future__ import annotations

import sqlite3

import pytest
import requests

from acetate import audio, candidate_deezer, candidates, db


def _conn(tmp_path) -> sqlite3.Connection:
    conn = db.connect(tmp_path / "acetate-test.db")
    db.init_db(conn)
    return conn


def _candidate(
    conn: sqlite3.Connection,
    *,
    artist: str = "New Artist",
    title: str = "New Track",
    source: str = "lastfm_track_similar",
    raw: dict | None = None,
) -> str:
    key = candidates.add_candidate_source(
        conn,
        artist=artist,
        title=title,
        source=source,
        reason="test",
        strength=0.9,
        raw=raw,
        known=set(),
    )
    conn.commit()
    candidates.score_candidates(conn)
    assert key is not None
    return key


def test_resolve_deezer_stores_match_preview_and_external_ids(tmp_path) -> None:
    conn = _conn(tmp_path)
    key = _candidate(
        conn,
        source="ytmusic_radio",
        raw={"length": "3:25"},
    )

    def search(_artist: str, _title: str) -> list[dict]:
        return [
            {
                "id": 1,
                "artist": {"name": "Wrong Artist"},
                "title": "New Track",
                "duration": 205,
                "preview": "https://example.test/wrong.mp3",
            },
            {
                "id": 42,
                "artist": {"name": "New Artist"},
                "title": "New Track",
                "duration": 205,
                "preview": "https://example.test/preview.mp3",
                "isrc": "USABC1234567",
            },
        ]

    result = candidate_deezer.resolve(conn, batch=1, search=search)

    assert result["matched"] == 1
    assert result["preview"] == 1
    resolution = conn.execute(
        "SELECT * FROM candidate_resolutions WHERE candidate_key=?", (key,)
    ).fetchone()
    assert resolution["status"] == "matched"
    assert resolution["external_id"] == "42"
    assert resolution["expected_duration_s"] == 205
    assert resolution["matched_duration_s"] == 205
    assert resolution["has_preview"] == 1
    assert (
        conn.execute(
            "SELECT status FROM candidate_recordings WHERE candidate_key=?", (key,)
        ).fetchone()[0]
        == "resolved"
    )
    external_ids = {
        (row["provider"], row["external_id"])
        for row in conn.execute(
            "SELECT provider, external_id FROM candidate_external_ids WHERE candidate_key=?", (key,)
        )
    }
    assert external_ids == {("deezer", "42"), ("isrc", "USABC1234567")}


def test_deezer_title_fallback_requires_matching_duration() -> None:
    results = [
        {
            "id": 42,
            "artist": {"name": "New Artist"},
            "title": "New Track",
            "duration": 204,
            "preview": "",
        }
    ]

    match = candidate_deezer._select_match(
        results,
        artist="New Artist",
        title="New Track (Radio Edit)",
        expected_duration_s=205,
    )

    assert match is not None
    assert match[1] == "search_title_variant"
    assert (
        candidate_deezer._select_match(
            results,
            artist="New Artist",
            title="New Track (Radio Edit)",
            expected_duration_s=None,
        )
        is None
    )
    assert (
        candidate_deezer._select_match(
            results,
            artist="New Artist",
            title="New Track (Radio Edit)",
            expected_duration_s=220,
        )
        is None
    )

    punctuation_match = candidate_deezer._select_match(
        [{**results[0], "title": "New Track - Radio Edit"}],
        artist="New Artist",
        title="New Track (Radio Edit)",
        expected_duration_s=None,
    )
    assert punctuation_match is not None
    assert punctuation_match[1] == "search_exact"


def test_deezer_miss_is_cached_and_not_retried(tmp_path) -> None:
    conn = _conn(tmp_path)
    key = _candidate(conn)
    calls = 0

    def search(_artist: str, _title: str) -> list[dict]:
        nonlocal calls
        calls += 1
        return [
            {
                "id": 7,
                "artist": {"name": "Someone Else"},
                "title": "New Track",
                "duration": 200,
            }
        ]

    first = candidate_deezer.resolve(conn, batch=1, search=search)
    second = candidate_deezer.resolve(conn, batch=1, search=search)

    assert first["provider_miss"] == 1
    assert second["seen"] == 0
    assert calls == 1
    assert (
        conn.execute(
            "SELECT status FROM candidate_resolutions WHERE candidate_key=?",
            (key,),
        ).fetchone()[0]
        == "provider_miss"
    )
    assert (
        conn.execute(
            "SELECT status FROM candidate_recordings WHERE candidate_key=?", (key,)
        ).fetchone()[0]
        == "metadata"
    )


def test_deezer_errors_retry_then_park(tmp_path) -> None:
    conn = _conn(tmp_path)
    key = _candidate(conn)

    def search(_artist: str, _title: str) -> list[dict]:
        raise requests.Timeout("slow")

    results = [candidate_deezer.resolve(conn, batch=1, search=search) for _ in range(4)]

    assert [result["seen"] for result in results] == [1, 1, 1, 0]
    resolution = conn.execute(
        "SELECT status, attempts FROM candidate_resolutions WHERE candidate_key=?", (key,)
    ).fetchone()
    assert resolution["status"] == "error"
    assert resolution["attempts"] == candidate_deezer.MAX_ATTEMPTS


def test_deezer_search_error_payload_is_not_an_empty_result(monkeypatch) -> None:
    monkeypatch.setattr(
        audio,
        "_get_json",
        lambda *_args, **_kwargs: {
            "error": {"type": "QuotaException", "code": 4, "message": "slow down"}
        },
    )

    with pytest.raises(requests.RequestException, match="Deezer search response"):
        audio.deezer_search("Artist", "Track")


def test_deezer_track_distinguishes_missing_from_transient_error(monkeypatch) -> None:
    monkeypatch.setattr(
        audio,
        "_get_json",
        lambda *_args, **_kwargs: {
            "error": {"type": "DataException", "code": 800, "message": "no data"}
        },
    )
    assert audio.deezer_track("42") is None

    monkeypatch.setattr(
        audio,
        "_get_json",
        lambda *_args, **_kwargs: {
            "error": {"type": "QuotaException", "code": 4, "message": "slow down"}
        },
    )
    with pytest.raises(requests.RequestException, match="Deezer track response"):
        audio.deezer_track("42")
