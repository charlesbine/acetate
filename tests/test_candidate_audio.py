from __future__ import annotations

import hashlib
import sqlite3

import requests

from acetate import candidate_audio, candidates, db


def _conn(tmp_path) -> sqlite3.Connection:
    conn = db.connect(tmp_path / "acetate-test.db")
    db.init_db(conn)
    return conn


def _resolved_candidate(
    conn: sqlite3.Connection,
    *,
    artist: str = "New Artist",
    title: str = "New Track",
    strength: float = 0.9,
    preview_url: str = "https://example.test/preview.mp3",
    external_id: str = "42",
) -> str:
    key = candidates.add_candidate_source(
        conn,
        artist=artist,
        title=title,
        source="test",
        reason="test",
        strength=strength,
        known=set(),
    )
    assert key is not None
    candidates.score_candidates(conn)
    conn.execute("UPDATE candidate_recordings SET status='resolved' WHERE candidate_key=?", (key,))
    conn.execute(
        """
        INSERT INTO candidate_resolutions(
            candidate_key, provider, status, external_id, has_preview,
            preview_url, matched_duration_s, attempts, updated_at
        )
        VALUES(?, 'deezer', 'matched', ?, 1, ?, 205, 1, 'now')
        """,
        (key, external_id, preview_url),
    )
    conn.commit()
    return key


def _mp3() -> bytes:
    return b"ID3" + b"\x00" * candidate_audio.MIN_PREVIEW_BYTES


def _refresh(preview_url: str = "https://example.test/preview.mp3"):
    return lambda _external_id: {"id": 42, "preview": preview_url}


def test_fetch_candidate_preview_is_validated_stored_and_idempotent(tmp_path) -> None:
    conn = _conn(tmp_path)
    key = _resolved_candidate(conn)
    calls = 0

    def download(_url: str) -> tuple[bytes, str]:
        nonlocal calls
        calls += 1
        return _mp3(), "audio/mpeg"

    first = candidate_audio.fetch_previews(
        conn, batch=1, download=download, refresh=_refresh(), workers=1
    )
    second = candidate_audio.fetch_previews(
        conn, batch=1, download=download, refresh=_refresh(), workers=1
    )

    assert first == {
        "seen": 1,
        "fetched": 1,
        "refreshed": 0,
        "provider_miss": 0,
        "source_not_covered": 0,
        "error": 0,
        "remaining": 0,
    }
    assert second["seen"] == 0
    assert calls == 1
    row = conn.execute("SELECT * FROM candidate_audio WHERE candidate_key=?", (key,)).fetchone()
    assert row["status"] == "fetched"
    assert row["size_bytes"] == len(_mp3())
    assert row["sha256"] == hashlib.sha256(_mp3()).hexdigest()
    assert (tmp_path / row["path"]).read_bytes() == _mp3()
    assert (
        conn.execute(
            "SELECT status FROM candidate_recordings WHERE candidate_key=?", (key,)
        ).fetchone()[0]
        == "preview"
    )
    assert dict(
        conn.execute(
            "SELECT model, status FROM candidate_embedding_runs "
            "WHERE candidate_key=? ORDER BY model",
            (key,),
        )
    ) == {
        "clap-music": "pending",
        "discogs-effnet": "pending",
        "discogs400": "pending",
    }


def test_bad_candidate_preview_retries_then_parks(tmp_path) -> None:
    conn = _conn(tmp_path)
    key = _resolved_candidate(conn)

    def download(_url: str) -> tuple[bytes, str]:
        return b"not an mp3", "text/html"

    results = [
        candidate_audio.fetch_previews(
            conn, batch=1, download=download, refresh=_refresh(), workers=1
        )
        for _ in range(4)
    ]

    assert [result["seen"] for result in results] == [1, 1, 1, 0]
    row = conn.execute(
        "SELECT status, attempts, error FROM candidate_audio WHERE candidate_key=?", (key,)
    ).fetchone()
    assert row["status"] == "error"
    assert row["attempts"] == candidate_audio.MAX_ATTEMPTS
    assert "content type" in row["error"]
    assert (
        conn.execute(
            "SELECT status FROM candidate_recordings WHERE candidate_key=?", (key,)
        ).fetchone()[0]
        == "resolved"
    )


def test_fetch_candidate_previews_uses_score_order_and_batch(tmp_path) -> None:
    conn = _conn(tmp_path)
    low = _resolved_candidate(
        conn,
        artist="Low Artist",
        title="Low Track",
        strength=0.2,
        preview_url="https://example.test/low.mp3",
        external_id="41",
    )
    high = _resolved_candidate(
        conn,
        artist="High Artist",
        title="High Track",
        strength=1.0,
        preview_url="https://example.test/high.mp3",
        external_id="42",
    )
    urls = []

    def download(url: str) -> tuple[bytes, str]:
        urls.append(url)
        return _mp3(), "audio/mpeg"

    result = candidate_audio.fetch_previews(
        conn,
        batch=1,
        download=download,
        refresh=lambda external_id: {
            "id": external_id,
            "preview": (
                "https://example.test/high.mp3"
                if external_id == "42"
                else "https://example.test/low.mp3"
            ),
        },
        workers=1,
    )

    assert result["fetched"] == 1
    assert urls == ["https://example.test/high.mp3"]
    assert (
        conn.execute(
            "SELECT status FROM candidate_recordings WHERE candidate_key=?", (high,)
        ).fetchone()[0]
        == "preview"
    )
    assert (
        conn.execute(
            "SELECT status FROM candidate_recordings WHERE candidate_key=?", (low,)
        ).fetchone()[0]
        == "resolved"
    )


def test_fetch_candidate_preview_refreshes_expired_signed_url(tmp_path) -> None:
    conn = _conn(tmp_path)
    key = _resolved_candidate(conn, preview_url="https://cdn.test/expired.mp3")
    urls = []

    def download(url: str) -> tuple[bytes, str]:
        urls.append(url)
        return _mp3(), "audio/mpeg"

    result = candidate_audio.fetch_previews(
        conn,
        batch=1,
        download=download,
        refresh=_refresh("https://cdn.test/fresh.mp3"),
        workers=1,
    )

    assert result["fetched"] == 1
    assert result["refreshed"] == 1
    assert urls == ["https://cdn.test/fresh.mp3"]
    resolution = conn.execute(
        "SELECT preview_url, has_preview, status FROM candidate_resolutions WHERE candidate_key=?",
        (key,),
    ).fetchone()
    assert tuple(resolution) == ("https://cdn.test/fresh.mp3", 1, "matched")
    assert (
        conn.execute(
            "SELECT source_url FROM candidate_audio WHERE candidate_key=?", (key,)
        ).fetchone()[0]
        == "https://cdn.test/fresh.mp3"
    )


def test_refresh_distinguishes_provider_miss_from_no_preview(tmp_path) -> None:
    conn = _conn(tmp_path)
    missing = _resolved_candidate(conn, artist="Missing", title="Gone", external_id="41")
    no_preview = _resolved_candidate(conn, artist="Present", title="No Preview", external_id="42")

    def refresh(external_id: str):
        if external_id == "41":
            return None
        if external_id == "42":
            return {"id": 42, "preview": None}
        raise AssertionError(external_id)

    result = candidate_audio.fetch_previews(conn, batch=2, refresh=refresh, workers=1)

    assert result["provider_miss"] == 1
    assert result["source_not_covered"] == 1
    missing_resolution = conn.execute(
        "SELECT status, has_preview FROM candidate_resolutions WHERE candidate_key=?",
        (missing,),
    ).fetchone()
    no_preview_resolution = conn.execute(
        "SELECT status, has_preview FROM candidate_resolutions WHERE candidate_key=?",
        (no_preview,),
    ).fetchone()
    assert tuple(missing_resolution) == ("provider_miss", 0)
    assert tuple(no_preview_resolution) == ("matched", 0)
    assert (
        conn.execute(
            "SELECT status FROM candidate_recordings WHERE candidate_key=?", (missing,)
        ).fetchone()[0]
        == "metadata"
    )
    assert (
        conn.execute(
            "SELECT status FROM candidate_audio WHERE candidate_key=?", (no_preview,)
        ).fetchone()[0]
        == "source_not_covered"
    )


def test_refresh_request_error_remains_retryable_not_unavailable(tmp_path) -> None:
    conn = _conn(tmp_path)
    key = _resolved_candidate(conn)

    def refresh(_external_id: str):
        raise requests.Timeout("temporary Deezer failure")

    result = candidate_audio.fetch_previews(conn, batch=1, refresh=refresh, workers=1)

    assert result["error"] == 1
    assert result["provider_miss"] == 0
    assert result["source_not_covered"] == 0
    assert tuple(
        conn.execute(
            "SELECT status, has_preview FROM candidate_resolutions WHERE candidate_key=?",
            (key,),
        ).fetchone()
    ) == ("matched", 1)
    assert tuple(
        conn.execute(
            "SELECT status, attempts FROM candidate_audio WHERE candidate_key=?",
            (key,),
        ).fetchone()
    ) == ("pending", 1)


def test_repair_parked_403s_is_dry_run_then_targets_only_403_errors(tmp_path) -> None:
    conn = _conn(tmp_path)
    parked = _resolved_candidate(conn, artist="Parked", title="Expired URL", external_id="41")
    unrelated = _resolved_candidate(conn, artist="Other", title="Bad Content", external_id="42")
    conn.executemany(
        """
        INSERT INTO candidate_audio(
            candidate_key, status, source, source_id, source_url,
            attempts, error, updated_at
        )
        VALUES(?, 'error', 'deezer_preview', ?, 'https://example.test/old.mp3',
               3, ?, 'now')
        """,
        [
            (
                parked,
                "41",
                "403 Client Error: Forbidden for url: https://example.test/expired.mp3",
            ),
            (unrelated, "42", "preview is not recognizable MP3 data"),
        ],
    )
    conn.commit()
    calls = []

    def refresh(external_id: str):
        calls.append(external_id)
        return {"id": external_id, "preview": "https://example.test/fresh.mp3"}

    dry_run = candidate_audio.repair_parked_403s(
        conn, apply=False, batch=10, refresh=refresh, workers=1
    )
    applied = candidate_audio.repair_parked_403s(
        conn,
        apply=True,
        batch=10,
        refresh=refresh,
        download=lambda _url: (_mp3(), "audio/mpeg"),
        workers=1,
    )

    assert dry_run == {"mode": "dry_run", "eligible": 1, "batch": 1}
    assert calls == ["41"]
    assert applied["mode"] == "applied"
    assert applied["eligible_before"] == 1
    assert applied["fetched"] == 1
    assert applied["remaining"] == 0
    assert (
        conn.execute(
            "SELECT status FROM candidate_audio WHERE candidate_key=?", (parked,)
        ).fetchone()[0]
        == "fetched"
    )
    assert (
        conn.execute(
            "SELECT status FROM candidate_audio WHERE candidate_key=?", (unrelated,)
        ).fetchone()[0]
        == "error"
    )
