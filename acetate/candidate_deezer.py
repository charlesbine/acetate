"""Resolve outside-library candidates to concrete Deezer tracks.

This pass only resolves identity and preview availability. It does not download
audio; preview fetching remains a separately budgeted promotion step.
"""

from __future__ import annotations

import json
import re
import sqlite3
from statistics import median
from typing import Any, Callable

import requests

from . import audio, candidates, db
from .canon import _norm, title_lookup_variants

PROVIDER = "deezer"
MAX_ATTEMPTS = 3
DURATION_TOLERANCE_S = 5

SearchFn = Callable[[str, str], list[dict[str, Any]]]
_MATCH_SEPARATORS = re.compile(r"[^\w]+", re.UNICODE)


def _duration_value(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and ":" in value:
        try:
            parts = [int(part) for part in value.split(":")]
        except ValueError:
            return None
        if len(parts) == 2:
            return float(parts[0] * 60 + parts[1])
        if len(parts) == 3:
            return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _known_duration_s(conn: sqlite3.Connection, candidate_key: str) -> float | None:
    values = []
    for row in conn.execute(
        """
        SELECT raw_json
        FROM candidate_sources
        WHERE candidate_key = ? AND source = 'ytmusic_radio' AND raw_json IS NOT NULL
        """,
        (candidate_key,),
    ):
        try:
            raw = json.loads(row["raw_json"])
        except (TypeError, json.JSONDecodeError):
            continue
        value = _duration_value(raw.get("duration_seconds", raw.get("duration", raw.get("length"))))
        if value is not None and value > 0:
            values.append(value)
    return float(median(values)) if values else None


def _artist_names(result: dict[str, Any]) -> set[str]:
    names = {_norm((result.get("artist") or {}).get("name"))}
    names.update(
        _norm(item.get("name")) for item in result.get("contributors") or [] if item.get("name")
    )
    return {name for name in names if name}


def _match_text(value: str | None) -> str:
    """Normalize punctuation while preserving words such as version labels."""
    return _MATCH_SEPARATORS.sub(" ", (value or "").casefold()).strip()


def _select_match(
    results: list[dict[str, Any]],
    *,
    artist: str,
    title: str,
    expected_duration_s: float | None,
) -> tuple[dict[str, Any], str, float] | None:
    artist_norm = _norm(artist)
    title_variants = title_lookup_variants(title)
    if not artist_norm or not title_variants:
        return None

    matches = []
    for rank, result in enumerate(results):
        if artist_norm not in _artist_names(result):
            continue
        result_title = _norm(result.get("title"))
        result_match_text = _match_text(result.get("title"))
        exact_title = result_title == title_variants[0] or result_match_text == _match_text(title)
        fallback_title = result_title in title_variants[1:] or result_match_text in {
            _match_text(variant) for variant in title_variants[1:]
        }
        if not exact_title and not fallback_title:
            continue

        matched_duration_s = _duration_value(result.get("duration"))
        if expected_duration_s is not None:
            if matched_duration_s is None:
                continue
            if abs(matched_duration_s - expected_duration_s) > DURATION_TOLERANCE_S:
                continue
        elif fallback_title:
            continue

        method = "search_exact" if exact_title else "search_title_variant"
        confidence = 0.98 if expected_duration_s is not None and exact_title else 0.90
        if fallback_title:
            confidence = 0.80
        matches.append((confidence, bool(result.get("preview")), -rank, result, method))

    if not matches:
        return None
    confidence, _preview, _rank, result, method = max(matches, key=lambda item: item[:3])
    return result, method, confidence


def _candidate_rows(
    conn: sqlite3.Connection,
    *,
    batch: int,
    min_score: float,
    include_attempted: bool,
) -> list[sqlite3.Row]:
    attempted_filter = (
        ""
        if include_attempted
        else """
        AND (
            resolution.candidate_key IS NULL
            OR (resolution.status = 'error' AND resolution.attempts < ?)
        )
    """
    )
    params: list[Any] = [PROVIDER, min_score]
    if not include_attempted:
        params.append(MAX_ATTEMPTS)
    params.append(batch)
    return conn.execute(
        f"""
        SELECT cr.candidate_key, cr.artist, cr.title, cr.score, cr.source_count
        FROM candidate_recordings cr
        LEFT JOIN candidate_resolutions resolution
          ON resolution.candidate_key = cr.candidate_key AND resolution.provider = ?
        WHERE cr.status = 'metadata'
          AND cr.score >= ?
          AND cr.artist IS NOT NULL AND cr.artist != ''
          AND cr.title IS NOT NULL AND cr.title != ''
          {attempted_filter}
        ORDER BY cr.score DESC, cr.source_count DESC, cr.last_seen DESC
        LIMIT ?
        """,
        params,
    ).fetchall()


def _record_resolution(
    conn: sqlite3.Connection,
    *,
    candidate_key: str,
    status: str,
    external_id: str | None = None,
    method: str | None = None,
    confidence: float | None = None,
    expected_duration_s: float | None = None,
    matched_duration_s: float | None = None,
    preview_url: str | None = None,
    isrc: str | None = None,
    raw: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO candidate_resolutions(
            candidate_key, provider, status, external_id, method, confidence,
            expected_duration_s, matched_duration_s, has_preview, preview_url,
            isrc, attempts, error, raw_json, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
        ON CONFLICT(candidate_key, provider) DO UPDATE SET
            status=excluded.status,
            external_id=excluded.external_id,
            method=excluded.method,
            confidence=excluded.confidence,
            expected_duration_s=excluded.expected_duration_s,
            matched_duration_s=excluded.matched_duration_s,
            has_preview=excluded.has_preview,
            preview_url=excluded.preview_url,
            isrc=excluded.isrc,
            attempts=candidate_resolutions.attempts + 1,
            error=excluded.error,
            raw_json=excluded.raw_json,
            updated_at=excluded.updated_at
        """,
        (
            candidate_key,
            PROVIDER,
            status,
            external_id,
            method,
            confidence,
            expected_duration_s,
            matched_duration_s,
            int(bool(preview_url)),
            preview_url,
            isrc,
            error[:300] if error else None,
            json.dumps(raw, ensure_ascii=False, sort_keys=True) if raw is not None else None,
            db.utcnow(),
        ),
    )


def _remaining(conn: sqlite3.Connection, min_score: float) -> int:
    return conn.execute(
        """
        SELECT COUNT(*)
        FROM candidate_recordings cr
        LEFT JOIN candidate_resolutions resolution
          ON resolution.candidate_key = cr.candidate_key AND resolution.provider = ?
        WHERE cr.status = 'metadata' AND cr.score >= ?
          AND (
            resolution.candidate_key IS NULL
            OR (resolution.status = 'error' AND resolution.attempts < ?)
          )
        """,
        (PROVIDER, min_score, MAX_ATTEMPTS),
    ).fetchone()[0]


def resolve(
    conn: sqlite3.Connection,
    *,
    batch: int = 100,
    min_score: float = 0.0,
    include_attempted: bool = False,
    search: SearchFn | None = None,
    progress=None,
) -> dict[str, int]:
    """Resolve top metadata candidates through Deezer search, resumably."""
    search = search or audio.deezer_search
    rows = _candidate_rows(
        conn,
        batch=batch,
        min_score=min_score,
        include_attempted=include_attempted,
    )
    counts = {
        "seen": 0,
        "matched": 0,
        "preview": 0,
        "duplicate": 0,
        "provider_miss": 0,
        "error": 0,
    }

    for index, row in enumerate(rows, start=1):
        counts["seen"] += 1
        expected_duration_s = _known_duration_s(conn, row["candidate_key"])
        try:
            results = search(row["artist"], row["title"])
            match = _select_match(
                results,
                artist=row["artist"],
                title=row["title"],
                expected_duration_s=expected_duration_s,
            )
            if match is None:
                _record_resolution(
                    conn,
                    candidate_key=row["candidate_key"],
                    status="provider_miss",
                    expected_duration_s=expected_duration_s,
                    raw={"results": results},
                    error="no conservative Deezer match",
                )
                counts["provider_miss"] += 1
            else:
                result, method, confidence = match
                external_id = str(result["id"])
                preview_url = result.get("preview") or None
                isrc = result.get("isrc") or None
                _record_resolution(
                    conn,
                    candidate_key=row["candidate_key"],
                    status="matched",
                    external_id=external_id,
                    method=method,
                    confidence=confidence,
                    expected_duration_s=expected_duration_s,
                    matched_duration_s=_duration_value(result.get("duration")),
                    preview_url=preview_url,
                    isrc=isrc,
                    raw={"selected": result, "results": results},
                )
                candidates.add_external_id(
                    conn,
                    candidate_key=row["candidate_key"],
                    provider=PROVIDER,
                    external_id=external_id,
                    id_type="track_id",
                    confidence=confidence,
                    raw=result,
                )
                if isrc:
                    candidates.add_external_id(
                        conn,
                        candidate_key=row["candidate_key"],
                        provider="isrc",
                        external_id=isrc,
                        id_type="isrc",
                        confidence=confidence,
                        raw={"deezer_id": external_id},
                    )
                duplicate_key = candidates.library_recording_for_isrc(conn, isrc) if isrc else None
                if duplicate_key:
                    candidates.mark_library_duplicate(
                        conn,
                        candidate_key=row["candidate_key"],
                        library_recording_key=duplicate_key,
                        method="isrc",
                        external_id=isrc,
                    )
                    counts["duplicate"] += 1
                else:
                    conn.execute(
                        "UPDATE candidate_recordings SET status='resolved', updated_at=? "
                        "WHERE candidate_key=? AND status='metadata'",
                        (db.utcnow(), row["candidate_key"]),
                    )
                counts["matched"] += 1
                counts["preview"] += int(bool(preview_url))
        except requests.RequestException as exc:
            _record_resolution(
                conn,
                candidate_key=row["candidate_key"],
                status="error",
                expected_duration_s=expected_duration_s,
                error=str(exc),
            )
            counts["error"] += 1
        conn.commit()
        if progress:
            progress(index, len(rows), counts)

    counts["remaining"] = _remaining(conn, min_score)
    return counts


def status(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        f"resolution:{row['provider']}:{row['status']}": row["n"]
        for row in conn.execute(
            "SELECT provider, status, COUNT(*) AS n "
            "FROM candidate_resolutions GROUP BY provider, status"
        )
    }
