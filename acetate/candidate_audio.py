"""Promote resolved recommendation candidates into a local preview cache."""

from __future__ import annotations

import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import requests

from . import audio, db
from .audio import USER_AGENT

MAX_ATTEMPTS = 3
MIN_PREVIEW_BYTES = 10_000
WORKERS = 4
PARKED_403_PATTERN = "%403 Client Error:%Forbidden%"

DownloadFn = Callable[[str], tuple[bytes, str | None]]
RefreshFn = Callable[[str], dict[str, Any] | None]


class PreviewUnavailableError(ValueError):
    """The resolved provider entity exists, but no preview is currently offered."""


class ProviderMissError(ValueError):
    """The previously resolved provider entity no longer exists."""


def _data_root(conn: sqlite3.Connection) -> Path:
    for row in conn.execute("PRAGMA database_list"):
        if row["name"] == "main" and row["file"]:
            return Path(row["file"]).resolve().parent
    raise ValueError("candidate audio cache requires a file-backed SQLite database")


def _download_preview(url: str) -> tuple[bytes, str | None]:
    response = requests.get(url, timeout=30, headers=USER_AGENT)
    response.raise_for_status()
    return response.content, response.headers.get("Content-Type")


def _looks_like_mp3(content: bytes) -> bool:
    if content.startswith(b"ID3"):
        return True
    scan_end = min(len(content) - 1, 4096)
    return any(
        content[index] == 0xFF and content[index + 1] & 0xE0 == 0xE0
        for index in range(max(0, scan_end))
    )


def _store_preview(
    data_root: Path,
    candidate_key: str,
    content: bytes,
    content_type: str | None,
) -> tuple[str, int, str]:
    media_type = (content_type or "").partition(";")[0].strip().lower()
    if media_type and not (
        media_type.startswith("audio/") or media_type == "application/octet-stream"
    ):
        raise ValueError(f"unexpected preview content type: {media_type}")
    if len(content) < MIN_PREVIEW_BYTES:
        raise ValueError(f"preview too small ({len(content)} bytes)")
    if not _looks_like_mp3(content):
        raise ValueError("preview is not recognizable MP3 data")

    key_hash = hashlib.sha1(candidate_key.encode()).hexdigest()
    rel = f"audio/candidates/{key_hash[:2]}/{key_hash}.mp3"
    destination = data_root / rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".mp3.part")
    temporary.write_bytes(content)
    temporary.replace(destination)
    return rel, len(content), hashlib.sha256(content).hexdigest()


def _candidate_rows(
    conn: sqlite3.Connection,
    *,
    batch: int,
    min_score: float,
    include_attempted: bool,
    only_parked_403: bool,
) -> list[sqlite3.Row]:
    if only_parked_403:
        attempted_filter = """
        AND ca.status = 'error'
        AND ca.error LIKE ?
        """
    elif include_attempted:
        attempted_filter = ""
    else:
        attempted_filter = """
        AND (
            ca.candidate_key IS NULL
            OR (ca.status = 'pending' AND ca.attempts < ?)
        )
        """
    params: list[object] = [min_score]
    if only_parked_403:
        params.append(PARKED_403_PATTERN)
    elif not include_attempted:
        params.append(MAX_ATTEMPTS)
    params.append(batch)
    return conn.execute(
        f"""
        SELECT cr.candidate_key, cr.artist, cr.title, cr.score,
               resolution.external_id, resolution.preview_url,
               resolution.matched_duration_s
        FROM candidate_recordings cr
        JOIN candidate_resolutions resolution
          ON resolution.candidate_key = cr.candidate_key
         AND resolution.provider = 'deezer'
         AND resolution.status = 'matched'
         AND resolution.has_preview = 1
        LEFT JOIN candidate_audio ca ON ca.candidate_key = cr.candidate_key
        WHERE cr.status = 'resolved'
          AND cr.score >= ?
          AND resolution.preview_url IS NOT NULL
          AND resolution.preview_url != ''
          {attempted_filter}
        ORDER BY cr.score DESC, cr.source_count DESC, cr.last_seen DESC
        LIMIT ?
        """,
        params,
    ).fetchall()


def _record_success(
    conn: sqlite3.Connection,
    *,
    row: sqlite3.Row,
    preview_url: str,
    path: str,
    content_type: str | None,
    size_bytes: int,
    sha256: str,
) -> None:
    now = db.utcnow()
    conn.execute(
        """
        INSERT INTO candidate_audio(
            candidate_key, status, source, source_id, source_url, path, duration_s,
            content_type, size_bytes, sha256, attempts, error, updated_at
        )
        VALUES(?, 'fetched', 'deezer_preview', ?, ?, ?, ?, ?, ?, ?, 1, NULL, ?)
        ON CONFLICT(candidate_key) DO UPDATE SET
            status='fetched',
            source=excluded.source,
            source_id=excluded.source_id,
            source_url=excluded.source_url,
            path=excluded.path,
            duration_s=excluded.duration_s,
            content_type=excluded.content_type,
            size_bytes=excluded.size_bytes,
            sha256=excluded.sha256,
            attempts=candidate_audio.attempts + 1,
            error=NULL,
            updated_at=excluded.updated_at
        """,
        (
            row["candidate_key"],
            row["external_id"],
            preview_url,
            path,
            row["matched_duration_s"],
            content_type,
            size_bytes,
            sha256,
            now,
        ),
    )
    conn.execute(
        """
        UPDATE candidate_resolutions
        SET status='matched', has_preview=1, preview_url=?, error=NULL, updated_at=?
        WHERE candidate_key=? AND provider='deezer'
        """,
        (preview_url, now, row["candidate_key"]),
    )
    conn.execute(
        "UPDATE candidate_recordings SET status='preview', updated_at=? "
        "WHERE candidate_key=? AND status='resolved'",
        (now, row["candidate_key"]),
    )
    conn.executemany(
        """
        INSERT OR IGNORE INTO candidate_embedding_runs(
            candidate_key, model, status, attempts, updated_at
        )
        VALUES(?, ?, 'pending', 0, ?)
        """,
        [(row["candidate_key"], model, now) for model in audio.EMBED_MODELS],
    )


def _record_failure(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    error: Exception,
    *,
    preview_url: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO candidate_audio(
            candidate_key, status, source, source_id, source_url, duration_s,
            attempts, error, updated_at
        )
        VALUES(?, 'pending', 'deezer_preview', ?, ?, ?, 1, ?, ?)
        ON CONFLICT(candidate_key) DO UPDATE SET
            status=CASE
                WHEN candidate_audio.attempts + 1 >= ? THEN 'error'
                ELSE 'pending'
            END,
            source=excluded.source,
            source_id=excluded.source_id,
            source_url=excluded.source_url,
            duration_s=excluded.duration_s,
            attempts=candidate_audio.attempts + 1,
            error=excluded.error,
            updated_at=excluded.updated_at
        """,
        (
            row["candidate_key"],
            row["external_id"],
            preview_url or row["preview_url"],
            row["matched_duration_s"],
            str(error)[:300],
            db.utcnow(),
            MAX_ATTEMPTS,
        ),
    )


def _record_unavailable(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    error: PreviewUnavailableError | ProviderMissError,
) -> None:
    now = db.utcnow()
    provider_miss = isinstance(error, ProviderMissError)
    conn.execute(
        """
        UPDATE candidate_resolutions
        SET status=?, has_preview=0, preview_url=NULL, error=?, updated_at=?
        WHERE candidate_key=? AND provider='deezer'
        """,
        (
            "provider_miss" if provider_miss else "matched",
            str(error)[:300],
            now,
            row["candidate_key"],
        ),
    )
    conn.execute(
        """
        INSERT INTO candidate_audio(
            candidate_key, status, source, source_id, duration_s,
            attempts, error, updated_at
        )
        VALUES(?, 'source_not_covered', 'deezer_preview', ?, ?, 1, ?, ?)
        ON CONFLICT(candidate_key) DO UPDATE SET
            status='source_not_covered',
            source=excluded.source,
            source_id=excluded.source_id,
            source_url=NULL,
            duration_s=excluded.duration_s,
            attempts=candidate_audio.attempts + 1,
            error=excluded.error,
            updated_at=excluded.updated_at
        """,
        (
            row["candidate_key"],
            row["external_id"],
            row["matched_duration_s"],
            str(error)[:300],
            now,
        ),
    )
    if provider_miss:
        conn.execute(
            "UPDATE candidate_recordings SET status='metadata', updated_at=? "
            "WHERE candidate_key=? AND status='resolved'",
            (now, row["candidate_key"]),
        )


def _remaining(conn: sqlite3.Connection, min_score: float) -> int:
    return conn.execute(
        """
        SELECT COUNT(*)
        FROM candidate_recordings cr
        JOIN candidate_resolutions resolution
          ON resolution.candidate_key = cr.candidate_key
         AND resolution.provider = 'deezer'
         AND resolution.status = 'matched'
         AND resolution.has_preview = 1
        LEFT JOIN candidate_audio ca ON ca.candidate_key = cr.candidate_key
        WHERE cr.status = 'resolved' AND cr.score >= ?
          AND resolution.preview_url IS NOT NULL AND resolution.preview_url != ''
          AND (
            ca.candidate_key IS NULL
            OR (ca.status = 'pending' AND ca.attempts < ?)
          )
        """,
        (min_score, MAX_ATTEMPTS),
    ).fetchone()[0]


def parked_403_count(conn: sqlite3.Connection, min_score: float = 0.0) -> int:
    """Count retryable preview rows parked by signed-URL HTTP 403 failures."""
    return conn.execute(
        """
        SELECT COUNT(*)
        FROM candidate_audio ca
        JOIN candidate_recordings cr USING(candidate_key)
        JOIN candidate_resolutions resolution
          ON resolution.candidate_key = cr.candidate_key
         AND resolution.provider = 'deezer'
         AND resolution.status = 'matched'
         AND resolution.has_preview = 1
        WHERE ca.status = 'error'
          AND ca.error LIKE ?
          AND cr.status = 'resolved'
          AND cr.score >= ?
        """,
        (PARKED_403_PATTERN, min_score),
    ).fetchone()[0]


def fetch_previews(
    conn: sqlite3.Connection,
    *,
    batch: int = 25,
    min_score: float = 0.0,
    include_attempted: bool = False,
    only_parked_403: bool = False,
    download: DownloadFn | None = None,
    refresh: RefreshFn | None = None,
    workers: int = WORKERS,
    progress=None,
) -> dict[str, int]:
    """Download top resolved candidate previews into their separate cache."""
    if include_attempted and only_parked_403:
        raise ValueError("include_attempted and only_parked_403 are mutually exclusive")
    rows = _candidate_rows(
        conn,
        batch=batch,
        min_score=min_score,
        include_attempted=include_attempted,
        only_parked_403=only_parked_403,
    )
    data_root = _data_root(conn)
    download = download or _download_preview
    refresh = refresh or audio.deezer_track
    counts = {
        "seen": 0,
        "fetched": 0,
        "refreshed": 0,
        "provider_miss": 0,
        "source_not_covered": 0,
        "error": 0,
    }

    def job(row: sqlite3.Row):
        preview_url = None
        try:
            current = refresh(str(row["external_id"]))
            if current is None:
                raise ProviderMissError(f"Deezer track {row['external_id']} no longer exists")
            preview_url = current.get("preview")
            if not preview_url:
                raise PreviewUnavailableError(f"Deezer track {row['external_id']} has no preview")
            content, content_type = download(str(preview_url))
            stored = _store_preview(data_root, row["candidate_key"], content, content_type)
            return row, str(preview_url), stored, content_type, None
        except (requests.RequestException, ValueError, OSError) as exc:
            return row, preview_url, None, None, exc

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for index, (row, preview_url, stored, content_type, error) in enumerate(
            executor.map(job, rows), start=1
        ):
            counts["seen"] += 1
            if error is None:
                path, size_bytes, sha256 = stored
                counts["refreshed"] += int(preview_url != row["preview_url"])
                _record_success(
                    conn,
                    row=row,
                    preview_url=preview_url,
                    path=path,
                    content_type=content_type,
                    size_bytes=size_bytes,
                    sha256=sha256,
                )
                counts["fetched"] += 1
            elif isinstance(error, ProviderMissError):
                _record_unavailable(conn, row, error)
                counts["provider_miss"] += 1
            elif isinstance(error, PreviewUnavailableError):
                _record_unavailable(conn, row, error)
                counts["source_not_covered"] += 1
            else:
                _record_failure(conn, row, error, preview_url=preview_url)
                counts["error"] += 1
            conn.commit()
            if progress:
                progress(index, len(rows), counts)

    counts["remaining"] = (
        parked_403_count(conn, min_score) if only_parked_403 else _remaining(conn, min_score)
    )
    return counts


def repair_parked_403s(
    conn: sqlite3.Connection,
    *,
    apply: bool = False,
    batch: int = 60,
    min_score: float = 0.0,
    download: DownloadFn | None = None,
    refresh: RefreshFn | None = None,
    workers: int = WORKERS,
    progress=None,
) -> dict[str, Any]:
    """Revalidate only previews parked by signed-URL HTTP 403 failures."""
    batch = max(1, int(batch))
    eligible = parked_403_count(conn, min_score)
    if not apply:
        return {
            "mode": "dry_run",
            "eligible": eligible,
            "batch": min(batch, eligible),
        }
    result = fetch_previews(
        conn,
        batch=batch,
        min_score=min_score,
        only_parked_403=True,
        download=download,
        refresh=refresh,
        workers=workers,
        progress=progress,
    )
    return {"mode": "applied", "eligible_before": eligible, **result}


def status(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        f"candidate_audio:{row['status']}": row["n"]
        for row in conn.execute("SELECT status, COUNT(*) AS n FROM candidate_audio GROUP BY status")
    }
