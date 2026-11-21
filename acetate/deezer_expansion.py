"""Resumable Deezer artist, discography, and album-track expansion."""

from __future__ import annotations

import json
import math
import sqlite3
from typing import Any, Callable

import requests

from . import audio, candidates, db
from .canon import _norm

ARTIST_TOP_SOURCE = "deezer_artist_top_track"
SAME_ARTIST_SOURCE = "deezer_same_artist"
SAME_ALBUM_SOURCE = "deezer_same_album"
MAX_ATTEMPTS = 3
MISS_CODES = {800}
PAGE_SIZE = 100

GetFn = Callable[..., dict[str, Any] | list[Any] | None]


class DeezerAPIError(requests.RequestException):
    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


def _get(path: str, **params) -> dict[str, Any] | list[Any] | None:
    return audio._get_json(f"{audio.DEEZER_API}{path}", **params)


def _int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _edge_strength(rank: int) -> float:
    return 1.0 / max(1.0, math.sqrt(rank))


def _error(payload: Any) -> DeezerAPIError:
    detail = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(detail, dict):
        return DeezerAPIError(f"invalid Deezer response: {str(payload)[:200]}")
    return DeezerAPIError(
        str(detail.get("message") or detail.get("type") or "Deezer API error"),
        _int(detail.get("code")),
    )


def _collection(get: GetFn, path: str, limit: int) -> dict[str, Any]:
    """Fetch a bounded collection while retaining every raw response page."""
    if limit <= 0:
        return {"data": [], "total": 0, "pages": []}
    rows: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    index = 0
    total = 0
    while len(rows) < limit:
        page_limit = min(PAGE_SIZE, limit - len(rows))
        payload = get(path, limit=page_limit, index=index)
        if not isinstance(payload, dict) or payload.get("error") is not None:
            raise _error(payload)
        page_rows = [item for item in payload.get("data") or [] if isinstance(item, dict)]
        pages.append(payload)
        rows.extend(page_rows)
        total = _int(payload.get("total")) or max(total, len(rows))
        if not payload.get("next") or not page_rows:
            break
        index += len(page_rows)
    return {"data": rows[:limit], "total": total, "pages": pages}


def _seed_artist_rows(
    conn: sqlite3.Connection,
    *,
    batch: int,
    min_plays: int,
    include_fetched: bool,
) -> tuple[list[dict[str, Any]], int]:
    rows = conn.execute(
        """
        WITH ranked AS (
            SELECT dt.artist_id, COALESCE(da.name, rs.artist) AS artist,
                   dt.recording_key, rs.plays, rs.liked,
                   SUM(rs.plays) OVER (PARTITION BY dt.artist_id) AS total_plays,
                   MAX(rs.liked) OVER (PARTITION BY dt.artist_id) AS any_liked,
                   ROW_NUMBER() OVER (
                       PARTITION BY dt.artist_id
                       ORDER BY rs.liked DESC, rs.plays DESC, dt.recording_key
                   ) AS row_num
            FROM deezer_track dt
            JOIN recording_stats rs USING(recording_key)
            LEFT JOIN deezer_artist da USING(artist_id)
            WHERE dt.artist_id IS NOT NULL AND (rs.liked = 1 OR rs.plays >= ?)
        )
        SELECT artist_id, artist, recording_key AS seed_recording_key,
               total_plays, any_liked
        FROM ranked
        WHERE row_num = 1
        ORDER BY any_liked DESC, total_plays DESC, artist_id
        """,
        (min_plays,),
    ).fetchall()
    pending = []
    for row in rows:
        cached = conn.execute(
            "SELECT status, attempts FROM deezer_artist_expansion WHERE artist_id=?",
            (row["artist_id"],),
        ).fetchone()
        if not include_fetched and cached:
            if cached["status"] in ("fetched", "miss"):
                continue
            if cached["status"] == "error" and cached["attempts"] >= MAX_ATTEMPTS:
                continue
        pending.append(dict(row))
    return pending[:batch], len(pending)


def _ensure_artist_attempt(conn: sqlite3.Connection, seed: dict[str, Any]) -> sqlite3.Row:
    conn.execute(
        """
        INSERT INTO deezer_artist_expansion(
            artist_id, artist, seed_recording_key, status, attempts, updated_at
        ) VALUES(?, ?, ?, 'pending', 1, ?)
        ON CONFLICT(artist_id) DO UPDATE SET
            artist=excluded.artist,
            seed_recording_key=excluded.seed_recording_key,
            status='pending',
            attempts=deezer_artist_expansion.attempts + 1,
            error=NULL,
            updated_at=excluded.updated_at
        """,
        (seed["artist_id"], seed["artist"], seed["seed_recording_key"], db.utcnow()),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM deezer_artist_expansion WHERE artist_id=?",
        (seed["artist_id"],),
    ).fetchone()


def _fetch_artist_surfaces(
    conn: sqlite3.Connection,
    *,
    row: sqlite3.Row,
    top_limit: int,
    album_limit: int,
    include_fetched: bool,
    get: GetFn,
) -> dict[str, dict[str, Any]]:
    payloads = {}
    surfaces = (
        ("top_json", f"/artist/{row['artist_id']}/top", top_limit),
        ("albums_json", f"/artist/{row['artist_id']}/albums", album_limit),
    )
    for column, path, limit in surfaces:
        if row[column] and not include_fetched:
            payloads[column] = json.loads(row[column])
            continue
        payload = _collection(get, path, limit)
        payloads[column] = payload
        conn.execute(
            f"UPDATE deezer_artist_expansion SET {column}=?, updated_at=? WHERE artist_id=?",
            (
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                db.utcnow(),
                row["artist_id"],
            ),
        )
        conn.commit()
    return payloads


def _external_ids(track: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    if track.get("id") is not None:
        out.append(
            {
                "provider": "deezer",
                "external_id": str(track["id"]),
                "id_type": "track_id",
                "confidence": 1.0,
                "raw": track,
            }
        )
    if track.get("isrc"):
        out.append(
            {
                "provider": "isrc",
                "external_id": str(track["isrc"]),
                "id_type": "isrc",
                "confidence": 1.0,
                "raw": {"deezer_id": track.get("id")},
            }
        )
    return out


def _normalize_artist(
    conn: sqlite3.Connection,
    *,
    row: sqlite3.Row,
    payloads: dict[str, dict[str, Any]],
    known: set[str],
) -> dict[str, Any]:
    artist_id = row["artist_id"]
    artist = row["artist"]
    now = db.utcnow()
    candidate_keys = set()

    conn.execute("DELETE FROM deezer_artist_top_tracks WHERE artist_id=?", (artist_id,))
    top_count = 0
    for rank, track in enumerate(payloads["top_json"]["data"], start=1):
        title = track.get("title")
        track_artist = (track.get("artist") or {}).get("name") or artist
        track_id = _int(track.get("id"))
        if track_id is None or not title or not track_artist:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO deezer_artist_top_tracks(
                artist_id, track_id, artist, title, rank, album_id, duration_s,
                preview_url, raw_json, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artist_id,
                track_id,
                track_artist,
                title,
                rank,
                _int((track.get("album") or {}).get("id")),
                _int(track.get("duration")),
                track.get("preview") or None,
                json.dumps(track, ensure_ascii=False, sort_keys=True),
                now,
            ),
        )
        candidate_key = candidates.add_candidate_source(
            conn,
            artist=track_artist,
            title=title,
            source=ARTIST_TOP_SOURCE,
            reason="artist_top",
            seed_recording_key=row["seed_recording_key"] or "",
            seed_artist=artist,
            seed_external_id=str(artist_id),
            source_rank=rank,
            strength=_edge_strength(rank),
            raw=track,
            external_ids=_external_ids(track),
            known=known,
        )
        if candidate_key:
            candidate_keys.add(candidate_key)
        top_count += 1

    conn.execute("DELETE FROM deezer_artist_albums WHERE artist_id=?", (artist_id,))
    album_count = 0
    for rank, album in enumerate(payloads["albums_json"]["data"], start=1):
        album_id = _int(album.get("id"))
        title = album.get("title")
        if album_id is None or not title:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO deezer_artist_albums(
                artist_id, album_id, artist, album, rank, record_type, release_date,
                fans, tracklist_url, raw_json, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artist_id,
                album_id,
                artist,
                title,
                rank,
                album.get("record_type"),
                album.get("release_date"),
                _int(album.get("fans")),
                album.get("tracklist"),
                json.dumps(album, ensure_ascii=False, sort_keys=True),
                now,
            ),
        )
        conn.execute(
            """
            INSERT INTO deezer_album_expansion(
                album_id, album, artist_id, artist, status, attempts, updated_at
            ) VALUES(?, ?, ?, ?, 'pending', 0, ?)
            ON CONFLICT(album_id) DO UPDATE SET
                album=COALESCE(excluded.album, deezer_album_expansion.album),
                artist_id=COALESCE(deezer_album_expansion.artist_id, excluded.artist_id),
                artist=COALESCE(deezer_album_expansion.artist, excluded.artist),
                updated_at=excluded.updated_at
            """,
            (album_id, title, artist_id, artist, now),
        )
        album_count += 1

    conn.execute(
        "UPDATE deezer_artist_expansion SET status='fetched', error=NULL, updated_at=? "
        "WHERE artist_id=?",
        (now, artist_id),
    )
    return {
        "top_tracks": top_count,
        "artist_albums": album_count,
        "candidate_keys": candidate_keys,
    }


def _enqueue_library_albums(conn: sqlite3.Connection, min_plays: int) -> int:
    rows = conn.execute(
        """
        WITH ranked AS (
            SELECT dt.album_id, COALESCE(dal.title, rs.title) AS album,
                   dt.artist_id, COALESCE(dar.name, rs.artist) AS artist,
                   ROW_NUMBER() OVER (
                       PARTITION BY dt.album_id
                       ORDER BY rs.liked DESC, rs.plays DESC, dt.recording_key
                   ) AS row_num
            FROM deezer_track dt
            JOIN recording_stats rs USING(recording_key)
            LEFT JOIN deezer_album dal USING(album_id)
            LEFT JOIN deezer_artist dar USING(artist_id)
            WHERE dt.album_id IS NOT NULL AND (rs.liked = 1 OR rs.plays >= ?)
        )
        SELECT album_id, album, artist_id, artist FROM ranked WHERE row_num = 1
        """,
        (min_plays,),
    ).fetchall()
    before = conn.execute("SELECT COUNT(*) FROM deezer_album_expansion").fetchone()[0]
    conn.executemany(
        """
        INSERT OR IGNORE INTO deezer_album_expansion(
            album_id, album, artist_id, artist, status, attempts, updated_at
        ) VALUES(?, ?, ?, ?, 'pending', 0, ?)
        """,
        [
            (row["album_id"], row["album"], row["artist_id"], row["artist"], db.utcnow())
            for row in rows
        ],
    )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM deezer_album_expansion").fetchone()[0]
    return after - before


def _album_rows(
    conn: sqlite3.Connection,
    *,
    batch: int,
    min_plays: int,
    include_fetched: bool,
) -> tuple[list[sqlite3.Row], int]:
    status_filter = (
        ""
        if include_fetched
        else ("WHERE (dae.status = 'pending' OR (dae.status = 'error' AND dae.attempts < ?))")
    )
    status_params: list[Any] = [] if include_fetched else [MAX_ATTEMPTS]
    query = f"""
        WITH artist_priority AS (
            SELECT album_id, MIN(rank) AS artist_rank
            FROM deezer_artist_albums
            GROUP BY album_id
        ),
        library_priority AS (
            SELECT dt.album_id, SUM(rs.plays) AS library_plays
            FROM deezer_track dt
            JOIN recording_stats rs USING(recording_key)
            WHERE rs.liked = 1 OR rs.plays >= ?
            GROUP BY dt.album_id
        )
        SELECT dae.*, (ap.album_id IS NOT NULL) AS artist_album,
               COALESCE(ap.artist_rank, 999999) AS artist_rank,
               COALESCE(lp.library_plays, 0) AS library_plays
        FROM deezer_album_expansion dae
        LEFT JOIN artist_priority ap USING(album_id)
        LEFT JOIN library_priority lp USING(album_id)
        {status_filter}
        ORDER BY artist_album DESC, artist_rank, library_plays DESC, dae.album_id
        LIMIT ?
    """
    rows = conn.execute(query, [min_plays, *status_params, max(0, batch)]).fetchall()
    if include_fetched:
        total = conn.execute("SELECT COUNT(*) FROM deezer_album_expansion").fetchone()[0]
    else:
        total = conn.execute(
            """
            SELECT COUNT(*) FROM deezer_album_expansion
            WHERE status='pending' OR (status='error' AND attempts < ?)
            """,
            (MAX_ATTEMPTS,),
        ).fetchone()[0]
    return rows, total


def _ensure_album_attempt(conn: sqlite3.Connection, album_id: int) -> sqlite3.Row:
    conn.execute(
        """
        UPDATE deezer_album_expansion
        SET status='pending', attempts=attempts + 1, error=NULL, updated_at=?
        WHERE album_id=?
        """,
        (db.utcnow(), album_id),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM deezer_album_expansion WHERE album_id=?", (album_id,)
    ).fetchone()


def _library_album_seed(
    conn: sqlite3.Connection,
    album_id: int,
    min_plays: int,
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT dt.recording_key, rs.artist, rs.title
        FROM deezer_track dt
        JOIN recording_stats rs USING(recording_key)
        WHERE dt.album_id=? AND (rs.liked = 1 OR rs.plays >= ?)
        ORDER BY rs.liked DESC, rs.plays DESC, dt.recording_key
        LIMIT 1
        """,
        (album_id, min_plays),
    ).fetchone()


def _album_artist_seeds(conn: sqlite3.Connection, album_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT daa.artist_id, dae.artist, dae.seed_recording_key, rs.title AS seed_title
        FROM deezer_artist_albums daa
        JOIN deezer_artist_expansion dae USING(artist_id)
        LEFT JOIN recording_stats rs ON rs.recording_key = dae.seed_recording_key
        WHERE daa.album_id=?
        ORDER BY daa.rank, daa.artist_id
        """,
        (album_id,),
    ).fetchall()


def _normalize_album(
    conn: sqlite3.Connection,
    *,
    row: sqlite3.Row,
    payload: dict[str, Any],
    known: set[str],
    min_plays: int,
) -> dict[str, Any]:
    album_id = row["album_id"]
    now = db.utcnow()
    library_seed = _library_album_seed(conn, album_id, min_plays)
    artist_seeds = _album_artist_seeds(conn, album_id)
    candidate_keys = set()
    duplicate_keys = set()
    source_edges = 0
    track_count = 0

    conn.execute("DELETE FROM deezer_album_tracks WHERE album_id=?", (album_id,))
    for rank, track in enumerate(payload["data"], start=1):
        track_id = _int(track.get("id"))
        title = track.get("title")
        track_artist = (track.get("artist") or {}).get("name")
        track_artist_id = _int((track.get("artist") or {}).get("id"))
        if track_id is None or not title or not track_artist:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO deezer_album_tracks(
                album_id, track_id, artist_id, artist, title, rank, disk_number,
                track_position, duration_s, isrc, preview_url, raw_json, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                album_id,
                track_id,
                track_artist_id,
                track_artist,
                title,
                rank,
                _int(track.get("disk_number")),
                _int(track.get("track_position")),
                _int(track.get("duration")),
                track.get("isrc") or None,
                track.get("preview") or None,
                json.dumps(track, ensure_ascii=False, sort_keys=True),
                now,
            ),
        )
        track_count += 1

        added_keys = []
        if library_seed:
            candidate_key = candidates.add_candidate_source(
                conn,
                artist=track_artist,
                title=title,
                source=SAME_ALBUM_SOURCE,
                reason="library_album_track",
                seed_recording_key=library_seed["recording_key"],
                seed_artist=library_seed["artist"],
                seed_title=library_seed["title"],
                seed_external_id=str(album_id),
                source_rank=rank,
                strength=_edge_strength(rank),
                raw=track,
                external_ids=_external_ids(track),
                known=known,
            )
            if candidate_key:
                added_keys.append(candidate_key)
                source_edges += 1

        for seed in artist_seeds:
            same_artist = (
                track_artist_id == seed["artist_id"]
                if track_artist_id is not None
                else _norm(track_artist) == _norm(seed["artist"])
            )
            if not same_artist:
                continue
            candidate_key = candidates.add_candidate_source(
                conn,
                artist=track_artist,
                title=title,
                source=SAME_ARTIST_SOURCE,
                reason="artist_album_track",
                seed_recording_key=seed["seed_recording_key"] or "",
                seed_artist=seed["artist"],
                seed_title=seed["seed_title"],
                seed_external_id=f"{seed['artist_id']}:{album_id}",
                source_rank=rank,
                strength=_edge_strength(rank),
                raw=track,
                external_ids=_external_ids(track),
                known=known,
            )
            if candidate_key:
                added_keys.append(candidate_key)
                source_edges += 1

        for candidate_key in added_keys:
            candidate_keys.add(candidate_key)
            isrc = track.get("isrc")
            duplicate_key = candidates.library_recording_for_isrc(conn, isrc) if isrc else None
            if duplicate_key:
                candidates.mark_library_duplicate(
                    conn,
                    candidate_key=candidate_key,
                    library_recording_key=duplicate_key,
                    method="isrc",
                    external_id=isrc,
                )
                duplicate_keys.add(candidate_key)

    conn.execute(
        "UPDATE deezer_album_expansion SET status='fetched', error=NULL, updated_at=? "
        "WHERE album_id=?",
        (now, album_id),
    )
    return {
        "tracks": track_count,
        "source_edges": source_edges,
        "candidate_keys": candidate_keys,
        "duplicate_keys": duplicate_keys,
    }


def _mark_failure(
    conn: sqlite3.Connection,
    *,
    table: str,
    id_column: str,
    entity_id: int,
    attempts: int,
    error: Exception,
) -> str:
    miss = isinstance(error, DeezerAPIError) and error.code in MISS_CODES
    status = "miss" if miss else ("error" if attempts >= MAX_ATTEMPTS else "pending")
    conn.execute(
        f"UPDATE {table} SET status=?, error=?, updated_at=? WHERE {id_column}=?",
        (status, str(error)[:300], db.utcnow(), entity_id),
    )
    conn.commit()
    return status


def harvest(
    conn: sqlite3.Connection,
    *,
    artist_batch: int = 5,
    album_batch: int = 25,
    min_plays: int = 5,
    top_limit: int = 25,
    album_limit: int = 10,
    track_limit: int = 100,
    include_fetched: bool = False,
    get: GetFn | None = None,
    progress=None,
) -> dict[str, int]:
    """Expand high-signal Deezer artists and a bounded album-track queue."""
    get = get or _get
    known = {
        row["recording_key"]
        for row in conn.execute(
            "SELECT recording_key FROM recording_stats WHERE recording_key IS NOT NULL"
        )
    }
    candidate_keys = set()
    duplicate_keys = set()
    counts = {
        "artist_seen": 0,
        "artist_fetched": 0,
        "artist_miss": 0,
        "artist_error": 0,
        "top_tracks": 0,
        "artist_albums": 0,
        "albums_enqueued": 0,
        "album_seen": 0,
        "album_fetched": 0,
        "album_miss": 0,
        "album_error": 0,
        "album_tracks": 0,
        "source_edges": 0,
    }

    artist_rows, _artist_pending = _seed_artist_rows(
        conn,
        batch=artist_batch,
        min_plays=min_plays,
        include_fetched=include_fetched,
    )
    for index, seed in enumerate(artist_rows, start=1):
        counts["artist_seen"] += 1
        row = _ensure_artist_attempt(conn, seed)
        try:
            payloads = _fetch_artist_surfaces(
                conn,
                row=row,
                top_limit=top_limit,
                album_limit=album_limit,
                include_fetched=include_fetched,
                get=get,
            )
            normalized = _normalize_artist(conn, row=row, payloads=payloads, known=known)
            counts["artist_fetched"] += 1
            counts["top_tracks"] += normalized["top_tracks"]
            counts["artist_albums"] += normalized["artist_albums"]
            candidate_keys.update(normalized["candidate_keys"])
            conn.commit()
        except requests.RequestException as exc:
            status = _mark_failure(
                conn,
                table="deezer_artist_expansion",
                id_column="artist_id",
                entity_id=row["artist_id"],
                attempts=row["attempts"],
                error=exc,
            )
            counts["artist_miss" if status == "miss" else "artist_error"] += 1
        if progress:
            progress("artist", index, len(artist_rows), counts)

    counts["albums_enqueued"] = _enqueue_library_albums(conn, min_plays)
    album_rows, _album_pending = _album_rows(
        conn,
        batch=album_batch,
        min_plays=min_plays,
        include_fetched=include_fetched,
    )
    for index, album_seed in enumerate(album_rows, start=1):
        counts["album_seen"] += 1
        row = _ensure_album_attempt(conn, album_seed["album_id"])
        try:
            payload = _collection(get, f"/album/{row['album_id']}/tracks", track_limit)
            conn.execute(
                "UPDATE deezer_album_expansion SET tracks_json=?, updated_at=? WHERE album_id=?",
                (
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    db.utcnow(),
                    row["album_id"],
                ),
            )
            conn.commit()
            normalized = _normalize_album(
                conn,
                row=row,
                payload=payload,
                known=known,
                min_plays=min_plays,
            )
            counts["album_fetched"] += 1
            counts["album_tracks"] += normalized["tracks"]
            counts["source_edges"] += normalized["source_edges"]
            candidate_keys.update(normalized["candidate_keys"])
            duplicate_keys.update(normalized["duplicate_keys"])
            conn.commit()
        except requests.RequestException as exc:
            status = _mark_failure(
                conn,
                table="deezer_album_expansion",
                id_column="album_id",
                entity_id=row["album_id"],
                attempts=row["attempts"],
                error=exc,
            )
            counts["album_miss" if status == "miss" else "album_error"] += 1
        if progress:
            progress("album", index, len(album_rows), counts)

    if candidate_keys:
        candidates.score_candidates(conn, candidate_keys=candidate_keys)
    _artist_rows, artist_remaining = _seed_artist_rows(
        conn,
        batch=1,
        min_plays=min_plays,
        include_fetched=False,
    )
    album_remaining = conn.execute(
        """
        SELECT COUNT(*) FROM deezer_album_expansion
        WHERE status='pending' OR (status='error' AND attempts < ?)
        """,
        (MAX_ATTEMPTS,),
    ).fetchone()[0]
    counts["candidates"] = len(candidate_keys)
    counts["duplicates"] = len(duplicate_keys)
    counts["artist_remaining"] = artist_remaining
    counts["album_remaining"] = album_remaining
    return counts


def status(conn: sqlite3.Connection) -> dict[str, int]:
    out = {
        f"deezer_expand:artist:{row['status']}": row["n"]
        for row in conn.execute(
            "SELECT status, COUNT(*) AS n FROM deezer_artist_expansion GROUP BY status"
        )
    }
    for row in conn.execute(
        "SELECT status, COUNT(*) AS n FROM deezer_album_expansion GROUP BY status"
    ):
        out[f"deezer_expand:album:{row['status']}"] = row["n"]
    out["deezer_expand:artist_top_tracks"] = conn.execute(
        "SELECT COUNT(*) FROM deezer_artist_top_tracks"
    ).fetchone()[0]
    out["deezer_expand:artist_albums"] = conn.execute(
        "SELECT COUNT(*) FROM deezer_artist_albums"
    ).fetchone()[0]
    out["deezer_expand:album_tracks"] = conn.execute(
        "SELECT COUNT(*) FROM deezer_album_tracks"
    ).fetchone()[0]
    return out
