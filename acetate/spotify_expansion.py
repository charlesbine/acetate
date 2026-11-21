"""Resumable Spotify artist-album and album-track candidate expansion."""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from typing import Any

import requests
import spotipy

from . import candidates, db

GROUPS = ("album", "single", "appears_on", "compilation")
MAX_ATTEMPTS = 3
PAGE_SIZE = 50
MIN_REQUEST_INTERVAL_S = 0.35
MAX_RATE_LIMIT_RETRIES = 3
_last_request_at = 0.0

TOP_SOURCE = "spotify_artist_top_track"
SAME_ARTIST_SOURCE = "spotify_same_artist"
SAME_ALBUM_SOURCE = "spotify_same_album"
APPEARS_ON_SOURCE = "spotify_appears_on_album"
COPRESENCE_SOURCE = "spotify_compilation_copresence"


def _retry_after_s(error: spotipy.SpotifyException) -> float:
    headers = getattr(error, "headers", None) or {}
    value = next(
        (header_value for key, header_value in headers.items() if key.casefold() == "retry-after"),
        None,
    )
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 1.0


def _spotify_call(
    call,
    *,
    min_interval_s: float = MIN_REQUEST_INTERVAL_S,
    sleep=time.sleep,
    monotonic=time.monotonic,
):
    """Pace catalog calls and honor Spotify's 429 Retry-After response."""
    global _last_request_at
    wait = min_interval_s - (monotonic() - _last_request_at)
    if wait > 0:
        sleep(wait)
    _last_request_at = monotonic()
    retries = 0
    while True:
        try:
            return call()
        except spotipy.SpotifyException as exc:
            if exc.http_status != 429 or retries >= MAX_RATE_LIMIT_RETRIES:
                raise
            retries += 1
            sleep(_retry_after_s(exc))
            _last_request_at = monotonic()


def _edge_strength(rank: int, *, weak: bool = False) -> float:
    value = 1.0 / max(1.0, math.sqrt(rank))
    return value * 0.5 if weak else value


def _collect(fetch, limit: int) -> dict[str, Any]:
    """Fetch a bounded Spotify paging surface and retain all raw response pages."""
    if limit <= 0:
        return {"items": [], "total": 0, "pages": []}
    items: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    offset = 0
    total = 0
    while len(items) < limit:
        page_limit = min(PAGE_SIZE, limit - len(items))
        payload = fetch(page_limit, offset)
        if not isinstance(payload, dict):
            raise ValueError(f"invalid Spotify response: {str(payload)[:200]}")
        page_items = [item for item in payload.get("items") or [] if isinstance(item, dict)]
        pages.append(payload)
        items.extend(page_items)
        total = int(payload.get("total") or max(total, len(items)))
        if not payload.get("next") or not page_items:
            break
        offset += len(page_items)
    return {"items": items[:limit], "total": total, "pages": pages}


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
            SELECT ta.artist_id, a.name AS artist, rs.recording_key, rs.plays, rs.liked,
                   SUM(rs.plays) OVER (PARTITION BY ta.artist_id) AS total_plays,
                   MAX(rs.liked) OVER (PARTITION BY ta.artist_id) AS any_liked,
                   ROW_NUMBER() OVER (
                       PARTITION BY ta.artist_id
                       ORDER BY rs.liked DESC, rs.plays DESC, rs.recording_key
                   ) AS row_num
            FROM recording_stats rs
            JOIN track_artists ta ON ta.track_id = rs.rep_track_id
            JOIN artists a ON a.artist_id = ta.artist_id
            WHERE a.name IS NOT NULL AND a.name != ''
              AND (rs.liked = 1 OR rs.plays >= ?)
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
            "SELECT status, attempts FROM spotify_artist_expansion WHERE artist_id=?",
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
        INSERT INTO spotify_artist_expansion(
            artist_id, artist, seed_recording_key, status, attempts, updated_at
        ) VALUES(?, ?, ?, 'pending', 1, ?)
        ON CONFLICT(artist_id) DO UPDATE SET
            artist=excluded.artist,
            seed_recording_key=excluded.seed_recording_key,
            status='pending',
            attempts=spotify_artist_expansion.attempts + 1,
            error=NULL,
            updated_at=excluded.updated_at
        """,
        (seed["artist_id"], seed["artist"], seed["seed_recording_key"], db.utcnow()),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM spotify_artist_expansion WHERE artist_id=?",
        (seed["artist_id"],),
    ).fetchone()


def _capability(conn: sqlite3.Connection, surface: str) -> str | None:
    row = conn.execute(
        "SELECT status FROM spotify_expansion_capabilities WHERE surface=?", (surface,)
    ).fetchone()
    return row["status"] if row else None


def _set_capability(
    conn: sqlite3.Connection,
    *,
    surface: str,
    status: str,
    error: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO spotify_expansion_capabilities(surface, status, error, updated_at)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(surface) DO UPDATE SET
            status=excluded.status, error=excluded.error, updated_at=excluded.updated_at
        """,
        (surface, status, error[:300] if error else None, db.utcnow()),
    )
    conn.commit()


def _fetch_top_tracks(
    conn: sqlite3.Connection,
    sp: spotipy.Spotify,
    artist_id: str,
) -> tuple[dict[str, Any] | None, bool]:
    """Return optional top tracks; cache the current dev-mode 403 globally."""
    if _capability(conn, "artist_top_tracks") == "unavailable":
        return None, True
    client_logger = logging.getLogger("spotipy.client")
    logger_was_disabled = client_logger.disabled
    client_logger.disabled = True
    try:
        payload = _spotify_call(lambda: sp.artist_top_tracks(artist_id, country="US"))
    except spotipy.SpotifyException as exc:
        if exc.http_status == 403:
            _set_capability(
                conn,
                surface="artist_top_tracks",
                status="unavailable",
                error=str(exc),
            )
            return None, True
        raise
    finally:
        client_logger.disabled = logger_was_disabled
    _set_capability(conn, surface="artist_top_tracks", status="available")
    return payload, False


def _fetch_artist_surfaces(
    conn: sqlite3.Connection,
    *,
    row: sqlite3.Row,
    sp: spotipy.Spotify,
    album_limit: int,
    include_fetched: bool,
) -> tuple[dict[str, Any], bool]:
    if row["albums_json"] and not include_fetched:
        albums = json.loads(row["albums_json"])
    else:
        albums = {
            group: _collect(
                lambda limit, offset, group=group: _spotify_call(
                    lambda: sp.artist_albums(
                        row["artist_id"], album_type=group, limit=limit, offset=offset
                    )
                ),
                album_limit,
            )
            for group in GROUPS
        }
        conn.execute(
            "UPDATE spotify_artist_expansion SET albums_json=?, updated_at=? WHERE artist_id=?",
            (json.dumps(albums, ensure_ascii=False, sort_keys=True), db.utcnow(), row["artist_id"]),
        )
        conn.commit()

    top_unavailable = False
    if row["top_json"] and not include_fetched:
        top = json.loads(row["top_json"])
    else:
        top, top_unavailable = _fetch_top_tracks(conn, sp, row["artist_id"])
        if top is not None:
            conn.execute(
                "UPDATE spotify_artist_expansion SET top_json=?, updated_at=? WHERE artist_id=?",
                (
                    json.dumps(top, ensure_ascii=False, sort_keys=True),
                    db.utcnow(),
                    row["artist_id"],
                ),
            )
            conn.commit()
    return {"albums": albums, "top": top}, top_unavailable


def _primary_artist(track: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
    artists = [item for item in track.get("artists") or [] if isinstance(item, dict)]
    return ((artists[0].get("name") if artists else None), artists)


def _external_ids(track: dict[str, Any]) -> list[dict[str, Any]]:
    if not track.get("id"):
        return []
    return [
        {
            "provider": "spotify",
            "external_id": str(track["id"]),
            "id_type": "track_id",
            "confidence": 1.0,
            "raw": track,
        }
    ]


def _normalize_artist(
    conn: sqlite3.Connection,
    *,
    row: sqlite3.Row,
    payloads: dict[str, Any],
    known: set[str],
) -> dict[str, Any]:
    artist_id = row["artist_id"]
    artist = row["artist"]
    now = db.utcnow()
    candidate_keys = set()
    top_count = 0
    album_count = 0

    conn.execute("DELETE FROM spotify_artist_top_tracks WHERE artist_id=?", (artist_id,))
    for rank, track in enumerate((payloads["top"] or {}).get("tracks") or [], start=1):
        title = track.get("name")
        track_artist, _artists = _primary_artist(track)
        track_id = track.get("id")
        if not track_id or not title or not track_artist:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO spotify_artist_top_tracks(
                artist_id, track_id, artist, title, rank, album_id, duration_ms, raw_json, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artist_id,
                track_id,
                track_artist,
                title,
                rank,
                (track.get("album") or {}).get("id"),
                track.get("duration_ms"),
                json.dumps(track, ensure_ascii=False, sort_keys=True),
                now,
            ),
        )
        candidate_key = candidates.add_candidate_source(
            conn,
            artist=track_artist,
            title=title,
            source=TOP_SOURCE,
            reason="artist_top",
            seed_recording_key=row["seed_recording_key"] or "",
            seed_artist=artist,
            seed_external_id=artist_id,
            source_rank=rank,
            strength=_edge_strength(rank),
            raw=track,
            external_ids=_external_ids(track),
            known=known,
        )
        if candidate_key:
            candidate_keys.add(candidate_key)
        top_count += 1

    conn.execute("DELETE FROM spotify_artist_albums WHERE artist_id=?", (artist_id,))
    for group in GROUPS:
        for rank, album in enumerate(
            (payloads["albums"].get(group) or {}).get("items") or [], start=1
        ):
            album_id = album.get("id")
            title = album.get("name")
            if not album_id or not title:
                continue
            conn.execute(
                """
                INSERT OR REPLACE INTO spotify_artist_albums(
                    artist_id, album_id, album_group, artist, album, rank, album_type,
                    release_date, total_tracks, raw_json, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artist_id,
                    album_id,
                    group,
                    artist,
                    title,
                    rank,
                    album.get("album_type"),
                    album.get("release_date"),
                    album.get("total_tracks"),
                    json.dumps(album, ensure_ascii=False, sort_keys=True),
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO spotify_album_expansion(album_id, album, status, attempts, updated_at)
                VALUES(?, ?, 'pending', 0, ?)
                ON CONFLICT(album_id) DO UPDATE SET
                    album=COALESCE(excluded.album, spotify_album_expansion.album),
                    updated_at=excluded.updated_at
                """,
                (album_id, title, now),
            )
            album_count += 1

    conn.execute(
        "UPDATE spotify_artist_expansion SET status='fetched', error=NULL, updated_at=? "
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
            SELECT t.album_id, a.name AS album,
                   ROW_NUMBER() OVER (
                       PARTITION BY t.album_id
                       ORDER BY rs.liked DESC, rs.plays DESC, rs.recording_key
                   ) AS row_num
            FROM recording_stats rs
            JOIN tracks t ON t.track_id = rs.rep_track_id
            LEFT JOIN albums a ON a.album_id = t.album_id
            WHERE t.album_id IS NOT NULL AND (rs.liked = 1 OR rs.plays >= ?)
        )
        SELECT album_id, album FROM ranked WHERE row_num = 1
        """,
        (min_plays,),
    ).fetchall()
    before = conn.execute("SELECT COUNT(*) FROM spotify_album_expansion").fetchone()[0]
    conn.executemany(
        """
        INSERT OR IGNORE INTO spotify_album_expansion(album_id, album, status, attempts, updated_at)
        VALUES(?, ?, 'pending', 0, ?)
        """,
        [(row["album_id"], row["album"], db.utcnow()) for row in rows],
    )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM spotify_album_expansion").fetchone()[0]
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
        else ("WHERE (sae.status = 'pending' OR (sae.status = 'error' AND sae.attempts < ?))")
    )
    status_params: list[Any] = [] if include_fetched else [MAX_ATTEMPTS]
    query = f"""
        WITH artist_priority AS (
            SELECT album_id,
                   MIN(CASE album_group
                       WHEN 'appears_on' THEN 0
                       WHEN 'album' THEN 1
                       WHEN 'single' THEN 2
                       ELSE 3
                   END) AS group_priority,
                   MIN(rank) AS artist_rank
            FROM spotify_artist_albums
            GROUP BY album_id
        ),
        library_priority AS (
            SELECT t.album_id, SUM(rs.plays) AS library_plays
            FROM recording_stats rs
            JOIN tracks t ON t.track_id = rs.rep_track_id
            WHERE rs.liked = 1 OR rs.plays >= ?
            GROUP BY t.album_id
        )
        SELECT sae.*, (ap.album_id IS NOT NULL) AS artist_album,
               COALESCE(ap.group_priority, 999999) AS group_priority,
               COALESCE(ap.artist_rank, 999999) AS artist_rank,
               COALESCE(lp.library_plays, 0) AS library_plays
        FROM spotify_album_expansion sae
        LEFT JOIN artist_priority ap USING(album_id)
        LEFT JOIN library_priority lp USING(album_id)
        {status_filter}
        ORDER BY artist_album DESC, group_priority, artist_rank, library_plays DESC, sae.album_id
        LIMIT ?
    """
    rows = conn.execute(query, [min_plays, *status_params, max(0, batch)]).fetchall()
    if include_fetched:
        total = conn.execute("SELECT COUNT(*) FROM spotify_album_expansion").fetchone()[0]
    else:
        total = conn.execute(
            """
            SELECT COUNT(*) FROM spotify_album_expansion
            WHERE status='pending' OR (status='error' AND attempts < ?)
            """,
            (MAX_ATTEMPTS,),
        ).fetchone()[0]
    return rows, total


def _ensure_album_attempt(conn: sqlite3.Connection, album_id: str) -> sqlite3.Row:
    conn.execute(
        """
        UPDATE spotify_album_expansion
        SET status='pending', attempts=attempts + 1, error=NULL, updated_at=?
        WHERE album_id=?
        """,
        (db.utcnow(), album_id),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM spotify_album_expansion WHERE album_id=?", (album_id,)
    ).fetchone()


def _library_album_seed(
    conn: sqlite3.Connection,
    album_id: str,
    min_plays: int,
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT rs.recording_key, rs.artist, rs.title
        FROM recording_stats rs
        JOIN tracks t ON t.track_id = rs.rep_track_id
        WHERE t.album_id=? AND (rs.liked = 1 OR rs.plays >= ?)
        ORDER BY rs.liked DESC, rs.plays DESC, rs.recording_key
        LIMIT 1
        """,
        (album_id, min_plays),
    ).fetchone()


def _album_artist_seeds(conn: sqlite3.Connection, album_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT saa.artist_id, sae.artist, sae.seed_recording_key, rs.title AS seed_title,
               saa.album_group
        FROM spotify_artist_albums saa
        JOIN spotify_artist_expansion sae USING(artist_id)
        LEFT JOIN recording_stats rs ON rs.recording_key = sae.seed_recording_key
        WHERE saa.album_id=?
        ORDER BY saa.album_group, saa.rank, saa.artist_id
        """,
        (album_id,),
    ).fetchall()


def _track_artist_ids(artists: list[dict[str, Any]]) -> set[str]:
    return {str(artist["id"]) for artist in artists if artist.get("id")}


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
    source_edges = 0
    track_count = 0

    conn.execute("DELETE FROM spotify_album_tracks WHERE album_id=?", (album_id,))
    for rank, track in enumerate(payload["items"], start=1):
        track_id = track.get("id")
        title = track.get("name")
        track_artist, artists = _primary_artist(track)
        if not track_id or not title or not track_artist:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO spotify_album_tracks(
                album_id, track_id, artist, artists_json, title, rank, disc_number,
                track_number, duration_ms, raw_json, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                album_id,
                track_id,
                track_artist,
                json.dumps(artists, ensure_ascii=False, sort_keys=True),
                title,
                rank,
                track.get("disc_number"),
                track.get("track_number"),
                track.get("duration_ms"),
                json.dumps(track, ensure_ascii=False, sort_keys=True),
                now,
            ),
        )
        track_count += 1
        added_keys = []
        external_ids = _external_ids(track)
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
                seed_external_id=album_id,
                source_rank=rank,
                strength=_edge_strength(rank),
                raw=track,
                external_ids=external_ids,
                known=known,
            )
            if candidate_key:
                added_keys.append(candidate_key)
                source_edges += 1

        track_artist_ids = _track_artist_ids(artists)
        for seed in artist_seeds:
            includes_seed = seed["artist_id"] in track_artist_ids
            group = seed["album_group"]
            if group in ("album", "single"):
                if not includes_seed:
                    continue
                source = SAME_ARTIST_SOURCE
                reason = "artist_album_track"
                weak = False
            elif group == "appears_on":
                source = APPEARS_ON_SOURCE if includes_seed else COPRESENCE_SOURCE
                reason = "appears_on_seed_artist" if includes_seed else "appears_on_copresence"
                weak = not includes_seed
            else:
                source = APPEARS_ON_SOURCE if includes_seed else COPRESENCE_SOURCE
                reason = "compilation_seed_artist" if includes_seed else "compilation_copresence"
                weak = not includes_seed
            candidate_key = candidates.add_candidate_source(
                conn,
                artist=track_artist,
                title=title,
                source=source,
                reason=reason,
                seed_recording_key=seed["seed_recording_key"] or "",
                seed_artist=seed["artist"],
                seed_title=seed["seed_title"],
                seed_external_id=f"{seed['artist_id']}:{album_id}:{group}",
                source_rank=rank,
                strength=_edge_strength(rank, weak=weak),
                raw=track,
                external_ids=external_ids,
                known=known,
            )
            if candidate_key:
                added_keys.append(candidate_key)
                source_edges += 1
        candidate_keys.update(added_keys)

    conn.execute(
        "UPDATE spotify_album_expansion SET status='fetched', error=NULL, updated_at=? "
        "WHERE album_id=?",
        (now, album_id),
    )
    return {"tracks": track_count, "source_edges": source_edges, "candidate_keys": candidate_keys}


def _mark_failure(
    conn: sqlite3.Connection,
    *,
    table: str,
    id_column: str,
    entity_id: str,
    attempts: int,
    error: Exception,
) -> str:
    miss = isinstance(error, spotipy.SpotifyException) and error.http_status == 404
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
    sp: spotipy.Spotify,
    artist_batch: int = 3,
    album_batch: int = 20,
    min_plays: int = 5,
    album_limit: int = 10,
    track_limit: int = 100,
    include_fetched: bool = False,
    progress=None,
) -> dict[str, int]:
    """Expand high-signal Spotify artists and a bounded album-track queue."""
    known = {
        row["recording_key"]
        for row in conn.execute(
            "SELECT recording_key FROM recording_stats WHERE recording_key IS NOT NULL"
        )
    }
    candidate_keys = set()
    counts = {
        "artist_seen": 0,
        "artist_fetched": 0,
        "artist_miss": 0,
        "artist_error": 0,
        "top_tracks": 0,
        "top_unavailable": 0,
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
            payloads, top_unavailable = _fetch_artist_surfaces(
                conn,
                row=row,
                sp=sp,
                album_limit=album_limit,
                include_fetched=include_fetched,
            )
            normalized = _normalize_artist(conn, row=row, payloads=payloads, known=known)
            counts["artist_fetched"] += 1
            counts["top_tracks"] += normalized["top_tracks"]
            counts["top_unavailable"] += int(top_unavailable)
            counts["artist_albums"] += normalized["artist_albums"]
            candidate_keys.update(normalized["candidate_keys"])
            conn.commit()
        except (requests.RequestException, spotipy.SpotifyException) as exc:
            status = _mark_failure(
                conn,
                table="spotify_artist_expansion",
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
            payload = _collect(
                lambda limit, offset: _spotify_call(
                    lambda: sp.album_tracks(row["album_id"], limit=limit, offset=offset)
                ),
                track_limit,
            )
            conn.execute(
                "UPDATE spotify_album_expansion SET tracks_json=?, updated_at=? WHERE album_id=?",
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
            conn.commit()
        except (requests.RequestException, spotipy.SpotifyException) as exc:
            status = _mark_failure(
                conn,
                table="spotify_album_expansion",
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
        SELECT COUNT(*) FROM spotify_album_expansion
        WHERE status='pending' OR (status='error' AND attempts < ?)
        """,
        (MAX_ATTEMPTS,),
    ).fetchone()[0]
    counts["candidates"] = len(candidate_keys)
    counts["artist_remaining"] = artist_remaining
    counts["album_remaining"] = album_remaining
    return counts


def status(conn: sqlite3.Connection) -> dict[str, int]:
    out = {
        f"spotify_expand:artist:{row['status']}": row["n"]
        for row in conn.execute(
            "SELECT status, COUNT(*) AS n FROM spotify_artist_expansion GROUP BY status"
        )
    }
    for row in conn.execute(
        "SELECT status, COUNT(*) AS n FROM spotify_album_expansion GROUP BY status"
    ):
        out[f"spotify_expand:album:{row['status']}"] = row["n"]
    for row in conn.execute("SELECT surface, status FROM spotify_expansion_capabilities"):
        out[f"spotify_expand:capability:{row['surface']}:{row['status']}"] = 1
    out["spotify_expand:artist_top_tracks"] = conn.execute(
        "SELECT COUNT(*) FROM spotify_artist_top_tracks"
    ).fetchone()[0]
    out["spotify_expand:artist_albums"] = conn.execute(
        "SELECT COUNT(*) FROM spotify_artist_albums"
    ).fetchone()[0]
    out["spotify_expand:album_tracks"] = conn.execute(
        "SELECT COUNT(*) FROM spotify_album_tracks"
    ).fetchone()[0]
    return out
