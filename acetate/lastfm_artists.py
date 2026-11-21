"""Resumable Last.fm artist metadata and candidate expansion cache."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Callable

import requests

from . import candidates, db, lastfm
from .canon import _norm, recording_key

SOURCE = "lastfm_artist_top_track"
MAX_ATTEMPTS = 3
DEFAULT_PAUSE_S = 1.0
SURFACES = (
    ("artist.getinfo", "info_json"),
    ("artist.getsimilar", "similar_json"),
    ("artist.gettoptracks", "top_tracks_json"),
    ("artist.gettopalbums", "top_albums_json"),
)

CallFn = Callable[..., dict[str, Any]]


def _as_list(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    if isinstance(value, dict):
        return [value]
    return [item for item in value if isinstance(item, dict)]


def _int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _seed_rows(
    conn: sqlite3.Connection,
    *,
    batch: int,
    min_plays: int,
    include_fetched: bool,
) -> tuple[list[dict[str, Any]], int]:
    rows = conn.execute(
        """
        SELECT artist, SUM(plays) AS total_plays, MAX(plays) AS max_plays,
               MAX(liked) AS liked
        FROM recording_stats
        WHERE artist IS NOT NULL AND artist != '' AND (liked = 1 OR plays >= ?)
        GROUP BY artist
        ORDER BY liked DESC, total_plays DESC, max_plays DESC
        """,
        (min_plays,),
    ).fetchall()
    seen = set()
    pending = []
    for row in rows:
        artist_key = _norm(row["artist"])
        if not artist_key or artist_key in seen:
            continue
        seen.add(artist_key)
        cached = conn.execute(
            "SELECT status, attempts FROM lastfm_artist_cache WHERE artist_key=?",
            (artist_key,),
        ).fetchone()
        if not include_fetched and cached:
            if cached["status"] in ("fetched", "miss"):
                continue
            if cached["status"] == "error" and cached["attempts"] >= MAX_ATTEMPTS:
                continue
        pending.append(
            {
                "artist_key": artist_key,
                "artist": row["artist"],
                "total_plays": row["total_plays"],
                "liked": row["liked"],
            }
        )
    return pending[:batch], len(pending)


def _ensure_attempt(conn: sqlite3.Connection, artist_key: str, artist: str) -> sqlite3.Row:
    now = db.utcnow()
    conn.execute(
        """
        INSERT INTO lastfm_artist_cache(artist_key, artist, status, attempts, updated_at)
        VALUES(?, ?, 'pending', 1, ?)
        ON CONFLICT(artist_key) DO UPDATE SET
            artist=excluded.artist,
            status='pending',
            attempts=lastfm_artist_cache.attempts + 1,
            error=NULL,
            updated_at=excluded.updated_at
        """,
        (artist_key, artist, now),
    )
    conn.commit()
    return conn.execute(
        "SELECT * FROM lastfm_artist_cache WHERE artist_key=?", (artist_key,)
    ).fetchone()


def _surface_params(method: str, artist: str, limit: int) -> dict[str, Any]:
    params: dict[str, Any] = {"artist": artist}
    if method in ("artist.getsimilar", "artist.gettoptracks", "artist.gettopalbums"):
        params["limit"] = limit
    return params


def _fetch_missing_surfaces(
    conn: sqlite3.Connection,
    *,
    row: sqlite3.Row,
    artist: str,
    limit: int,
    call: CallFn,
    refresh: bool,
) -> dict[str, dict[str, Any]]:
    payloads = {}
    for method, column in SURFACES:
        cached = row[column]
        if cached and not refresh:
            payloads[column] = json.loads(cached)
            continue
        payload = call(method, **_surface_params(method, artist, limit))
        payloads[column] = payload
        conn.execute(
            f"UPDATE lastfm_artist_cache SET {column}=?, updated_at=? WHERE artist_key=?",
            (
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                db.utcnow(),
                row["artist_key"],
            ),
        )
        conn.commit()
    return payloads


def _replace_normalized(
    conn: sqlite3.Connection,
    *,
    artist_key: str,
    requested_artist: str,
    payloads: dict[str, dict[str, Any]],
    known: set[str],
) -> dict[str, int]:
    info = payloads["info_json"].get("artist") or {}
    canonical_artist = info.get("name") or requested_artist
    stats = info.get("stats") or {}
    tags = [
        tag.get("name") for tag in _as_list((info.get("tags") or {}).get("tag")) if tag.get("name")
    ]
    conn.execute(
        """
        UPDATE lastfm_artist_cache
        SET corrected_artist=?, mbid=?, listeners=?, playcount=?, tags_json=?,
            bio_summary=?, status='fetched', error=NULL, updated_at=?
        WHERE artist_key=?
        """,
        (
            canonical_artist,
            info.get("mbid") or None,
            _int(stats.get("listeners")),
            _int(stats.get("playcount")),
            json.dumps(tags, ensure_ascii=False),
            (info.get("bio") or {}).get("summary"),
            db.utcnow(),
            artist_key,
        ),
    )

    now = db.utcnow()
    similar_rows = _as_list((payloads["similar_json"].get("similarartists") or {}).get("artist"))
    conn.execute("DELETE FROM lastfm_artist_similar WHERE artist_key=?", (artist_key,))
    for rank, item in enumerate(similar_rows, start=1):
        name = item.get("name")
        if not name:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO lastfm_artist_similar(
                artist_key, sim_artist_key, sim_artist, match, rank, mbid, url, raw_json, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artist_key,
                _norm(name),
                name,
                float(item.get("match") or 0),
                rank,
                item.get("mbid") or None,
                item.get("url"),
                json.dumps(item, ensure_ascii=False, sort_keys=True),
                now,
            ),
        )

    track_rows = _as_list((payloads["top_tracks_json"].get("toptracks") or {}).get("track"))
    conn.execute("DELETE FROM lastfm_artist_top_tracks WHERE artist_key=?", (artist_key,))
    candidate_count = 0
    for rank, item in enumerate(track_rows, start=1):
        title = item.get("name")
        track_artist = (item.get("artist") or {}).get("name") or canonical_artist
        if not title or not track_artist:
            continue
        track_key = recording_key(track_artist, title)
        conn.execute(
            """
            INSERT OR REPLACE INTO lastfm_artist_top_tracks(
                artist_key, track_key, artist, title, rank, listeners, playcount,
                mbid, url, raw_json, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artist_key,
                track_key,
                track_artist,
                title,
                rank,
                _int(item.get("listeners")),
                _int(item.get("playcount")),
                item.get("mbid") or None,
                item.get("url"),
                json.dumps(item, ensure_ascii=False, sort_keys=True),
                now,
            ),
        )
        candidate_key = candidates.add_candidate_source(
            conn,
            artist=track_artist,
            title=title,
            source=SOURCE,
            reason="artist.getTopTracks",
            seed_artist=canonical_artist,
            seed_external_id=artist_key,
            source_rank=rank,
            strength=1.0,
            raw=item,
            known=known,
        )
        candidate_count += int(candidate_key is not None)

    album_rows = _as_list((payloads["top_albums_json"].get("topalbums") or {}).get("album"))
    conn.execute("DELETE FROM lastfm_artist_top_albums WHERE artist_key=?", (artist_key,))
    for rank, item in enumerate(album_rows, start=1):
        album = item.get("name")
        album_artist = (item.get("artist") or {}).get("name") or canonical_artist
        if not album or not album_artist:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO lastfm_artist_top_albums(
                artist_key, album_key, artist, album, rank, playcount,
                mbid, url, raw_json, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artist_key,
                recording_key(album_artist, album),
                album_artist,
                album,
                rank,
                _int(item.get("playcount")),
                item.get("mbid") or None,
                item.get("url"),
                json.dumps(item, ensure_ascii=False, sort_keys=True),
                now,
            ),
        )
    return {
        "similar": len(similar_rows),
        "tracks": len(track_rows),
        "albums": len(album_rows),
        "candidates": candidate_count,
    }


def _mark_failure(
    conn: sqlite3.Connection,
    artist_key: str,
    attempts: int,
    error: Exception,
) -> str:
    miss = isinstance(error, lastfm.LastfmAPIError) and error.code == 6
    status = "miss" if miss else ("error" if attempts >= MAX_ATTEMPTS else "pending")
    conn.execute(
        "UPDATE lastfm_artist_cache SET status=?, error=?, updated_at=? WHERE artist_key=?",
        (status, str(error)[:300], db.utcnow(), artist_key),
    )
    conn.commit()
    return status


def harvest(
    conn: sqlite3.Connection,
    *,
    batch: int = 10,
    min_plays: int = 5,
    limit: int = 25,
    include_fetched: bool = False,
    pause_s: float = DEFAULT_PAUSE_S,
    call: CallFn | None = None,
    progress=None,
) -> dict[str, int]:
    """Cache four Last.fm artist surfaces and promote top-track candidates."""
    call = call or lastfm._call
    rows, _pending_total = _seed_rows(
        conn,
        batch=batch,
        min_plays=min_plays,
        include_fetched=include_fetched,
    )
    known = {
        row["recording_key"]
        for row in conn.execute(
            "SELECT recording_key FROM recording_stats WHERE recording_key IS NOT NULL"
        )
    }
    counts = {
        "seen": 0,
        "fetched": 0,
        "miss": 0,
        "error": 0,
        "similar": 0,
        "tracks": 0,
        "albums": 0,
        "candidates": 0,
    }
    for index, seed in enumerate(rows, start=1):
        started = time.monotonic()
        counts["seen"] += 1
        row = _ensure_attempt(conn, seed["artist_key"], seed["artist"])
        try:
            payloads = _fetch_missing_surfaces(
                conn,
                row=row,
                artist=seed["artist"],
                limit=limit,
                call=call,
                refresh=include_fetched,
            )
            normalized = _replace_normalized(
                conn,
                artist_key=seed["artist_key"],
                requested_artist=seed["artist"],
                payloads=payloads,
                known=known,
            )
            counts["fetched"] += 1
            for key in ("similar", "tracks", "albums", "candidates"):
                counts[key] += normalized[key]
            conn.commit()
        except requests.RequestException as exc:
            status = _mark_failure(conn, seed["artist_key"], row["attempts"], exc)
            counts["miss" if status == "miss" else "error"] += 1
        if progress:
            progress(index, len(rows), counts)
        wait = pause_s - (time.monotonic() - started)
        if wait > 0:
            time.sleep(wait)

    if counts["candidates"]:
        scored = candidates.score_candidates(conn)
        counts["scored"] = scored["scored"]
    else:
        counts["scored"] = 0

    _pending_rows, remaining = _seed_rows(
        conn,
        batch=1,
        min_plays=min_plays,
        include_fetched=False,
    )
    counts["remaining"] = remaining
    return counts


def status(conn: sqlite3.Connection) -> dict[str, int]:
    out = {
        f"lastfm_artist:{row['status']}": row["n"]
        for row in conn.execute(
            "SELECT status, COUNT(*) AS n FROM lastfm_artist_cache GROUP BY status"
        )
    }
    out["lastfm_artist:similar_edges"] = conn.execute(
        "SELECT COUNT(*) FROM lastfm_artist_similar"
    ).fetchone()[0]
    out["lastfm_artist:top_tracks"] = conn.execute(
        "SELECT COUNT(*) FROM lastfm_artist_top_tracks"
    ).fetchone()[0]
    out["lastfm_artist:top_albums"] = conn.execute(
        "SELECT COUNT(*) FROM lastfm_artist_top_albums"
    ).fetchone()[0]
    return out
