"""YouTube Music radio candidate harvesting.

This uses ytmusicapi's unofficial YouTube Music client surface. Treat it as an
opportunistic metadata source: cache the raw response, store provenance, and do
not depend on the endpoint existing forever.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any

import duckdb

try:
    from ytmusicapi import YTMusic
except ImportError:  # optional extra: `uv sync --extra ytmusic`
    YTMusic = None

from . import candidates, db
from .canon import _norm, recording_key, title_lookup_variants

SOURCE = "ytmusic_radio"
REASON = "watch_radio"
LAION_SOURCE = "laion_disco_12m"
_MISSING_YTMUSIC_MSG = (
    "Optional dependency 'ytmusicapi' is not installed. "
    "Install it with `uv sync --extra ytmusic` (or `pip install 'acetate[ytmusic]'`)."
)


def is_available() -> bool:
    """Return True when the optional ytmusicapi extra is installed."""
    return YTMusic is not None


def _catalog_glob() -> str:
    return str(db.db_path().parent / "catalogs" / "laion-disco-12m" / "*.parquet")


def _catalog_exists() -> bool:
    return any(Path(db.db_path().parent / "catalogs" / "laion-disco-12m").glob("*.parquet"))


def _seed_rows(
    conn: sqlite3.Connection, *, batch: int, min_plays: int, include_fetched: bool
) -> list[sqlite3.Row]:
    fetched_filter = (
        ""
        if include_fetched
        else """
        AND NOT EXISTS (
            SELECT 1 FROM ytmusic_radio_seeds y
            WHERE y.recording_key = rs.recording_key AND y.status = 'fetched'
        )
    """
    )
    return conn.execute(
        f"""
        SELECT rs.recording_key, rs.artist, rs.title, rs.plays, rs.liked,
               COALESCE(MAX(CASE WHEN t.track_id = rs.rep_track_id THEN t.duration_ms END),
                        MAX(t.duration_ms)) AS duration_ms
        FROM recording_stats rs
        LEFT JOIN track_recording tr ON tr.recording_key = rs.recording_key
        LEFT JOIN tracks t ON t.track_id = tr.track_id
        WHERE (rs.liked = 1 OR rs.plays >= ?)
          AND rs.artist IS NOT NULL AND rs.title IS NOT NULL
          {fetched_filter}
        GROUP BY rs.recording_key
        ORDER BY rs.liked DESC, rs.plays DESC
        LIMIT ?
        """,
        (min_plays, batch),
    ).fetchall()


# Genres that define the "techno space" the seeds should center on. Fave / hot
# playlist membership brings in everything else the user actually rotates.
_TECHNO_SPACE_LIKE = "%techno%"
_TECHNO_SPACE_IN = (
    "melodic house",
    "progressive house",
    "tech house",
    "minimal",
    "minimal techno",
    "melodic house and techno",
)


def _seed_rows_for_keys(
    conn: sqlite3.Connection, keys: list[str], *, include_fetched: bool, limit: int
) -> list[sqlite3.Row]:
    """Seed rows for an explicit, pre-ordered key list (priority order preserved),
    dropping already-fetched seeds unless include_fetched, capped at `limit`."""
    fetched = set()
    if not include_fetched:
        fetched = {
            r[0]
            for r in conn.execute(
                "SELECT recording_key FROM ytmusic_radio_seeds WHERE status='fetched'"
            )
        }
    out: list[sqlite3.Row] = []
    for key in keys:
        if key in fetched:
            continue
        row = conn.execute(
            """SELECT rs.recording_key, rs.artist, rs.title, rs.plays, rs.liked,
                      COALESCE(MAX(CASE WHEN t.track_id = rs.rep_track_id THEN t.duration_ms END),
                               MAX(t.duration_ms)) AS duration_ms
               FROM recording_stats rs
               LEFT JOIN track_recording tr ON tr.recording_key = rs.recording_key
               LEFT JOIN tracks t ON t.track_id = tr.track_id
               WHERE rs.recording_key = ? AND rs.artist IS NOT NULL AND rs.title IS NOT NULL
               GROUP BY rs.recording_key""",
            (key,),
        ).fetchone()
        if row:
            out.append(row)
        if len(out) >= limit:
            break
    return out


def priority_seeds(
    conn: sqlite3.Connection, *, min_plays: int = 3, hot_playlists: int = 15, limit: int = 500
) -> list[str]:
    """Ordered recording_keys to harvest first: the union of (a) tracks in the
    techno/melodic-techno genre space, (b) members of fave-labeled playlists, and
    (c) members of the most-played own playlists — ranked most-played first. This
    focuses YouTube's radio on the music the user actually lives in."""
    rows = conn.execute(
        """
        WITH hot AS (
            SELECT playlist_id FROM playlist_stats
            WHERE is_own = 1 AND COALESCE(persona, '') NOT IN ('archive', 'deleted', 'one-shot')
            ORDER BY COALESCE(member_plays_180d, 0) DESC LIMIT ?
        ),
        picked AS (
            SELECT playlist_id FROM hot
            UNION SELECT playlist_id FROM playlist_labels WHERE label = 'fave'
        ),
        genre_keys AS (
            SELECT DISTINCT tr.recording_key
            FROM track_genres g JOIN track_recording tr ON tr.track_id = g.track_id
            WHERE g.source = 'lastfm'
              AND (lower(g.genre) LIKE ? OR lower(g.genre) IN (%s))
        ),
        pl_keys AS (
            SELECT DISTINCT recording_key FROM playlist_members
            WHERE playlist_id IN (SELECT playlist_id FROM picked) AND recording_key IS NOT NULL
        ),
        pool AS (
            SELECT recording_key FROM genre_keys
            UNION SELECT recording_key FROM pl_keys
        )
        SELECT rs.recording_key
        FROM recording_stats rs JOIN pool p ON p.recording_key = rs.recording_key
        WHERE rs.artist IS NOT NULL AND rs.title IS NOT NULL
          AND (rs.liked = 1 OR rs.plays >= ?)
        ORDER BY rs.plays DESC, rs.liked DESC
        LIMIT ?
        """
        % ",".join("?" * len(_TECHNO_SPACE_IN)),
        (hot_playlists, _TECHNO_SPACE_LIKE, *_TECHNO_SPACE_IN, min_plays, limit),
    ).fetchall()
    return [r[0] for r in rows]


def background_seeds(
    conn: sqlite3.Connection, *, min_plays: int = 5, priority_min_plays: int = 3
) -> list[str]:
    """The full seed order for the standing background harvest: the priority pool
    first (techno space + fave/hot playlists, most-played), then the rest of the
    liked-or-most-played universe by plays. Covers everything over time, best-fit
    first. `harvest_radio(seed_keys=...)` skips already-fetched, so each run just
    advances the frontier."""
    pri = priority_seeds(conn, min_plays=priority_min_plays, limit=1_000_000)
    seen = set(pri)
    rest = [
        r[0]
        for r in conn.execute(
            "SELECT recording_key FROM recording_stats "
            "WHERE artist IS NOT NULL AND title IS NOT NULL AND (liked = 1 OR plays >= ?) "
            "ORDER BY plays DESC, liked DESC",
            (min_plays,),
        )
    ]
    return pri + [k for k in rest if k not in seen]


def stale_fetched_seeds(conn: sqlite3.Connection, limit: int) -> list[str]:
    """Oldest-fetched seeds — re-harvested for drift once the frontier is covered
    (the radio queue changes over time; repeated sightings are signal)."""
    return [
        r[0]
        for r in conn.execute(
            "SELECT recording_key FROM ytmusic_radio_seeds WHERE status='fetched' "
            "ORDER BY updated_at ASC LIMIT ?",
            (limit,),
        )
    ]


def _duration_ok(result: dict[str, Any], duration_ms: int | None) -> bool:
    if not duration_ms:
        return True
    duration_seconds = result.get("duration_seconds")
    if duration_seconds is None:
        return True
    return abs(float(duration_seconds) - duration_ms / 1000) <= 10


def _duration_match(duration_s: int | None, duration_ms: int | None) -> float:
    if duration_s is None or duration_ms is None:
        return 0.2
    delta = abs(float(duration_s) - duration_ms / 1000)
    if delta <= 3:
        return 1.0
    if delta <= 10:
        return 0.7
    if delta <= 20:
        return 0.2
    return -1.0


def _title_variants(title: str) -> list[str]:
    return title_lookup_variants(title)


def _artist_match(artists: list[str], artist: str) -> float:
    artist_norm = _norm(artist)
    names = [_norm(name) for name in artists]
    if artist_norm in names:
        return 1.0
    if any(artist_norm in name or name in artist_norm for name in names):
        return 0.6
    return -1.0


def _candidate_score(result: dict[str, Any], *, artist: str, duration_ms: int | None) -> float:
    names = [_norm(item.get("name")) for item in result.get("artists", []) if item.get("name")]
    artist_norm = _norm(artist)
    artist_score = 0.0
    if artist_norm in names:
        artist_score = 3.0
    elif any(artist_norm in name or name in artist_norm for name in names):
        artist_score = 1.5
    duration_score = 1.0 if _duration_ok(result, duration_ms) else -2.0
    return artist_score + duration_score


def _row_payload(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    return {
        "song_id": row["song_id"],
        "title": row["title"],
        "artist_names": list(row["artist_names"] or []),
        "album_name": row["album_name"],
        "album_id": row["album_id"],
        "duration": row["duration"],
    }


def _cache_seed_match(
    conn: sqlite3.Connection,
    *,
    recording_key_value: str,
    row: dict[str, Any] | None,
    source: str,
    confidence: float | None,
    status: str,
) -> None:
    conn.execute(
        """
        INSERT INTO ytmusic_seed_matches(
            recording_key, song_id, title, artists_json, album_name, album_id,
            duration_s, source, confidence, status, raw_json, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(recording_key) DO UPDATE SET
            song_id=excluded.song_id,
            title=excluded.title,
            artists_json=excluded.artists_json,
            album_name=excluded.album_name,
            album_id=excluded.album_id,
            duration_s=excluded.duration_s,
            source=excluded.source,
            confidence=excluded.confidence,
            status=excluded.status,
            raw_json=excluded.raw_json,
            updated_at=excluded.updated_at
        """,
        (
            recording_key_value,
            row.get("song_id") if row else None,
            row.get("title") if row else None,
            json.dumps(row.get("artist_names"), sort_keys=True) if row else None,
            row.get("album_name") if row else None,
            row.get("album_id") if row else None,
            row.get("duration") if row else None,
            source,
            confidence,
            status,
            json.dumps(row, sort_keys=True) if row else None,
            db.utcnow(),
        ),
    )


def _lookup_cached_seed(conn: sqlite3.Connection, recording_key_value: str) -> str | None:
    row = conn.execute(
        """
        SELECT song_id FROM ytmusic_seed_matches
        WHERE recording_key = ? AND status = 'matched' AND song_id IS NOT NULL
        """,
        (recording_key_value,),
    ).fetchone()
    return row["song_id"] if row else None


def _lookup_laion(
    conn: sqlite3.Connection,
    *,
    recording_key_value: str,
    artist: str,
    title: str,
    duration_ms: int | None,
) -> str | None:
    if not _catalog_exists():
        return None
    title_variants = _title_variants(title)
    query = """
        SELECT song_id, title, artist_names, album_name, album_id, duration
        FROM read_parquet(?)
        WHERE lower(trim(title)) = ?
        LIMIT 50
    """
    con = duckdb.connect(":memory:")
    columns = ["song_id", "title", "artist_names", "album_name", "album_id", "duration"]
    rows = []
    matched_variant = None
    for variant in title_variants:
        rows = con.execute(query, [_catalog_glob(), variant]).fetchall()
        if rows:
            matched_variant = variant
            break
    best_row: dict[str, Any] | None = None
    best_score = -10.0
    for values in rows:
        row = dict(zip(columns, values, strict=True))
        exact_title = _norm(row.get("title")) == _norm(title)
        title_score = 0.75 if exact_title else 0.25
        score = (
            3.0 * _artist_match(row.get("artist_names") or [], artist)
            + _duration_match(row.get("duration"), duration_ms)
            + title_score
        )
        if score > best_score:
            best_score = score
            best_row = row
    if best_row and best_score >= 3.25:
        best_row["matched_title_variant"] = matched_variant
        best_row["original_title"] = title
        confidence = min(1.0, max(0.0, best_score / 4.0))
        _cache_seed_match(
            conn,
            recording_key_value=recording_key_value,
            row=best_row,
            source=LAION_SOURCE,
            confidence=confidence,
            status="matched",
        )
        return best_row["song_id"]
    if rows:
        if best_row:
            best_row["matched_title_variant"] = matched_variant
            best_row["original_title"] = title
        _cache_seed_match(
            conn,
            recording_key_value=recording_key_value,
            row=best_row,
            source=LAION_SOURCE,
            confidence=max(0.0, best_score / 4.0),
            status="miss",
        )
    else:
        _cache_seed_match(
            conn,
            recording_key_value=recording_key_value,
            row={
                "song_id": None,
                "title": title,
                "artist_names": [artist] if artist else [],
                "album_name": None,
                "album_id": None,
                "duration": int(duration_ms / 1000) if duration_ms else None,
                "miss_reason": "no exact title rows",
            },
            source=LAION_SOURCE,
            confidence=0.0,
            status="miss",
        )
    return None


def _resolve_video_id(
    client: YTMusic, artist: str, title: str, duration_ms: int | None
) -> tuple[str | None, dict[str, Any] | None]:
    results = client.search(f"{artist} {title}", filter="songs", limit=5)
    if not results:
        return None, None
    scored = [
        (_candidate_score(result, artist=artist, duration_ms=duration_ms), index, result)
        for index, result in enumerate(results)
        if result.get("videoId")
    ]
    if not scored:
        return None, None
    score, _index, result = max(scored, key=lambda item: (item[0], -item[1]))
    if score < 1.0:
        return None, result
    return result["videoId"], result


def _resolve_seed_video_id(
    conn: sqlite3.Connection,
    client: YTMusic,
    seed: sqlite3.Row,
) -> tuple[str | None, dict[str, Any] | None, str]:
    cached = _lookup_cached_seed(conn, seed["recording_key"])
    if cached:
        return cached, {"song_id": cached, "source": "cache"}, "cache"
    laion = _lookup_laion(
        conn,
        recording_key_value=seed["recording_key"],
        artist=seed["artist"],
        title=seed["title"],
        duration_ms=seed["duration_ms"],
    )
    if laion:
        return laion, {"song_id": laion, "source": LAION_SOURCE}, LAION_SOURCE
    video_id, search_result = _resolve_video_id(
        client, seed["artist"], seed["title"], seed["duration_ms"]
    )
    if video_id:
        artist_names = (
            [item.get("name") for item in search_result.get("artists", []) if item.get("name")]
            if search_result
            else []
        )
        _cache_seed_match(
            conn,
            recording_key_value=seed["recording_key"],
            row={
                "song_id": video_id,
                "title": search_result.get("title") if search_result else seed["title"],
                "artist_names": artist_names,
                "album_name": (search_result.get("album") or {}).get("name")
                if search_result
                else None,
                "album_id": (search_result.get("album") or {}).get("id") if search_result else None,
                "duration": search_result.get("duration_seconds") if search_result else None,
            },
            source="search",
            confidence=0.65,
            status="matched",
        )
    return video_id, search_result, "search"


def _track_artist(track: dict[str, Any]) -> str | None:
    artists = track.get("artists") or []
    if not artists:
        return None
    return artists[0].get("name")


def _edge_strength(rank: int) -> float:
    return 1.0 / max(1.0, math.sqrt(rank))


def _mark_seed(
    conn: sqlite3.Connection,
    *,
    recording_key_value: str,
    video_id: str | None,
    playlist_id: str | None,
    status: str,
    result_count: int = 0,
    raw: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO ytmusic_radio_seeds(
            recording_key, video_id, playlist_id, status, result_count, attempts,
            error, raw_json, updated_at
        )
        VALUES(?, ?, ?, ?, ?, 1, ?, ?, ?)
        ON CONFLICT(recording_key) DO UPDATE SET
            video_id=COALESCE(excluded.video_id, ytmusic_radio_seeds.video_id),
            playlist_id=COALESCE(excluded.playlist_id, ytmusic_radio_seeds.playlist_id),
            status=excluded.status,
            result_count=excluded.result_count,
            attempts=ytmusic_radio_seeds.attempts + 1,
            error=excluded.error,
            raw_json=COALESCE(excluded.raw_json, ytmusic_radio_seeds.raw_json),
            updated_at=excluded.updated_at
        """,
        (
            recording_key_value,
            video_id,
            playlist_id,
            status,
            result_count,
            error[:300] if error else None,
            json.dumps(raw, sort_keys=True) if raw else None,
            db.utcnow(),
        ),
    )


def harvest_radio(
    conn: sqlite3.Connection,
    *,
    batch: int = 25,
    radio_limit: int = 50,
    min_plays: int = 20,
    include_fetched: bool = False,
    pause_s: float = 1.0,
    seed_keys: list[str] | None = None,
    progress=None,
) -> dict[str, int]:
    if YTMusic is None:
        raise RuntimeError(_MISSING_YTMUSIC_MSG)
    client = YTMusic()
    known = {
        row["recording_key"]
        for row in conn.execute(
            "SELECT recording_key FROM recording_stats WHERE recording_key IS NOT NULL"
        )
    }
    if seed_keys is not None:
        rows = _seed_rows_for_keys(conn, seed_keys, include_fetched=include_fetched, limit=batch)
    else:
        rows = _seed_rows(conn, batch=batch, min_plays=min_plays, include_fetched=include_fetched)
    counts = {"seeds": 0, "resolved": 0, "fetched": 0, "miss": 0, "error": 0, "candidates": 0}

    for index, seed in enumerate(rows):
        counts["seeds"] += 1
        try:
            video_id, search_result, resolver = _resolve_seed_video_id(conn, client, seed)
            if not video_id:
                _mark_seed(
                    conn,
                    recording_key_value=seed["recording_key"],
                    video_id=None,
                    playlist_id=None,
                    status="miss",
                    raw={"search_result": search_result} if search_result else None,
                    error="no acceptable YT Music song result",
                )
                counts["miss"] += 1
                conn.commit()
                continue
            counts["resolved"] += 1
            watch = client.get_watch_playlist(videoId=video_id, radio=True, limit=radio_limit)
            playlist_id = watch.get("playlistId")
            result_count = 0
            for rank, track in enumerate(watch.get("tracks") or [], start=1):
                title = track.get("title")
                artist = _track_artist(track)
                if not title or not artist:
                    continue
                if recording_key(artist, title) == seed["recording_key"]:
                    continue
                candidate_key = candidates.add_candidate_source(
                    conn,
                    artist=artist,
                    title=title,
                    source=SOURCE,
                    reason=REASON,
                    seed_recording_key=seed["recording_key"],
                    seed_artist=seed["artist"],
                    seed_title=seed["title"],
                    seed_external_id=video_id,
                    source_rank=rank,
                    strength=_edge_strength(rank),
                    raw=track,
                    external_ids=[
                        {
                            "provider": "ytmusic",
                            "external_id": track.get("videoId"),
                            "id_type": "videoId",
                            "confidence": 0.8,
                            "raw": track,
                        }
                    ],
                    known=known,
                )
                if candidate_key:
                    result_count += 1
            _mark_seed(
                conn,
                recording_key_value=seed["recording_key"],
                video_id=video_id,
                playlist_id=playlist_id,
                status="fetched",
                result_count=result_count,
                raw={"resolver": resolver, "search_result": search_result, "watch": watch},
            )
            counts["fetched"] += 1
            counts["candidates"] += result_count
            conn.commit()
        except Exception as exc:
            _mark_seed(
                conn,
                recording_key_value=seed["recording_key"],
                video_id=None,
                playlist_id=None,
                status="error",
                error=str(exc),
            )
            counts["error"] += 1
            conn.commit()
        if progress:
            progress(index + 1, len(rows), counts)
        if pause_s > 0:
            time.sleep(pause_s)

    scored = candidates.score_candidates(conn)
    counts["scored"] = scored["scored"]
    return counts


def status(conn: sqlite3.Connection) -> dict[str, int]:
    out = {
        f"ytmusic_seed:{row['status']}": row["n"]
        for row in conn.execute(
            "SELECT status, COUNT(*) AS n FROM ytmusic_radio_seeds GROUP BY status"
        )
    }
    for row in conn.execute(
        "SELECT source, status, COUNT(*) AS n FROM ytmusic_seed_matches GROUP BY source, status"
    ):
        out[f"ytmusic_match:{row['source']}:{row['status']}"] = row["n"]
    return out


def resolve_laion_seeds(
    conn: sqlite3.Connection,
    *,
    batch: int = 100,
    min_plays: int = 20,
    include_matched: bool = False,
    progress=None,
) -> dict[str, int]:
    rows = _seed_rows(conn, batch=batch, min_plays=min_plays, include_fetched=True)
    counts = {"seen": 0, "matched": 0, "miss": 0}
    for index, row in enumerate(rows):
        if (
            not include_matched
            and conn.execute(
                "SELECT 1 FROM ytmusic_seed_matches WHERE recording_key = ?",
                (row["recording_key"],),
            ).fetchone()
        ):
            continue
        counts["seen"] += 1
        found = _lookup_laion(
            conn,
            recording_key_value=row["recording_key"],
            artist=row["artist"],
            title=row["title"],
            duration_ms=row["duration_ms"],
        )
        if found:
            counts["matched"] += 1
        else:
            counts["miss"] += 1
        if index % 25 == 0:
            conn.commit()
            if progress:
                progress(index, len(rows), counts)
    conn.commit()
    return counts
