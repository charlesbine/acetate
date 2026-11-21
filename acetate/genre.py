"""Genre classification + discovery.

Tags favorite tracks with Last.fm genres (cached in track_genres, raw weights
kept), and ranks your genres play-weighted for the "discover, then pick" menu.
Non-genre tags (geography, moods, decades) are filtered at report time so the
cache stays raw and re-classifiable.
"""

from __future__ import annotations

import os
import json
import sqlite3

import requests

from . import db, lastfm

MAX_ATTEMPTS = 3

# Tags that aren't musical genres — excluded from the discovery menu / playlists.
NON_GENRE = {
    "belgian",
    "belgium",
    "switzerland",
    "swiss",
    "french",
    "france",
    "german",
    "germany",
    "british",
    "uk",
    "usa",
    "american",
    "spanish",
    "italian",
    "italy",
    "dutch",
    "netherlands",
    "canadian",
    "australian",
    "swedish",
    "norwegian",
    "finnish",
    "danish",
    "icelandic",
    "scottish",
    "irish",
    "japanese",
    "brazilian",
    "polish",
    "russian",
    "ukrainian",
    "female vocalists",
    "male vocalists",
    "female vocalist",
    "instrumental",
    "vocal",
    "seen live",
    "favorites",
    "favourite",
    "favorite",
    "favourites",
    "beautiful",
    "chill",
    "chillout",
    "cool",
    "awesome",
    "love",
    "sexy",
    "catchy",
    "epic",
    "melancholic",
    "happy",
    "sad",
    "party",
    "summer",
    "relaxing",
    "mellow",
    "smooth",
    "easy listening",
    "00s",
    "10s",
    "90s",
    "80s",
    "70s",
    "60s",
    "2010s",
    "2020s",
    "2000s",
    "1990s",
    "1980s",
    "all",
    "other",
    "amazing",
    "best",
    "good music",
    "my music",
    "music",
    "spotify",
}


def is_genre(tag: str) -> bool:
    return tag not in NON_GENRE and len(tag) <= 40


# Specificity order for assigning each track ONE "primary" genre: micro-genres
# first, broad catch-alls last. A track's primary genre is the highest-priority
# tag it carries (above a confidence floor). This is the tunable lever.
GENRE_PRIORITY = [
    "melodic techno",
    "melodic house",
    "minimal techno",
    "tech house",
    "deep house",
    "progressive house",
    "progressive trance",
    "electro house",
    "nu disco",
    "nu-disco",
    "trip-hop",
    "downtempo",
    "idm",
    "dubstep",
    "drum and bass",
    "breakbeat",
    "future garage",
    "ambient",
    "disco",
    "nu jazz",
    "funk",
    "trance",
    "techno",
    "house",
    "electro",
    "minimal",
    "electronica",
    "edm",
    "dance",
    "indie pop",
    "indie rock",
    "indie",
    "alternative",
    "rock",
    "pop",
    "trap",
    "hip-hop",
    "hip hop",
    "rap",
    "experimental",
    "folk",
    "singer-songwriter",
]
_PRIORITY_RANK = {g: i for i, g in enumerate(GENRE_PRIORITY)}


def primary_genre(tags, min_confidence: float = 10) -> str | None:
    """Most-specific genre present in `tags` (iterable of (genre, confidence))
    above the confidence floor, or None if only broad/untracked tags."""
    cands = [(_PRIORITY_RANK[g], g) for g, c in tags if g in _PRIORITY_RANK and c >= min_confidence]
    return min(cands)[1] if cands else None


def _favorite_rows(conn: sqlite3.Connection, min_plays: int, limit: int | None):
    """Most-played tracks (from GDPR labels) — the quick discovery-menu source."""
    sql = """
        SELECT p.track_id, l.artist_name, l.track_name, COUNT(*) AS plays
        FROM play_events p JOIN track_labels l ON l.track_id = p.track_id
        WHERE p.track_id IS NOT NULL AND (p.ms_played IS NULL OR p.ms_played >= 30000)
          AND l.artist_name IS NOT NULL AND l.track_name IS NOT NULL
        GROUP BY p.track_id HAVING COUNT(*) >= ?
        ORDER BY plays DESC
    """
    params: tuple = (min_plays,)
    if limit:
        sql += " LIMIT ?"
        params = (min_plays, limit)
    return conn.execute(sql, params).fetchall()


def _all_rows(conn: sqlite3.Connection):
    """Every track we can name — GDPR labels UNION playlist/library dims — most
    played first. This is the full backfill universe."""
    return conn.execute("""
        WITH plays AS (
            SELECT track_id, COUNT(*) AS plays FROM play_events
            WHERE track_id IS NOT NULL AND (ms_played IS NULL OR ms_played >= 30000)
            GROUP BY track_id
        ),
        names AS (
            SELECT track_id, artist_name, track_name FROM track_labels
            WHERE artist_name IS NOT NULL AND track_name IS NOT NULL
            UNION
            SELECT t.track_id, a.name, t.name
            FROM tracks t
            JOIN track_artists ta ON ta.track_id = t.track_id AND ta.position = 0
            JOIN artists a ON a.artist_id = ta.artist_id
            WHERE t.name IS NOT NULL AND a.name IS NOT NULL
        )
        SELECT n.track_id, MIN(n.artist_name) AS artist_name, MIN(n.track_name) AS track_name,
               COALESCE(p.plays, 0) AS plays
        FROM names n LEFT JOIN plays p ON p.track_id = n.track_id
        GROUP BY n.track_id
        ORDER BY plays DESC
    """).fetchall()


def universe_count(conn: sqlite3.Connection) -> int:
    """Size of the nameable-track universe — same set as _all_rows but a plain COUNT,
    so it skips materializing 48k rows + the 222k-row play aggregation (used by status)."""
    return conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT track_id FROM track_labels
            WHERE artist_name IS NOT NULL AND track_name IS NOT NULL
            UNION
            SELECT t.track_id FROM tracks t
            JOIN track_artists ta ON ta.track_id = t.track_id AND ta.position = 0
            JOIN artists a ON a.artist_id = ta.artist_id
            WHERE t.name IS NOT NULL AND a.name IS NOT NULL
        )
    """).fetchone()[0]


def _already_tagged(conn: sqlite3.Connection) -> set[str]:
    return {
        r[0]
        for r in conn.execute(
            """
        SELECT DISTINCT track_id FROM track_genres WHERE source='lastfm'
        UNION
        SELECT track_id FROM lastfm_tag_runs
        WHERE status IN ('available', 'provider_miss', 'unknown_legacy')
           OR (status='error' AND attempts >= ?)
    """,
            (MAX_ATTEMPTS,),
        )
    }


def _start_tag_run(conn: sqlite3.Connection, track_id: str) -> int:
    conn.execute(
        """
        INSERT INTO lastfm_tag_runs(track_id, status, attempts, updated_at)
        VALUES(?, 'pending', 1, ?)
        ON CONFLICT(track_id) DO UPDATE SET
            status='pending', attempts=lastfm_tag_runs.attempts + 1,
            error=NULL, detail_json=NULL, updated_at=excluded.updated_at
    """,
        (track_id, db.utcnow()),
    )
    conn.commit()
    return conn.execute(
        "SELECT attempts FROM lastfm_tag_runs WHERE track_id=?", (track_id,)
    ).fetchone()[0]


def _finish_tag_run(
    conn: sqlite3.Connection,
    track_id: str,
    status: str,
    *,
    detail: dict | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        "UPDATE lastfm_tag_runs SET status=?, error=?, detail_json=?, updated_at=? "
        "WHERE track_id=?",
        (
            status,
            error[:300] if error else None,
            json.dumps(detail, sort_keys=True) if detail else None,
            db.utcnow(),
            track_id,
        ),
    )


def _run_tagging(
    conn: sqlite3.Connection, rows, batch: int | None = None, max_tags: int = 6, progress=None
) -> dict:
    """Tag the not-yet-tagged tracks in `rows` via Last.fm. Resumable; `batch`
    caps how many to do this run."""
    if not os.environ.get("LASTFM_API_KEY"):
        raise RuntimeError("LASTFM_API_KEY not set (resolve it via .env/1Password first).")
    now = db.utcnow()
    done = _already_tagged(conn)
    todo = [r for r in rows if r["track_id"] not in done]
    if batch:
        todo = todo[:batch]
    newly = 0
    outcomes = {"available": 0, "provider_miss": 0, "error": 0}
    for i, t in enumerate(todo):
        attempts = _start_tag_run(conn, t["track_id"])
        try:
            lookup = lastfm.best_tags_result(t["artist_name"], t["track_name"])
        except requests.RequestException as exc:
            status = "error" if attempts >= MAX_ATTEMPTS else "pending"
            _finish_tag_run(conn, t["track_id"], status, error=str(exc))
            outcomes["error"] += 1
            conn.commit()
            continue
        tags = lookup.tags
        conn.execute(
            "DELETE FROM track_genres WHERE track_id=? AND source IN ('lastfm','lastfm_none')",
            (t["track_id"],),
        )
        genre_rows = [
            (t["track_id"], name, "lastfm", float(count), now) for name, count in tags[:max_tags]
        ]
        if genre_rows:
            conn.executemany(
                "INSERT OR IGNORE INTO track_genres(track_id, genre, source, confidence, assigned_at) "
                "VALUES(?,?,?,?,?)",
                genre_rows,
            )
            newly += 1
        else:
            conn.execute(  # sentinel: don't re-query an untaggable track next run
                "INSERT OR IGNORE INTO track_genres(track_id, genre, source, confidence, assigned_at) "
                "VALUES(?,?,?,?,?)",
                (t["track_id"], "_none_", "lastfm_none", 0.0, now),
            )
        _finish_tag_run(
            conn,
            t["track_id"],
            lookup.status,
            detail={"result": lookup.detail, "tag_count": len(tags)},
        )
        outcomes[lookup.status] += 1
        if i % 50 == 0:
            conn.commit()
            if progress:
                progress(i, len(todo), newly)
    conn.commit()
    return {
        "universe": len(rows),
        "already_tagged": len(done),
        "to_do": len(todo),
        "newly_tagged": newly,
        **outcomes,
    }


def tag_favorites(
    conn: sqlite3.Connection, min_plays: int = 10, limit: int | None = None, progress=None
) -> dict:
    """Tag your most-played favorites (the quick discovery-menu pass)."""
    return _run_tagging(conn, _favorite_rows(conn, min_plays, limit), progress=progress)


def tag_backfill(conn: sqlite3.Connection, batch: int | None = None, progress=None) -> dict:
    """Tag your ENTIRE library, favorites first, resumable. Re-run to continue;
    once caught up it only tags newly-ingested tracks."""
    return _run_tagging(conn, _all_rows(conn), batch=batch, progress=progress)


def top_genres(
    conn: sqlite3.Connection, min_plays: int = 10, limit: int = 40
) -> list[tuple[str, int, int]]:
    """Your genres ranked by total plays of tracks carrying each tag."""
    rows = conn.execute(
        """
        WITH fav AS (
            SELECT track_id, COUNT(*) AS plays FROM play_events
            WHERE track_id IS NOT NULL AND (ms_played IS NULL OR ms_played >= 30000)
            GROUP BY track_id HAVING COUNT(*) >= ?
        )
        SELECT g.genre, COUNT(DISTINCT g.track_id) AS tracks, SUM(f.plays) AS plays
        FROM track_genres g JOIN fav f ON f.track_id = g.track_id
        WHERE g.source = 'lastfm'
        GROUP BY g.genre ORDER BY plays DESC
    """,
        (min_plays,),
    ).fetchall()
    out = [(r["genre"], r["tracks"], r["plays"]) for r in rows if is_genre(r["genre"])]
    return out[:limit]
