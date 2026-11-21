"""Discovery — outside-library recommendations, consumed two ways.

Per-playlist: embedded candidates scored through the taste membership engine
against a specific authored playlist ("new for Melodic Techno") → proposals
ledger, judged on /review.

Weekly playlist: the top reranked candidates that resolve to a Spotify track id
become the acetate-owned 🔭 Discovery playlist, rotated so nothing repeats
until the pool cycles. Every surfaced track is logged to recommendation_events,
so subsequent plays grade the recommender automatically (proposals.feedback).
"""

from __future__ import annotations

import sqlite3

import spotipy

from . import db, proposals, taste

DISCOVERY_KIND = "discovery"
MIN_BLEND = 0.0


def spotify_track_id(conn: sqlite3.Connection, candidate_key: str) -> str | None:
    row = conn.execute(
        "SELECT external_id FROM candidate_external_ids "
        "WHERE candidate_key=? AND provider='spotify' AND id_type='track_id' "
        "ORDER BY confidence DESC LIMIT 1",
        (candidate_key,),
    ).fetchone()
    return row["external_id"] if row else None


def _embedded_candidates(conn: sqlite3.Connection) -> list[dict]:
    """Non-duplicate embedded candidates with their rerank blend (when ranked)."""
    return [
        dict(r)
        for r in conn.execute("""
        SELECT cr.candidate_key, cr.artist, cr.title, rr.combined_score
        FROM candidate_recordings cr
        LEFT JOIN candidate_rerank rr ON rr.candidate_key = cr.candidate_key
        WHERE cr.status='embedded'
          AND cr.candidate_key NOT IN (SELECT candidate_key FROM candidate_duplicates)""")
    ]


def for_playlist(
    conn: sqlite3.Connection, playlist_id: str, name: str | None = None, limit: int = 15
) -> dict:
    """Score embedded candidates against one authored playlist's region and file
    the best fits as add-proposals. Score = 70% playlist membership + 30% the
    global rerank blend (provenance + library affinity), so a candidate needs
    both "sounds like this playlist" and independent evidence to rank."""
    region = taste.load_region(conn, playlist_id, name)
    if region is None:
        return {"proposed": 0, "skipped": "too few embedded members"}
    cands = _embedded_candidates(conn)
    if not cands:
        return {"proposed": 0, "skipped": "no embedded candidates"}
    fit = taste.membership(conn, region, [c["candidate_key"] for c in cands])
    scored = []
    for c in cands:
        m = fit.get(c["candidate_key"])
        if m is None:
            continue
        blend = 0.7 * m + 0.3 * float(c["combined_score"] or 0.0)
        scored.append((blend, m, c))
    scored.sort(key=lambda t: -t[0])
    proposed = 0
    for blend, m, c in scored[:limit]:
        tid = spotify_track_id(conn, c["candidate_key"])
        if proposals.propose(
            conn,
            playlist_id=playlist_id,
            playlist_name=region.name,
            action="add",
            kind="candidate",
            recording_key=c["candidate_key"],
            track_id=tid,
            artist=c["artist"],
            title=c["title"],
            score=round(blend, 4),
            source="discovery",
            reasons=[
                f"outside-library: membership {round(m, 3)} for “{region.name}”, "
                f"rerank blend {round(float(c['combined_score'] or 0), 3)}"
            ],
        ):
            proposals.log_event(
                conn,
                recording_key=c["candidate_key"],
                track_id=tid,
                surface="proposal:discovery",
                target=region.name,
                score=round(blend, 4),
            )
            proposed += 1
    conn.commit()
    return {"proposed": proposed, "playlist": region.name, "candidates_scored": len(scored)}


def playlist_pool(conn: sqlite3.Connection, size: int) -> tuple[list[tuple[str, str, float]], int]:
    """Top reranked, non-duplicate candidates with Spotify ids, rotated:
    never-surfaced first, then least-recently surfaced. Returns
    [(candidate_key, spotify_track_id, score)] and how many lacked a Spotify id."""
    rows = conn.execute(
        """
        SELECT rr.candidate_key, rr.combined_score, mpt.last_surfaced_at
        FROM candidate_rerank rr
        JOIN candidate_recordings cr ON cr.candidate_key = rr.candidate_key
        LEFT JOIN managed_playlist_tracks mpt
          ON mpt.kind=? AND mpt.key='default' AND mpt.track_id = rr.candidate_key
        WHERE cr.status='embedded'
          AND cr.candidate_key NOT IN (SELECT candidate_key FROM candidate_duplicates)
        ORDER BY (mpt.last_surfaced_at IS NOT NULL), mpt.last_surfaced_at,
                 rr.combined_score DESC""",
        (DISCOVERY_KIND,),
    ).fetchall()
    picked: list[tuple[str, str, float]] = []
    missing_spotify = 0
    for r in rows:
        if len(picked) >= size:
            break
        tid = spotify_track_id(conn, r["candidate_key"])
        if not tid:
            missing_spotify += 1
            continue
        picked.append((r["candidate_key"], tid, float(r["combined_score"])))
    return picked, missing_spotify


def build_discovery_playlist(conn: sqlite3.Connection, sp: spotipy.Spotify, size: int = 30) -> dict:
    """Sync the 🔭 Discovery managed playlist from the rotated candidate pool.
    managed_playlist_tracks stores candidate_keys (not Spotify ids) so rotation
    state survives id re-resolution."""
    from . import playlists

    kind, key, now = DISCOVERY_KIND, "default", db.utcnow()
    picked, missing = playlist_pool(conn, size)
    name = "🔭 Discovery"
    if not picked and not playlists._existing(conn, kind, key):
        return {"name": name, "tracks": 0, "skipped": True, "no_spotify_id": missing}
    pid = playlists._get_or_create(
        conn,
        sp,
        kind,
        key,
        name,
        "Music outside your library that acetate thinks you'll like — provenance-first "
        "candidates reranked by sonic fit. Auto-generated by acetate.",
    )
    playlists._set_tracks(conn, sp, pid, [tid for _ck, tid, _s in picked])
    conn.execute(
        "UPDATE managed_playlist_tracks SET currently_in=0 WHERE kind=? AND key=?", (kind, key)
    )
    for ck, tid, score in picked:
        conn.execute(
            "INSERT INTO managed_playlist_tracks(kind,key,track_id,first_added_at,"
            "last_surfaced_at,surface_count,currently_in) VALUES(?,?,?,?,?,1,1) "
            "ON CONFLICT(kind,key,track_id) DO UPDATE SET "
            "last_surfaced_at=excluded.last_surfaced_at, surface_count=surface_count+1, "
            "currently_in=1",
            (kind, key, ck, now, now),
        )
        proposals.log_event(
            conn,
            recording_key=ck,
            track_id=tid,
            surface="discovery_playlist",
            target=name,
            score=score,
        )
    conn.commit()
    return {"name": name, "tracks": len(picked), "playlist_id": pid, "no_spotify_id": missing}
