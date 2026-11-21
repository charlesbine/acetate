"""Artist discovery from a playlist by reverse-neighbor voting.

Each playlist recording votes for outside-library artists found in its cached
Last.fm similar-track edges.  An artist gets at most one vote per seed, which
prevents a large catalog around one song from masquerading as broad playlist
fit.  The result is artist-first, but keeps representative track connections
so every recommendation is explainable.
"""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict

from . import taste
from .canon import normalize_text

VALID_NOVELTY = {"outside_playlist", "outside_library"}


def find_playlist(conn: sqlite3.Connection, query: str) -> tuple[dict | None, list[str]]:
    """Resolve an authored playlist id, exact name, or unique name fragment.

    Returns ``(playlist, alternatives)``.  Alternatives is populated when the
    query is ambiguous so callers can report the useful choices.
    """
    playlists = taste.authored_playlists(conn)
    needle = query.strip().lower()
    exact = [
        playlist
        for playlist in playlists
        if playlist["playlist_id"] == query or (playlist["name"] or "").lower() == needle
    ]
    if exact:
        return exact[0], []
    matches = [
        playlist for playlist in playlists if needle and needle in (playlist["name"] or "").lower()
    ]
    if len(matches) == 1:
        return matches[0], []
    return None, [playlist["name"] for playlist in matches[:8]]


def _familiar_artists(
    conn: sqlite3.Connection,
    playlist_id: str,
    novelty: str,
) -> set[str]:
    if novelty == "outside_library":
        rows = conn.execute("SELECT DISTINCT artist FROM recording_stats")
        return {normalize_text(row[0]) for row in rows if row[0]}
    keys = taste.playlist_recordings(conn, playlist_id)
    if not keys:
        return set()
    familiar: set[str] = set()
    for index in range(0, len(keys), 500):
        chunk = keys[index : index + 500]
        marks = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT DISTINCT artist FROM recording_stats WHERE recording_key IN ({marks})",
            chunk,
        )
        familiar.update(normalize_text(row[0]) for row in rows if row[0])
    return familiar


def discover(
    conn: sqlite3.Connection,
    playlist: str,
    *,
    limit: int = 20,
    novelty: str = "outside_library",
    min_seed_coverage: int = 2,
) -> dict:
    """Rank artists repeatedly adjacent to distinct tracks in ``playlist``.

    Score components are normalized rank-weighted votes (50%), distinct seed
    coverage (30%), and mean edge strength (20%).  Only the strongest candidate
    track by an artist may vote for a given playlist seed.
    """
    if novelty not in VALID_NOVELTY:
        raise ValueError(f"novelty must be one of {sorted(VALID_NOVELTY)}")
    limit = max(1, min(int(limit), 100))
    min_seed_coverage = max(1, int(min_seed_coverage))
    resolved, alternatives = find_playlist(conn, playlist)
    if not resolved:
        return {
            "error": "ambiguous playlist" if alternatives else "playlist not found",
            "query": playlist,
            "matches": alternatives,
            "artists": [],
        }

    playlist_id = resolved["playlist_id"]
    member_keys = taste.playlist_recordings(conn, playlist_id)
    if not member_keys:
        return {
            "playlist": resolved,
            "artists": [],
            "reason": "playlist has no resolved recordings",
        }

    familiar = _familiar_artists(conn, playlist_id, novelty)
    best_by_artist_seed: dict[tuple[str, str], dict] = {}
    for index in range(0, len(member_keys), 400):
        chunk = member_keys[index : index + 400]
        marks = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"""
            SELECT cs.seed_recording_key, cs.seed_artist, cs.seed_title,
                   cs.source_rank, cs.strength, cr.candidate_key,
                   cr.artist, cr.title
            FROM candidate_sources cs
            JOIN candidate_recordings cr USING(candidate_key)
            LEFT JOIN candidate_duplicates cd USING(candidate_key)
            WHERE cs.source='lastfm_track_similar'
              AND cs.seed_recording_key IN ({marks})
              AND cd.candidate_key IS NULL
              AND cr.status != 'duplicate'
            """,
            chunk,
        )
        for row in rows:
            artist_key = normalize_text(row["artist"])
            if not artist_key or artist_key in familiar:
                continue
            strength = max(0.0, min(float(row["strength"] or 0.0), 1.0))
            rank = max(1, int(row["source_rank"] or 1))
            vote = strength / math.sqrt(rank)
            key = (artist_key, row["seed_recording_key"])
            current = best_by_artist_seed.get(key)
            if current is None or vote > current["vote"]:
                best_by_artist_seed[key] = {
                    "artist": row["artist"],
                    "candidate_key": row["candidate_key"],
                    "candidate_title": row["title"],
                    "seed_recording_key": row["seed_recording_key"],
                    "seed_artist": row["seed_artist"],
                    "seed_title": row["seed_title"],
                    "strength": strength,
                    "rank": rank,
                    "vote": vote,
                }

    grouped: dict[str, list[dict]] = defaultdict(list)
    for (artist_key, _seed), connection in best_by_artist_seed.items():
        grouped[artist_key].append(connection)
    eligible = {
        artist_key: connections
        for artist_key, connections in grouped.items()
        if len(connections) >= min_seed_coverage
    }
    if not eligible:
        return {
            "playlist": resolved,
            "playlist_recordings": len(member_keys),
            "novelty": novelty,
            "artists": [],
            "reason": "no artists met the distinct-seed threshold",
        }

    raw = []
    for artist_key, connections in eligible.items():
        ordered = sorted(connections, key=lambda item: -item["vote"])
        raw.append(
            {
                "artist_key": artist_key,
                "artist": ordered[0]["artist"],
                "seed_count": len(ordered),
                "coverage": len(ordered) / len(member_keys),
                "weighted_votes": sum(item["vote"] for item in ordered),
                "mean_similarity": sum(item["strength"] for item in ordered) / len(ordered),
                "connections": ordered,
            }
        )

    max_votes = max(item["weighted_votes"] for item in raw) or 1.0
    max_seeds = max(item["seed_count"] for item in raw) or 1
    for item in raw:
        item["score"] = (
            0.5 * item["weighted_votes"] / max_votes
            + 0.3 * item["seed_count"] / max_seeds
            + 0.2 * item["mean_similarity"]
        )
        item["score"] = round(item["score"], 4)
        item["coverage"] = round(item["coverage"], 4)
        item["mean_similarity"] = round(item["mean_similarity"], 4)
        item["weighted_votes"] = round(item["weighted_votes"], 4)
        item["connections"] = item["connections"][:5]
        del item["artist_key"]
    raw.sort(key=lambda item: (-item["score"], -item["seed_count"], item["artist"].lower()))

    return {
        "playlist": resolved,
        "playlist_recordings": len(member_keys),
        "novelty": novelty,
        "min_seed_coverage": min_seed_coverage,
        "artists_considered": len(grouped),
        "artists": raw[:limit],
    }
