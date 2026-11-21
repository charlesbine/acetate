"""Audit and conservatively repair legacy enrichment provenance.

Repairs only facts recoverable from stored payloads. Empty legacy Last.fm rows
remain ``unknown_legacy`` because historical aggregate logs cannot prove whether
the provider returned no data or the request failed.
"""

from __future__ import annotations

import sqlite3

from . import db


def audit(conn: sqlite3.Connection) -> dict[str, int]:
    def one(sql: str) -> int:
        return int(conn.execute(sql).fetchone()[0])

    return {
        "lastfm_tags_with_data_without_ledger": one("""
            SELECT COUNT(DISTINCT g.track_id) FROM track_genres g
            WHERE g.source='lastfm' AND NOT EXISTS (
                SELECT 1 FROM lastfm_tag_runs r WHERE r.track_id=g.track_id)
        """),
        "lastfm_none_ambiguous_without_ledger": one("""
            SELECT COUNT(DISTINCT g.track_id) FROM track_genres g
            WHERE g.source='lastfm_none' AND NOT EXISTS (
                SELECT 1 FROM lastfm_tag_runs r WHERE r.track_id=g.track_id)
        """),
        "lastfm_info_with_stats_without_ledger": one("""
            SELECT COUNT(*) FROM lastfm_info i
            WHERE (i.listeners IS NOT NULL OR i.playcount IS NOT NULL) AND NOT EXISTS (
                SELECT 1 FROM lastfm_track_runs r
                WHERE r.recording_key=i.recording_key AND r.surface='info')
        """),
        "lastfm_info_null_ambiguous_without_ledger": one("""
            SELECT COUNT(*) FROM lastfm_info i
            WHERE i.listeners IS NULL AND i.playcount IS NULL AND NOT EXISTS (
                SELECT 1 FROM lastfm_track_runs r
                WHERE r.recording_key=i.recording_key AND r.surface='info')
        """),
        "lastfm_similar_with_edges_without_ledger": one("""
            SELECT COUNT(DISTINCT s.recording_key) FROM lastfm_similar s
            WHERE NOT EXISTS (SELECT 1 FROM lastfm_track_runs r
                WHERE r.recording_key=s.recording_key AND r.surface='similar')
        """),
        "embeddings_without_ledger": one("""
            SELECT COUNT(*) FROM embeddings e JOIN recording_stats rs USING(recording_key)
            WHERE NOT EXISTS (SELECT 1 FROM embedding_runs r
                WHERE r.recording_key=e.recording_key AND r.model=e.model)
        """),
        "library_embedding_work_without_ledger": one("""
            WITH models(model) AS (
                VALUES ('discogs-effnet'), ('discogs400'), ('clap-music')
            )
            SELECT COUNT(*) FROM audio_cache a CROSS JOIN models m
            WHERE a.status='fetched'
              AND NOT EXISTS (
                  SELECT 1 FROM embeddings e
                  WHERE e.recording_key=a.recording_key AND e.model=m.model)
              AND NOT EXISTS (
                  SELECT 1 FROM embedding_runs r
                  WHERE r.recording_key=a.recording_key AND r.model=m.model)
        """),
        "candidate_embeddings_without_ledger": one("""
            SELECT COUNT(*) FROM embeddings e
            JOIN candidate_recordings c ON c.candidate_key=e.recording_key
            WHERE NOT EXISTS (
                SELECT 1 FROM candidate_embedding_runs r
                WHERE r.candidate_key=e.recording_key AND r.model=e.model)
        """),
        "candidate_embedding_work_without_ledger": one("""
            WITH models(model) AS (
                VALUES ('discogs-effnet'), ('discogs400'), ('clap-music')
            )
            SELECT COUNT(*) FROM candidate_audio a CROSS JOIN models m
            WHERE a.status='fetched'
              AND NOT EXISTS (
                  SELECT 1 FROM embeddings e
                  WHERE e.recording_key=a.candidate_key AND e.model=m.model)
              AND NOT EXISTS (
                  SELECT 1 FROM candidate_embedding_runs r
                  WHERE r.candidate_key=a.candidate_key AND r.model=m.model)
        """),
        "spotify_features_without_coverage": one("""
            SELECT COUNT(*) FROM spotify_features f
            WHERE NOT EXISTS (SELECT 1 FROM spotify_feature_coverage c
                WHERE c.recording_key=f.recording_key)
        """),
        "deezer_tracks_without_ledger": one("""
            SELECT COUNT(*) FROM deezer_track d
            WHERE d.deezer_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM deezer_metadata_runs r
                WHERE r.entity_type='track' AND r.entity_id=CAST(d.deezer_id AS TEXT))
        """),
        "deezer_albums_without_ledger": one("""
            SELECT COUNT(*) FROM deezer_album d
            WHERE NOT EXISTS (
                SELECT 1 FROM deezer_metadata_runs r
                WHERE r.entity_type='album' AND r.entity_id=CAST(d.album_id AS TEXT))
        """),
        "deezer_artists_without_ledger": one("""
            SELECT COUNT(*) FROM deezer_artist d
            WHERE NOT EXISTS (
                SELECT 1 FROM deezer_metadata_runs r
                WHERE r.entity_type='artist' AND r.entity_id=CAST(d.artist_id AS TEXT))
        """),
    }


def repair_legacy(conn: sqlite3.Connection, *, apply: bool = False) -> dict:
    before = audit(conn)
    if not apply:
        return {"mode": "dry_run", "would_repair": before}
    now = db.utcnow()
    changed: dict[str, int] = {}

    def run(name: str, sql: str, params: tuple = ()) -> None:
        cur = conn.execute(sql, params)
        changed[name] = max(0, cur.rowcount)

    run(
        "lastfm_tags_available",
        """
        INSERT OR IGNORE INTO lastfm_tag_runs(
            track_id, status, attempts, detail_json, updated_at)
        SELECT DISTINCT track_id, 'available', 1,
               '{"provenance":"recovered_from_stored_tags"}', ?
        FROM track_genres WHERE source='lastfm'
    """,
        (now,),
    )
    run(
        "lastfm_tags_unknown_legacy",
        """
        INSERT OR IGNORE INTO lastfm_tag_runs(
            track_id, status, attempts, detail_json, updated_at)
        SELECT DISTINCT track_id, 'unknown_legacy', 1,
               '{"reason":"legacy_none_conflated_empty_miss_and_error"}', assigned_at
        FROM track_genres WHERE source='lastfm_none'
    """,
    )
    run(
        "lastfm_info_available",
        """
        INSERT OR IGNORE INTO lastfm_track_runs(
            recording_key, surface, status, attempts, detail_json, updated_at)
        SELECT recording_key, 'info', 'available', 1,
               '{"provenance":"recovered_from_stored_stats"}', updated_at
        FROM lastfm_info WHERE listeners IS NOT NULL OR playcount IS NOT NULL
    """,
    )
    run(
        "lastfm_info_unknown_legacy",
        """
        INSERT OR IGNORE INTO lastfm_track_runs(
            recording_key, surface, status, attempts, detail_json, updated_at)
        SELECT recording_key, 'info', 'unknown_legacy', 1,
               '{"reason":"legacy_null_conflated_miss_and_error"}', updated_at
        FROM lastfm_info WHERE listeners IS NULL AND playcount IS NULL
    """,
    )
    run(
        "lastfm_similar_available",
        """
        INSERT OR IGNORE INTO lastfm_track_runs(
            recording_key, surface, status, attempts, detail_json, updated_at)
        SELECT recording_key, 'similar', 'available', 1,
               json_object('provenance','recovered_from_stored_edges',
                           'result_count',COUNT(*)), MAX(updated_at)
        FROM lastfm_similar GROUP BY recording_key
    """,
    )
    run(
        "embeddings_available",
        """
        INSERT OR IGNORE INTO embedding_runs(
            recording_key, model, status, attempts, updated_at)
        SELECT e.recording_key, e.model, 'available', 1, e.updated_at
        FROM embeddings e JOIN recording_stats rs USING(recording_key)
    """,
    )
    run(
        "library_embeddings_pending",
        """
        WITH models(model) AS (
            VALUES ('discogs-effnet'), ('discogs400'), ('clap-music')
        )
        INSERT OR IGNORE INTO embedding_runs(
            recording_key, model, status, attempts, updated_at)
        SELECT a.recording_key, m.model, 'pending', 0, ?
        FROM audio_cache a CROSS JOIN models m
        WHERE a.status='fetched'
          AND NOT EXISTS (
              SELECT 1 FROM embeddings e
              WHERE e.recording_key=a.recording_key AND e.model=m.model)
    """,
        (now,),
    )
    run(
        "candidate_embeddings_available",
        """
        INSERT OR IGNORE INTO candidate_embedding_runs(
            candidate_key, model, status, attempts, updated_at)
        SELECT e.recording_key, e.model, 'available', 1, e.updated_at
        FROM embeddings e
        JOIN candidate_recordings c ON c.candidate_key=e.recording_key
    """,
    )
    run(
        "candidate_embeddings_pending",
        """
        WITH models(model) AS (
            VALUES ('discogs-effnet'), ('discogs400'), ('clap-music')
        )
        INSERT OR IGNORE INTO candidate_embedding_runs(
            candidate_key, model, status, attempts, updated_at)
        SELECT a.candidate_key, m.model, 'pending', 0, ?
        FROM candidate_audio a CROSS JOIN models m
        WHERE a.status='fetched'
          AND NOT EXISTS (
              SELECT 1 FROM embeddings e
              WHERE e.recording_key=a.candidate_key AND e.model=m.model)
    """,
        (now,),
    )
    run(
        "spotify_features_available",
        """
        INSERT OR IGNORE INTO spotify_feature_coverage(
            recording_key, status, source, attempts, updated_at)
        SELECT recording_key, 'available', 'legacy_static_archive_import', 1, updated_at
        FROM spotify_features
    """,
    )
    run(
        "deezer_tracks_available",
        """
        INSERT OR IGNORE INTO deezer_metadata_runs(
            entity_type, entity_id, status, attempts, updated_at)
        SELECT 'track', CAST(deezer_id AS TEXT), 'available', 1, updated_at
        FROM deezer_track WHERE deezer_id IS NOT NULL
    """,
    )
    run(
        "deezer_albums_available",
        """
        INSERT OR IGNORE INTO deezer_metadata_runs(
            entity_type, entity_id, status, attempts, updated_at)
        SELECT 'album', CAST(album_id AS TEXT), 'available', 1, updated_at
        FROM deezer_album
    """,
    )
    run(
        "deezer_artists_available",
        """
        INSERT OR IGNORE INTO deezer_metadata_runs(
            entity_type, entity_id, status, attempts, updated_at)
        SELECT 'artist', CAST(artist_id AS TEXT), 'available', 1, updated_at
        FROM deezer_artist
    """,
    )
    conn.commit()
    return {"mode": "applied", "changed": changed, "remaining": audit(conn)}
