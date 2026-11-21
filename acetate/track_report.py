"""One-call, provenance-aware dossier for a library recording.

Every enrichment surface returns an explicit state. A missing row is never
silently promoted to a provider miss, and derived audio styles stay separate
from classic Spotify/Echo Nest feature archives.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

from . import similar
from .canon import _norm

EMBED_MODELS = ("discogs-effnet", "discogs400", "clap-music")
AVAILABLE_STATES = {"available", "embedded", "fetched", "matched", "resolved"}
PROVIDER_MISS_STATES = {"miss", "provider_miss"}


def _json(value: str | None, fallback: Any = None) -> Any:
    try:
        return json.loads(value) if value else fallback
    except (TypeError, json.JSONDecodeError):
        return fallback


def _state(state: str, **values: Any) -> dict:
    return {"state": state, **{k: v for k, v in values.items() if v is not None}}


def _normalized_state(status: str) -> str:
    if status in AVAILABLE_STATES:
        return "available"
    if status in PROVIDER_MISS_STATES:
        return "provider_miss"
    if status == "unavailable":
        return "source_not_covered"
    return status


def _run_state(row: sqlite3.Row | None, *, absent: str = "not_scheduled") -> dict:
    if not row:
        return _state(absent)
    keys = set(row.keys())
    out = _state(
        _normalized_state(row["status"]),
        attempts=row["attempts"] if "attempts" in keys else None,
        updated_at=row["updated_at"] if "updated_at" in keys else None,
    )
    if "error" in keys and row["error"]:
        out["error"] = row["error"]
    detail = _json(row["detail_json"]) if "detail_json" in keys else None
    if detail:
        out["detail"] = detail
    return out


def _spotify_catalog(track: sqlite3.Row | None) -> dict:
    if not track:
        return _state("not_scheduled", reason="no_hydrated_spotify_track_row")
    payload = _json(track["raw_json"], {})
    if not payload:
        return _state("unknown_legacy", updated_at=track["updated_at"])
    if "popularity" not in payload:
        return _state(
            "source_not_covered",
            reason="spotify_track_payload_omitted_popularity",
            updated_at=track["updated_at"],
        )
    return _state(
        "available",
        popularity=payload.get("popularity"),
        provider_returned_null=payload.get("popularity") is None,
        updated_at=track["updated_at"],
    )


def _spotify_track_metadata(track: sqlite3.Row | None) -> dict:
    if not track:
        return _state("not_scheduled", reason="no_hydrated_spotify_track_row")
    return _state(
        "available",
        duration_ms=track["duration_ms"],
        album=track["album"],
        release_date=track["release_date"],
        updated_at=track["updated_at"],
    )


def _classic_features(conn: sqlite3.Connection, recording_key: str) -> dict:
    row = conn.execute(
        "SELECT * FROM spotify_features WHERE recording_key=?", (recording_key,)
    ).fetchone()
    coverage = conn.execute(
        "SELECT * FROM spotify_feature_coverage WHERE recording_key=?", (recording_key,)
    ).fetchone()
    if row:
        fields = {
            k: row[k] for k in row.keys() if k not in ("recording_key", "track_id", "updated_at")
        }
        return _state(
            "available",
            source="static_spotify_features_archive",
            source_track_id=row["track_id"],
            updated_at=row["updated_at"],
            values=fields,
        )
    if coverage:
        return _state(
            coverage["status"],
            source=coverage["source"],
            attempts=coverage["attempts"],
            error=coverage["error"],
            updated_at=coverage["updated_at"],
        )
    return _state(
        "not_scheduled",
        source="static_spotify_features_archive",
        reason="no_import_coverage_record",
    )


def _lastfm_tags(conn: sqlite3.Connection, track_ids: list[str]) -> dict:
    if not track_ids:
        return _state("not_scheduled", reason="no_spotify_track_id")
    qm = ",".join("?" * len(track_ids))
    tags = [
        dict(r)
        for r in conn.execute(
            f"SELECT track_id, genre, confidence FROM track_genres "
            f"WHERE track_id IN ({qm}) AND source='lastfm' ORDER BY confidence DESC",
            track_ids,
        )
    ]
    runs = conn.execute(
        f"SELECT * FROM lastfm_tag_runs WHERE track_id IN ({qm}) "
        "ORDER BY CASE status WHEN 'available' THEN 0 WHEN 'provider_miss' THEN 1 "
        "WHEN 'error' THEN 2 ELSE 3 END, updated_at DESC LIMIT 1",
        track_ids,
    ).fetchone()
    if tags:
        out = _state("available", tags=tags)
        if runs:
            out["ledger"] = _run_state(runs)
        else:
            out["provenance"] = "inferred_from_legacy_tag_rows"
        return out
    if runs:
        return _run_state(runs)
    sentinel = conn.execute(
        f"SELECT assigned_at FROM track_genres WHERE track_id IN ({qm}) "
        "AND source='lastfm_none' ORDER BY assigned_at DESC LIMIT 1",
        track_ids,
    ).fetchone()
    if sentinel:
        return _state(
            "unknown_legacy",
            reason="legacy_lastfm_none_conflated_empty_miss_and_error",
            updated_at=sentinel["assigned_at"],
        )
    return _state("not_scheduled")


def _lastfm_track_surface(conn: sqlite3.Connection, recording_key: str, surface: str) -> dict:
    run = conn.execute(
        "SELECT * FROM lastfm_track_runs WHERE recording_key=? AND surface=?",
        (recording_key, surface),
    ).fetchone()
    if surface == "info":
        data = conn.execute(
            "SELECT * FROM lastfm_info WHERE recording_key=?", (recording_key,)
        ).fetchone()
        if run:
            out = _run_state(run)
            if data:
                out["listeners"], out["playcount"] = data["listeners"], data["playcount"]
            return out
        if data and (data["listeners"] is not None or data["playcount"] is not None):
            return _state(
                "available",
                listeners=data["listeners"],
                playcount=data["playcount"],
                provenance="inferred_from_legacy_payload",
                updated_at=data["updated_at"],
            )
        if data:
            return _state(
                "unknown_legacy",
                reason="legacy_null_row_conflated_miss_and_request_error",
                updated_at=data["updated_at"],
            )
        return _state("not_scheduled")
    count = conn.execute(
        "SELECT COUNT(*) FROM lastfm_similar WHERE recording_key=?", (recording_key,)
    ).fetchone()[0]
    if run:
        out = _run_state(run)
        out["result_count"] = count
        return out
    if count:
        return _state("available", result_count=count, provenance="inferred_from_legacy_edges")
    return _state("not_scheduled")


def _audio(conn: sqlite3.Connection, recording_key: str, *, eligible: bool) -> dict:
    row = conn.execute(
        "SELECT * FROM audio_cache WHERE recording_key=?", (recording_key,)
    ).fetchone()
    if not row:
        return _state(
            "not_scheduled" if eligible else "not_eligible",
            reason=(
                "not_in_promoted_audio_ledger"
                if eligible
                else "audio_universe_requires_liked_or_at_least_5_plays"
            ),
        )
    status = row["status"]
    if status == "fetched":
        status = "available"
    elif status == "miss":
        status = "source_not_covered"
    return _state(
        status,
        source=row["source"],
        source_id=row["source_id"],
        path=row["path"],
        attempts=row["attempts"],
        error=row["error"],
        updated_at=row["updated_at"],
    )


def _embedding(conn: sqlite3.Connection, recording_key: str, model: str, audio: dict) -> dict:
    vector = conn.execute(
        "SELECT dim, updated_at FROM embeddings WHERE recording_key=? AND model=?",
        (recording_key, model),
    ).fetchone()
    if vector:
        return _state("available", dim=vector["dim"], updated_at=vector["updated_at"])
    run = conn.execute(
        "SELECT * FROM embedding_runs WHERE recording_key=? AND model=?", (recording_key, model)
    ).fetchone()
    if run:
        return _run_state(run)
    if audio["state"] == "available":
        return _state("pending", reason="audio_fetched_waiting_for_embedding_worker")
    return _state("blocked", blocked_by=f"audio_preview:{audio['state']}")


def _simple_provider_surface(
    row: sqlite3.Row | None,
    *,
    values: tuple[str, ...] = (),
    absent: str = "not_scheduled",
) -> dict:
    if not row:
        return _state(absent)
    keys = set(row.keys())
    out = _state(
        _normalized_state(row["status"]) if "status" in keys else "available",
        attempts=row["attempts"] if "attempts" in keys else None,
        error=row["error"] if "error" in keys else None,
        updated_at=row["updated_at"] if "updated_at" in keys else None,
    )
    for field in values:
        if field in keys and row[field] is not None:
            out[field] = row[field]
    return out


def _deezer_metadata(conn: sqlite3.Connection, recording_key: str, audio: dict) -> dict:
    track = conn.execute(
        """
        SELECT dt.*, da.title AS album_title, da.label, da.genres,
               da.release_date AS album_release_date,
               dar.name AS artist_name, dar.nb_fan, dar.related_json
        FROM deezer_track dt
        LEFT JOIN deezer_album da USING(album_id)
        LEFT JOIN deezer_artist dar USING(artist_id)
        WHERE dt.recording_key=?
        """,
        (recording_key,),
    ).fetchone()
    if not track:
        if audio["state"] != "available":
            return {"track": _state("blocked", blocked_by=f"audio_preview:{audio['state']}")}
        if not str(audio.get("source") or "").startswith("deezer"):
            return {"track": _state("not_eligible", reason="preview_source_is_not_deezer")}
        run = conn.execute(
            "SELECT * FROM deezer_metadata_runs WHERE entity_type='track' AND entity_id=?",
            (str(audio.get("source_id") or ""),),
        ).fetchone()
        return {"track": _run_state(run)}

    track_state = _state(
        "available",
        deezer_id=track["deezer_id"],
        bpm=track["bpm"],
        gain=track["gain"],
        rank=track["rank"],
        release_date=track["release_date"],
        updated_at=track["updated_at"],
    )

    def related_state(entity_type: str, entity_id: Any, fields: tuple[str, ...]) -> dict:
        if entity_id is None:
            return _state("source_not_covered", reason=f"deezer_track_omitted_{entity_type}_id")
        has_payload = any(track[field] is not None for field in fields)
        if has_payload:
            return _state(
                "available",
                **{field: track[field] for field in fields if track[field] is not None},
            )
        run = conn.execute(
            "SELECT * FROM deezer_metadata_runs WHERE entity_type=? AND entity_id=?",
            (entity_type, str(entity_id)),
        ).fetchone()
        return _run_state(run)

    return {
        "track": track_state,
        "album": related_state(
            "album",
            track["album_id"],
            ("album_title", "label", "genres", "album_release_date"),
        ),
        "artist": related_state(
            "artist",
            track["artist_id"],
            ("artist_name", "nb_fan", "related_json"),
        ),
    }


def report(conn: sqlite3.Connection, query: str) -> dict:
    match = re.search(r"(?:open\.spotify\.com/track/|spotify:track:)?([A-Za-z0-9]{22})", query)
    seed = None
    if match:
        seed = conn.execute(
            """
            SELECT rs.* FROM track_recording tr
            JOIN recording_stats rs USING(recording_key)
            WHERE tr.track_id=? LIMIT 1
        """,
            (match.group(1),),
        ).fetchone()
    seed = seed or similar.find_recording(conn, query)
    if not seed:
        return {"error": "no recording in the library matches that query"}
    recording_key = seed["recording_key"]
    track_ids = [
        r[0]
        for r in conn.execute(
            "SELECT track_id FROM track_recording WHERE recording_key=? ORDER BY track_id=? DESC",
            (recording_key, seed["rep_track_id"]),
        ).fetchall()
    ]
    track = conn.execute(
        "SELECT t.*, a.name album, a.release_date, a.images_json, a.raw_json album_raw "
        "FROM tracks t LEFT JOIN albums a USING(album_id) WHERE t.track_id=?",
        (seed["rep_track_id"],),
    ).fetchone()
    if not track and track_ids:
        track = conn.execute(
            "SELECT t.*, a.name album, a.release_date, a.images_json, a.raw_json album_raw "
            "FROM tracks t LEFT JOIN albums a USING(album_id) WHERE t.track_id=?",
            (track_ids[0],),
        ).fetchone()

    album_raw = _json(track["album_raw"], {}) if track else {}
    images = _json(track["images_json"], []) if track else []
    playlists = [
        dict(r)
        for r in conn.execute(
            """
        SELECT ps.playlist_id, ps.name, ps.persona, ps.user_label, ps.is_managed,
               ps.n_tracks, pm.added_at
        FROM playlist_members pm JOIN playlist_stats ps USING(playlist_id)
        WHERE pm.recording_key=? ORDER BY ps.is_managed, ps.name
    """,
            (recording_key,),
        ).fetchall()
    ]

    artist_rows = [
        dict(r)
        for r in conn.execute(
            """
        SELECT a.artist_id, a.name, a.genres_json, a.popularity, a.updated_at
        FROM track_artists ta JOIN artists a USING(artist_id)
        WHERE ta.track_id=? ORDER BY ta.position
    """,
            (seed["rep_track_id"],),
        ).fetchall()
    ]
    spotify_genres = []
    for artist in artist_rows:
        spotify_genres.extend(_json(artist["genres_json"], []) or [])
    if spotify_genres:
        artist_genre_state = _state("available", genres=sorted(set(spotify_genres)))
    elif artist_rows and any(a["genres_json"] is not None for a in artist_rows):
        artist_genre_state = _state("available", genres=[], reason="provider_returned_empty_genres")
    else:
        artist_genre_state = _state("not_scheduled", reason="artist_not_hydrated_with_genres")
    spotify_artist_metadata = (
        _state(
            "available",
            artists=[
                {
                    "artist_id": artist["artist_id"],
                    "name": artist["name"],
                    "popularity": artist["popularity"],
                    "updated_at": artist["updated_at"],
                }
                for artist in artist_rows
            ],
        )
        if artist_rows
        else _state("not_scheduled", reason="spotify_artists_not_hydrated")
    )

    artist_cache = conn.execute(
        "SELECT * FROM lastfm_artist_cache WHERE artist_key=?", (_norm(seed["artist"]),)
    ).fetchone()
    lastfm_artist = (
        _state(
            _normalized_state(artist_cache["status"]),
            attempts=artist_cache["attempts"],
            error=artist_cache["error"],
            tags=_json(artist_cache["tags_json"], []),
            updated_at=artist_cache["updated_at"],
        )
        if artist_cache
        else _state("not_scheduled")
    )

    audio = _audio(conn, recording_key, eligible=bool(seed["liked"] or seed["plays"] >= 5))
    embeddings = {model: _embedding(conn, recording_key, model, audio) for model in EMBED_MODELS}
    styles = [
        dict(r)
        for r in conn.execute(
            "SELECT rank, style, score FROM recording_styles WHERE recording_key=? ORDER BY rank",
            (recording_key,),
        ).fetchall()
    ]
    style_state = (
        _state("available", values=styles)
        if styles
        else _state("blocked", blocked_by=f"discogs400:{embeddings['discogs400']['state']}")
    )
    sync = conn.execute(
        "SELECT source_created_at, applied_at, bundle_name FROM enrichment_sync "
        "ORDER BY applied_at DESC LIMIT 1"
    ).fetchone()
    mbid = conn.execute(
        "SELECT * FROM recording_mbid WHERE recording_key=?", (recording_key,)
    ).fetchone()
    ytmusic_match = conn.execute(
        "SELECT * FROM ytmusic_seed_matches WHERE recording_key=?", (recording_key,)
    ).fetchone()
    ytmusic_radio = conn.execute(
        "SELECT * FROM ytmusic_radio_seeds WHERE recording_key=?", (recording_key,)
    ).fetchone()
    embedding_request = conn.execute(
        "SELECT status, attempts, error, updated_at FROM embedding_requests WHERE recording_key=?",
        (recording_key,),
    ).fetchone()

    return {
        "track": {
            "recording_key": recording_key,
            "track_id": seed["rep_track_id"],
            "artist": seed["artist"],
            "title": seed["title"],
            "duration_ms": track["duration_ms"] if track else None,
            "album": track["album"] if track else None,
            "release_date": track["release_date"] if track else None,
            "album_type": album_raw.get("album_type"),
            "album_total_tracks": album_raw.get("total_tracks"),
            "image_url": images[0].get("url") if images else None,
            "spotify_url": (
                f"https://open.spotify.com/track/{seed['rep_track_id']}"
                if seed["rep_track_id"]
                else None
            ),
        },
        "personal": {
            "plays": seed["plays"],
            "first_played_at": seed["first_at"],
            "last_played_at": seed["last_at"],
            "liked": bool(seed["liked"]),
            "liked_at": seed["liked_at"],
            "homed": bool(seed["homed"]),
            "playlists": playlists,
        },
        "surfaces": {
            "spotify_track_metadata": _spotify_track_metadata(track),
            "spotify_catalog_popularity": _spotify_catalog(track),
            "spotify_artist_metadata": spotify_artist_metadata,
            "spotify_artist_genres": artist_genre_state,
            "classic_spotify_features": _classic_features(conn, recording_key),
            "lastfm_tags": _lastfm_tags(conn, track_ids),
            "lastfm_track_info": _lastfm_track_surface(conn, recording_key, "info"),
            "lastfm_similar": _lastfm_track_surface(conn, recording_key, "similar"),
            "lastfm_artist": lastfm_artist,
            "deezer_metadata": _deezer_metadata(conn, recording_key, audio),
            "musicbrainz_recording": _simple_provider_surface(
                mbid, values=("mbid", "method", "isrc")
            ),
            "ytmusic_seed_resolution": _simple_provider_surface(
                ytmusic_match,
                values=("song_id", "source", "confidence"),
            ),
            "ytmusic_radio": _simple_provider_surface(
                ytmusic_radio,
                values=("video_id", "playlist_id", "result_count"),
            ),
            "audio_preview": audio,
            "embeddings": embeddings,
            "embedding_request": _run_state(embedding_request),
            "audio_styles": style_state,
        },
        "enrichment_snapshot": dict(sync) if sync else None,
        "evidence_rules": [
            "not_scheduled, pending, blocked, and source_not_covered are not provider_miss",
            "do not infer obscurity or genre from absent enrichment",
            "classic Spotify features and audio embeddings are independent surfaces",
        ],
    }
