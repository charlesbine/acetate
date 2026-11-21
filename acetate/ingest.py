"""Turn raw Spotify objects into normalized dimension rows.

Shared by the TinyDB migration and the live snapshotter so both populate
tracks/albums/artists identically.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

Json = dict[str, Any]


def _is_real_track(track: Json | None) -> bool:
    """True for a usable track object (skip None / local / podcast episodes)."""
    return bool(track) and track.get("id") is not None and track.get("type", "track") == "track"


def upsert_album(conn: sqlite3.Connection, album: Json | None, now: str) -> str | None:
    if not album or not album.get("id"):
        return None
    conn.execute(
        """
        INSERT INTO albums(album_id, name, release_date, images_json, raw_json, updated_at)
        VALUES(:id, :name, :release_date, :images, :raw, :now)
        ON CONFLICT(album_id) DO UPDATE SET
            name=excluded.name, release_date=excluded.release_date,
            images_json=excluded.images_json, raw_json=excluded.raw_json,
            updated_at=excluded.updated_at
        """,
        {
            "id": album["id"],
            "name": album.get("name"),
            "release_date": album.get("release_date"),
            "images": json.dumps(album.get("images")) if album.get("images") else None,
            "raw": json.dumps(album),
            "now": now,
        },
    )
    return album["id"]


def upsert_artist(conn: sqlite3.Connection, artist: Json, now: str) -> str | None:
    """Upsert an artist *reference* (id/name). Does not overwrite genres, which
    are filled separately by the genre phase from the full artist endpoint."""
    if not artist.get("id"):
        return None
    conn.execute(
        """
        INSERT INTO artists(artist_id, name, uri, updated_at)
        VALUES(:id, :name, :uri, :now)
        ON CONFLICT(artist_id) DO UPDATE SET
            name=excluded.name, uri=excluded.uri, updated_at=excluded.updated_at
        """,
        {"id": artist["id"], "name": artist.get("name"), "uri": artist.get("uri"), "now": now},
    )
    return artist["id"]


def upsert_track(conn: sqlite3.Connection, track: Json | None, now: str) -> str | None:
    """Upsert a track plus its album and artists. Returns the track_id, or None
    if the object isn't a real track."""
    if not _is_real_track(track):
        return None
    assert track is not None
    album_id = upsert_album(conn, track.get("album"), now)
    isrc = (track.get("external_ids") or {}).get("isrc")
    conn.execute(
        """
        INSERT INTO tracks(track_id, name, album_id, duration_ms, popularity, isrc, uri, raw_json, updated_at)
        VALUES(:id, :name, :album_id, :duration, :pop, :isrc, :uri, :raw, :now)
        ON CONFLICT(track_id) DO UPDATE SET
            name=excluded.name, album_id=excluded.album_id, duration_ms=excluded.duration_ms,
            popularity=excluded.popularity, isrc=excluded.isrc, uri=excluded.uri,
            raw_json=excluded.raw_json, updated_at=excluded.updated_at
        """,
        {
            "id": track["id"],
            "name": track.get("name"),
            "album_id": album_id,
            "duration": track.get("duration_ms"),
            "pop": track.get("popularity"),
            "isrc": isrc,
            "uri": track.get("uri"),
            "raw": json.dumps(track),
            "now": now,
        },
    )
    for position, artist in enumerate(track.get("artists") or []):
        artist_id = upsert_artist(conn, artist, now)
        if artist_id:
            conn.execute(
                """
                INSERT INTO track_artists(track_id, artist_id, position)
                VALUES(?, ?, ?)
                ON CONFLICT(track_id, artist_id) DO UPDATE SET position=excluded.position
                """,
                (track["id"], artist_id, position),
            )
    return track["id"]
