"""Live Spotify capture: poll recently-played into play_events, and snapshot
playlists + saved tracks into the append-only backup tables.

Ported from the legacy acetate.py but writes to SQLite via ingest, hydrating the
track/artist/album dimensions as a side effect.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Iterator

import spotipy

from . import db
from .ingest import upsert_track

Json = dict[str, Any]


# --- pagination helpers -----------------------------------------------------


def _paginate(fetch, limit: int) -> Iterator[Json]:
    """Yield items from a Spotify paging endpoint. `fetch(limit, offset)` returns
    a paging object with 'items' and 'next'."""
    offset = 0
    while True:
        page = fetch(limit, offset)
        yield from page["items"]
        if page.get("next") is None:
            break
        offset += limit


def _track_of(item: Json) -> Json | None:
    """The track object inside a list item. Playlist items nest it under 'item'
    since the Feb-2026 API rename; saved-tracks/recently-played/legacy use 'track'."""
    return item.get("item") or item.get("track")


def _item_track_id(item: Json) -> str | None:
    """The track_id for a list item, or None for local/podcast/removed entries."""
    tr = _track_of(item)
    if tr and tr.get("id") and tr.get("type", "track") == "track":
        return tr["id"]
    return None


def _content_hash(track_ids: list[str | None]) -> str:
    """Order-sensitive hash of a tracklist — the change-detection signature."""
    h = hashlib.sha1()
    for tid in track_ids:
        h.update(((tid or "\x00") + "\n").encode())
    return h.hexdigest()


# --- recently played (play history) ----------------------------------------


def poll_recently_played(conn: sqlite3.Connection, sp: spotipy.Spotify) -> dict[str, int]:
    """Fetch the last <=50 plays and store new ones. Full track objects come back
    here, so we hydrate the track dimension too. Idempotent via the unique
    (played_at, track_id, source) constraint."""
    now = db.utcnow()
    resp = sp.current_user_recently_played(limit=50)
    items = resp.get("items", [])
    inserted = 0
    for item in items:
        track = _track_of(item)
        track_id = upsert_track(conn, track, now)
        ctx = item.get("context") or {}
        cur = conn.execute(
            "INSERT OR IGNORE INTO play_events(played_at, track_id, ms_played, context_type, context_uri, source) "
            "VALUES(?,?,?,?,?, 'recently_played')",
            (item.get("played_at"), track_id, None, ctx.get("type"), ctx.get("uri")),
        )
        inserted += cur.rowcount
    conn.commit()
    return {"fetched": len(items), "inserted": inserted}


# --- backups (deletion trace) ----------------------------------------------


def backup_saved_tracks(conn: sqlite3.Connection, sp: spotipy.Spotify, backup_time: str) -> dict:
    """Snapshot liked songs — but only if the set changed since last backup
    (compared by content hash in kv_state). Captures likes *and* un-likes."""
    now = db.utcnow()
    items = list(
        _paginate(lambda lim, off: sp.current_user_saved_tracks(limit=lim, offset=off), 50)
    )
    chash = _content_hash([_item_track_id(it) for it in items])
    prev = conn.execute("SELECT value FROM kv_state WHERE key='saved_tracks_hash'").fetchone()
    if prev and prev[0] == chash:
        return {"saved_tracks": len(items), "saved_changed": False}
    for item in items:
        track_id = upsert_track(conn, _track_of(item), now)
        conn.execute(
            "INSERT OR IGNORE INTO saved_track_snapshots(backup_time, track_id, added_at, raw_json) "
            "VALUES(?,?,?,?)",
            (
                backup_time,
                track_id,
                item.get("added_at"),
                json.dumps({k: v for k, v in item.items() if k not in ("track", "item")}),
            ),
        )
    conn.execute(
        "INSERT INTO kv_state(key, value, updated_at) VALUES('saved_tracks_hash', ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (chash, now),
    )
    conn.commit()
    return {"saved_tracks": len(items), "saved_changed": True}


def _all_playlist_items(sp: spotipy.Spotify, playlist_id: str) -> list[Json]:
    return list(
        _paginate(lambda lim, off: sp.playlist_items(playlist_id, limit=lim, offset=off), 100)
    )


def backup_playlists(
    conn: sqlite3.Connection, sp: spotipy.Spotify, backup_time: str, only_mine: bool = False
) -> dict[str, int]:
    """Snapshot playlists, storing full membership only for playlists whose content
    changed since their last snapshot (content-hash diff — we don't trust Spotify's
    snapshot_id). Every playlist gets a tiny seen-log row each backup so deletions
    and change-times are precise. Commits per playlist to keep write locks short."""
    now = db.utcnow()
    me_id = sp.me()["id"]
    n_pl = n_changed = n_tracks = 0
    for pl in _paginate(lambda lim, off: sp.current_user_playlists(limit=lim, offset=off), 50):
        if only_mine and (pl.get("owner") or {}).get("id") != me_id:
            continue
        n_pl += 1
        items = _all_playlist_items(sp, pl["id"])
        chash = _content_hash([_item_track_id(it) for it in items])

        conn.execute(
            "INSERT OR REPLACE INTO playlist_seen(backup_time, playlist_id, content_hash) "
            "VALUES(?,?,?)",
            (backup_time, pl["id"], chash),
        )
        prev = conn.execute(
            "SELECT content_hash FROM playlist_snapshots WHERE playlist_id=? "
            "ORDER BY backup_time DESC LIMIT 1",
            (pl["id"],),
        ).fetchone()
        if prev and prev[0] == chash:
            conn.commit()  # unchanged — keep only the heartbeat
            continue

        n_changed += 1
        conn.execute(
            """INSERT OR IGNORE INTO playlist_snapshots(
                 backup_time, playlist_id, name, description, owner_id, owner_name,
                 public, collaborative, spotify_snapshot_id, total_tracks, content_hash, raw_json)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                backup_time,
                pl["id"],
                pl.get("name"),
                pl.get("description"),
                (pl.get("owner") or {}).get("id"),
                (pl.get("owner") or {}).get("display_name"),
                int(bool(pl.get("public"))) if pl.get("public") is not None else None,
                int(bool(pl.get("collaborative"))),
                pl.get("snapshot_id"),
                ((pl.get("tracks") or pl.get("items") or {}).get("total")),
                chash,
                json.dumps({k: v for k, v in pl.items() if k not in ("tracks", "items")}),
            ),
        )
        for position, item in enumerate(items):
            track_id = upsert_track(conn, _track_of(item), now)
            conn.execute(
                """INSERT INTO playlist_track_snapshots(
                     backup_time, playlist_id, position, track_id, added_at, added_by, raw_json)
                   VALUES(?,?,?,?,?,?,?)""",
                (
                    backup_time,
                    pl["id"],
                    position,
                    track_id,
                    item.get("added_at"),
                    (item.get("added_by") or {}).get("id"),
                    json.dumps({k: v for k, v in item.items() if k not in ("track", "item")}),
                ),
            )
            n_tracks += 1
        conn.commit()
    return {"playlists": n_pl, "changed": n_changed, "tracks": n_tracks}


def hydrate_artists(
    conn: sqlite3.Connection, sp: spotipy.Spotify, limit: int | None = None
) -> dict:
    """Fetch artist objects (images + genres + popularity) one at a time — Spotify's
    Feb-2026 changes 403 the bulk endpoint but the single GET still works. Most-played
    artists first; resumable (skips already-hydrated)."""
    ids = [
        r[0]
        for r in conn.execute("""
        WITH ap AS (SELECT ta.artist_id, COUNT(*) c FROM track_artists ta
                    JOIN play_events p ON p.track_id=ta.track_id GROUP BY ta.artist_id)
        SELECT a.artist_id FROM artists a LEFT JOIN ap ON ap.artist_id=a.artist_id
        WHERE a.raw_json IS NULL ORDER BY COALESCE(ap.c,0) DESC""")
    ]
    if limit:
        ids = ids[:limit]
    now = db.utcnow()
    hydrated = 0
    for aid in ids:
        try:
            a = sp.artist(aid)
        except Exception:
            continue
        if not a or not a.get("id"):
            continue
        conn.execute(
            "UPDATE artists SET name=COALESCE(?, name), genres_json=?, popularity=?, "
            "raw_json=?, updated_at=? WHERE artist_id=?",
            (
                a.get("name"),
                json.dumps(a.get("genres") or []),
                a.get("popularity"),
                json.dumps(a),
                now,
                a["id"],
            ),
        )
        hydrated += 1
        if hydrated % 50 == 0:
            conn.commit()
    conn.commit()
    return {"candidates": len(ids), "hydrated": hydrated}


def run_backup(conn: sqlite3.Connection, sp: spotipy.Spotify, only_mine: bool = False) -> dict:
    backup_time = db.utcnow()
    saved = backup_saved_tracks(conn, sp, backup_time)
    playlists = backup_playlists(conn, sp, backup_time, only_mine=only_mine)
    return {"backup_time": backup_time, **saved, **playlists}
