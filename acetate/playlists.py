"""Generate and maintain acetate-owned playlists.

Each playlist is keyed by (kind, key) in managed_playlists so re-running updates
the same Spotify playlist in place instead of creating duplicates.

Phase 4 (monthly) lives here now; the genre builders (Phase 5) will join later.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from datetime import date, timedelta

import spotipy

from . import db

# A "real play" for ranking: GDPR streams need >=30s; live-poll events have a
# NULL ms_played (recently-played doesn't report duration) and always count.
REAL_PLAY = "(ms_played IS NULL OR ms_played >= 30000)"


def _create_playlist(
    sp: spotipy.Spotify, name: str, description: str, public: bool = False
) -> dict:
    # spotipy's user_playlist_create still posts to the removed /users/{id}/playlists,
    # so call the Feb-2026 endpoint directly.
    # Request private. NOTE: Spotify's API sets/reports playlist `public` unreliably
    # (a long-standing quirk); toggle visibility in-app if profile exposure matters.
    return sp._post(
        "me/playlists", payload={"name": name, "public": public, "description": description}
    )


def _get_or_create(
    conn: sqlite3.Connection, sp: spotipy.Spotify, kind: str, key: str, name: str, description: str
) -> str:
    row = conn.execute(
        "SELECT playlist_id, name FROM managed_playlists WHERE kind=? AND key=?", (kind, key)
    ).fetchone()
    if row:
        # Keep the Spotify playlist's name in sync if the naming scheme changed.
        if row["name"] != name:
            _assert_managed(conn, row["playlist_id"])
            sp.playlist_change_details(row["playlist_id"], name=name, description=description)
            conn.execute(
                "UPDATE managed_playlists SET name=?, updated_at=? WHERE kind=? AND key=?",
                (name, db.utcnow(), kind, key),
            )
            conn.commit()
        return row["playlist_id"]
    pl = _create_playlist(sp, name, description)
    now = db.utcnow()
    conn.execute(
        "INSERT INTO managed_playlists(kind, key, playlist_id, name, created_at, updated_at) "
        "VALUES(?,?,?,?,?,?)",
        (kind, key, pl["id"], name, now, now),
    )
    conn.commit()
    return pl["id"]


def _assert_managed(conn: sqlite3.Connection, playlist_id: str) -> None:
    """HARD SAFETY GATE — the single chokepoint every playlist-content write passes through.
    acetate only ever modifies playlists it created itself (tracked in managed_playlists).
    Touching any other playlist — i.e. one of YOUR own — raises and writes nothing, so a bug
    upstream can never wipe or alter a playlist acetate didn't make. There is also no
    unfollow/delete code anywhere in acetate, so it cannot remove a Spotify playlist at all."""
    if not playlist_id:
        raise RuntimeError("refusing to modify a playlist with no id")
    if not conn.execute(
        "SELECT 1 FROM managed_playlists WHERE playlist_id=?", (playlist_id,)
    ).fetchone():
        raise RuntimeError(
            f"REFUSING to modify playlist {playlist_id!r}: it is not acetate-managed. "
            "acetate never touches playlists it didn't create."
        )


def _set_tracks(
    conn: sqlite3.Connection, sp: spotipy.Spotify, playlist_id: str, track_ids: list[str]
) -> None:
    """Make the playlist's contents exactly `track_ids` (Spotify caps writes at 100/call)."""
    _assert_managed(conn, playlist_id)  # never overwrite a playlist acetate didn't create
    uris = [f"spotify:track:{tid}" for tid in track_ids]
    sp.playlist_replace_items(playlist_id, uris[:100])
    for i in range(100, len(uris), 100):
        sp.playlist_add_items(playlist_id, uris[i : i + 100])
    conn.execute(
        "UPDATE managed_playlists SET updated_at=? WHERE playlist_id=?", (db.utcnow(), playlist_id)
    )
    conn.commit()


def _existing(conn: sqlite3.Connection, kind: str, key: str) -> str | None:
    row = conn.execute(
        "SELECT playlist_id FROM managed_playlists WHERE kind=? AND key=?", (kind, key)
    ).fetchone()
    return row["playlist_id"] if row else None


def build_liked_month(conn: sqlite3.Connection, sp: spotipy.Spotify, year_month: str) -> dict:
    """Playlist of tracks liked during year_month (YYYY-MM), from the latest backup."""
    bt = conn.execute("SELECT MAX(backup_time) FROM saved_track_snapshots").fetchone()[0]
    if not bt:
        raise RuntimeError("No saved-track snapshot yet — run `acetate backup` first.")
    track_ids = [
        r["track_id"]
        for r in conn.execute(
            "SELECT track_id FROM saved_track_snapshots "
            "WHERE backup_time=? AND track_id IS NOT NULL AND substr(added_at,1,7)=? ORDER BY added_at",
            (bt, year_month),
        )
    ]
    name = f"📆 Liked · {year_month}"
    if not track_ids and not _existing(conn, "liked_month", year_month):
        return {"name": name, "tracks": 0, "skipped": True}
    pid = _get_or_create(
        conn,
        sp,
        "liked_month",
        year_month,
        name,
        f"Songs you liked in {year_month}. Auto-generated by acetate.",
    )
    _set_tracks(conn, sp, pid, track_ids)
    return {"name": name, "tracks": len(track_ids), "playlist_id": pid}


def liked_years(conn: sqlite3.Connection) -> list[str]:
    """Years that have liked songs (from the latest saved-tracks snapshot), newest first."""
    bt = conn.execute("SELECT MAX(backup_time) FROM saved_track_snapshots").fetchone()[0]
    if not bt:
        return []
    return [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT substr(added_at,1,4) y FROM saved_track_snapshots "
            "WHERE backup_time=? AND added_at IS NOT NULL ORDER BY y DESC",
            (bt,),
        )
    ]


def build_liked_year(conn: sqlite3.Connection, sp: spotipy.Spotify, year: str | int) -> dict:
    """Playlist of every track liked during a calendar year, oldest-liked first."""
    year = str(year)
    bt = conn.execute("SELECT MAX(backup_time) FROM saved_track_snapshots").fetchone()[0]
    if not bt:
        raise RuntimeError("No saved-track snapshot yet — run `acetate pull library` first.")
    track_ids = [
        r["track_id"]
        for r in conn.execute(
            "SELECT track_id FROM saved_track_snapshots "
            "WHERE backup_time=? AND track_id IS NOT NULL AND substr(added_at,1,4)=? ORDER BY added_at",
            (bt, year),
        )
    ]
    name = f"📆 Liked {year}"
    if not track_ids and not _existing(conn, "liked_year", year):
        return {"name": name, "tracks": 0, "skipped": True}
    pid = _get_or_create(
        conn,
        sp,
        "liked_year",
        year,
        name,
        f"Every song you liked in {year}. Auto-generated by acetate.",
    )
    _set_tracks(conn, sp, pid, track_ids)
    return {"name": name, "tracks": len(track_ids), "playlist_id": pid}


def build_top_played_month(
    conn: sqlite3.Connection, sp: spotipy.Spotify, year_month: str, limit: int = 50
) -> dict:
    """Playlist of the most-played tracks during year_month (YYYY-MM)."""
    track_ids = [
        r["rid"]
        for r in conn.execute(
            f"SELECT tr.rep_track_id rid, COUNT(*) plays FROM play_events p "
            f"JOIN track_recording tr ON tr.track_id=p.track_id "
            f"WHERE p.track_id IS NOT NULL AND {REAL_PLAY} AND substr(p.played_at,1,7)=? "
            f"GROUP BY tr.recording_key ORDER BY plays DESC LIMIT ?",
            (year_month, limit),
        )
    ]
    name = f"🔥 Top Played · {year_month}"
    if not track_ids and not _existing(conn, "top_played_month", year_month):
        return {"name": name, "tracks": 0, "skipped": True}
    pid = _get_or_create(
        conn,
        sp,
        "top_played_month",
        year_month,
        name,
        f"Your most-played tracks in {year_month}. Auto-generated by acetate.",
    )
    _set_tracks(conn, sp, pid, track_ids)
    return {"name": name, "tracks": len(track_ids), "playlist_id": pid}


def build_lost_gems(
    conn: sqlite3.Connection, sp: spotipy.Spotify, size: int = 50, faded_days: int = 183
) -> dict:
    """Rotating playlist of faded orphaned gems (liked + played + never playlisted +
    silent). Each run surfaces the `size` least-recently-shown candidates, so the pool
    cycles over time. Rotation state lives in managed_playlist_tracks."""
    from . import stats

    kind, key, now = "lost_gems", "default", db.utcnow()
    pool = [
        g["track_id"] for g in stats.lost_gems(conn, faded_days=faded_days, limit=None)
    ]  # plays-desc
    name = "💎 Lost Gems"
    if not pool and not _existing(conn, kind, key):
        return {"name": name, "tracks": 0, "skipped": True}
    # rotation: never-surfaced first, then oldest last_surfaced; stable sort keeps plays-desc tiebreak
    seen = {
        r["track_id"]: r["last_surfaced_at"]
        for r in conn.execute(
            "SELECT track_id, last_surfaced_at FROM managed_playlist_tracks WHERE kind=? AND key=?",
            (kind, key),
        )
    }
    pool.sort(key=lambda t: (seen.get(t) is not None, seen.get(t) or ""))
    chosen = pool[:size]
    pid = _get_or_create(
        conn,
        sp,
        kind,
        key,
        name,
        "Liked favorites you've not played in a while — a fresh batch each refresh. Auto-generated by acetate.",
    )
    _set_tracks(conn, sp, pid, chosen)
    conn.execute(
        "UPDATE managed_playlist_tracks SET currently_in=0 WHERE kind=? AND key=?", (kind, key)
    )
    for tid in chosen:
        conn.execute(
            "INSERT INTO managed_playlist_tracks(kind,key,track_id,first_added_at,last_surfaced_at,surface_count,currently_in) "
            "VALUES(?,?,?,?,?,1,1) ON CONFLICT(kind,key,track_id) DO UPDATE SET "
            "last_surfaced_at=excluded.last_surfaced_at, surface_count=surface_count+1, currently_in=1",
            (kind, key, tid, now, now),
        )
    conn.commit()
    return {"name": name, "tracks": len(chosen), "playlist_id": pid, "pool": len(pool)}


def build_on_loop(
    conn: sqlite3.Connection, sp: spotipy.Spotify, days: int = 21, size: int = 40
) -> dict:
    """Transient 'on repeat right now' playlist: the most-played recordings of the last
    `days`, recency-weighted (a play today outweighs one three weeks ago). Fully replaced
    each run — a snapshot of your current rotation, not a rotation/accretion like Lost Gems."""
    start = (date.today() - timedelta(days=days)).isoformat()
    track_ids = [
        r["rid"]
        for r in conn.execute(
            f"SELECT tr.rep_track_id rid, "
            f"       SUM(julianday(substr(p.played_at,1,10)) - julianday(?) + 1) score "
            f"FROM play_events p JOIN track_recording tr ON tr.track_id=p.track_id "
            f"WHERE p.played_at >= ? AND p.track_id IS NOT NULL AND {REAL_PLAY} "
            f"GROUP BY tr.recording_key ORDER BY score DESC, COUNT(*) DESC LIMIT ?",
            (start, start, size),
        )
    ]
    name = "🔁 On Loop"
    if not track_ids and not _existing(conn, "on_loop", "default"):
        return {"name": name, "tracks": 0, "skipped": True}
    pid = _get_or_create(
        conn,
        sp,
        "on_loop",
        "default",
        name,
        f"What you've had on repeat the last {days} days — recency-weighted. "
        "Auto-generated by acetate; refreshed daily.",
    )
    _set_tracks(conn, sp, pid, track_ids)
    return {"name": name, "tracks": len(track_ids), "playlist_id": pid, "window_days": days}


def build_smart(conn: sqlite3.Connection, sp: spotipy.Spotify, row: dict) -> dict:
    """Serialize first publication across the separate web and scheduler processes."""
    from . import smart

    spec = row["spec"] if isinstance(row["spec"], dict) else json.loads(row["spec"])
    if not smart.is_curated(spec):
        return _build_smart(conn, sp, row)
    # The database is shared by the web and scheduler containers. A database-adjacent
    # OS lock survives helper commits, releases on process exit, and leaves failed
    # publications' already-created Spotify IDs available for the next retry.
    import fcntl

    filename = next(r["file"] for r in conn.execute("PRAGMA database_list") if r["name"] == "main")
    if not filename:  # An in-memory archive cannot be shared between processes.
        return _build_smart(conn, sp, row)
    lock_path = Path(filename).with_name(Path(filename).name + ".curated-publish.lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _build_smart(conn, sp, row)


def _build_smart(conn: sqlite3.Connection, sp: spotipy.Spotify, row: dict) -> dict:
    """Sync a rule, or publish a fixed selection once, through the same managed writer."""
    from . import smart

    spec = row["spec"] if isinstance(row["spec"], dict) else json.loads(row["spec"])
    track_ids = smart.evaluate(conn, spec)
    name, key = row["name"], str(row["id"])
    if smart.is_curated(spec):
        current = smart.get(conn, row["id"])
        if current and current["published_at"]:
            return {
                "name": name,
                "tracks": len(track_ids),
                "playlist_id": _existing(conn, "smart", key),
                "skipped": True,
            }
    if not track_ids and not _existing(conn, "smart", key):
        return {"name": name, "tracks": 0, "skipped": True}
    description = (
        f"{smart.describe(spec)}. Fixed selection; no automatic refreshes."
        if smart.is_curated(spec)
        else f"{smart.describe(spec)}. Smart Playlist · auto-generated by acetate."
    )
    pid = _get_or_create(conn, sp, "smart", key, name, description)
    _set_tracks(conn, sp, pid, track_ids)
    conn.execute("UPDATE smart_playlists SET published_at=? WHERE id=?", (db.utcnow(), row["id"]))
    conn.commit()
    return {"name": name, "tracks": len(track_ids), "playlist_id": pid}
