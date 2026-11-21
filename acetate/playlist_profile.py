"""Playlist profiles — what each playlist IS to the user.

271 playlists are not 271 taste statements. Some are living rotations, some are
archival dumps (old Liked-songs exports, Shazam mirrors), some were made once
for one night and never touched again. This module flattens the latest snapshot
of every playlist into playlist_members, then derives one playlist_stats row
each: size, liked overlap, add-cadence (one sitting vs curated over years),
member-play affinity, direct played-from-this-playlist counts (Spotify context
from the poller — only exists for polled plays), and a persona:

  acetate   made by acetate (managed_playlists)
  archive   big dump added in a burst — a backup of taste, not a use of it
  one-shot  small burst playlist never used since (the "sets from that show")
  rotation  alive: recent adds, recent member listening, or played-from recently
  dormant   curated over time, but not touched lately
  deleted   gone from the latest backup (survives via the snapshot trace)

playlist_labels holds the user's own verdicts ('fave', 'archive', ...) which
always beat the heuristics. The chat agent reads all of this via plain SQL.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping

from . import db

REAL_PLAY = "(ms_played IS NULL OR ms_played >= 30000)"
RECENT_DAYS = 180
BURST_ADD_DAYS = 3  # members added on ≤ this many distinct days = one sitting
ARCHIVE_MIN_TRACKS = 200  # a burst this large is a dump, not a party playlist
ROTATION_PLAYS_PER_TRACK = 0.5  # recent member plays / size ≥ this = still alive

KNOWN_LABELS = ("fave", "rotation", "archive", "one-shot", "dormant", "mute")

_ARCHIVE_TITLE = re.compile(
    r"liked songs?|^songs? (?:as of|\d)|liked since|old shazam|my shazam|"
    r"soundhou?nd|\barchive\b|my top 20\d\d|rewind|playlist in a bottle|"
    r"my songs 20\d\d|liked from radio|disc weekly agg|jqbx.*starred",
    re.IGNORECASE,
)
_EVENT_TITLE = re.compile(
    r"^for\s|^pour\s|\bmiami music week\b|\bmmw\b|\bultra\b|spin class|"
    r"\blive at\b|\basot\b|\bconcert\b|\bset\b|\brome\b|\biceland\b|"
    r"drive to |christmas|^c[-&]s\b|^dw\s?20|^fkr:|\bwarehouse\b|"
    r"^20\d\d-\d\d(?:-\d\d)?\b",
    re.IGNORECASE,
)
_TASTE_TITLE = re.compile(
    r"techno|house|trance|ambient|lofi|chill|workout|^sport$|synthwave|"
    r"outrun|rock|rap|hip-hop|rnb|classical|salsa|tango|\bedm\b|dance|"
    r"\bpop\b|acoustic|energy|piano|lounge|club|tropical|electro|arab|"
    r"melanchol|beat posay|^posay$|smoky vocalist|vapor soul|drop!",
    re.IGNORECASE,
)
_JUNK_TITLE = re.compile(
    r"^new playlist(?: \d+)?$|^my playlist #\d+$|^untitled playlist$|^~+$|"
    r"^lol$|i don't listen to this|^useless$|shitty club music|jqbx.*queue",
    re.IGNORECASE,
)


def suggest_label(stats: Mapping[str, object]) -> dict:
    """Conservative title + usage recommendation for the web review flow.

    This is intentionally a suggestion, not an automatic truth. High confidence
    means the title states its intent (archive/event) or the playlist is empty;
    usage-only classifications remain medium/low so the user's label wins.
    """
    name = str(stats.get("name") or "").strip()
    n_tracks = int(stats.get("n_tracks") or 0)
    recent = int(stats.get("member_plays_180d") or 0)
    add_days = int(stats.get("add_days") or 0)
    persona = str(stats.get("persona") or "other")
    cohesion = stats.get("cohesion")

    if n_tracks == 0:
        return {"label": "mute", "confidence": "high", "reasons": ["empty in the latest backup"]}
    if _JUNK_TITLE.search(name) and recent < 5:
        return {
            "label": "mute",
            "confidence": "high",
            "reasons": ["temporary or disposable title", f"{recent} recent member plays"],
        }
    if _ARCHIVE_TITLE.search(name):
        reasons = ["title reads as a snapshot or ingestion archive", f"{n_tracks} tracks"]
        if add_days <= 3:
            reasons.append(f"loaded in {add_days} day{'s' if add_days != 1 else ''}")
        return {"label": "archive", "confidence": "high", "reasons": reasons}
    slash_date = "/" in name and any(ch.isdigit() for ch in name)
    if slash_date or _EVENT_TITLE.search(name):
        return {
            "label": "one-shot",
            "confidence": "high",
            "reasons": ["title reads as a person, event, trip, date, or set", f"{n_tracks} tracks"],
        }
    if persona == "archive":
        return {
            "label": "archive",
            "confidence": "medium",
            "reasons": [
                "large burst-loaded collection",
                f"{n_tracks} tracks in {add_days} add days",
            ],
        }
    if persona == "rotation" and _TASTE_TITLE.search(name):
        reasons = ["stable genre, mood, or use-case title", f"{recent} member plays in 180d"]
        if cohesion is not None:
            reasons.append(f"sonic cohesion {float(cohesion):.3f}")
        return {"label": "rotation", "confidence": "high", "reasons": reasons}
    if persona == "dormant" and _TASTE_TITLE.search(name):
        reasons = ["meaningful taste title, but not actively maintained"]
        if cohesion is not None:
            reasons.append(f"sonic cohesion {float(cohesion):.3f}")
        return {"label": "dormant", "confidence": "medium", "reasons": reasons}
    if persona == "one-shot":
        return {
            "label": "one-shot",
            "confidence": "medium",
            "reasons": ["assembled in one short burst", f"{recent} recent member plays"],
        }
    return {
        "label": None,
        "confidence": "low",
        "reasons": ["usage does not reveal whether this is a destination or history"],
    }


def _user_owner_id(conn: sqlite3.Connection) -> str | None:
    from . import taste

    return taste.user_owner_id(conn)


def rebuild(conn: sqlite3.Connection) -> dict:
    """Regenerate playlist_members + playlist_stats from the snapshot tables."""
    now = db.utcnow()
    owner = _user_owner_id(conn)

    # --- members: latest track snapshot per playlist, recording-mapped ---
    conn.execute("DELETE FROM playlist_members")
    conn.execute("""
        INSERT OR IGNORE INTO playlist_members(playlist_id, track_id, recording_key,
                                               position, added_at)
        SELECT pts.playlist_id, pts.track_id, tr.recording_key, pts.position, pts.added_at
        FROM playlist_track_snapshots pts
        JOIN (SELECT playlist_id, MAX(backup_time) bt FROM playlist_track_snapshots
              GROUP BY playlist_id) latest
          ON latest.playlist_id = pts.playlist_id AND latest.bt = pts.backup_time
        LEFT JOIN track_recording tr ON tr.track_id = pts.track_id
        WHERE pts.track_id IS NOT NULL""")

    # --- one stats row per snapshotted playlist ---
    latest_seen = conn.execute("SELECT MAX(backup_time) FROM playlist_seen").fetchone()[0]
    managed = {r[0] for r in conn.execute("SELECT playlist_id FROM managed_playlists")}
    labels = {
        r["playlist_id"]: r["label"]
        for r in conn.execute("SELECT playlist_id, label FROM playlist_labels")
    }
    recent_cut = conn.execute(f"SELECT date('now', '-{RECENT_DAYS} day')").fetchone()[0]

    conn.execute("DELETE FROM playlist_stats")
    rows = conn.execute("""
        SELECT ps.playlist_id, ps.name, ps.owner_id
        FROM playlist_snapshots ps
        JOIN (SELECT playlist_id, MAX(backup_time) bt FROM playlist_snapshots
              GROUP BY playlist_id) latest
          ON latest.playlist_id = ps.playlist_id AND latest.bt = ps.backup_time""").fetchall()
    n = 0
    for pl in rows:
        pid = pl["playlist_id"]
        m = conn.execute(
            """
            SELECT COUNT(*) n_tracks,
                   SUM(COALESCE(rs.liked, 0)) n_liked,
                   MIN(pm.added_at) first_add, MAX(pm.added_at) last_add,
                   COUNT(DISTINCT substr(pm.added_at, 1, 10)) add_days,
                   SUM(COALESCE(rs.plays, 0)) member_plays
            FROM playlist_members pm
            LEFT JOIN recording_stats rs ON rs.recording_key = pm.recording_key
            WHERE pm.playlist_id = ?""",
            (pid,),
        ).fetchone()
        recent = conn.execute(
            f"""
            SELECT COUNT(*) FROM play_events p
            JOIN track_recording tr ON tr.track_id = p.track_id
            WHERE {REAL_PLAY} AND p.played_at >= ?
              AND tr.recording_key IN
                  (SELECT recording_key FROM playlist_members WHERE playlist_id = ?
                   AND recording_key IS NOT NULL)""",
            (recent_cut, pid),
        ).fetchone()[0]
        ctx = conn.execute(
            f"""
            SELECT COUNT(*) n, MAX(played_at) last FROM play_events
            WHERE context_type = 'playlist' AND context_uri = ? AND {REAL_PLAY}""",
            (f"spotify:playlist:{pid}",),
        ).fetchone()
        seen = conn.execute(
            "SELECT MAX(backup_time) FROM playlist_seen WHERE playlist_id = ?", (pid,)
        ).fetchone()[0]
        is_deleted = int(latest_seen is not None and seen is not None and seen < latest_seen)
        stats = {
            "playlist_id": pid,
            "name": pl["name"],
            "owner_id": pl["owner_id"],
            "is_own": int(owner is not None and pl["owner_id"] == owner),
            "is_managed": int(pid in managed),
            "is_deleted": is_deleted,
            "n_tracks": m["n_tracks"] or 0,
            "n_liked": m["n_liked"] or 0,
            "first_added_at": m["first_add"],
            "last_added_at": m["last_add"],
            "add_days": m["add_days"] or 0,
            "member_plays": m["member_plays"] or 0,
            "member_plays_180d": recent,
            "plays_from": ctx["n"] or 0,
            "last_played_from": ctx["last"],
            "user_label": labels.get(pid),
        }
        stats["persona"] = classify(stats, recent_cut)
        conn.execute(
            """
            INSERT INTO playlist_stats(playlist_id, name, owner_id, is_own, is_managed,
                is_deleted, n_tracks, n_liked, first_added_at, last_added_at, add_days,
                member_plays, member_plays_180d, plays_from, last_played_from, persona,
                user_label, built_at)
            VALUES(:playlist_id, :name, :owner_id, :is_own, :is_managed, :is_deleted,
                   :n_tracks, :n_liked, :first_added_at, :last_added_at, :add_days,
                   :member_plays, :member_plays_180d, :plays_from, :last_played_from,
                   :persona, :user_label, :built_at)""",
            {**stats, "built_at": now},
        )
        n += 1
    conn.commit()
    members = conn.execute("SELECT COUNT(*) FROM playlist_members").fetchone()[0]
    return {"playlists": n, "members": members}


def classify(s: dict, recent_cut: str) -> str:
    """Persona heuristic. The user's label (mapped onto a persona name when it
    is one) always wins; heuristics fill the rest."""
    if s.get("user_label") in ("fave", "rotation"):
        return "rotation"
    if s.get("user_label") in ("archive", "one-shot", "dormant"):
        return s["user_label"]
    if s["is_managed"]:
        return "acetate"
    if s["is_deleted"]:
        return "deleted"
    n = s["n_tracks"] or 0
    if n == 0:
        return "other"
    burst = (s["add_days"] or 0) <= BURST_ADD_DAYS
    recently_added = bool(s["last_added_at"] and s["last_added_at"] >= recent_cut)
    recently_played_from = bool(s["last_played_from"] and s["last_played_from"] >= recent_cut)
    alive = (
        recently_added
        or recently_played_from
        or (s["member_plays_180d"] or 0) >= ROTATION_PLAYS_PER_TRACK * n
    )
    if alive:
        return "rotation"
    if burst and n >= ARCHIVE_MIN_TRACKS:
        return "archive"
    if burst:
        return "one-shot"
    return "dormant"


def set_label(
    conn: sqlite3.Connection, playlist_id: str, label: str, note: str | None = None
) -> None:
    """Record the user's own verdict on a playlist; rebuild folds it in."""
    conn.execute(
        "INSERT INTO playlist_labels(playlist_id, label, note, updated_at) VALUES(?,?,?,?) "
        "ON CONFLICT(playlist_id) DO UPDATE SET label=excluded.label, note=excluded.note, "
        "updated_at=excluded.updated_at",
        (playlist_id, label, note, db.utcnow()),
    )
    conn.execute("UPDATE playlist_stats SET user_label=? WHERE playlist_id=?", (label, playlist_id))
    row = conn.execute(
        "SELECT * FROM playlist_stats WHERE playlist_id=?", (playlist_id,)
    ).fetchone()
    if row:
        recent_cut = conn.execute(f"SELECT date('now', '-{RECENT_DAYS} day')").fetchone()[0]
        conn.execute(
            "UPDATE playlist_stats SET persona=? WHERE playlist_id=?",
            (classify(dict(row) | {"user_label": label}, recent_cut), playlist_id),
        )
    conn.commit()


def clear_label(conn: sqlite3.Connection, playlist_id: str) -> None:
    """Remove a manual verdict and return the playlist to heuristic profiling."""
    conn.execute("DELETE FROM playlist_labels WHERE playlist_id=?", (playlist_id,))
    conn.execute("UPDATE playlist_stats SET user_label=NULL WHERE playlist_id=?", (playlist_id,))
    row = conn.execute(
        "SELECT * FROM playlist_stats WHERE playlist_id=?", (playlist_id,)
    ).fetchone()
    if row:
        recent_cut = conn.execute(f"SELECT date('now', '-{RECENT_DAYS} day')").fetchone()[0]
        conn.execute(
            "UPDATE playlist_stats SET persona=? WHERE playlist_id=?",
            (classify(dict(row) | {"user_label": None}, recent_cut), playlist_id),
        )
    conn.commit()


def overview(
    conn: sqlite3.Connection, persona: str | None = None, own_only: bool = True, limit: int = 300
) -> list[dict]:
    where, params = ["1=1"], []
    if persona:
        where.append("persona=?")
        params.append(persona)
    if own_only:
        where.append("is_own=1")
    return [
        dict(r)
        for r in conn.execute(
            f"SELECT * FROM playlist_stats WHERE {' AND '.join(where)} "
            f"ORDER BY member_plays_180d DESC, member_plays DESC LIMIT ?",
            (*params, limit),
        )
    ]
