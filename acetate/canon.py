"""Canonicalize Spotify's duplicate track IDs into one logical recording per song.

Spotify hands out multiple track IDs for the same recording (single vs album vs
compilation), and play history / likes scatter across them. We key each track_id
to a `recording_key` = normalized(artist) + "\\t" + normalized(title), pick the
most-played id as the group's representative, and store it in track_recording.
Every track-level aggregation joins through this so plays/last-played/discovery
collapse duplicates. (ISRC would be more precise but is coverage-gapped — GDPR-only
tracks carry no ISRC — so artist+title is the workhorse key.)
"""

from __future__ import annotations

import re
import sqlite3

_WS = re.compile(r"\s+")
_VERSION_TOKEN = re.compile(
    r"\b("
    r"radio|edit|extended|mix|remix|remaster(?:ed)?|version|club|original|single|"
    r"instrumental|acoustic|live|vip|dub|rework|bootleg|mono|stereo"
    r")\b",
    re.I,
)
_TRAILING_VERSION = re.compile(r"\s+-\s+[^-]*$")


def normalize_text(s: str | None) -> str:
    """Normalize an artist or title for stable display-name identity matching."""
    return _WS.sub(" ", (s or "").strip().lower())


_norm = normalize_text


def recording_key(artist: str | None, title: str | None) -> str:
    return f"{_norm(artist)}\t{_norm(title)}"


def title_lookup_variants(title: str) -> list[str]:
    """Return conservative lookup-only aliases without changing identity."""
    variants: list[str] = []

    def add(value: str | None) -> None:
        norm = _norm(value)
        if norm and norm not in variants:
            variants.append(norm)

    add(title)
    if not title:
        return variants

    for match in re.finditer(r"[\[(]([^\])]+)[\])]", title):
        if _VERSION_TOKEN.search(match.group(1)):
            add((title[: match.start()] + title[match.end() :]).strip())

    trailing = title.rsplit(" - ", 1)
    if len(trailing) == 2 and _VERSION_TOKEN.search(trailing[1]):
        add(trailing[0])

    no_trailing = _TRAILING_VERSION.sub("", title)
    if no_trailing != title and _VERSION_TOKEN.search(title.rsplit(" - ", 1)[-1]):
        add(no_trailing)

    return variants


def canonicalize(conn: sqlite3.Connection) -> dict:
    """Rebuild track_recording from track_labels (GDPR) + the hydrated dims."""
    names: dict[str, tuple[str, str]] = {}
    # GDPR labels first (cover every played track), then dims for poll/playlist-only ids.
    for tid, art, ti in conn.execute(
        "SELECT track_id, artist_name, track_name FROM track_labels "
        "WHERE artist_name IS NOT NULL AND track_name IS NOT NULL"
    ):
        names[tid] = (art, ti)
    for tid, art, ti in conn.execute(
        "SELECT t.track_id, a.name, t.name FROM tracks t "
        "JOIN track_artists ta ON ta.track_id=t.track_id AND ta.position=0 "
        "JOIN artists a ON a.artist_id=ta.artist_id "
        "WHERE t.name IS NOT NULL AND a.name IS NOT NULL"
    ):
        names.setdefault(tid, (art, ti))

    plays = {
        tid: c
        for tid, c in conn.execute(
            "SELECT track_id, COUNT(*) FROM play_events WHERE track_id IS NOT NULL GROUP BY track_id"
        )
    }

    groups: dict[str, list[tuple[str, str, str]]] = {}
    for tid, (art, ti) in names.items():
        groups.setdefault(recording_key(art, ti), []).append((tid, art, ti))

    rows = []
    for key, members in groups.items():
        rep_tid, rep_art, rep_ti = max(members, key=lambda m: (plays.get(m[0], 0), m[0]))
        for tid, _a, _t in members:
            rows.append((tid, key, rep_tid, rep_art, rep_ti))

    conn.execute("DELETE FROM track_recording")
    conn.executemany(
        "INSERT INTO track_recording(track_id, recording_key, rep_track_id, artist, title) "
        "VALUES(?,?,?,?,?)",
        rows,
    )
    conn.commit()
    refresh_recording_stats(conn)  # keep the rollup in sync with the new recording map
    return {"tracks": len(names), "recordings": len(groups)}


def refresh_recording_stats(conn: sqlite3.Connection) -> int:
    """Rebuild recording_stats: one row per recording with plays / first / last play and
    liked / homed flags precomputed. Cheap (~0.6s) — run after canonicalize and after a
    poll that ingested plays, so the dashboard's all-time aggregations are instant reads."""
    conn.execute("DELETE FROM recording_stats")
    conn.execute("""
        INSERT INTO recording_stats(recording_key, rep_track_id, artist, title,
                                    plays, first_at, last_at, liked, homed, liked_at)
        WITH pt AS (
            SELECT track_id, COUNT(*) plays, MIN(played_at) first_at, MAX(played_at) last_at
            FROM play_events WHERE track_id IS NOT NULL AND (ms_played IS NULL OR ms_played >= 30000)
            GROUP BY track_id),
        lk AS (SELECT track_id, MIN(added_at) added_at FROM saved_track_snapshots
               WHERE backup_time=(SELECT MAX(backup_time) FROM saved_track_snapshots) AND track_id IS NOT NULL
               GROUP BY track_id),
        hm AS (SELECT DISTINCT track_id FROM playlist_track_snapshots
               WHERE track_id IS NOT NULL AND playlist_id NOT IN (SELECT playlist_id FROM managed_playlists))
        SELECT tr.recording_key, MAX(tr.rep_track_id), MAX(tr.artist), MAX(tr.title),
               COALESCE(SUM(pt.plays), 0), MIN(pt.first_at), MAX(pt.last_at),
               MAX(lk.track_id IS NOT NULL), MAX(hm.track_id IS NOT NULL), MIN(lk.added_at)
        FROM track_recording tr
        LEFT JOIN pt ON pt.track_id = tr.track_id
        LEFT JOIN lk ON lk.track_id = tr.track_id
        LEFT JOIN hm ON hm.track_id = tr.track_id
        GROUP BY tr.recording_key""")
    conn.commit()
    return conn.execute("SELECT COUNT(*) FROM recording_stats").fetchone()[0]


def ensure_recording_stats(conn: sqlite3.Connection) -> None:
    """Build the rollup if it's empty (first boot after deploy) or stale (a schema change
    like liked_at was migrated in but never repopulated)."""
    try:
        empty = conn.execute("SELECT COUNT(*) FROM recording_stats").fetchone()[0] == 0
        stale = (not empty) and bool(
            conn.execute(
                "SELECT (SELECT COUNT(*) FROM recording_stats WHERE liked=1) > 0 "
                "AND (SELECT COUNT(*) FROM recording_stats WHERE liked_at IS NOT NULL) = 0"
            ).fetchone()[0]
        )
        if empty or stale:
            refresh_recording_stats(conn)
    except sqlite3.OperationalError:
        pass  # table not created yet; init_db will create it
