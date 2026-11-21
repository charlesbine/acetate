"""Read-only stats queries for the dashboard home page."""

from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta

from . import genre

REAL = "(ms_played IS NULL OR ms_played >= 30000)"  # a "real" play: >=30s, or a live-poll event


def _art(conn: sqlite3.Connection, track_ids: list[str]) -> dict[str, str]:
    """Best-available album-art URL per track, from hydrated track objects."""
    if not track_ids:
        return {}
    qm = ",".join("?" * len(track_ids))
    out: dict[str, str] = {}
    for tid, raw in conn.execute(
        f"SELECT track_id, raw_json FROM tracks WHERE track_id IN ({qm}) AND raw_json IS NOT NULL",
        track_ids,
    ):
        try:
            imgs = (json.loads(raw).get("album") or {}).get("images") or []
            if imgs:
                out[tid] = (imgs[len(imgs) // 2] if len(imgs) > 1 else imgs[0])["url"]
        except Exception:
            pass
    return out


def _artist_images(conn: sqlite3.Connection, names: list[str]) -> dict[str, str]:
    """artist name → image URL, for artists hydrated from the artist endpoint."""
    if not names:
        return {}
    qm = ",".join("?" * len(names))
    out: dict[str, str] = {}
    for name, raw in conn.execute(
        f"SELECT name, raw_json FROM artists WHERE name IN ({qm}) AND raw_json IS NOT NULL", names
    ):
        try:
            imgs = json.loads(raw).get("images") or []
            if imgs:
                out[name] = (imgs[len(imgs) // 2] if len(imgs) > 1 else imgs[0])["url"]
        except Exception:
            pass
    return out


def totals(conn: sqlite3.Connection) -> dict:
    r = conn.execute(f"""
        SELECT COUNT(*) plays, COALESCE(SUM(ms_played),0) ms,
               COUNT(DISTINCT substr(played_at,1,4)) years
        FROM play_events WHERE track_id IS NOT NULL AND {REAL}""").fetchone()
    # distinct *recordings* played (duplicate IDs collapsed) — from the precomputed rollup
    tracks = conn.execute("SELECT COUNT(*) FROM recording_stats WHERE plays>0").fetchone()[0]
    artists = conn.execute(f"""
        SELECT COUNT(DISTINCT l.artist_name) FROM play_events p
        JOIN track_labels l ON l.track_id=p.track_id WHERE {REAL}""").fetchone()[0]
    bt = conn.execute("SELECT MAX(backup_time) FROM saved_track_snapshots").fetchone()[0]
    liked = (
        conn.execute(
            "SELECT COUNT(*) FROM saved_track_snapshots WHERE backup_time=?", (bt,)
        ).fetchone()[0]
        if bt
        else 0
    )
    return {
        "plays": r["plays"],
        "hours": round(r["ms"] / 3_600_000),
        "years": r["years"],
        "tracks": tracks,
        "artists": artists,
        "liked": liked,
    }


def by_year(conn: sqlite3.Connection) -> list[dict]:
    return [
        {"year": r["y"], "plays": r["plays"], "hours": round(r["ms"] / 3_600_000)}
        for r in conn.execute(f"""
        SELECT substr(played_at,1,4) y, COUNT(*) plays, COALESCE(SUM(ms_played),0) ms
        FROM play_events WHERE track_id IS NOT NULL AND {REAL} GROUP BY y ORDER BY y""")
    ]


def top_tracks(conn: sqlite3.Connection, since: str | None = None, limit: int = 10) -> list[dict]:
    if since is None:  # all-time → precomputed rollup (instant)
        rows = conn.execute(
            "SELECT rep_track_id tid, artist, title, plays FROM recording_stats "
            "WHERE plays>0 ORDER BY plays DESC LIMIT ?",
            (limit,),
        ).fetchall()
    else:  # period → aggregate play_events directly (time-filtered, covering index)
        rows = conn.execute(
            f"""
            SELECT tr.rep_track_id tid, tr.artist artist, tr.title title, COUNT(*) plays
            FROM play_events p JOIN track_recording tr ON tr.track_id=p.track_id
            WHERE p.track_id IS NOT NULL AND {REAL} AND p.played_at >= ?
            GROUP BY tr.recording_key ORDER BY plays DESC LIMIT ?""",
            (since, limit),
        ).fetchall()
    art = _art(conn, [r["tid"] for r in rows])
    return [
        {"artist": r["artist"], "title": r["title"], "plays": r["plays"], "art": art.get(r["tid"])}
        for r in rows
    ]


def top_artists(conn: sqlite3.Connection, since: str | None = None, limit: int = 10) -> list[dict]:
    where, params = f"WHERE p.track_id IS NOT NULL AND {REAL} AND l.artist_name IS NOT NULL", []
    if since:
        where += " AND p.played_at >= ?"
        params.append(since)
    rows = conn.execute(
        f"""
        SELECT l.artist_name artist, COUNT(*) plays, COUNT(DISTINCT p.track_id) tracks
        FROM play_events p JOIN track_labels l ON l.track_id=p.track_id
        {where} GROUP BY l.artist_name ORDER BY plays DESC LIMIT ?""",
        (*params, limit),
    ).fetchall()
    imgs = _artist_images(conn, [r["artist"] for r in rows])
    return [
        {
            "artist": r["artist"],
            "plays": r["plays"],
            "tracks": r["tracks"],
            "art": imgs.get(r["artist"]),
        }
        for r in rows
    ]


def top_genres(conn: sqlite3.Connection, limit: int = 12) -> list[dict]:
    return [
        {"genre": g, "tracks": t, "plays": p}
        for g, t, p in genre.top_genres(conn, min_plays=1, limit=limit)
    ]


def tracks_of_the_year(conn: sqlite3.Connection) -> list[dict]:
    """Your #1 most-played track for each calendar year."""
    rows = conn.execute(f"""
        WITH yt AS (
            SELECT substr(p.played_at,1,4) yr, tr.recording_key rk, tr.rep_track_id rid,
                   tr.artist artist, tr.title title, COUNT(*) plays,
                   ROW_NUMBER() OVER (PARTITION BY substr(p.played_at,1,4)
                                      ORDER BY COUNT(*) DESC, tr.recording_key) rn
            FROM play_events p JOIN track_recording tr ON tr.track_id=p.track_id
            WHERE p.track_id IS NOT NULL AND {REAL}
            GROUP BY yr, tr.recording_key)
        SELECT yr, rid tid, plays, artist, title FROM yt WHERE rn=1 ORDER BY yr DESC""").fetchall()
    art = _art(conn, [r["tid"] for r in rows])
    return [
        {
            "year": r["yr"],
            "plays": r["plays"],
            "artist": r["artist"],
            "title": r["title"],
            "art": art.get(r["tid"]),
        }
        for r in rows
    ]


def by_hour(conn: sqlite3.Connection) -> list[dict]:
    """Plays by hour-of-day (UTC), 0–23 with gaps filled."""
    counts = {
        int(r["h"]): r["plays"]
        for r in conn.execute(f"""
        SELECT substr(played_at,12,2) h, COUNT(*) plays
        FROM play_events WHERE track_id IS NOT NULL AND {REAL} GROUP BY h""")
    }
    return [{"hour": h, "plays": counts.get(h, 0)} for h in range(24)]


def by_weekday(conn: sqlite3.Connection) -> list[dict]:
    """Plays by day-of-week (Mon→Sun)."""
    counts = {
        int(r["w"]): r["plays"]
        for r in conn.execute(f"""
        SELECT strftime('%w', substr(played_at,1,10)) w, COUNT(*) plays
        FROM play_events WHERE track_id IS NOT NULL AND {REAL} GROUP BY w""")
    }
    names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    order = [1, 2, 3, 4, 5, 6, 0]  # Mon..Sun
    return [{"day": names[i], "plays": counts.get(i, 0)} for i in order]


def discovery_by_year(conn: sqlite3.Connection) -> list[dict]:
    """How many distinct recordings you first heard in each year (from the rollup)."""
    return [
        {"year": r["yr"], "n": r["n"]}
        for r in conn.execute(
            "SELECT substr(first_at,1,4) yr, COUNT(*) n FROM recording_stats "
            "WHERE first_at IS NOT NULL GROUP BY yr ORDER BY yr"
        )
    ]


def recent_plays(conn: sqlite3.Connection, limit: int = 12) -> list[dict]:
    rows = conn.execute(
        """
        SELECT p.track_id tid, l.artist_name artist, l.track_name title, p.played_at
        FROM play_events p JOIN track_labels l ON l.track_id=p.track_id
        WHERE p.track_id IS NOT NULL ORDER BY p.played_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    art = _art(conn, [r["tid"] for r in rows])
    return [
        {
            "artist": r["artist"],
            "title": r["title"],
            "played_at": r["played_at"],
            "art": art.get(r["tid"]),
        }
        for r in rows
    ]


# --------------------------------------------------------------------------- #
#  Period drill-down (year / quarter / month / week)
# --------------------------------------------------------------------------- #


def period_range(kind: str, key: str) -> tuple[date, date]:
    """[start, end) dates for a period key. year=2025, quarter=2025-Q2, month=2025-07, week=2025-W31."""
    if kind == "year":
        y = int(key)
        return date(y, 1, 1), date(y + 1, 1, 1)
    if kind == "quarter":
        y, q = key.split("-Q")
        y, m = int(y), (int(q) - 1) * 3 + 1
        end = date(y + 1, 1, 1) if m == 10 else date(y, m + 3, 1)
        return date(y, m, 1), end
    if kind == "month":
        y, m = (int(x) for x in key.split("-"))
        end = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
        return date(y, m, 1), end
    if kind == "week":
        y, w = key.split("-W")
        start = date.fromisocalendar(int(y), int(w), 1)
        return start, start + timedelta(days=7)
    raise ValueError(kind)


def available_periods(conn: sqlite3.Connection, kind: str) -> list[str]:
    """Period keys that actually have plays, newest first."""
    q = f"WHERE track_id IS NOT NULL AND {REAL}"
    if kind == "year":
        return [
            r[0]
            for r in conn.execute(
                f"SELECT DISTINCT substr(played_at,1,4) FROM play_events {q} ORDER BY 1 DESC"
            )
        ]
    if kind == "month":
        return [
            r[0]
            for r in conn.execute(
                f"SELECT DISTINCT substr(played_at,1,7) FROM play_events {q} ORDER BY 1 DESC"
            )
        ]
    if kind == "quarter":
        months = [
            r[0]
            for r in conn.execute(f"SELECT DISTINCT substr(played_at,1,7) FROM play_events {q}")
        ]
        keys = {f"{m[:4]}-Q{(int(m[5:7]) - 1) // 3 + 1}" for m in months}
        return sorted(keys, reverse=True)
    if kind == "week":
        weeks = set()
        for (d,) in conn.execute(f"SELECT DISTINCT substr(played_at,1,10) FROM play_events {q}"):
            try:
                iso = date.fromisoformat(d).isocalendar()
                weeks.add(f"{iso.year}-W{iso.week:02d}")
            except ValueError:
                pass
        return sorted(weeks, reverse=True)
    return []


def period_summary(
    conn: sqlite3.Connection, start: str, end: str, prev_start: str, prev_end: str
) -> dict:
    """Everything for one period: totals, deltas, top tracks/artists/genres,
    new discoveries, busiest day, and an hourly clock."""
    rng = f"p.track_id IS NOT NULL AND {REAL} AND p.played_at>=? AND p.played_at<?"

    def scalar(sql: str, params: tuple) -> int:
        r = conn.execute(sql, params).fetchone()
        return r[0] if r and r[0] is not None else 0

    plays = scalar(f"SELECT COUNT(*) FROM play_events p WHERE {rng}", (start, end))
    prev_plays = scalar(f"SELECT COUNT(*) FROM play_events p WHERE {rng}", (prev_start, prev_end))
    ms = scalar(f"SELECT COALESCE(SUM(ms_played),0) FROM play_events p WHERE {rng}", (start, end))
    tracks = scalar(
        f"SELECT COUNT(DISTINCT tr.recording_key) FROM play_events p "
        f"JOIN track_recording tr ON tr.track_id=p.track_id WHERE {rng}",
        (start, end),
    )
    artists = scalar(
        f"""SELECT COUNT(DISTINCT l.artist_name) FROM play_events p
        JOIN track_labels l ON l.track_id=p.track_id WHERE {rng}""",
        (start, end),
    )

    tt = conn.execute(
        f"""SELECT tr.rep_track_id tid, tr.artist artist, tr.title title, COUNT(*) plays
        FROM play_events p JOIN track_recording tr ON tr.track_id=p.track_id WHERE {rng}
        GROUP BY tr.recording_key ORDER BY plays DESC LIMIT 10""",
        (start, end),
    ).fetchall()
    ta = conn.execute(
        f"""SELECT l.artist_name artist, COUNT(*) plays, COUNT(DISTINCT p.track_id) tracks
        FROM play_events p JOIN track_labels l ON l.track_id=p.track_id
        WHERE {rng} AND l.artist_name IS NOT NULL
        GROUP BY l.artist_name ORDER BY plays DESC LIMIT 10""",
        (start, end),
    ).fetchall()

    new_count = scalar(
        f"""WITH f AS (SELECT tr.recording_key rk, MIN(p.played_at) fp
        FROM play_events p JOIN track_recording tr ON tr.track_id=p.track_id
        WHERE p.track_id IS NOT NULL AND {REAL} GROUP BY tr.recording_key)
        SELECT COUNT(*) FROM f WHERE fp>=? AND fp<?""",
        (start, end),
    )
    new_top = conn.execute(
        f"""WITH f AS (SELECT tr.recording_key rk, MIN(p.played_at) fp
        FROM play_events p JOIN track_recording tr ON tr.track_id=p.track_id
        WHERE p.track_id IS NOT NULL AND {REAL} GROUP BY tr.recording_key)
        SELECT tr.rep_track_id tid, tr.artist artist, tr.title title, COUNT(*) plays
        FROM play_events p JOIN track_recording tr ON tr.track_id=p.track_id JOIN f ON f.rk=tr.recording_key
        WHERE p.played_at>=? AND p.played_at<? AND f.fp>=? AND f.fp<? AND {REAL}
        GROUP BY tr.recording_key ORDER BY plays DESC LIMIT 6""",
        (start, end, start, end),
    ).fetchall()

    grows = conn.execute(
        f"""SELECT g.genre, COUNT(DISTINCT g.track_id) tracks, COUNT(*) plays
        FROM play_events p JOIN track_genres g ON g.track_id=p.track_id
        WHERE p.played_at>=? AND p.played_at<? AND {REAL} AND g.source='lastfm'
        GROUP BY g.genre ORDER BY plays DESC""",
        (start, end),
    ).fetchall()
    genres = [
        {"genre": r["genre"], "tracks": r["tracks"], "plays": r["plays"]}
        for r in grows
        if genre.is_genre(r["genre"])
    ][:10]

    bd = conn.execute(
        f"""SELECT substr(played_at,1,10) d, COUNT(*) n FROM play_events p
        WHERE {rng} GROUP BY d ORDER BY n DESC LIMIT 1""",
        (start, end),
    ).fetchone()
    hc = {
        int(r["h"]): r["plays"]
        for r in conn.execute(
            f"SELECT substr(played_at,12,2) h, COUNT(*) plays FROM play_events p WHERE {rng} GROUP BY h",
            (start, end),
        )
    }

    art = _art(conn, [r["tid"] for r in tt] + [r["tid"] for r in new_top])
    a_imgs = _artist_images(conn, [r["artist"] for r in ta])

    def mk(rows):
        return [
            {
                "artist": r["artist"],
                "title": r["title"],
                "plays": r["plays"],
                "art": art.get(r["tid"]),
            }
            for r in rows
        ]

    return {
        "plays": plays,
        "hours": round(ms / 3_600_000),
        "tracks": tracks,
        "artists": artists,
        "prev_plays": prev_plays,
        "delta": (round((plays - prev_plays) / prev_plays * 100) if prev_plays else None),
        "top_tracks": mk(tt),
        "top_artists": [
            {
                "artist": r["artist"],
                "plays": r["plays"],
                "tracks": r["tracks"],
                "art": a_imgs.get(r["artist"]),
            }
            for r in ta
        ],
        "new_count": new_count,
        "new_top": mk(new_top),
        "genres": genres,
        "busiest": ({"date": bd["d"], "n": bd["n"]} if bd else None),
        "hours_clock": [{"hour": h, "plays": hc.get(h, 0)} for h in range(24)],
    }


# --------------------------------------------------------------------------- #
#  Lost Gems — liked favorites that never made it into a real playlist and have
#  faded (the north-star "liked but not playlisted, now lost" problem).
# --------------------------------------------------------------------------- #


def _faded_cutoff(faded_days: int) -> str:
    return (date.today() - timedelta(days=faded_days)).isoformat()


def lost_gems_summary(conn: sqlite3.Connection, faded_days: int = 183) -> dict:
    """Counts: liked total (track IDs), orphaned *recordings* (liked, never in a real
    playlist), and faded (orphaned + played + silent). Recording-aware: a song homed
    or played under any of its duplicate IDs counts."""
    cutoff = _faded_cutoff(faded_days)
    liked = conn.execute(
        "SELECT COUNT(DISTINCT track_id) FROM saved_track_snapshots "
        "WHERE backup_time=(SELECT MAX(backup_time) FROM saved_track_snapshots) AND track_id IS NOT NULL"
    ).fetchone()[0]
    r = conn.execute(
        "SELECT (SELECT COUNT(*) FROM recording_stats WHERE liked=1 AND homed=0) AS orphan, "
        "(SELECT COUNT(*) FROM recording_stats WHERE liked=1 AND homed=0 AND plays>0 AND last_at < ?) AS faded",
        (cutoff,),
    ).fetchone()
    return {"liked": liked, "orphan": r["orphan"], "faded": r["faded"]}


def lost_gems(
    conn: sqlite3.Connection, faded_days: int = 183, limit: int | None = 100
) -> list[dict]:
    """Faded orphaned gems, one row per recording (duplicate IDs collapsed): liked,
    never playlisted, played, silent for `faded_days`, ranked by total plays."""
    cutoff = _faded_cutoff(faded_days)
    sql = (
        "SELECT rep_track_id tid, artist, title, plays, last_at last FROM recording_stats "
        "WHERE liked=1 AND homed=0 AND plays>0 AND last_at < ? ORDER BY plays DESC"
    )
    params: tuple = (cutoff,)
    if limit:
        sql += " LIMIT ?"
        params = (cutoff, limit)
    rows = conn.execute(sql, params).fetchall()
    art = _art(conn, [r["tid"] for r in rows])
    return [
        {
            "track_id": r["tid"],
            "artist": r["artist"],
            "title": r["title"],
            "plays": r["plays"],
            "last": r["last"],
            "art": art.get(r["tid"]),
        }
        for r in rows
    ]


# Sibling "lost favorite" lenses — all liked-but-never-playlisted, sliced different ways.
# Each is a cheap filter over the recording_stats rollup.
GEM_CATEGORIES = {
    "faded": {
        "label": "Faded",
        "heading": "faded gems · most-played first",
        "desc": "Liked and actually played — then gone quiet for 6 months or more.",
    },
    "never_played": {
        "label": "Never played",
        "heading": "never played · liked but untouched",
        "desc": "Liked and never once played — the deepest cuts in your library.",
    },
    "under": {
        "label": "Under-explored",
        "heading": "barely sampled · 1–3 plays",
        "desc": "Liked, but you barely gave them a chance (1–3 plays).",
    },
    "reups": {
        "label": "Re-ups",
        "heading": "former heavy rotation · gone quiet",
        "desc": "Played heavily (15+ plays), then dropped cold for 6 months+. Time to revive.",
    },
}


def _gem_where(cat: str, cutoff: str) -> tuple[str, tuple]:
    base = "liked=1 AND homed=0"
    if cat == "never_played":
        return f"{base} AND plays=0", ()
    if cat == "under":
        return f"{base} AND plays BETWEEN 1 AND 3", ()
    if cat == "reups":
        return f"{base} AND plays>=15 AND last_at < ?", (cutoff,)
    return f"{base} AND plays>0 AND last_at < ?", (cutoff,)  # faded (default)


def gems(
    conn: sqlite3.Connection, cat: str = "faded", faded_days: int = 183, limit: int | None = 80
) -> list[dict]:
    """Lost-favorite recordings for one category (see GEM_CATEGORIES)."""
    cutoff = _faded_cutoff(faded_days)
    where, params = _gem_where(cat, cutoff)
    order = "artist, title" if cat == "never_played" else "plays DESC, last_at DESC"
    sql = (
        f"SELECT rep_track_id tid, artist, title, plays, last_at last "
        f"FROM recording_stats WHERE {where} ORDER BY {order}"
    )
    if limit:
        sql += " LIMIT ?"
        params = (*params, limit)
    rows = conn.execute(sql, params).fetchall()
    art = _art(conn, [r["tid"] for r in rows])
    return [
        {
            "track_id": r["tid"],
            "artist": r["artist"],
            "title": r["title"],
            "plays": r["plays"],
            "last": r["last"],
            "art": art.get(r["tid"]),
        }
        for r in rows
    ]


def gem_counts(conn: sqlite3.Connection, faded_days: int = 183) -> dict[str, int]:
    cutoff = _faded_cutoff(faded_days)
    return {
        cat: conn.execute(f"SELECT COUNT(*) FROM recording_stats WHERE {w}", p).fetchone()[0]
        for cat in GEM_CATEGORIES
        for (w, p) in [_gem_where(cat, cutoff)]
    }
