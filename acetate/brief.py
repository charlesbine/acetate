"""One-call playlist intelligence for the chat agent — report + fit.

The chat agent's only window on playlists used to be raw SQL, which cannot
reach the embedding signal (taste_regions centroids are BLOBs; cosine isn't
computable in SQLite) — so it approximated sonic fit by string-matching style
labels, badly, across a dozen queries. These entrypoints wrap the taste/health
engine into tool-sized calls:

  report(conn, query)       resolve a playlist by name → identity, sonic health,
                            worst-fitting members WITH engagement (so "sonic
                            outlier" isn't confused with "unwanted"), gap-fill
                            candidates, name-family siblings, and stale links
                            (the playlist holds a dead variant of a song the
                            user still plays under another recording)
  fit(conn, query, tracks)  membership scores for named tracks vs the region,
                            with percentiles against the current members

Both are read-only and work on the agent's mode=ro connection.
"""

from __future__ import annotations

import re
import sqlite3

import numpy as np

from . import health, similar, taste

REAL_PLAY = "(ms_played IS NULL OR ms_played >= 30000)"
STALE_DAYS = 540  # a member unplayed this long is a dead-variant suspect...
FRESH_DAYS = 365  # ...when a same-song sibling recording played this recently

# Words that title variants add or drop freely ('Sunset - Original Mix' is the
# same song as 'Sunset'). Remix/remixer words are NOT listed on purpose — a
# remix is a different recording, not a variant.
_NOISE = {
    "original",
    "mix",
    "version",
    "edit",
    "radio",
    "extended",
    "club",
    "remaster",
    "remastered",
    "re",
    "mastered",
    "single",
    "album",
}


def _norm_title(title: str) -> str:
    """Collapse a track title to its variant-invariant core: lowercase, drop
    parentheticals (feat credits), punctuation, and packaging words."""
    t = re.sub(r"[(\[].*?[)\]]", " ", title.lower())
    t = re.sub(r"[^\w]+", " ", t, flags=re.UNICODE)
    return " ".join(w for w in t.split() if w not in _NOISE)


def resolve(conn: sqlite3.Connection, query: str, limit: int = 8) -> list[dict]:
    """Playlists matching `query` — an exact playlist_id, else a name search
    (all words, any order; falls back to any-word). Index 0 is the best guess
    (rotation first, then most-played); the rest are usually the name family
    ('Beat posay' / 'Beat posay (2)' / 'Posay')."""
    q = (query or "").strip()
    if not q:
        return []
    cols = (
        "playlist_id, name, persona, user_label, n_tracks, n_liked, "
        "member_plays, member_plays_180d, first_added_at, last_added_at"
    )
    row = conn.execute(f"SELECT {cols} FROM playlist_stats WHERE playlist_id=?", (q,)).fetchone()
    if row:
        return [dict(row)]
    words = [w for w in re.split(r"\W+", q.lower()) if w]
    if not words:
        return []
    order = "ORDER BY (persona='rotation') DESC, member_plays DESC LIMIT ?"
    where = " AND ".join(["instr(lower(name), ?) > 0"] * len(words))
    rows = conn.execute(
        f"SELECT {cols} FROM playlist_stats WHERE {where} {order}", (*words, limit)
    ).fetchall()
    long_words = [w for w in words if len(w) >= 4]  # 'no' would match inside 'techno'
    if not rows and len(words) > 1 and long_words:
        where = " OR ".join(["instr(lower(name), ?) > 0"] * len(long_words))
        rows = conn.execute(
            f"SELECT {cols} FROM playlist_stats WHERE {where} {order}", (*long_words, limit)
        ).fetchall()
    return [dict(r) for r in rows]


def _plays_180d(conn: sqlite3.Connection, recording_key: str) -> int:
    return conn.execute(
        f"SELECT COUNT(*) FROM play_events p "
        f"JOIN track_recording tr ON tr.track_id = p.track_id "
        f"WHERE tr.recording_key = ? AND {REAL_PLAY} "
        f"AND p.played_at >= date('now', '-180 day')",
        (recording_key,),
    ).fetchone()[0]


def _member_context(conn: sqlite3.Connection, playlist_id: str, row: dict) -> dict:
    """Engagement fields for a misfit row: is this sonic outlier actually
    unwanted, or a deliberate taste statement the user still plays?"""
    rs = conn.execute(
        "SELECT plays, liked, last_at FROM recording_stats WHERE recording_key=?",
        (row["recording_key"],),
    ).fetchone()
    added = conn.execute(
        "SELECT MIN(added_at) FROM playlist_members WHERE playlist_id=? AND recording_key=?",
        (playlist_id, row["recording_key"]),
    ).fetchone()[0]
    return {
        **row,
        "plays": rs["plays"] if rs else None,
        "liked": rs["liked"] if rs else None,
        "last_at": rs["last_at"] if rs else None,
        "added_at": added,
        "plays_180d": _plays_180d(conn, row["recording_key"]),
    }


def _homes(conn: sqlite3.Connection, row: dict) -> dict:
    """Where a gap-fill candidate already lives — the user may have homed it
    in a sibling vibe on purpose."""
    names = [
        r["name"]
        for r in conn.execute(
            "SELECT ps.name FROM playlist_members pm "
            "JOIN playlist_stats ps ON ps.playlist_id = pm.playlist_id "
            "WHERE pm.recording_key = ? AND ps.persona IN ('rotation','dormant','one-shot') "
            "ORDER BY ps.member_plays_180d DESC LIMIT 4",
            (row["recording_key"],),
        )
    ]
    last = conn.execute(
        "SELECT last_at FROM recording_stats WHERE recording_key=?", (row["recording_key"],)
    ).fetchone()
    return {**row, "last_at": last["last_at"] if last else None, "in_playlists": names}


def stale_links(conn: sqlite3.Connection, playlist_id: str, limit: int = 10) -> list[dict]:
    """Members whose recording looks dead while a same-artist, same-song
    variant recording is alive — the playlist entry went stale (relinked or
    region-pulled track), the song did not. The fix is swapping the entry,
    not removing the song."""
    out: list[dict] = []
    members = conn.execute(
        "SELECT rs.recording_key, rs.artist, rs.title, rs.plays, rs.last_at "
        "FROM playlist_members pm JOIN recording_stats rs USING(recording_key) "
        "WHERE pm.playlist_id = ? "
        "AND (rs.last_at IS NULL OR rs.last_at < date('now', ?))",
        (playlist_id, f"-{STALE_DAYS} day"),
    ).fetchall()
    for m in members:
        want = _norm_title(m["title"])
        if not want:
            continue
        for v in conn.execute(
            "SELECT title, plays, last_at, rep_track_id FROM recording_stats "
            "WHERE artist = ? AND recording_key != ? AND last_at >= date('now', ?)",
            (m["artist"], m["recording_key"], f"-{FRESH_DAYS} day"),
        ):
            if _norm_title(v["title"]) == want:
                out.append(
                    {
                        "member": f"{m['artist']} — {m['title']}",
                        "member_plays": m["plays"],
                        "member_last_at": m["last_at"],
                        "live_variant": v["title"],
                        "variant_plays": v["plays"],
                        "variant_last_at": v["last_at"],
                        "variant_track_id": v["rep_track_id"],
                    }
                )
                break
        if len(out) >= limit:
            break
    return out


def report(conn: sqlite3.Connection, query: str) -> dict:
    """The one-call playlist briefing behind the chat agent's playlist_report
    tool. Everything the agent used to re-derive with ~10 SQL queries — plus
    the embedding-side judgments SQL can't make at all."""
    matches = resolve(conn, query)
    if not matches:
        return {
            "error": f"no playlist matches {query!r} — try query_db over "
            "playlist_stats to browse names"
        }
    best = matches[0]
    pid = best["playlist_id"]
    out: dict = {"playlist": best}
    if len(matches) > 1:
        out["other_name_matches"] = [
            {k: m[k] for k in ("playlist_id", "name", "persona", "n_tracks", "member_plays")}
            for m in matches[1:6]
        ]
    out["members_played_180d"] = conn.execute(
        f"SELECT COUNT(DISTINCT tr.recording_key) FROM play_events p "
        f"JOIN track_recording tr ON tr.track_id = p.track_id "
        f"JOIN playlist_members pm ON pm.recording_key = tr.recording_key "
        f"  AND pm.playlist_id = ? "
        f"WHERE {REAL_PLAY} AND p.played_at >= date('now', '-180 day')",
        (pid,),
    ).fetchone()[0]

    region = taste.load_region(conn, pid, best["name"])
    if region is None:
        out["note"] = (
            "too few embedded members for sonic analysis — misfits/"
            "gap-fill/fit unavailable; answer from stats and SQL"
        )
        return out

    ent, styles = health.style_entropy(conn, region.member_keys)
    scores = region.member_scores
    out["sonic"] = {
        "cohesion": round(region.spread, 3),
        "style_entropy": round(ent, 3),
        "top_styles": [f"{s} ({n})" for s, n in styles[:8]],
        "member_score_p10": round(float(np.percentile(scores, 10)), 3),
        "member_score_median": round(float(np.median(scores)), 3),
    }
    split = health.split_check(region)
    if split:
        groups = []
        for g in split["groups"]:
            sample = conn.execute(
                "SELECT artist, title FROM recording_stats WHERE recording_key IN "
                f"({','.join('?' * min(3, len(g['member_keys'])))})",
                g["member_keys"][:3],
            ).fetchall()
            groups.append(
                {
                    "size": g["size"],
                    "cohesion": g["cohesion"],
                    "examples": [f"{r['artist']} — {r['title']}" for r in sample],
                }
            )
        out["split_suggestion"] = {"gain": split["gain"], "groups": groups}

    out["misfits"] = [_member_context(conn, pid, m) for m in taste.misfits(conn, region, limit=8)]
    out["gap_fill"] = [_homes(conn, a) for a in taste.additions(conn, region, limit=12)]
    out["stale_links"] = stale_links(conn, pid)
    return out


def fit(conn: sqlite3.Connection, playlist_query: str, tracks: list[str]) -> dict:
    """Membership scores for named tracks against a playlist's region — the
    chat agent's playlist_fit tool. Percentile is vs the playlist's own
    members: below ~15 fits worse than nearly everything already in it."""
    matches = resolve(conn, playlist_query)
    if not matches:
        return {"error": f"no playlist matches {playlist_query!r}"}
    best = matches[0]
    region = taste.load_region(conn, best["playlist_id"], best["name"])
    if region is None:
        return {"error": f"'{best['name']}' has too few embedded members to score against"}
    member_scores = np.sort(region.member_scores)
    members = set(taste.playlist_recordings(conn, best["playlist_id"]))

    rows: list[tuple[str, sqlite3.Row | None]] = [
        (str(q), similar.find_recording(conn, str(q))) for q in tracks[:20]
    ]
    scores = taste.membership(conn, region, [r["recording_key"] for _q, r in rows if r is not None])
    results = []
    for q, r in rows:
        if r is None:
            results.append({"query": q, "error": "no library match"})
            continue
        item = {
            "query": q,
            "artist": r["artist"],
            "title": r["title"],
            "track_id": r["rep_track_id"],
            "plays": r["plays"],
            "liked": r["liked"],
            "in_playlist": r["recording_key"] in members,
        }
        s = scores.get(r["recording_key"])
        if s is None:
            item["note"] = "not embedded yet — no sonic score"
        else:
            item["score"] = round(s, 3)
            item["percentile_vs_members"] = int(round(100 * float((member_scores < s).mean())))
        results.append(item)
    return {
        "playlist": {"playlist_id": best["playlist_id"], "name": best["name"]},
        "member_score_p10": round(float(np.percentile(member_scores, 10)), 3),
        "member_score_median": round(float(np.median(member_scores)), 3),
        "tracks": results,
    }
