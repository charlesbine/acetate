"""Smart Playlists — saved rules that compile to SQL over recording_stats.

A *spec* is a small JSON object of filters + sort + limit. evaluate() turns it into
a deterministic list of representative track IDs (duplicate-collapsed). The scheduler
syncs the result to a Spotify playlist (see playlists.build_smart). The hardcoded
monthly/gems/on-loop builders are all expressible as specs.

Spec fields (all optional unless noted):
  liked              true → only liked; false → only un-liked
  homed              true → only in a real playlist; false → never playlisted (orphans)
  played             true → has >=1 real play; false → never played
  min_plays/max_plays   integer bounds on total real plays
  liked_before_days  liked more than N days ago        (uses recording_stats.liked_at)
  liked_after_days   liked within the last N days
  not_played_days    silent: last play older than N days (or never played)
  played_within_days played at least once in the last N days
  first_played_before_days / first_played_after_days   discovery-date bounds
  genres             list[str] — recording has any of these Last.fm genres (case-insensitive)
  sort               plays_desc (default) | plays_asc | recent | oldest_played
                     | liked_recent | liked_oldest | discovered_recent | discovered_oldest | random
  limit              integer (default 50)

A spec may instead be a *similarity source* — "candidates for <playlist>", ranked by
taste-model membership against that playlist's embedding region rather than by SQL:
  candidates_for     str — anchor playlist name (exact, else unique substring match)
  pool               library | outside | both (default both) — own tracks not yet in the
                     playlist (gap-fill), embedded outside-library candidates, or both
  min_membership     float — membership floor (default: the playlist's own member p10,
                     i.e. "as strong a fit as your weakest 10% of real members")
  exclude_members    bool (default true) — drop tracks already in the anchor playlist
  limit              integer (default 40)
This source needs only numpy over the stored embeddings table — no heavy ML env — so it
runs on the scheduler like any other Smart Playlist.

Or a spec may be an *explicit track list* — a hand-picked set (e.g. a chat-agent
curation verified with playlist_fit). This is exact: what you list is what the playlist
becomes, in order.
  track_ids          list[str] — Spotify track ids, kept verbatim and in order
  limit              optional cap on how many to keep
When track_ids is present, all filter/sort fields are ignored — the list IS the playlist.
These fixed selections belong to Curated Playlists: publish once, without daily refreshes.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta

from . import db

_SORTS = {
    "plays_desc": "plays DESC",
    "plays_asc": "plays ASC",
    "recent": "last_at DESC",
    "oldest_played": "last_at ASC",
    "liked_recent": "liked_at DESC",
    "liked_oldest": "liked_at ASC",
    "discovered_recent": "first_at DESC",
    "discovered_oldest": "first_at ASC",
    "random": "RANDOM()",
}


def _ago(days: int) -> str:
    return (date.today() - timedelta(days=int(days))).isoformat()


def is_curated(spec: dict) -> bool:
    """Explicit tracks are a fixed selection; every other spec is a live rule."""
    return "track_ids" in spec


def build_query(spec: dict) -> tuple[str, list]:
    """Compile a spec to (sql, params) returning rep_track_id rows from recording_stats."""
    where: list[str] = ["1=1"]
    p: list = []

    def flag(key: str, col: str) -> None:
        if spec.get(key) is True:
            where.append(f"{col}=1")
        elif spec.get(key) is False:
            where.append(f"{col}=0")

    flag("liked", "liked")
    flag("homed", "homed")
    if spec.get("played") is True:
        where.append("plays > 0")
    elif spec.get("played") is False:
        where.append("plays = 0")
    if spec.get("min_plays") is not None:
        where.append("plays >= ?")
        p.append(int(spec["min_plays"]))
    if spec.get("max_plays") is not None:
        where.append("plays <= ?")
        p.append(int(spec["max_plays"]))
    if spec.get("liked_before_days"):
        where.append("liked_at IS NOT NULL AND liked_at < ?")
        p.append(_ago(spec["liked_before_days"]))
    if spec.get("liked_after_days"):
        where.append("liked_at >= ?")
        p.append(_ago(spec["liked_after_days"]))
    if spec.get("not_played_days"):
        where.append("(last_at IS NULL OR last_at < ?)")
        p.append(_ago(spec["not_played_days"]))
    if spec.get("played_within_days"):
        where.append("last_at >= ?")
        p.append(_ago(spec["played_within_days"]))
    if spec.get("first_played_before_days"):
        where.append("first_at IS NOT NULL AND first_at < ?")
        p.append(_ago(spec["first_played_before_days"]))
    if spec.get("first_played_after_days"):
        where.append("first_at >= ?")
        p.append(_ago(spec["first_played_after_days"]))
    if spec.get("genres"):
        genres = [g.lower() for g in spec["genres"]]
        qs = ",".join("?" * len(genres))
        where.append(
            "recording_key IN (SELECT tr.recording_key FROM track_recording tr "
            f"JOIN track_genres g ON g.track_id = tr.track_id WHERE lower(g.genre) IN ({qs}))"
        )
        p.extend(genres)

    order = _SORTS.get(spec.get("sort", "plays_desc"), "plays DESC")
    limit = int(spec.get("limit", 50))
    sql = (
        f"SELECT rep_track_id FROM recording_stats WHERE {' AND '.join(where)} "
        f"ORDER BY {order} LIMIT ?"
    )
    return sql, [*p, limit]


def evaluate(conn: sqlite3.Connection, spec: dict) -> list[str]:
    """Representative track IDs matching the spec (duplicate-collapsed, deterministic
    except for sort=random). A 'candidates_for' spec routes to the similarity source;
    a 'track_ids' spec is an explicit hand-picked list."""
    if is_curated(spec):
        return _explicit_tracks(spec)
    if spec.get("candidates_for"):
        return _evaluate_candidates(conn, spec)
    sql, params = build_query(spec)
    return [r[0] for r in conn.execute(sql, params)]


def _explicit_tracks(spec: dict) -> list[str]:
    """An explicit, ordered Spotify-track-id list — deduped, order preserved,
    optionally capped by `limit`. The list IS the playlist (no SQL, no ranking)."""
    ids = spec["track_ids"]
    if not isinstance(ids, list) or any(not isinstance(tid, str) or not tid.strip() for tid in ids):
        raise ValueError("track_ids must be a list of nonempty Spotify track ID strings")
    seen: set[str] = set()
    out: list[str] = []
    for tid in ids:
        if tid not in seen:
            seen.add(tid)
            out.append(tid)
    limit = spec.get("limit")
    return out[: int(limit)] if limit else out


def _resolve_anchor(conn: sqlite3.Connection, name: str) -> dict | None:
    """Find the authored playlist named `name` (exact case-insensitive match, else a
    unique substring match). Ambiguous or missing → None."""
    from . import taste

    nl = name.lower().strip()
    playlists = taste.authored_playlists(conn)
    exact = [p for p in playlists if (p["name"] or "").lower() == nl]
    if exact:
        return exact[0]
    subs = [p for p in playlists if nl in (p["name"] or "").lower()]
    return subs[0] if len(subs) == 1 else None


def _candidate_spotify_id(conn: sqlite3.Connection, candidate_key: str) -> str | None:
    row = conn.execute(
        "SELECT external_id FROM candidate_external_ids "
        "WHERE candidate_key=? AND provider='spotify' AND id_type='track_id' "
        "ORDER BY confidence DESC LIMIT 1",
        (candidate_key,),
    ).fetchone()
    return row[0] if row else None


def _evaluate_candidates(conn: sqlite3.Connection, spec: dict) -> list[str]:
    """Rank library gap-fill and/or outside-library candidates by taste membership
    against an anchor playlist's region. Returns Spotify track ids, best fit first."""
    import numpy as np

    from . import taste

    anchor = _resolve_anchor(conn, spec["candidates_for"])
    if anchor is None:
        return []
    region = taste.load_region(conn, anchor["playlist_id"], anchor["name"])
    if region is None:  # too few embedded members to define a region
        return []

    pool = spec.get("pool", "both")
    limit = int(spec.get("limit", 40))
    exclude_members = spec.get("exclude_members", True)

    lib_keys: list[str] = []
    cand_keys: list[str] = []
    if pool in ("library", "both"):
        lib_keys = [r[0] for r in conn.execute("SELECT recording_key FROM recording_stats")]
        if exclude_members:
            members = set(taste.playlist_recordings(conn, anchor["playlist_id"]))
            lib_keys = [k for k in lib_keys if k not in members]
    if pool in ("outside", "both"):
        cand_keys = [
            r[0]
            for r in conn.execute(
                "SELECT candidate_key FROM candidate_recordings WHERE status='embedded' "
                "AND candidate_key NOT IN (SELECT candidate_key FROM candidate_duplicates)"
            )
        ]

    lib_set = set(lib_keys)
    scores = taste.membership(conn, region, lib_keys + cand_keys)
    if not scores:
        return []

    thr = spec.get("min_membership")
    if thr is None:
        ms = region.member_scores
        thr = float(np.percentile(ms, 10)) if ms is not None and len(ms) else 0.0
    thr = float(thr)

    ranked = sorted(((s, k) for k, s in scores.items() if s >= thr), key=lambda t: -t[0])

    # resolve rep_track_id for the library keys still in contention (bulk, chunked)
    lib_ids: dict[str, str] = {}
    ranked_lib = [k for _s, k in ranked if k in lib_set]
    for i in range(0, len(ranked_lib), 500):
        chunk = ranked_lib[i : i + 500]
        qm = ",".join("?" * len(chunk))
        for r in conn.execute(
            f"SELECT recording_key, rep_track_id FROM recording_stats "
            f"WHERE recording_key IN ({qm})",
            chunk,
        ):
            if r["rep_track_id"]:
                lib_ids[r["recording_key"]] = r["rep_track_id"]

    out: list[str] = []
    seen: set[str] = set()
    for _s, k in ranked:
        if len(out) >= limit:
            break
        tid = lib_ids.get(k) if k in lib_set else _candidate_spotify_id(conn, k)
        if tid and tid not in seen:
            seen.add(tid)
            out.append(tid)
    return out


def preview(conn: sqlite3.Connection, spec: dict, limit: int = 6) -> list[dict]:
    """Top matches with names — a UI preview of what a rule currently contains."""
    ids = evaluate(conn, {**spec, "limit": limit})
    if not ids:
        return []
    qm = ",".join("?" * len(ids))
    rows = {
        r["rep_track_id"]: r
        for r in conn.execute(
            f"SELECT rep_track_id, artist, title FROM recording_stats WHERE rep_track_id IN ({qm})",
            ids,
        )
    }
    return [{"artist": rows[i]["artist"], "title": rows[i]["title"]} for i in ids if i in rows]


def describe(spec: dict) -> str:
    """One-line human summary of a spec (for the UI / playlist description)."""
    if is_curated(spec):
        n = len(_explicit_tracks(spec))
        return f"{n} hand-picked track{'' if n == 1 else 's'}"
    if spec.get("candidates_for"):
        pool = spec.get("pool", "both")
        pool_txt = {
            "library": "your library",
            "outside": "outside library",
            "both": "library + outside",
        }.get(pool, pool)
        thr = spec.get("min_membership")
        thr_txt = f"membership ≥{thr}" if thr is not None else "membership ≥ member p10"
        return " · ".join(
            [
                f"candidates for “{spec['candidates_for']}”",
                pool_txt,
                thr_txt,
                f"top {spec.get('limit', 40)}",
            ]
        )
    b: list[str] = []
    if spec.get("liked") is True:
        b.append("liked")
    if spec.get("liked") is False:
        b.append("not liked")
    if spec.get("homed") is True:
        b.append("in a playlist")
    if spec.get("homed") is False:
        b.append("never playlisted")
    if spec.get("played") is True:
        b.append("played")
    if spec.get("played") is False:
        b.append("never played")
    if spec.get("min_plays") is not None:
        b.append(f"≥{spec['min_plays']} plays")
    if spec.get("max_plays") is not None:
        b.append(f"≤{spec['max_plays']} plays")
    if spec.get("liked_before_days"):
        b.append(f"liked >{spec['liked_before_days']}d ago")
    if spec.get("liked_after_days"):
        b.append(f"liked ≤{spec['liked_after_days']}d ago")
    if spec.get("not_played_days"):
        b.append(f"silent {spec['not_played_days']}d+")
    if spec.get("played_within_days"):
        b.append(f"played ≤{spec['played_within_days']}d")
    if spec.get("first_played_before_days"):
        b.append(f"discovered >{spec['first_played_before_days']}d ago")
    if spec.get("first_played_after_days"):
        b.append(f"discovered ≤{spec['first_played_after_days']}d ago")
    if spec.get("genres"):
        b.append("genre: " + ", ".join(spec["genres"]))
    b.append(f"top {spec.get('limit', 50)} by {spec.get('sort', 'plays_desc').replace('_', ' ')}")
    return " · ".join(b)


# Starter specs — handy presets + examples for the LLM authoring layer later.
EXAMPLES = {
    "Forgotten Techno": {
        "liked": True,
        "homed": False,
        "liked_before_days": 365,
        "not_played_days": 183,
        "genres": ["techno"],
        "sort": "plays_desc",
        "limit": 50,
    },
    "Fresh Likes": {"liked": True, "liked_after_days": 60, "sort": "liked_recent", "limit": 40},
    "Deep Cuts": {"liked": True, "homed": False, "max_plays": 3, "sort": "random", "limit": 50},
    "Revive Old Favorites": {
        "liked": True,
        "min_plays": 15,
        "not_played_days": 365,
        "sort": "plays_desc",
        "limit": 50,
    },
    "Candidates for Melodic techno": {
        "candidates_for": "Melodic techno",
        "pool": "both",
        "exclude_members": True,
        "limit": 40,
    },
}


# --------------------------------------------------------------------------- #
#  CRUD over the smart_playlists table
# --------------------------------------------------------------------------- #


def create(conn: sqlite3.Connection, name: str, spec: dict, *, commit: bool = True) -> int:
    now = db.utcnow()
    cur = conn.execute(
        "INSERT INTO smart_playlists(name, spec, enabled, created_at, updated_at) VALUES(?,?,1,?,?)",
        (name, json.dumps(spec), now, now),
    )
    if commit:
        conn.commit()
    return cur.lastrowid


def list_all(conn: sqlite3.Connection) -> list[dict]:
    out = []
    for r in conn.execute("SELECT * FROM smart_playlists ORDER BY id"):
        d = dict(r)
        d["spec"] = json.loads(d["spec"])
        out.append(d)
    return out


def get(conn: sqlite3.Connection, sid: int) -> dict | None:
    r = conn.execute("SELECT * FROM smart_playlists WHERE id=?", (sid,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["spec"] = json.loads(d["spec"])
    return d


def update(
    conn: sqlite3.Connection,
    sid: int,
    name: str | None = None,
    spec: dict | None = None,
    enabled: bool | None = None,
) -> None:
    cur = get(conn, sid)
    if not cur:
        return
    conn.execute(
        "UPDATE smart_playlists SET name=?, spec=?, enabled=?, updated_at=? WHERE id=?",
        (
            name if name is not None else cur["name"],
            json.dumps(spec) if spec is not None else json.dumps(cur["spec"]),
            cur["enabled"] if enabled is None else int(enabled),
            db.utcnow(),
            sid,
        ),
    )
    conn.commit()


def delete(conn: sqlite3.Connection, sid: int) -> None:
    conn.execute("DELETE FROM smart_playlists WHERE id=?", (sid,))
    conn.execute("DELETE FROM managed_playlists WHERE kind='smart' AND key=?", (str(sid),))
    conn.commit()
