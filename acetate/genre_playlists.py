"""Genre playlists — the north-star build, framed as confidence tiers.

A genre is classified by three independent signals, combined per recording:

  region   taste membership against anchor playlists — YOUR definition of the
           genre ("melodic techno" = your own region of embedding space, not
           anyone's taxonomy). Anchors auto-detect by name match, or pass them.
  style    discogs400 activations whose label matches the genre's keywords
           (content signal, dense — covers everything embedded)
  lastfm   track_genres tags (human signal, sparse until the backfill lands)

High-confidence matches append to the 🎧 genre playlist (append-only via
managed_playlist_tracks, so nothing ever silently leaves); borderline ones go
to the proposals ledger for review — and each verdict is a future training label.
"""

from __future__ import annotations

import sqlite3

from . import db, proposals, taste

AUTO_THRESHOLD = 0.70  # confidence ≥ this AND ≥2 signals → auto-append
REVIEW_THRESHOLD = 0.45  # confidence ≥ this → proposals queue
REGION_W, STYLE_W, LASTFM_W = 0.5, 0.25, 0.25
REGION_FLOOR = 0.55  # membership below this contributes nothing


def anchor_playlists(conn: sqlite3.Connection, genre: str) -> list[dict]:
    """Authored playlists whose name contains the genre string — the labeled
    regions that define what this genre means to YOU."""
    g = genre.lower()
    return [pl for pl in taste.authored_playlists(conn) if g in (pl["name"] or "").lower()]


def _universe(conn: sqlite3.Connection, min_plays: int = 5) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT recording_key, rep_track_id, artist, title FROM recording_stats "
        "WHERE liked=1 OR plays>=?",
        (min_plays,),
    ).fetchall()


def _lastfm_signal(conn: sqlite3.Connection, tags: list[str]) -> dict[str, float]:
    """recording_key → strongest matching Last.fm tag weight, scaled to [0,1]."""
    qs = ",".join("?" * len(tags))
    out: dict[str, float] = {}
    for r in conn.execute(
        f"""
        SELECT tr.recording_key, MAX(g.confidence) conf
        FROM track_genres g JOIN track_recording tr ON tr.track_id = g.track_id
        WHERE g.source='lastfm' AND lower(g.genre) IN ({qs})
        GROUP BY tr.recording_key""",
        [t.lower() for t in tags],
    ):
        out[r["recording_key"]] = min(1.0, float(r["conf"] or 0) / 100.0)
    return out


def _style_signal(conn: sqlite3.Connection, keywords: list[str]) -> dict[str, float]:
    """recording_key → summed top-8 discogs400 activation over matching styles."""
    like = " OR ".join("lower(style) LIKE ?" for _ in keywords)
    out: dict[str, float] = {}
    for r in conn.execute(
        f"SELECT recording_key, SUM(score) s FROM recording_styles "
        f"WHERE {like} GROUP BY recording_key",
        [f"%{k.lower()}%" for k in keywords],
    ):
        out[r["recording_key"]] = min(1.0, float(r["s"]))
    return out


def _region_signal(
    conn: sqlite3.Connection, anchors: list[dict], keys: list[str]
) -> dict[str, float]:
    """recording_key → best anchor-playlist membership, rescaled so scores at
    or below REGION_FLOOR contribute nothing (raw cosine blends sit high)."""
    best: dict[str, float] = {}
    for pl in anchors:
        region = taste.load_region(conn, pl["playlist_id"], pl["name"])
        if region is None:
            continue
        for key, score in taste.membership(conn, region, keys).items():
            scaled = max(0.0, (score - REGION_FLOOR) / (1 - REGION_FLOOR))
            if scaled > best.get(key, 0.0):
                best[key] = scaled
    return best


def classify(
    conn: sqlite3.Connection,
    genre: str,
    *,
    tags: list[str] | None = None,
    style_keywords: list[str] | None = None,
    anchors: list[dict] | None = None,
    min_plays: int = 5,
) -> list[dict]:
    """Score the library universe for one genre. Returns rows with per-signal
    scores, blended confidence, signal count, and tier ('auto'|'review'|None)."""
    tags = tags or [genre]
    # discogs400 labels are taxonomy heads ("Techno", "Progressive House"), so a
    # multi-word genre also matches on its individual words
    style_keywords = style_keywords or [genre, *(w for w in genre.split() if len(w) >= 4)]
    anchors = anchors if anchors is not None else anchor_playlists(conn, genre)
    universe = _universe(conn, min_plays)
    keys = [r["recording_key"] for r in universe]

    lastfm = _lastfm_signal(conn, tags)
    style = _style_signal(conn, style_keywords)
    region = _region_signal(conn, anchors, keys)

    out = []
    for r in universe:
        key = r["recording_key"]
        sl, ss, sr = lastfm.get(key, 0.0), style.get(key, 0.0), region.get(key, 0.0)
        n_signals = sum(1 for s in (sl, ss, sr) if s > 0.05)
        if n_signals == 0:
            continue
        conf = REGION_W * sr + STYLE_W * ss + LASTFM_W * sl
        tier = None
        if conf >= AUTO_THRESHOLD and n_signals >= 2:
            tier = "auto"
        elif conf >= REVIEW_THRESHOLD:
            tier = "review"
        if tier:
            out.append(
                {
                    "recording_key": key,
                    "track_id": r["rep_track_id"],
                    "artist": r["artist"],
                    "title": r["title"],
                    "lastfm": round(sl, 3),
                    "style": round(ss, 3),
                    "region": round(sr, 3),
                    "confidence": round(conf, 3),
                    "signals": n_signals,
                    "tier": tier,
                }
            )
    out.sort(key=lambda d: -d["confidence"])
    return out


def sync(
    conn: sqlite3.Connection,
    sp,
    genre: str,
    *,
    tags: list[str] | None = None,
    style_keywords: list[str] | None = None,
    min_plays: int = 5,
    classify_rows: list[dict] | None = None,
    review_limit: int = 50,
) -> dict:
    """Classify, append auto-tier tracks to the 🎧 genre playlist (append-only —
    nothing is ever removed automatically), and file the top review-tier rows as
    proposals (capped — the borderline tail can be thousands deep)."""
    from . import playlists

    rows = (
        classify_rows
        if classify_rows is not None
        else classify(conn, genre, tags=tags, style_keywords=style_keywords, min_plays=min_plays)
    )
    kind, key, now = "genre", genre.lower(), db.utcnow()
    name = f"🎧 {genre.title()}"

    # append-only membership: everything ever added + this run's auto tier
    prior = [
        r["track_id"]
        for r in conn.execute(
            "SELECT track_id FROM managed_playlist_tracks WHERE kind=? AND key=? "
            "ORDER BY first_added_at",
            (kind, key),
        )
    ]
    auto = [r for r in rows if r["tier"] == "auto"]
    new = [r for r in auto if r["track_id"] not in set(prior)]
    track_ids = prior + [r["track_id"] for r in new]

    reviewed = 0
    # proposals key on the stable logical target, not the Spotify id, so verdicts
    # survive the playlist being created later (or recreated)
    playlist_id = f"genre:{key}"
    review_rows = [r for r in rows if r["tier"] == "review"][:review_limit]
    for r in review_rows:
        if proposals.propose(
            conn,
            playlist_id=playlist_id,
            playlist_name=name,
            action="add",
            kind="library",
            recording_key=r["recording_key"],
            track_id=r["track_id"],
            artist=r["artist"],
            title=r["title"],
            score=r["confidence"],
            source="genre",
            reasons=[
                f"{genre}: confidence {r['confidence']} from {r['signals']} signals "
                f"(region {r['region']}, style {r['style']}, lastfm {r['lastfm']})"
            ],
        ):
            reviewed += 1

    if not track_ids:
        conn.commit()
        return {"name": name, "tracks": 0, "skipped": True, "review_proposed": reviewed}

    pid = playlists._get_or_create(
        conn,
        sp,
        kind,
        key,
        name,
        f"Your {genre} — auto-classified from your own playlists, sound, and tags. "
        "Append-only; auto-generated by acetate.",
    )
    playlists._set_tracks(conn, sp, pid, track_ids)
    for r in new:
        conn.execute(
            "INSERT INTO managed_playlist_tracks(kind,key,track_id,first_added_at,"
            "last_surfaced_at,surface_count,currently_in) VALUES(?,?,?,?,?,1,1) "
            "ON CONFLICT(kind,key,track_id) DO UPDATE SET currently_in=1",
            (kind, key, r["track_id"], now, now),
        )
    conn.commit()
    return {
        "name": name,
        "tracks": len(track_ids),
        "appended": len(new),
        "review_proposed": reviewed,
        "playlist_id": pid,
    }
