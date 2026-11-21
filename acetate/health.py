"""Playlist Health — curation assist for user-authored playlists.

Three independent coherence lenses (a playlist can be sonically tight but
collaboratively scattered, which is itself a finding):

  sonic     mean member cosine to the region centroid (taste_regions.spread)
  style     entropy of members' rank-1 discogs400 styles (low = one sound)
  social    fraction of members with a Last.fm co-listening edge to another member

Plus the actionable outputs: misfits (members that don't fit, with z-scores),
gap-fill (liked library tracks that belong here), and a 2-means split check
("this is actually two playlists"). Misfit/gap-fill suggestions land in the
proposals ledger — acetate never edits an authored playlist directly (the
safety gate stands); you apply accepted changes yourself or via mirrors.
"""

from __future__ import annotations

import math
import sqlite3

import numpy as np

from . import proposals, taste

MISFIT_Z = -1.5  # members this far below the playlist mean are flagged
SPLIT_MIN_MEMBERS = 12  # don't suggest splitting tiny playlists
SPLIT_GAIN = 0.06  # 2-means must tighten mean cohesion by this much


def style_entropy(
    conn: sqlite3.Connection, member_keys: list[str]
) -> tuple[float, list[tuple[str, int]]]:
    """Normalized entropy (0=one style, 1=uniform) of members' top styles,
    plus the style histogram for display."""
    counts: dict[str, int] = {}
    for i in range(0, len(member_keys), 500):
        chunk = member_keys[i : i + 500]
        qm = ",".join("?" * len(chunk))
        for r in conn.execute(
            f"SELECT style FROM recording_styles WHERE rank=1 AND recording_key IN ({qm})", chunk
        ):
            s = r["style"].split("---")[-1]
            counts[s] = counts.get(s, 0) + 1
    total = sum(counts.values())
    if total == 0 or len(counts) < 2:
        return 0.0, sorted(counts.items(), key=lambda kv: -kv[1])
    ent = -sum((c / total) * math.log(c / total) for c in counts.values())
    return ent / math.log(len(counts)), sorted(counts.items(), key=lambda kv: -kv[1])


def colisten_density(conn: sqlite3.Connection, member_keys: list[str]) -> float:
    """Fraction of members with at least one lastfm_similar edge to another
    member — do people who play one of these also play the others?"""
    if len(member_keys) < 2:
        return 0.0
    member_set = set(member_keys)
    # which members have an edge landing on another member (either direction)
    linked: set[str] = set()
    for i in range(0, len(member_keys), 500):
        chunk = member_keys[i : i + 500]
        qm = ",".join("?" * len(chunk))
        for r in conn.execute(
            f"SELECT recording_key src, sim_recording_key dst FROM lastfm_similar "
            f"WHERE recording_key IN ({qm}) AND sim_recording_key IS NOT NULL",
            chunk,
        ):
            if r["dst"] in member_set:
                linked.add(r["src"])
                linked.add(r["dst"])
    connected = len(linked & member_set)
    return connected / len(member_keys)


def split_check(region: taste.Region) -> dict | None:
    """2-means over member vectors: if two clusters are each markedly tighter
    than the whole, the playlist is probably two playlists. Deterministic init
    (farthest pair) so re-runs agree."""
    mat = region.matrix
    n = mat.shape[0]
    if n < SPLIT_MIN_MEMBERS:
        return None
    sims = mat @ mat.T
    i, j = np.unravel_index(np.argmin(sims), sims.shape)  # farthest apart pair
    c = np.vstack([mat[i], mat[j]])
    assign = np.zeros(n, dtype=int)
    for _ in range(20):
        assign_new = (mat @ c.T).argmax(axis=1)
        if (assign_new == assign).all() and _ > 0:
            break
        assign = assign_new
        for g in (0, 1):
            if (assign == g).any():
                m = mat[assign == g].mean(axis=0)
                c[g] = m / (np.linalg.norm(m) + 1e-9)
    sizes = [(assign == g).sum() for g in (0, 1)]
    if min(sizes) < 4:
        return None
    whole = float((mat @ region.centroid).mean())
    parts = [float((mat[assign == g] @ c[g]).mean()) for g in (0, 1)]
    gain = min(parts) - whole
    if gain < SPLIT_GAIN:
        return None
    return {
        "gain": round(gain, 3),
        "groups": [
            {
                "size": int(sizes[g]),
                "cohesion": round(parts[g], 3),
                "member_keys": [region.member_keys[x] for x in np.where(assign == g)[0]],
            }
            for g in (0, 1)
        ],
    }


def report(conn: sqlite3.Connection, playlist_id: str, name: str | None = None) -> dict | None:
    """The full health readout for one playlist."""
    region = taste.load_region(conn, playlist_id, name)
    if region is None:
        return None
    ent, styles = style_entropy(conn, region.member_keys)
    density = colisten_density(conn, region.member_keys)
    mis = [m for m in taste.misfits(conn, region, limit=15) if m["z"] <= MISFIT_Z]
    split = split_check(region)
    return {
        "playlist_id": playlist_id,
        "name": region.name,
        "members": region.n_members,
        "embedded": len(region.member_keys),
        "sonic_cohesion": round(region.spread, 3),
        "style_entropy": round(ent, 3),
        "styles": styles[:8],
        "colisten_density": round(density, 3),
        "misfits": mis,
        "split": split,
    }


def overview(conn: sqlite3.Connection) -> list[dict]:
    """One row per taste region: the health list page (uses cached region stats;
    the per-playlist page computes the expensive parts)."""
    return taste.regions(conn)


def scan(
    conn: sqlite3.Connection,
    playlist_id: str,
    name: str | None = None,
    gapfill_limit: int = 15,
    *,
    include_misfits: bool = True,
    include_additions: bool = True,
) -> dict:
    """Run misfit + gap-fill for one playlist and file the results as proposals.
    Returns the counts; the /review queue is where they get judged."""
    region = taste.load_region(conn, playlist_id, name)
    if region is None:
        return {"proposed": 0, "skipped": "too few embedded members"}
    proposed = additions = removals = 0
    if include_misfits:
        for m in taste.misfits(conn, region, limit=15):
            if m["z"] > MISFIT_Z:
                continue
            if proposals.propose(
                conn,
                playlist_id=playlist_id,
                playlist_name=region.name,
                action="remove",
                kind="library",
                recording_key=m["recording_key"],
                track_id=m["track_id"],
                artist=m["artist"],
                title=m["title"],
                score=m["score"],
                source="health",
                reasons=[
                    f"fits the playlist worst (z={m['z']}, "
                    f"membership {m['score']} vs region spread {round(region.spread, 3)})"
                ],
            ):
                proposed += 1
                removals += 1
    if include_additions:
        for a in taste.additions(conn, region, limit=gapfill_limit):
            if proposals.propose(
                conn,
                playlist_id=playlist_id,
                playlist_name=region.name,
                action="add",
                kind="library",
                recording_key=a["recording_key"],
                track_id=a["track_id"],
                artist=a["artist"],
                title=a["title"],
                score=a["score"],
                source="health",
                reasons=[
                    f"liked, not in the playlist, membership {a['score']} ({a['plays']} plays)"
                ],
            ):
                proposed += 1
                additions += 1
    conn.commit()
    return {
        "proposed": proposed,
        "additions": additions,
        "removals": removals,
        "playlist": region.name,
    }
