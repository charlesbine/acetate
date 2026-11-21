"""Playlist-as-region taste model — the membership engine behind curation,
discovery, and genre playlists.

Every user-authored playlist is a labeled region of the discogs-effnet
embedding space. membership(track, playlist) scores how well a recording fits
that region: half centroid cosine (the playlist's center of gravity), half
mean top-k similarity to actual members (local shape — a playlist can be a
banana, not a ball). One function, many consumers:

  additions  = high membership, not in the playlist   (gap-fill / discovery)
  misfits    = low membership, in the playlist        (curation assist)
  coherence  = the distribution of member scores      (playlist health)
  genre fit  = membership against genre-anchor playlists

Regions are cached in taste_regions by build(); members are re-read from the
latest playlist snapshot at query time (cheap, always current). evaluate()
grades the engine with member holdouts vs sampled library negatives so the
centroid/k-NN baseline has real numbers before anything fancier replaces it.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from . import db

EMBED_MODEL = "discogs-effnet"
MIN_EMBEDDED = 5  # fewer embedded members than this → no usable region
CENTROID_WEIGHT = 0.5  # membership = 0.5*centroid_cos + 0.5*mean top-k member cos
KNN_K = 5


def user_owner_id(conn: sqlite3.Connection) -> str | None:
    """The Spotify user whose playlists count as "authored": the modal owner
    across snapshotted playlists (the /me/playlists pull is dominated by your
    own). Overridable via kv_state key 'spotify_user_id' if the heuristic is
    ever wrong."""
    row = conn.execute("SELECT value FROM kv_state WHERE key='spotify_user_id'").fetchone()
    if row and row["value"]:
        return row["value"]
    row = conn.execute("""
        SELECT owner_id FROM (
            SELECT playlist_id, owner_id, MAX(backup_time)
            FROM playlist_snapshots WHERE owner_id IS NOT NULL GROUP BY playlist_id)
        GROUP BY owner_id ORDER BY COUNT(*) DESC LIMIT 1""").fetchone()
    return row["owner_id"] if row else None


def authored_playlists(conn: sqlite3.Connection, owner_id: str | None = None) -> list[dict]:
    """Latest snapshot of every playlist the user authored (owner match,
    not acetate-managed), newest-snapshot fields. These are the taste labels."""
    owner_id = owner_id or user_owner_id(conn)
    if not owner_id:
        return []
    rows = conn.execute(
        """
        SELECT ps.playlist_id, ps.name, ps.total_tracks, ps.owner_id
        FROM playlist_snapshots ps
        JOIN (SELECT playlist_id, MAX(backup_time) bt FROM playlist_snapshots
              GROUP BY playlist_id) latest
          ON latest.playlist_id = ps.playlist_id AND latest.bt = ps.backup_time
        WHERE ps.owner_id = ?
          AND ps.playlist_id NOT IN (SELECT playlist_id FROM managed_playlists)
        ORDER BY ps.name""",
        (owner_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def playlist_recordings(conn: sqlite3.Connection, playlist_id: str) -> list[str]:
    """Distinct recording_keys in the playlist's latest track snapshot."""
    bt = conn.execute(
        "SELECT MAX(backup_time) FROM playlist_track_snapshots WHERE playlist_id=?", (playlist_id,)
    ).fetchone()[0]
    if not bt:
        return []
    return [
        r[0]
        for r in conn.execute(
            """
        SELECT DISTINCT tr.recording_key
        FROM playlist_track_snapshots pts
        JOIN track_recording tr ON tr.track_id = pts.track_id
        WHERE pts.playlist_id=? AND pts.backup_time=?""",
            (playlist_id, bt),
        )
    ]


def _vectors(
    conn: sqlite3.Connection, keys: list[str], model: str = EMBED_MODEL
) -> tuple[list[str], np.ndarray]:
    """Normalized embedding rows for the subset of `keys` that have one."""
    if not keys:
        return [], np.empty((0, 0), dtype=np.float32)
    found: list[str] = []
    vecs: list[np.ndarray] = []
    for i in range(0, len(keys), 500):  # chunk to stay under SQLite's variable cap
        chunk = keys[i : i + 500]
        qm = ",".join("?" * len(chunk))
        for r in conn.execute(
            f"SELECT recording_key, vector FROM embeddings "
            f"WHERE model=? AND recording_key IN ({qm})",
            (model, *chunk),
        ):
            v = np.frombuffer(r["vector"], dtype=np.float32)
            found.append(r["recording_key"])
            vecs.append(v / (np.linalg.norm(v) + 1e-9))
    if not vecs:
        return [], np.empty((0, 0), dtype=np.float32)
    return found, np.vstack(vecs)


@dataclass
class Region:
    playlist_id: str
    name: str
    member_keys: list[str]  # embedded members (region basis)
    matrix: np.ndarray  # (n, d) normalized member vectors
    centroid: np.ndarray  # (d,) normalized
    n_members: int = 0  # all members, embedded or not
    spread: float = 0.0  # mean member cosine to centroid
    member_scores: np.ndarray = field(default=None, repr=False)  # leave-self-out


def load_region(
    conn: sqlite3.Connection, playlist_id: str, name: str | None = None, model: str = EMBED_MODEL
) -> Region | None:
    """Build the in-memory region for one playlist from its latest snapshot.
    Returns None when too few members are embedded to define a region."""
    keys = playlist_recordings(conn, playlist_id)
    member_keys, mat = _vectors(conn, keys, model)
    if len(member_keys) < MIN_EMBEDDED:
        return None
    centroid = mat.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
    region = Region(
        playlist_id=playlist_id,
        name=name or playlist_id,
        member_keys=member_keys,
        matrix=mat,
        centroid=centroid,
        n_members=len(keys),
        spread=float((mat @ centroid).mean()),
    )
    region.member_scores = _score_rows(mat, region, leave_self_out=True)
    return region


def _score_rows(
    targets: np.ndarray, region: Region, *, leave_self_out: bool = False, k: int = KNN_K
) -> np.ndarray:
    """Membership scores for target vectors: blend of centroid cosine and mean
    top-k similarity to members. leave_self_out drops each row's self-match
    (targets ARE the members) so member scores aren't inflated by the trivial 1.0."""
    sims = targets @ region.matrix.T  # (t, n)
    if leave_self_out:
        np.fill_diagonal(sims, -1.0)
    kk = min(k, sims.shape[1] - (1 if leave_self_out else 0))
    kk = max(kk, 1)
    topk = np.sort(sims, axis=1)[:, -kk:].mean(axis=1)
    centroid_sim = targets @ region.centroid
    return CENTROID_WEIGHT * centroid_sim + (1 - CENTROID_WEIGHT) * topk


def membership(
    conn: sqlite3.Connection, region: Region, keys: list[str], model: str = EMBED_MODEL
) -> dict[str, float]:
    """Membership score for each of `keys` that has an embedding (library
    recordings and embedded candidates share the embeddings table, so both work)."""
    found, mat = _vectors(conn, keys, model)
    if not found:
        return {}
    scores = _score_rows(mat, region)
    return dict(zip(found, (float(s) for s in scores)))


def build(conn: sqlite3.Connection, owner_id: str | None = None, model: str = EMBED_MODEL) -> dict:
    """(Re)build taste_regions from the latest playlist snapshots. Idempotent;
    playlists that shrank below MIN_EMBEDDED members lose their region."""
    now = db.utcnow()
    playlists = authored_playlists(conn, owner_id)
    built = skipped = 0
    seen: list[str] = []
    for pl in playlists:
        region = load_region(conn, pl["playlist_id"], pl["name"], model)
        if region is None:
            skipped += 1
            continue
        p10 = float(np.percentile(region.member_scores, 10))
        conn.execute(
            """
            INSERT INTO taste_regions(playlist_id, name, owner_id, n_members, n_embedded,
                                      dim, centroid, spread, member_p10, built_at)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(playlist_id) DO UPDATE SET
                name=excluded.name, owner_id=excluded.owner_id,
                n_members=excluded.n_members, n_embedded=excluded.n_embedded,
                dim=excluded.dim, centroid=excluded.centroid, spread=excluded.spread,
                member_p10=excluded.member_p10, built_at=excluded.built_at""",
            (
                pl["playlist_id"],
                pl["name"],
                pl["owner_id"],
                region.n_members,
                len(region.member_keys),
                region.matrix.shape[1],
                region.centroid.astype(np.float32).tobytes(),
                region.spread,
                p10,
                now,
            ),
        )
        seen.append(pl["playlist_id"])
        built += 1
    if seen:  # drop regions for playlists that no longer qualify
        qm = ",".join("?" * len(seen))
        conn.execute(f"DELETE FROM taste_regions WHERE playlist_id NOT IN ({qm})", seen)
    conn.commit()
    return {"playlists": len(playlists), "regions_built": built, "too_small": skipped}


def regions(conn: sqlite3.Connection) -> list[dict]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT playlist_id, name, n_members, n_embedded, spread, member_p10, built_at "
            "FROM taste_regions ORDER BY n_embedded DESC"
        )
    ]


def library_pool(conn: sqlite3.Connection, model: str = EMBED_MODEL) -> list[str]:
    """Embedded library recordings — the universe additions are drawn from
    (and negatives are sampled from during eval)."""
    return [
        r[0]
        for r in conn.execute(
            "SELECT e.recording_key FROM embeddings e "
            "JOIN recording_stats rs USING(recording_key) WHERE e.model=?",
            (model,),
        )
    ]


def evaluate(
    conn: sqlite3.Connection,
    owner_id: str | None = None,
    holdout_frac: float = 0.2,
    negatives_per_pos: int = 20,
    seed: int = 7,
    model: str = EMBED_MODEL,
) -> list[dict]:
    """Grade the membership engine per playlist: hold out a fraction of members,
    build the region from the rest, rank held-out positives among sampled
    library negatives. Reports precision@10 (positives in the top 10 of the
    mixed pool) and AUC (probability a positive outranks a negative)."""
    rng = random.Random(seed)
    pool = library_pool(conn, model)
    results: list[dict] = []
    for pl in authored_playlists(conn, owner_id):
        keys = playlist_recordings(conn, pl["playlist_id"])
        member_keys, mat = _vectors(conn, keys, model)
        n_hold = int(len(member_keys) * holdout_frac)
        if len(member_keys) - n_hold < MIN_EMBEDDED or n_hold < 2:
            continue
        idx = list(range(len(member_keys)))
        rng.shuffle(idx)
        hold, train = idx[:n_hold], idx[n_hold:]
        train_mat = mat[train]
        centroid = train_mat.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        region = Region(
            playlist_id=pl["playlist_id"],
            name=pl["name"],
            member_keys=[member_keys[i] for i in train],
            matrix=train_mat,
            centroid=centroid,
        )

        member_set = set(member_keys)
        negatives = rng.sample(
            [k for k in pool if k not in member_set],
            min(negatives_per_pos * n_hold, max(0, len(pool) - len(member_set))),
        )
        if not negatives:
            continue
        pos_scores = _score_rows(mat[hold], region)
        _, neg_mat = _vectors(conn, negatives, model)
        neg_scores = _score_rows(neg_mat, region)

        merged = sorted([(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores], reverse=True)
        top10 = sum(lab for _s, lab in merged[:10])
        auc = float((pos_scores[:, None] > neg_scores[None, :]).mean())
        results.append(
            {
                "playlist": pl["name"],
                "members": len(member_keys),
                "held_out": n_hold,
                "negatives": len(negatives),
                "precision_at_10": top10 / 10.0,
                "auc": auc,
            }
        )
    return results


def additions(
    conn: sqlite3.Connection,
    region: Region,
    *,
    liked_only: bool = True,
    limit: int = 25,
    model: str = EMBED_MODEL,
) -> list[dict]:
    """Top library recordings by membership that are NOT in the playlist —
    the gap-fill / "belongs here" list. liked_only keeps it to tracks you've
    already vouched for; the discovery module handles outside-library."""
    member_set = set(playlist_recordings(conn, region.playlist_id))
    where = "AND rs.liked=1" if liked_only else ""
    pool = [
        r[0]
        for r in conn.execute(
            f"SELECT e.recording_key FROM embeddings e "
            f"JOIN recording_stats rs USING(recording_key) WHERE e.model=? {where}",
            (model,),
        )
    ]
    pool = [k for k in pool if k not in member_set]
    scores = membership(conn, region, pool, model)
    top = sorted(scores.items(), key=lambda kv: -kv[1])[:limit]
    out = []
    for key, score in top:
        rs = conn.execute(
            "SELECT artist, title, plays, rep_track_id FROM recording_stats WHERE recording_key=?",
            (key,),
        ).fetchone()
        if rs:
            out.append(
                {
                    "recording_key": key,
                    "artist": rs["artist"],
                    "title": rs["title"],
                    "plays": rs["plays"],
                    "track_id": rs["rep_track_id"],
                    "score": round(score, 4),
                }
            )
    return out


def misfits(conn: sqlite3.Connection, region: Region, limit: int = 10) -> list[dict]:
    """Members that fit the region worst (lowest leave-self-out membership),
    with how far below the playlist's typical score they sit."""
    scores = region.member_scores
    mean, std = float(scores.mean()), float(scores.std() + 1e-9)
    order = np.argsort(scores)[:limit]
    out = []
    for i in order:
        key = region.member_keys[int(i)]
        rs = conn.execute(
            "SELECT artist, title, plays, rep_track_id FROM recording_stats WHERE recording_key=?",
            (key,),
        ).fetchone()
        out.append(
            {
                "recording_key": key,
                "artist": rs["artist"] if rs else None,
                "title": rs["title"] if rs else None,
                "track_id": rs["rep_track_id"] if rs else None,
                "score": round(float(scores[i]), 4),
                "z": round((float(scores[i]) - mean) / std, 2),
            }
        )
    return out
