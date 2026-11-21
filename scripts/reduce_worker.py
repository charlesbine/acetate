# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["umap-learn>=0.5", "scikit-learn>=1.4", "numpy"]
# ///
"""Project the embedding space to 3D for the Hyperspace page.

Runs in its own uv script env (numba — under umap-learn — trails the newest
Python, so this pins to 3.12/3.13 while the embed worker needs 3.14). Invoke
via `acetate embed reduce` or directly:

    uv run scripts/reduce_worker.py

Reads all 'discogs-effnet' vectors, fits UMAP → 3D (cosine metric), clusters
the projected points with HDBSCAN, and rewrites hyperspace_points. Re-run
whenever embeddings change; coordinates are NOT stable across runs (UMAP is
stochastic even seeded when the input set changes) — the page reloads fully.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
SPACES = {"effnet": "discogs-effnet", "clap": "clap-music"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", choices=list(SPACES), default="effnet")
    args = ap.parse_args()
    model = SPACES[args.space]

    conn = sqlite3.connect(DATA / "acetate.db", timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    rows = conn.execute(
        "SELECT e.recording_key, e.dim, e.vector FROM embeddings e "
        "JOIN recording_stats rs USING(recording_key) WHERE e.model = ?",
        (model,),
    ).fetchall()
    if len(rows) < 50:
        print(f"only {len(rows)} embeddings — need at least 50 for a meaningful map")
        return 1

    keys = [r[0] for r in rows]
    mat = np.vstack([np.frombuffer(r[2], dtype=np.float32).reshape(r[1]) for r in rows])
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    print(f"fitting UMAP on {mat.shape[0]} x {mat.shape[1]} …", flush=True)

    import umap  # slow import (numba JIT) — after the data checks

    t0 = time.monotonic()
    coords = umap.UMAP(
        n_components=3, metric="cosine", n_neighbors=30, min_dist=0.1, random_state=42
    ).fit_transform(mat)
    print(f"display UMAP done in {time.monotonic() - t0:.0f}s; clustering …", flush=True)

    # Cluster in a HIGHER-dim UMAP, not the display 3D: squeezing to 3D creates
    # bridges between regions, and past ~10k points HDBSCAN merges everything
    # into one mega-cluster there. 8D keeps neighborhoods separable; 'leaf'
    # selection extracts fine-grained clusters instead of favoring the blob.
    t0 = time.monotonic()
    c8 = umap.UMAP(
        n_components=8, metric="cosine", n_neighbors=30, min_dist=0.0, random_state=42
    ).fit_transform(mat)
    from sklearn.cluster import HDBSCAN

    # 'leaf' selection, deliberately: 'eom' is catastrophically unstable at this
    # density — a few hundred extra points can bridge everything into one
    # mega-cluster (observed twice). Leaf extracts the fine-grained cores and
    # stays stable as the library grows.
    labels = HDBSCAN(
        min_cluster_size=max(25, len(keys) // 300), min_samples=5, cluster_selection_method="leaf"
    ).fit_predict(c8)
    # HDBSCAN's ~55% "noise" here is connective tissue between regions, which
    # reads as a gray fog on the map. Attach each noise point to its nearest
    # clustered neighbor (approximate_predict-style): clusters stay honest
    # cores, boundaries get the nearest core's color.
    noise = float(np.mean(labels == -1))
    noise_idx = np.where(labels == -1)[0]
    core_idx = np.where(labels >= 0)[0]
    if len(core_idx) and len(noise_idx):
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=1).fit(c8[core_idx])
        _, ni = nn.kneighbors(c8[noise_idx])
        labels[noise_idx] = labels[core_idx[ni[:, 0]]]
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(
        f"cluster UMAP+HDBSCAN done in {time.monotonic() - t0:.0f}s "
        f"({noise:.0%} boundary points attached to nearest core)",
        flush=True,
    )

    # Center + scale to a stable display range so the page needs no per-load fit.
    coords = coords - coords.mean(axis=0)
    coords = coords / (np.abs(coords).max() + 1e-9) * 100

    now = datetime.now(timezone.utc).isoformat()
    conn.execute("DELETE FROM hyperspace_points WHERE space = ?", (args.space,))
    conn.executemany(
        "INSERT INTO hyperspace_points(space, recording_key, x, y, z, cluster, updated_at) "
        "VALUES(?,?,?,?,?,?,?)",
        [
            (args.space, k, float(x), float(y), float(z), int(c), now)
            for k, (x, y, z), c in zip(keys, coords, labels)
        ],
    )
    conn.commit()
    print(
        f"wrote {len(keys)} points to space '{args.space}', "
        f"{n_clusters} clusters ({noise:.0%} noise)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
