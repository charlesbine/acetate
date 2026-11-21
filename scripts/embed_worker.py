# /// script
# requires-python = ">=3.14"
# dependencies = ["essentia-tensorflow", "numpy"]
# ///
"""Embed cached preview clips with discogs-effnet (+ the discogs400 style head).

Runs in its OWN uv script env (Python 3.14 — the only version with a macOS
arm64 essentia-tensorflow wheel) so the heavy TF deps never touch the app env
or the Docker image. Invoke via `acetate embed run` or directly:

    uv run scripts/embed_worker.py [--batch N] [--candidates-only]

Reads fetched library audio by default, or candidate_audio with
`--candidates-only`, and writes:
  embeddings(recording_key, 'discogs-effnet', 1280, mean-pooled float32 blob)
  embeddings(recording_key, 'discogs400',      400, mean activations blob)
  recording_styles(recording_key, rank 1..8, style, score)

Resumable and safe to re-run; commits in batches. Model weights auto-download
to data/models/ on first run (CC BY-NC-ND 4.0 — personal use only).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from acetate.embedding_guard import BUSY_EXIT, slot  # noqa: E402

DB_PATH = Path(os.environ.get("ACETATE_DB", str(REPO / "data" / "acetate.db"))).expanduser()
DATA = DB_PATH.parent
MODELS = DATA / "models"

MODEL_URLS = {
    "discogs-effnet-bs64-1.pb": "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",
    "genre_discogs400-discogs-effnet-1.pb": "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb",
    "genre_discogs400-discogs-effnet-1.json": "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json",
}

EMBED_MODEL = "discogs-effnet"
STYLE_MODEL = "discogs400"
TOP_STYLES = 8
MAX_ATTEMPTS = 3


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_models() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    for name, url in MODEL_URLS.items():
        dest = MODELS / name
        if not dest.exists():
            print(f"downloading {name} …", flush=True)
            urllib.request.urlretrieve(url, dest)


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.executescript((REPO / "acetate" / "schema.sql").read_text())
    return conn


def todo_rows(
    conn: sqlite3.Connection,
    batch: int | None,
    candidates_only: bool,
    recording_key: str | None = None,
):
    """Fetched clips lacking EffNet/style vectors, ordered by signal."""
    if recording_key is not None:
        candidate_filter = (
            "AND EXISTS (SELECT 1 FROM candidate_recordings cr "
            "WHERE cr.candidate_key=a.candidate_key "
            "AND cr.status NOT IN ('duplicate','dismissed')) "
            if candidates_only
            else ""
        )
        table, column, kind = (
            ("candidate_audio", "candidate_key", "candidate")
            if candidates_only
            else ("audio_cache", "recording_key", "library")
        )
        return conn.execute(
            f"SELECT {column} AS recording_key, path, ? AS kind FROM {table} a "
            f"WHERE {column}=? AND status='fetched' {candidate_filter}AND ("
            f"NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.recording_key=a.{column} AND model=?) "
            f"OR NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.recording_key=a.{column} AND model=?))",
            (kind, recording_key, EMBED_MODEL, STYLE_MODEL),
        ).fetchall()
    if candidates_only:
        rows = conn.execute(
            """
            SELECT a.candidate_key AS recording_key, a.path, 'candidate' AS kind
            FROM candidate_audio a
            JOIN candidate_recordings cr ON cr.candidate_key = a.candidate_key
            LEFT JOIN candidate_embedding_runs run
              ON run.candidate_key = a.candidate_key AND run.model = ?
            LEFT JOIN candidate_embedding_runs style_run
              ON style_run.candidate_key = a.candidate_key AND style_run.model = ?
            WHERE a.status = 'fetched' AND cr.status IN ('preview', 'embedded')
              AND NOT EXISTS (SELECT 1 FROM embedding_requests request
                              WHERE request.recording_key = a.candidate_key)
              AND (
                (
                  NOT EXISTS (SELECT 1 FROM embeddings e
                              WHERE e.recording_key = a.candidate_key AND e.model = ?)
                  AND (
                    run.candidate_key IS NULL
                    OR run.status = 'pending'
                    OR (run.status = 'error' AND run.attempts < ?)
                  )
                )
                OR (
                  NOT EXISTS (SELECT 1 FROM embeddings e
                              WHERE e.recording_key = a.candidate_key AND e.model = ?)
                  AND (
                    style_run.candidate_key IS NULL
                    OR style_run.status = 'pending'
                    OR (style_run.status = 'error' AND style_run.attempts < ?)
                  )
                )
              )
            ORDER BY cr.score DESC
            """,
            (
                EMBED_MODEL,
                STYLE_MODEL,
                EMBED_MODEL,
                MAX_ATTEMPTS,
                STYLE_MODEL,
                MAX_ATTEMPTS,
            ),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT a.recording_key, a.path, 'library' AS kind
            FROM audio_cache a
            LEFT JOIN recording_stats rs ON rs.recording_key = a.recording_key
            LEFT JOIN embedding_runs run
              ON run.recording_key = a.recording_key AND run.model = ?
            LEFT JOIN embedding_runs style_run
              ON style_run.recording_key = a.recording_key AND style_run.model = ?
            WHERE a.status = 'fetched'
              AND NOT EXISTS (SELECT 1 FROM embedding_requests request
                              WHERE request.recording_key = a.recording_key)
              AND (
                (
                  NOT EXISTS (SELECT 1 FROM embeddings e
                              WHERE e.recording_key = a.recording_key AND e.model = ?)
                  AND (
                    run.recording_key IS NULL
                    OR run.status = 'pending'
                    OR (run.status = 'error' AND run.attempts < ?)
                  )
                )
                OR (
                  NOT EXISTS (SELECT 1 FROM embeddings e
                              WHERE e.recording_key = a.recording_key AND e.model = ?)
                  AND (
                    style_run.recording_key IS NULL
                    OR style_run.status = 'pending'
                    OR (style_run.status = 'error' AND style_run.attempts < ?)
                  )
                )
              )
            ORDER BY COALESCE(rs.plays, 0) DESC
            """,
            (
                EMBED_MODEL,
                STYLE_MODEL,
                EMBED_MODEL,
                MAX_ATTEMPTS,
                STYLE_MODEL,
                MAX_ATTEMPTS,
            ),
        ).fetchall()
    return rows[:batch] if batch else rows


def mark_candidate_run(
    conn: sqlite3.Connection,
    candidate_key: str,
    model: str,
    status: str,
    error: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO candidate_embedding_runs(
            candidate_key, model, status, attempts, error, updated_at
        )
        VALUES(?, ?, ?, 1, ?, ?)
        ON CONFLICT(candidate_key, model) DO UPDATE SET
            status=excluded.status,
            attempts=candidate_embedding_runs.attempts + 1,
            error=excluded.error,
            updated_at=excluded.updated_at
        """,
        (candidate_key, model, status, error[:300] if error else None, utcnow()),
    )


def mark_library_run(
    conn: sqlite3.Connection,
    recording_key: str,
    model: str,
    status: str,
    error: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO embedding_runs(recording_key, model, status, attempts, error, updated_at)
        VALUES(?, ?, ?, 1, ?, ?)
        ON CONFLICT(recording_key, model) DO UPDATE SET
            status=excluded.status,
            attempts=embedding_runs.attempts + 1,
            error=excluded.error,
            updated_at=excluded.updated_at
        """,
        (recording_key, model, status, error[:300] if error else None, utcnow()),
    )


def _run() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--candidates-only", action="store_true")
    ap.add_argument(
        "--recording-key", help="Embed only this exact recording; used by bounded requests."
    )
    args = ap.parse_args()

    conn = connect()
    todo = todo_rows(conn, args.batch, args.candidates_only, args.recording_key)
    label = "candidate clips" if args.candidates_only else "clips"
    print(f"{len(todo)} {label} to embed", flush=True)
    if not todo:
        return 0
    ensure_models()
    # Import late: TF logs a bit and takes ~1s; do it after arg/model checks.
    from essentia.standard import MonoLoader, TensorflowPredict2D, TensorflowPredictEffnetDiscogs

    effnet = TensorflowPredictEffnetDiscogs(
        graphFilename=str(MODELS / "discogs-effnet-bs64-1.pb"), output="PartitionedCall:1"
    )
    head = TensorflowPredict2D(
        graphFilename=str(MODELS / "genre_discogs400-discogs-effnet-1.pb"),
        input="serving_default_model_Placeholder",
        output="PartitionedCall:0",
    )
    labels = json.loads((MODELS / "genre_discogs400-discogs-effnet-1.json").read_text())["classes"]

    done = errors = 0
    t0 = time.monotonic()

    for i, row in enumerate(todo):
        path = DATA / row["path"]
        try:
            audio = MonoLoader(filename=str(path), sampleRate=16000, resampleQuality=4)()
            if len(audio) < 16000 * 3:  # under ~3s can't fill one model patch
                raise ValueError(f"clip too short ({len(audio) / 16000:.1f}s)")
            emb = effnet(audio)  # (patches, 1280)
            acts = head(emb)  # (patches, 400)
            emb_mean = emb.mean(axis=0).astype(np.float32)
            acts_mean = acts.mean(axis=0).astype(np.float32)
            now = utcnow()
            conn.execute(
                "INSERT OR IGNORE INTO embeddings(recording_key, model, dim, vector, updated_at) "
                "VALUES(?,?,?,?,?)",
                (row["recording_key"], EMBED_MODEL, emb_mean.size, emb_mean.tobytes(), now),
            )
            conn.execute(
                "INSERT OR IGNORE INTO embeddings(recording_key, model, dim, vector, updated_at) "
                "VALUES(?,?,?,?,?)",
                (row["recording_key"], STYLE_MODEL, acts_mean.size, acts_mean.tobytes(), now),
            )
            conn.execute(
                "DELETE FROM recording_styles WHERE recording_key = ?", (row["recording_key"],)
            )
            top = np.argsort(-acts_mean)[:TOP_STYLES]
            conn.executemany(
                "INSERT INTO recording_styles(recording_key, rank, style, score) VALUES(?,?,?,?)",
                [
                    (row["recording_key"], r + 1, labels[j], float(acts_mean[j]))
                    for r, j in enumerate(top)
                ],
            )
            if row["kind"] == "candidate":
                mark_candidate_run(conn, row["recording_key"], EMBED_MODEL, "available")
                mark_candidate_run(conn, row["recording_key"], STYLE_MODEL, "available")
            else:
                mark_library_run(conn, row["recording_key"], EMBED_MODEL, "available")
                mark_library_run(conn, row["recording_key"], STYLE_MODEL, "available")
            done += 1
        except Exception as e:  # bad download, decode failure — park it, keep going
            if row["kind"] == "candidate":
                mark_candidate_run(conn, row["recording_key"], EMBED_MODEL, "error", f"embed: {e}")
                mark_candidate_run(conn, row["recording_key"], STYLE_MODEL, "error", f"embed: {e}")
            else:
                mark_library_run(conn, row["recording_key"], EMBED_MODEL, "error", f"embed: {e}")
                mark_library_run(conn, row["recording_key"], STYLE_MODEL, "error", f"embed: {e}")
            errors += 1
        # per-item commit: an open write txn holds the WAL write lock; the fetch and
        # mbid backfills write concurrently, so never hold it across inference calls
        conn.commit()
        if i % 50 == 0:
            rate = (i + 1) / (time.monotonic() - t0)
            print(
                f"  …{i}/{len(todo)} ({rate:.1f}/s, eta {(len(todo) - i) / rate / 60:.0f} min)",
                flush=True,
            )
    conn.commit()
    print(f"embedded {done}, errors {errors}")
    return 0


def main() -> int:
    with slot(DATA) as acquired:
        if not acquired:
            print("audio inference busy; try again later", flush=True)
            return BUSY_EXIT
        return _run()


if __name__ == "__main__":
    sys.exit(main())
