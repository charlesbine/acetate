# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["laion-clap>=1.1", "torch", "torchvision", "torchaudio", "numpy"]
# ///
"""CLAP text↔audio embeddings — natural-language retrieval over the library.

Runs in its own uv script env (torch + laion-clap never touch the app env).
The checkpoint is PINNED to the music-tuned LAION-CLAP release: vectors from
different CLAP checkpoints do NOT share a space (the DISCO-10M mistake), so
the checkpoint name is part of the embeddings row's model id.

    uv run scripts/clap_worker.py embed [--batch N] [--microbatch N] [--candidates-only]
    uv run scripts/clap_worker.py query --text "..." [--k 20]   # text search

`query` prints one JSON object: {"query": ..., "results": [{artist, title,
plays, liked, style, track_id, score}, ...]} so the CLI/agent can render it.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from acetate.embedding_guard import BUSY_EXIT, slot  # noqa: E402

DB_PATH = Path(os.environ.get("ACETATE_DB", str(REPO / "data" / "acetate.db"))).expanduser()
DATA = DB_PATH.parent
MODELS = DATA / "models"

CKPT = "music_audioset_epoch_15_esc_90.14.pt"
CKPT_URL = f"https://huggingface.co/lukewys/laion_clap/resolve/main/{CKPT}"
MODEL_NAME = "clap-music"  # if the checkpoint ever changes, change this too
DEFAULT_MICROBATCH = 8
MAX_ATTEMPTS = 3


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.executescript((REPO / "acetate" / "schema.sql").read_text())
    return conn


def load_model():
    MODELS.mkdir(parents=True, exist_ok=True)
    ckpt = MODELS / CKPT
    if not ckpt.exists():
        print(f"downloading {CKPT} (~2 GB, one-time) …", flush=True)
        urllib.request.urlretrieve(CKPT_URL, ckpt)
    import laion_clap

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(str(ckpt), verbose=False)
    return model


def mark_candidate_run(
    conn: sqlite3.Connection,
    candidate_key: str,
    status: str,
    error: str | None = None,
) -> None:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
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
        (candidate_key, MODEL_NAME, status, error[:300] if error else None, now),
    )


def mark_library_run(
    conn: sqlite3.Connection,
    recording_key: str,
    status: str,
    error: str | None = None,
) -> None:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
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
        (recording_key, MODEL_NAME, status, error[:300] if error else None, now),
    )


def cmd_embed(
    batch: int | None,
    candidates_only: bool = False,
    microbatch: int = DEFAULT_MICROBATCH,
    recording_key: str | None = None,
) -> int:
    if microbatch < 1:
        raise ValueError("microbatch must be at least 1")
    conn = connect()
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
        todo = conn.execute(
            f"SELECT {column} AS recording_key, path, ? AS kind FROM {table} a "
            f"WHERE {column}=? AND status='fetched' {candidate_filter}AND NOT EXISTS "
            f"(SELECT 1 FROM embeddings e WHERE e.recording_key=a.{column} AND model=?)",
            (kind, recording_key, MODEL_NAME),
        ).fetchall()
    elif candidates_only:
        todo = conn.execute(
            """
            SELECT a.candidate_key AS recording_key, a.path, 'candidate' AS kind
            FROM candidate_audio a
            JOIN candidate_recordings cr ON cr.candidate_key = a.candidate_key
            LEFT JOIN candidate_embedding_runs run
              ON run.candidate_key = a.candidate_key AND run.model = ?
            WHERE a.status = 'fetched' AND cr.status IN ('preview', 'embedded')
              AND NOT EXISTS (SELECT 1 FROM embedding_requests request
                              WHERE request.recording_key = a.candidate_key)
              AND NOT EXISTS (SELECT 1 FROM embeddings e
                              WHERE e.recording_key = a.candidate_key AND e.model = ?)
              AND (
                run.candidate_key IS NULL
                OR run.status = 'pending'
                OR (run.status = 'error' AND run.attempts < ?)
              )
            ORDER BY cr.score DESC
            """,
            (MODEL_NAME, MODEL_NAME, MAX_ATTEMPTS),
        ).fetchall()
    else:
        todo = conn.execute(
            """
            SELECT a.recording_key, a.path, 'library' AS kind
            FROM audio_cache a
            LEFT JOIN recording_stats rs ON rs.recording_key = a.recording_key
            LEFT JOIN embedding_runs run
              ON run.recording_key = a.recording_key AND run.model = ?
            WHERE a.status = 'fetched'
              AND NOT EXISTS (SELECT 1 FROM embedding_requests request
                              WHERE request.recording_key = a.recording_key)
              AND NOT EXISTS (SELECT 1 FROM embeddings e
                              WHERE e.recording_key = a.recording_key AND e.model = ?)
              AND (
                run.recording_key IS NULL
                OR run.status = 'pending'
                OR (run.status = 'error' AND run.attempts < ?)
              )
            ORDER BY COALESCE(rs.plays, 0) DESC
            """,
            (MODEL_NAME, MODEL_NAME, MAX_ATTEMPTS),
        ).fetchall()
    if batch:
        todo = todo[:batch]
    print(f"{len(todo)} clips to embed with {MODEL_NAME}", flush=True)
    if not todo:
        return 0

    model = load_model()
    done = errors = 0
    t0 = time.monotonic()
    for start in range(0, len(todo), microbatch):
        chunk = todo[start : start + microbatch]
        files = [str(DATA / r["path"]) for r in chunk]
        try:
            embs = model.get_audio_embedding_from_filelist(x=files, use_tensor=False)
        except Exception:
            # a bad file poisons the whole batch call — fall back to one-by-one
            embs = []
            for f in files:
                try:
                    embs.append(model.get_audio_embedding_from_filelist(x=[f], use_tensor=False)[0])
                except Exception:
                    embs.append(None)
                    errors += 1
        now_rows = []
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        for rec, emb in zip(chunk, embs):
            if emb is None:
                if rec["kind"] == "candidate":
                    mark_candidate_run(conn, rec["recording_key"], "error", "CLAP decode failed")
                else:
                    mark_library_run(conn, rec["recording_key"], "error", "CLAP decode failed")
                continue
            vec = np.asarray(emb, dtype=np.float32)
            now_rows.append((rec["recording_key"], MODEL_NAME, vec.size, vec.tobytes(), now))
            if rec["kind"] == "candidate":
                mark_candidate_run(conn, rec["recording_key"], "available")
            else:
                mark_library_run(conn, rec["recording_key"], "available")
            done += 1
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings(recording_key, model, dim, vector, updated_at) "
            "VALUES(?,?,?,?,?)",
            now_rows,
        )
        conn.commit()  # write+commit only after inference: lock held for ~ms
        rate = (start + len(chunk)) / (time.monotonic() - t0)
        print(
            f"  …{start + len(chunk)}/{len(todo)} ({rate:.1f}/s, "
            f"eta {(len(todo) - start - len(chunk)) / max(rate, 0.01) / 60:.0f} min)",
            flush=True,
        )
    print(f"embedded {done}, errors {errors}")
    return 0


def cmd_query(text: str, k: int) -> int:
    conn = connect()
    rows = conn.execute(
        "SELECT e.recording_key, e.vector FROM embeddings e "
        "JOIN recording_stats rs USING(recording_key) WHERE e.model = ?",
        (MODEL_NAME,),
    ).fetchall()
    if not rows:
        print(json.dumps({"error": f"no {MODEL_NAME} embeddings yet — run: acetate embed clap"}))
        return 1
    keys = [r["recording_key"] for r in rows]
    mat = np.vstack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9

    model = load_model()
    t = np.asarray(model.get_text_embedding([text], use_tensor=False)[0], dtype=np.float32)
    t /= np.linalg.norm(t) + 1e-9
    sims = mat @ t
    order = np.argsort(-sims)[:k]

    results = []
    for j in order:
        r = conn.execute(
            "SELECT rs.artist, rs.title, rs.plays, rs.liked, rs.rep_track_id, "
            "(SELECT style FROM recording_styles s WHERE s.recording_key = rs.recording_key "
            " AND s.rank = 1) AS style "
            "FROM recording_stats rs WHERE rs.recording_key = ?",
            (keys[j],),
        ).fetchone()
        if r:
            results.append(
                {
                    "artist": r["artist"],
                    "title": r["title"],
                    "plays": r["plays"],
                    "liked": bool(r["liked"]),
                    "style": r["style"],
                    "track_id": r["rep_track_id"],
                    "score": round(float(sims[j]), 4),
                }
            )
    print(
        json.dumps({"query": text, "model": MODEL_NAME, "searched": len(keys), "results": results})
    )
    return 0


def cmd_matrix(tracks: list[str], prompts: list[str], out: str) -> int:
    """Real prompt×track cosine matrix for the blog's CLAP demo. Repro:
    uv run scripts/clap_worker.py matrix --track "a - b" ... --prompt "..." ... --out f.json"""
    conn = connect()
    resolved = []
    for q in tracks:
        artist, _, title = (p.strip() for p in q.partition(" - "))
        row = conn.execute(
            "SELECT rs.recording_key, rs.artist, rs.title, e.vector FROM recording_stats rs "
            "JOIN embeddings e ON e.recording_key = rs.recording_key AND e.model = ? "
            "WHERE rs.artist LIKE ? AND rs.title LIKE ? ORDER BY rs.plays DESC LIMIT 1",
            (MODEL_NAME, f"%{artist}%", f"%{title or artist}%"),
        ).fetchone()
        if not row:
            print(f"skipping {q!r} (not embedded)", file=sys.stderr)
            continue
        vec = np.frombuffer(row["vector"], dtype=np.float32)
        resolved.append((row["artist"], row["title"], vec / (np.linalg.norm(vec) + 1e-9)))

    model = load_model()
    t_embs = np.asarray(model.get_text_embedding(prompts, use_tensor=False), dtype=np.float32)
    t_embs /= np.linalg.norm(t_embs, axis=1, keepdims=True) + 1e-9
    scores = [[round(float(t @ vec), 3) for _, _, vec in resolved] for t in t_embs]

    Path(out).write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "prompts": prompts,
                "tracks": [{"artist": a, "title": ti} for a, ti, _ in resolved],
                "scores": scores,
            },
            indent=1,
        )
    )
    print(f"wrote {out}: {len(prompts)}x{len(resolved)} matrix")
    return 0


def _run() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_embed = sub.add_parser("embed")
    p_embed.add_argument("--batch", type=int, default=None)
    p_embed.add_argument(
        "--microbatch",
        type=int,
        default=DEFAULT_MICROBATCH,
        help="Clips per inference call; keep bounded independently of total work.",
    )
    p_embed.add_argument("--candidates-only", action="store_true")
    p_embed.add_argument(
        "--recording-key", help="Embed only this exact recording; used by bounded requests."
    )
    p_query = sub.add_parser("query")
    p_query.add_argument("--text", required=True)
    p_query.add_argument("--k", type=int, default=20)
    p_matrix = sub.add_parser("matrix")
    p_matrix.add_argument("--track", action="append", required=True)
    p_matrix.add_argument("--prompt", action="append", required=True)
    p_matrix.add_argument("--out", required=True)
    p_text = sub.add_parser("textvec")  # raw text embeddings, for the blog's space map
    p_text.add_argument("--prompt", action="append", required=True)
    p_text.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "embed":
        return cmd_embed(args.batch, args.candidates_only, args.microbatch, args.recording_key)
    if args.cmd == "matrix":
        return cmd_matrix(args.track, args.prompt, args.out)
    if args.cmd == "textvec":
        model = load_model()
        embs = model.get_text_embedding(args.prompt, use_tensor=False)
        Path(args.out).write_text(
            json.dumps({"prompts": args.prompt, "vectors": [[float(x) for x in e] for e in embs]})
        )
        print(f"wrote {args.out}: {len(args.prompt)} text vectors")
        return 0
    return cmd_query(args.text, args.k)


def main() -> int:
    # Includes text queries/matrices: loading another CLAP beside an embedding
    # process would defeat the memory/concurrency bound.
    with slot(DATA) as acquired:
        if not acquired:
            print(json.dumps({"error": "audio inference busy; try again later"}))
            return BUSY_EXIT
        return _run()


if __name__ == "__main__":
    sys.exit(main())
