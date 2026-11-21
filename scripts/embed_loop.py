"""Serve bounded ad hoc requests and the existing candidate backfill, serially.

In the sidecar: python scripts/embed_loop.py
Locally: uv run python scripts/embed_loop.py --once --local
The local flag launches the existing PEP-723 worker environments through uv.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from acetate import db, embedding_requests as requests  # noqa: E402
from acetate.embedding_guard import BUSY_EXIT, slot  # noqa: E402


def heartbeat(conn, state: str) -> None:
    conn.execute(
        "INSERT INTO kv_state(key,value,updated_at) VALUES('embedding_worker',?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",
        (state, db.utcnow()),
    )
    conn.commit()


def worker_command(script: str, *args: str, local: bool = False) -> list[str]:
    prefix = ["uv", "run"] if local else [sys.executable]
    return [*prefix, str(REPO / "scripts" / script), *args]


def run_stage(command: list[str]) -> int:
    # Killing the process group also stops a Python child launched by local uv.
    # Waiting for it before returning keeps timeout recovery inside the lock contract.
    with subprocess.Popen(command, start_new_session=True) as child:
        try:
            return child.wait(timeout=requests.STAGE_TIMEOUT)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise


def process_one(conn, *, local: bool = False) -> bool:
    row = conn.execute(
        "SELECT * FROM embedding_requests WHERE status='queued' ORDER BY requested_at,id LIMIT 1"
    ).fetchone()
    if row is None:
        return False
    key = row["recording_key"]
    conn.execute(
        "UPDATE embedding_requests SET status='running',attempts=attempts+1,updated_at=? WHERE id=?",
        (db.utcnow(), row["id"]),
    )
    heartbeat(conn, "running")
    state, error = "available", None
    try:
        if requests.missing_models(conn, key):
            requests.prepare_audio(conn, row)
        commands = [
            ("EffNet", "embed_worker.py", []),
            ("CLAP", "clap_worker.py", ["embed", "--microbatch=1"]),
        ]
        for stage, script, args in commands:
            needed = requests.missing_models(conn, key)
            if not needed or (stage == "EffNet" and not set(needed) & set(requests.MODELS[:2])):
                continue
            if stage == "CLAP" and "clap-music" not in needed:
                continue
            if row["kind"] == "candidate":
                requests.require_eligible_candidate(conn, key)
                args = [*args, "--candidates-only"]
            result = run_stage(
                worker_command(script, *args, "--recording-key", key, "--batch=1", local=local)
            )
            if result == BUSY_EXIT:
                # Another CLI/background process owns the shared inference slot.
                # Yield without spending a failed attempt or launching more work.
                conn.execute(
                    "UPDATE embedding_requests SET status='queued',attempts=attempts-1,updated_at=? "
                    "WHERE id=?",
                    (db.utcnow(), row["id"]),
                )
                heartbeat(conn, "idle")
                return True
            if result:
                raise RuntimeError(f"{stage} worker failed (exit {result})")
        if requests.missing_models(conn, key):
            raise RuntimeError(
                "embedding worker finished with missing vectors; inspect the track ledgers"
            )
    except requests.PreviewUnavailable as exc:
        state, error = "source_not_covered", str(exc)
    except subprocess.TimeoutExpired:
        state, error = "error", f"embedding stage exceeded {requests.STAGE_TIMEOUT} seconds"
    except Exception as exc:
        state, error = "error", str(exc)[:300]
    conn.execute(
        "UPDATE embedding_requests SET status=?,error=?,updated_at=? WHERE id=?",
        (state, error, db.utcnow(), row["id"]),
    )
    heartbeat(conn, "idle")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Serve at most one request, then exit.")
    parser.add_argument(
        "--local", action="store_true", help="Use the workers' isolated uv environments."
    )
    args = parser.parse_args()
    with slot(db.db_path().parent, "requests") as acquired:
        if not acquired:
            return BUSY_EXIT
        # The controller polls a small queue. Inference subprocesses manage
        # their own connections for bulk reads.
        conn = db.connect(cache_kib=2048)
        db.init_db(conn)
        # Only this process holds the request slot. Interrupted requests need an
        # explicit retry, rather than silently restarting expensive work forever.
        conn.execute(
            "UPDATE embedding_requests SET status='error',error='worker interrupted; request an explicit retry', "
            "updated_at=? WHERE status='running'",
            (db.utcnow(),),
        )
        conn.commit()
        next_backfill = 0.0
        try:
            while True:
                heartbeat(conn, "idle")
                worked = process_one(conn, local=args.local)
                if args.once:
                    return 0
                if not worked and time.monotonic() >= next_backfill:
                    commands = [
                        worker_command(
                            "embed_worker.py",
                            "--candidates-only",
                            f"--batch={os.environ.get('EMBED_BATCH', '25')}",
                            local=args.local,
                        ),
                        worker_command(
                            "clap_worker.py",
                            "embed",
                            "--candidates-only",
                            f"--batch={os.environ.get('EMBED_CLAP_BATCH', '1000')}",
                            f"--microbatch={os.environ.get('EMBED_CLAP_MICROBATCH', '8')}",
                            local=args.local,
                        ),
                    ]
                    for command in commands:
                        heartbeat(conn, "backfill")
                        try:
                            run_stage(command)
                        except subprocess.TimeoutExpired:
                            print("embedding backfill stage timed out", flush=True)
                        # Honor requests arriving during a background stage before
                        # starting the next model. Models never reside together.
                        if conn.execute(
                            "SELECT 1 FROM embedding_requests WHERE status='queued' LIMIT 1"
                        ).fetchone():
                            break
                    next_backfill = time.monotonic() + max(
                        5, int(os.environ.get("EMBED_SLEEP", "900"))
                    )
                time.sleep(5)
        finally:
            heartbeat(conn, "stopped")
            conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
