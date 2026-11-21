# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb"]
# ///
"""Import classic Spotify audio features into spotify_features by exact
track-id join against a user-supplied feature archive Parquet file.

    uv run scripts/import_features.py --parquet /path/to/audio-features.parquet

The Parquet file must contain a `track_id` column along with the classic
audio-feature columns (`danceability`, `energy`, `valence`, `tempo`, `key`,
`mode`, `loudness`, `acousticness`, `instrumentalness`, `liveness`,
`speechiness`, `track_popularity`).

One row per recording; the canonical (rep) member's features win when several
member IDs match. Resumable/idempotent: INSERT OR REPLACE over the join.
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"

# (optional time_signature stays NULL when absent from the archive)
COLS = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "key",
    "mode",
    "loudness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "track_popularity",
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Join a user-supplied audio-features Parquet archive into acetate.db."
    )
    ap.add_argument(
        "--parquet",
        type=Path,
        required=True,
        help="Path to a local Parquet file containing track_id and classic audio-feature columns.",
    )
    ap.add_argument(
        "--db",
        type=Path,
        default=Path(os.environ.get("ACETATE_DB") or (DATA / "acetate.db")),
        help="Path to acetate SQLite database (defaults to ACETATE_DB or data/acetate.db).",
    )
    args = ap.parse_args()
    if not args.parquet.is_file():
        ap.error(f"Parquet file not found: {args.parquet}")

    conn = sqlite3.connect(args.db, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.executescript((REPO / "acetate" / "schema.sql").read_text())
    members = {}  # track_id -> (recording_key, is_rep)
    for tid, rk, rep in conn.execute(
        "SELECT track_id, recording_key, rep_track_id FROM track_recording"
    ):
        members[tid] = (rk, 1 if tid == rep else 0)
    print(f"{len(members)} member track ids in library", flush=True)

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id"])
        w.writerows([tid] for tid in members)
        ids_csv = f.name

    con = duckdb.connect()
    rows = con.execute(f"""
        SELECT DISTINCT p.track_id, {", ".join("p." + c for c in COLS)}
        FROM '{args.parquet}' p
        JOIN read_csv('{ids_csv}', header=true) ids USING (track_id)
    """).fetchall()
    print(f"{len(rows)} member ids matched in archive", flush=True)

    # best row per recording: rep member wins, else any member
    best: dict[str, tuple] = {}
    for row in rows:
        rk, is_rep = members[row[0]]
        if rk not in best or (is_rep and not best[rk][1]):
            best[rk] = (row, is_rep)

    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        "INSERT OR REPLACE INTO spotify_features(recording_key, track_id, "
        + ", ".join(COLS)
        + ", updated_at) VALUES("
        + ",".join("?" * (len(COLS) + 3))
        + ")",
        [(rk, *row, now) for rk, (row, _) in best.items()],
    )
    source = str(args.parquet)
    recording_keys = sorted({rk for rk, _ in members.values()})
    conn.executemany(
        """
        INSERT INTO spotify_feature_coverage(
            recording_key, status, source, attempts, error, updated_at)
        VALUES(?, ?, ?, 1, NULL, ?)
        ON CONFLICT(recording_key) DO UPDATE SET
            status=excluded.status, source=excluded.source,
            attempts=spotify_feature_coverage.attempts + 1,
            error=NULL, updated_at=excluded.updated_at
    """,
        [
            (rk, "available" if rk in best else "source_not_covered", source, now)
            for rk in recording_keys
        ],
    )
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM spotify_features").fetchone()[0]
    print(f"spotify_features now holds {n} recordings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
