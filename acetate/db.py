"""SQLite connection + schema management for acetate.

Stdlib-only so it can run anywhere (migration, poller, analysis) without the
full project environment.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
SCHEMA_VERSION = "1"

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "acetate.db"


def db_path() -> Path:
    """Resolve the DB location: $ACETATE_DB or the default data/ dir."""
    env = os.environ.get("ACETATE_DB")
    return Path(env).expanduser() if env else DEFAULT_DB_PATH


def utcnow() -> str:
    """ISO8601 UTC timestamp (seconds precision is plenty here)."""
    return datetime.now(timezone.utc).isoformat()


def connect(path: Path | None = None, *, cache_kib: int = 64 * 1024) -> sqlite3.Connection:
    """Open a connection with dict-like rows and a bounded page cache."""
    path = path or db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: streaming responses iterate generators across threadpool
    # threads, so a connection may be created on one step and used on the next. Access is
    # always sequential (never concurrent), so this is safe.
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(
        "PRAGMA busy_timeout = 30000"
    )  # wait, don't error, when scheduler+web write at once
    conn.execute(f"PRAGMA cache_size = {-cache_kib}")
    conn.execute("PRAGMA temp_store = MEMORY")  # keep the remaining GROUP BY temp B-trees in RAM
    conn.execute("PRAGMA mmap_size = 268435456")  # 256 MB memory-mapped reads
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Light, idempotent column additions for DBs created before a schema change.
    (CREATE TABLE IF NOT EXISTS won't add columns to an existing table.)"""
    chat_cols = {r[1] for r in conn.execute("PRAGMA table_info(chat_threads)")}
    if "summary" not in chat_cols:
        conn.execute("ALTER TABLE chat_threads ADD COLUMN summary TEXT")
    if "summary_through_id" not in chat_cols:
        conn.execute(
            "ALTER TABLE chat_threads ADD COLUMN summary_through_id INTEGER NOT NULL DEFAULT 0"
        )
    smart_cols = {r[1] for r in conn.execute("PRAGMA table_info(smart_playlists)")}
    if "published_at" not in smart_cols:
        conn.execute("ALTER TABLE smart_playlists ADD COLUMN published_at TEXT")
        # Existing linked curations must not be rewritten after this upgrade.
        # Backfill only during the column migration, so a failed new publication
        # (which may already have a Spotify ID) remains retryable on later boots.
        conn.execute(
            "UPDATE smart_playlists SET published_at=COALESCE("
            "(SELECT COALESCE(m.updated_at, m.created_at) FROM managed_playlists m "
            "WHERE m.kind='smart' AND m.key=CAST(smart_playlists.id AS TEXT)), ?) "
            "WHERE json_type(spec, '$.track_ids') IS NOT NULL AND EXISTS "
            "(SELECT 1 FROM managed_playlists m WHERE m.kind='smart' "
            "AND m.key=CAST(smart_playlists.id AS TEXT))",
            (utcnow(),),
        )
    cols = {r[1] for r in conn.execute("PRAGMA table_info(playlist_snapshots)")}
    if "content_hash" not in cols:
        conn.execute("ALTER TABLE playlist_snapshots ADD COLUMN content_hash TEXT")
    # idx_play_track(track_id) is superseded by the covering idx_play_track_ms.
    conn.execute("DROP INDEX IF EXISTS idx_play_track")
    # Widen idx_play_track_ms to include played_at if an earlier 2-column version exists.
    idx_cols = [r["name"] for r in conn.execute("PRAGMA index_info(idx_play_track_ms)")]
    if idx_cols and idx_cols != ["track_id", "ms_played", "played_at"]:
        conn.execute("DROP INDEX idx_play_track_ms")
        conn.execute(
            "CREATE INDEX idx_play_track_ms ON play_events(track_id, ms_played, played_at)"
        )
    # recording_stats gained liked_at (Smart Playlists) — add it to pre-existing rollups so the
    # rebuild INSERT (which now lists liked_at) matches; canon repopulates it on the next run.
    rs_cols = {r["name"] for r in conn.execute("PRAGMA table_info(recording_stats)")}
    if rs_cols and "liked_at" not in rs_cols:
        conn.execute("ALTER TABLE recording_stats ADD COLUMN liked_at TEXT")
    # hyperspace_points gained a `space` dimension (effnet|clap projections) with a new PK.
    # The table is a regenerable projection cache — drop the old shape; schema.sql recreates
    # it and `acetate embed reduce` repopulates.
    hp_cols = {r["name"] for r in conn.execute("PRAGMA table_info(hyperspace_points)")}
    if hp_cols and "space" not in hp_cols:
        conn.execute("DROP TABLE hyperspace_points")
        conn.executescript(SCHEMA_PATH.read_text())  # recreate in the new shape (idempotent)
    # Early candidate ledgers used the generic `miss`. Keep stored state aligned
    # with the typed enrichment vocabulary: only an attempted provider lookup
    # can become provider_miss; row absence still means not_scheduled.
    conn.execute("UPDATE candidate_resolutions SET status='provider_miss' WHERE status='miss'")
    conn.execute("UPDATE candidate_embedding_runs SET status='available' WHERE status='embedded'")
    # Candidate reranking is incremental. Existing databases need one nullable
    # fingerprint column; null rows are deliberately rebuilt on the first pass.
    rerank_cols = {r["name"] for r in conn.execute("PRAGMA table_info(candidate_rerank)")}
    if rerank_cols and "input_fingerprint" not in rerank_cols:
        conn.execute("ALTER TABLE candidate_rerank ADD COLUMN input_fingerprint TEXT")


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables if absent and record the schema version."""
    conn.executescript(SCHEMA_PATH.read_text())
    _migrate(conn)
    conn.execute(
        "INSERT INTO schema_meta(key, value) VALUES('version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (SCHEMA_VERSION,),
    )
    conn.commit()
