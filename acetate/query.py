"""Read-only SQL access — the safe substrate for the chat agent and any query tooling.

The connection is opened mode=ro (+ PRAGMA query_only), so the SQLite engine itself
rejects every write/DDL no matter what SQL arrives — the agent can read everything and
mutate nothing. On top of that we allow only a single SELECT/WITH/EXPLAIN statement,
cap returned rows, and abort queries that run too long.
"""

from __future__ import annotations

import sqlite3
import time

from . import db

# Bookkeeping tables not useful to a question-answering agent.
_HIDE = {"schema_meta", "kv_state", "playlist_seen", "managed_playlist_tracks"}


def readonly_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(
        f"file:{db.db_path()}?mode=ro", uri=True, timeout=10, check_same_thread=False
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")  # belt-and-suspenders on top of mode=ro
    return conn


def guard(sql: str) -> str:
    """Reject anything that isn't a single read-only statement.

    Shared with the DuckDB analytics path, which has no `mode=ro` of its own —
    there the engine-level protection is the read-only ATTACH.
    """
    s = sql.strip().rstrip(";").strip()
    if not s:
        raise ValueError("empty query")
    if ";" in s:
        raise ValueError("only a single statement is allowed")
    if s.split(None, 1)[0].lower() not in ("select", "with", "explain"):
        raise ValueError("only SELECT / WITH / EXPLAIN queries are allowed")
    return s


def run_query(sql: str, max_rows: int = 500, timeout_s: float = 5.0) -> dict:
    """Run a read-only query → {columns, rows, row_count, truncated}.

    Raises ValueError for a disallowed statement, sqlite3.OperationalError if the engine
    rejects a write that slipped past the guard (it can't succeed — mode=ro), or
    'query timed out' if it exceeds timeout_s."""
    s = guard(sql)
    conn = readonly_connect()
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = conn.execute(s)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(max_rows)
        more = cur.fetchone() is not None
        return {
            "columns": cols,
            "rows": [list(r) for r in rows],
            "row_count": len(rows),
            "truncated": more,
        }
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise sqlite3.OperationalError(f"query timed out after {timeout_s}s") from exc
        raise
    finally:
        conn.close()


def schema_summary(conn: sqlite3.Connection | None = None) -> str:
    """Compact `table(col, col, ...)` listing for the agent's system prompt."""
    own = conn is None
    conn = conn or readonly_connect()
    try:
        lines = []
        for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ):
            if name in _HIDE:
                continue
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({name})")]
            lines.append(f"{name}({', '.join(cols)})")
        return "\n".join(lines)
    finally:
        if own:
            conn.close()
