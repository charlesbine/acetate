"""Analytical read-only SQL — DuckDB over the same SQLite archive.

`query.run_query` is SQLite, which has no median, percentile, stddev or
correlation. So a question like "median gap between plays" had no expressible
answer, and the agent's only move was to pull raw rows and do the statistics in
its head. DuckDB attaches the archive read-only and brings those aggregates, so
the arithmetic happens in the engine and only the answer reaches the model.

SECURITY — the hardening below is not optional. A default DuckDB can read any
file the process can (`read_csv('/etc/passwd')` — and `.env` sits next to the
database) and can reach the network via httpfs. Three things contain it:

1. the archive is ATTACHed READ_ONLY, so the engine rejects writes to it;
2. `enable_external_access=false` removes all file and network reach outside
   the already-attached database;
3. `lock_configuration=true` stops a query from turning (2) back on.

Order matters: LOAD and ATTACH need the access that step 2 drops, so they run
first and the door is shut behind them. The extension is loaded from a
pre-populated directory with autoinstall off, so no call ever needs the network.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import duckdb

from . import db, query

# Where `INSTALL sqlite` was staged (see Dockerfile). Unset locally → DuckDB's
# default (~/.duckdb/extensions), which a dev machine populates on first use.
EXTENSION_DIR = os.environ.get("ACETATE_DUCKDB_EXTENSIONS")

# The attached catalog's name, and the default one, so the agent writes the same
# unqualified `FROM recording_stats` it writes for query_db.
CATALOG = "acetate"


def analytic_connect() -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB with the archive attached read-only, then locked down."""
    config: dict[str, Any] = {
        # Never reach out for an extension mid-request: it would need network we
        # are about to forbid, and it would turn a stats question into a stall.
        "autoinstall_known_extensions": False,
        "autoload_known_extensions": False,
    }
    if EXTENSION_DIR:
        config["extension_directory"] = EXTENSION_DIR
    conn = duckdb.connect(":memory:", config=config)
    conn.execute("LOAD sqlite")
    # ATTACH takes no bound parameters. The path is operator config, never model
    # input, but quote it properly anyway.
    path = str(db.db_path()).replace("'", "''")
    conn.execute(f"ATTACH '{path}' AS {CATALOG} (TYPE sqlite, READ_ONLY)")
    conn.execute(f"USE {CATALOG}")
    conn.execute("SET enable_external_access=false")
    conn.execute("SET lock_configuration=true")  # must be last — it freezes the above
    return conn


def run_analysis(sql: str, max_rows: int = 200, timeout_s: float = 10.0) -> dict:
    """Run one read-only analytical query → {columns, rows, row_count, truncated}.

    Raises ValueError for a disallowed statement, and duckdb.InterruptException
    if the query outruns timeout_s (DuckDB has no statement timeout, so a timer
    thread interrupts it)."""
    s = query.guard(sql)
    conn = analytic_connect()
    timer = threading.Timer(timeout_s, conn.interrupt)
    timer.start()
    try:
        cur = conn.execute(s)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(max_rows)
        more = bool(cur.fetchmany(1))
        return {
            "columns": cols,
            "rows": [list(r) for r in rows],
            "row_count": len(rows),
            "truncated": more,
        }
    except duckdb.InterruptException as exc:
        raise duckdb.InterruptException(
            f"query timed out after {timeout_s}s — aggregate more, scan less"
        ) from exc
    finally:
        timer.cancel()
        conn.close()
