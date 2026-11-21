# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    ACETATE_DB=/data/acetate.db \
    ACETATE_AUTH_MODE=authelia

WORKDIR /app
RUN pip install --no-cache-dir uv

# Install deps first (cached layer) using just the manifests, then the source.
# --no-dev: prod deps only (no ruff/ipykernel→jedi/debugpy). --frozen: use the lock as-is.
#
# The two syncs are the point: --no-install-project resolves and installs the
# DEPENDENCIES only, so that layer's cache key is the manifests alone and an
# ordinary source edit can't invalidate it. The source is copied after, and the
# second sync installs just the acetate package itself (near-instant). Copying
# the source before a single sync — as this did until 2026-07-30 — meant every
# one-line change re-downloaded and rebuilt all 54 packages on every deploy.
COPY pyproject.toml uv.lock* README.md ./
RUN uv sync --no-dev --frozen --no-install-project
COPY acetate ./acetate
RUN uv sync --no-dev --frozen

# Put the synced venv on PATH and call binaries directly. We deliberately avoid
# `uv run` at runtime: it re-syncs the *dev* group on every startup, which would
# reinstall jedi/debugpy/ruff and slow the boot.
ENV PATH="/app/.venv/bin:$PATH"

# Stage DuckDB's sqlite_scanner at build time. It is NOT statically linked into
# the wheel, and the analytics connection runs with external access disabled and
# autoinstall off — so if it were not already on disk here, the first stats
# question of the container's life would fail with no way to fetch it.
ENV ACETATE_DUCKDB_EXTENSIONS=/opt/duckdb-extensions
RUN python -c "import duckdb, os; \
    duckdb.connect(':memory:', config={'extension_directory': os.environ['ACETATE_DUCKDB_EXTENSIONS']}) \
    .execute('INSTALL sqlite')"

EXPOSE 8765
# Default = the web dashboard; the scheduler service overrides `command`.
CMD ["uvicorn", "acetate.web:app", "--host", "0.0.0.0", "--port", "8765"]
