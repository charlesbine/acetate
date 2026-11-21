"""Spotify OAuth for acetate.

Uses the Authorization Code flow with a cached refresh token so scheduled/headless
runs (the VPS) work unattended after a single interactive approval. App credentials
come from the environment (SPOTIPY_CLIENT_ID / SPOTIPY_CLIENT_SECRET /
SPOTIPY_REDIRECT_URI), loaded from .env and .env.local. We never print their values.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import spotipy
from dotenv import find_dotenv, load_dotenv
from loguru import logger
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import CacheFileHandler

from . import db

# Read everything we back up/analyze, and write the playlists we generate.
SCOPES = [
    "user-read-private",
    "user-library-read",
    "user-read-recently-played",
    "user-read-currently-playing",
    "user-top-read",
    "playlist-read-private",
    "playlist-modify-private",
    "playlist-modify-public",
]

REQUIRED_ENV = ["SPOTIPY_CLIENT_ID", "SPOTIPY_CLIENT_SECRET", "SPOTIPY_REDIRECT_URI"]
# Resolved from .env/1Password if present, but not required for core Spotify ops.
OPTIONAL_ENV = ["LASTFM_API_KEY", "OPENROUTER_API_KEY"]


def _token_cache_path() -> str:
    # Lives under data/ (gitignored). Holds the refresh token -> copy this to the VPS.
    return str(db.db_path().parent / ".spotify_token_cache")


_OP_CACHE: dict[str, str] = {}


def _resolve_op_refs() -> None:
    """Resolve any credential whose value is a 1Password secret reference
    (op://...) via the `op` CLI. Resolved values are cached in-process, so a
    long-running dashboard hits `op` once (and survives a later 1Password session
    timeout) rather than on every job."""
    for key in (*REQUIRED_ENV, *OPTIONAL_ENV):
        value = os.environ.get(key, "")
        if not value.startswith("op://"):
            continue
        if value in _OP_CACHE:
            os.environ[key] = _OP_CACHE[value]
            continue
        if not shutil.which("op"):
            logger.warning(
                "{} is a 1Password reference, but the `op` CLI is unavailable; leaving it unset",
                key,
            )
            os.environ.pop(key, None)
            continue
        result = subprocess.run(
            ["op", "read", value],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            logger.warning(
                "Couldn't resolve {} from 1Password; leaving it unset ({})",
                key,
                result.stderr.strip(),
            )
            os.environ.pop(key, None)
            continue
        _OP_CACHE[value] = os.environ[key] = result.stdout.strip()


def load_credentials() -> None:
    """Load .env and its local override, then resolve 1Password references.

    Both files are authoritative over stale ambient values.  ``.env.local`` is
    intentionally gitignored and wins over ``.env`` when present.
    """
    for filename in (".env", ".env.local"):
        path = find_dotenv(filename, usecwd=True)
        if path:
            load_dotenv(path, override=True)
    _resolve_op_refs()


def missing_env() -> list[str]:
    """Names (never values) of required credentials not present after loading."""
    load_credentials()
    return [k for k in REQUIRED_ENV if not os.environ.get(k)]


def client(open_browser: bool = True) -> spotipy.Spotify:
    """Build an authenticated Spotify client. On first use this triggers the
    interactive consent; afterwards it silently refreshes from the cache."""
    load_credentials()
    auth = SpotifyOAuth(
        scope=" ".join(SCOPES),
        open_browser=open_browser,
        cache_handler=CacheFileHandler(cache_path=_token_cache_path()),
    )
    return spotipy.Spotify(auth_manager=auth)


def has_cached_token() -> bool:
    return os.path.exists(_token_cache_path())
