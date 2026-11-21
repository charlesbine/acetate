"""Shared fixtures for the downstream-consumer tests (taste / health /
discovery / genre): a tiny synthetic library with two sonically distinct
clusters, two authored playlists, and embedded vectors."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from acetate import db

DIM = 8
MODEL = "discogs-effnet"
OWNER = "alice"
BT = "2026-07-01T00:00:00+00:00"


def cluster_vector(axis: int, wobble: float = 0.0, seed: int = 0) -> np.ndarray:
    """A unit vector near basis axis `axis`, deterministically perturbed."""
    rng = np.random.default_rng(seed)
    v = np.zeros(DIM, dtype=np.float32)
    v[axis] = 1.0
    v += wobble * rng.standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def add_recording(
    conn: sqlite3.Connection,
    key: str,
    artist: str,
    title: str,
    track_id: str,
    plays: int = 10,
    liked: int = 1,
) -> None:
    conn.execute(
        "INSERT INTO recording_stats(recording_key, rep_track_id, artist, title, "
        "plays, liked, homed) VALUES(?,?,?,?,?,?,0)",
        (key, track_id, artist, title, plays, liked),
    )
    conn.execute(
        "INSERT INTO track_recording(track_id, recording_key, rep_track_id, artist, title) "
        "VALUES(?,?,?,?,?)",
        (track_id, key, track_id, artist, title),
    )


def add_embedding(conn: sqlite3.Connection, key: str, vec: np.ndarray, model: str = MODEL) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO embeddings(recording_key, model, dim, vector, updated_at) "
        "VALUES(?,?,?,?,?)",
        (key, model, len(vec), vec.astype(np.float32).tobytes(), BT),
    )


def add_playlist(
    conn: sqlite3.Connection,
    playlist_id: str,
    name: str,
    track_ids: list[str],
    owner: str = OWNER,
    backup_time: str = BT,
) -> None:
    conn.execute(
        "INSERT INTO playlist_snapshots(backup_time, playlist_id, name, owner_id, "
        "total_tracks) VALUES(?,?,?,?,?)",
        (backup_time, playlist_id, name, owner, len(track_ids)),
    )
    for pos, tid in enumerate(track_ids):
        conn.execute(
            "INSERT INTO playlist_track_snapshots(backup_time, playlist_id, position, "
            "track_id) VALUES(?,?,?,?)",
            (backup_time, playlist_id, pos, tid),
        )


class FakeSpotify:
    """Records playlist writes; mimics the spotipy surface playlists.py touches."""

    def __init__(self) -> None:
        self.created: list[dict] = []
        self.replaced: dict[str, list[str]] = {}
        self.added: dict[str, list[str]] = {}
        self.removed: dict[str, list[str]] = {}
        self._n = 0
        self._snap = 0

    def _post(self, path, payload=None):
        self._n += 1
        pl = {"id": f"spotify-pl-{self._n}", **(payload or {})}
        self.created.append(pl)
        return pl

    def playlist_replace_items(self, playlist_id, uris):
        self.replaced[playlist_id] = list(uris)

    def playlist_add_items(self, playlist_id, uris):
        self.added.setdefault(playlist_id, []).extend(uris)

    def playlist_remove_all_occurrences_of_items(self, playlist_id, uris):
        self.removed.setdefault(playlist_id, []).extend(uris)

    def playlist(self, playlist_id, fields=None):
        self._snap += 1
        return {"snapshot_id": f"snap-{self._snap}"}

    def playlist_change_details(self, playlist_id, **kwargs):
        pass


@pytest.fixture
def conn(tmp_path) -> sqlite3.Connection:
    conn = db.connect(tmp_path / "acetate-test.db")
    db.init_db(conn)
    return conn


@pytest.fixture(autouse=True)
def cached_chat_catalog(monkeypatch) -> None:
    """Use deterministic catalog windows without calling the live public catalog.

    Registered test routes model a catalogued 128 Ki-token window. Unknown routes
    still exercise the production fallback; budget tests override this metadata.
    """
    from acetate import agent, chat_context

    monkeypatch.setattr(
        chat_context,
        "_catalog",
        {
            model["slug"].removesuffix(":nitro"): {"context_length": 131_072}
            for model in agent.MODELS
        },
    )
    monkeypatch.setattr(chat_context, "_catalog_expires", float("inf"))
    monkeypatch.delenv("ACETATE_CHAT_CONTEXT_TOKENS", raising=False)


@pytest.fixture
def library(conn) -> sqlite3.Connection:
    """Two clusters: 'techno' near axis 0 (playlist P1), 'folk' near axis 1
    (playlist P2). t9 is a liked techno track NOT in P1 (the gap-fill target);
    f_odd is a folk track planted INSIDE P1 (the misfit)."""
    techno_ids = []
    for i in range(8):
        key, tid = f"techno artist\tt{i}", f"tid-t{i}"
        add_recording(conn, key, "Techno Artist", f"T{i}", tid)
        add_embedding(conn, key, cluster_vector(0, 0.08, seed=i))
        techno_ids.append(tid)
    folk_ids = []
    for i in range(8):
        key, tid = f"folk artist\tf{i}", f"tid-f{i}"
        add_recording(conn, key, "Folk Artist", f"F{i}", tid)
        add_embedding(conn, key, cluster_vector(1, 0.08, seed=100 + i))
        folk_ids.append(tid)
    # gap-fill target: liked techno, not playlisted
    add_recording(conn, "techno artist\tt9", "Techno Artist", "T9", "tid-t9")
    add_embedding(conn, "techno artist\tt9", cluster_vector(0, 0.05, seed=999))
    # misfit: a folk track sitting inside the techno playlist
    add_recording(conn, "folk artist\tfodd", "Folk Artist", "Fodd", "tid-fodd")
    add_embedding(conn, "folk artist\tfodd", cluster_vector(1, 0.05, seed=888))

    add_playlist(conn, "P1", "Warehouse Techno", techno_ids + ["tid-fodd"])
    add_playlist(conn, "P2", "Quiet Folk", folk_ids)
    conn.commit()
    return conn
