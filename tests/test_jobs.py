"""Scheduler job glue — candidate_prep aggregates the light candidate steps and
isolates a failing one so a flaky external API can't sink the rest."""

from __future__ import annotations

import sqlite3

import pytest

from acetate import candidate_audio, candidate_deezer, candidate_embeddings, db, jobs


def test_run_candidate_prep_aggregates_and_isolates_errors(monkeypatch, conn) -> None:
    monkeypatch.setattr(db, "connect", lambda: conn)
    monkeypatch.setattr(candidate_deezer, "resolve", lambda conn, **k: {"matched": 3})

    def boom(conn, **k):
        raise RuntimeError("deezer down")

    monkeypatch.setattr(candidate_audio, "fetch_previews", boom)
    monkeypatch.setattr(candidate_embeddings, "finalize", lambda conn: 2)
    monkeypatch.setattr(candidate_embeddings, "rerank", lambda conn, **k: {"ranked": 5})

    res = jobs.run_candidate_prep()
    assert res["resolve"] == {"matched": 3}
    assert "error" in res["preview"]  # the failing step is isolated
    assert res["finalize"] == {"finalized": 2}  # and later steps still run
    assert res["rerank"] == {"ranked": 5}
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        conn.execute("SELECT 1")
