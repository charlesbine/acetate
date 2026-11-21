from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_clap_worker():
    path = Path(__file__).parents[1] / "scripts" / "clap_worker.py"
    spec = importlib.util.spec_from_file_location("clap_worker_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeConnection:
    def __init__(self, rows):
        self.rows = rows
        self.commits = 0
        self.inserted = []
        self.result = []

    def execute(self, sql, _params=()):
        if "SELECT a.candidate_key AS recording_key" in sql:
            self.result = self.rows
        else:
            self.result = []
        return self

    def fetchall(self):
        return self.result

    def executemany(self, _sql, rows):
        self.inserted.extend(rows)

    def commit(self):
        self.commits += 1


def test_clap_worker_loads_once_and_uses_bounded_microbatches(monkeypatch) -> None:
    worker = _load_clap_worker()
    rows = [
        {"recording_key": f"candidate-{index}", "path": f"{index}.mp3", "kind": "candidate"}
        for index in range(5)
    ]
    conn = _FakeConnection(rows)
    calls = []
    loads = 0

    class Model:
        def get_audio_embedding_from_filelist(self, *, x, use_tensor):
            assert use_tensor is False
            calls.append(list(x))
            return np.ones((len(x), 3), dtype=np.float32)

    def load_model():
        nonlocal loads
        loads += 1
        return Model()

    monkeypatch.setattr(worker, "connect", lambda: conn)
    monkeypatch.setattr(worker, "load_model", load_model)

    assert (
        worker.cmd_embed(
            batch=5,
            candidates_only=True,
            microbatch=2,
        )
        == 0
    )
    assert loads == 1
    assert [len(call) for call in calls] == [2, 2, 1]
    assert len(conn.inserted) == 5
    assert conn.commits == 3


def test_clap_worker_does_not_load_model_when_queue_is_empty(monkeypatch) -> None:
    worker = _load_clap_worker()
    monkeypatch.setattr(worker, "connect", lambda: _FakeConnection([]))
    monkeypatch.setattr(
        worker,
        "load_model",
        lambda: (_ for _ in ()).throw(AssertionError("model should stay unloaded")),
    )

    assert (
        worker.cmd_embed(
            batch=1000,
            candidates_only=True,
            microbatch=8,
        )
        == 0
    )
