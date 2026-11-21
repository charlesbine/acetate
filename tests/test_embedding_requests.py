from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from acetate import agent, audio, db, embedding_requests as requests, track_report, workflows
from acetate.embedding_guard import BUSY_EXIT, slot
from tests.conftest import add_embedding, add_recording, cluster_vector


def load_script(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def queue_db(conn, monkeypatch):
    path = conn.execute("PRAGMA database_list").fetchone()["file"]
    monkeypatch.setenv("ACETATE_DB", path)
    for i in range(30):
        # Explicit requests also cover infrequently played, unliked tracks.
        add_recording(conn, f"artist\tt{i}", "Artist", f"T{i}", f"tid-{i}", plays=1, liked=0)
    conn.commit()
    return conn


def enqueue(*tracks):
    return requests.enqueue(list(tracks), requested_by="test-admin")


def test_queue_is_exact_deduplicated_and_skips_available_vectors(queue_db):
    for model in requests.MODELS:
        add_embedding(queue_db, "artist\tt1", cluster_vector(0), model)
    queue_db.commit()
    first = enqueue("Artist - T0", "Artist — T0", "Artist - T1")
    second = enqueue("artist\tt0")

    assert len(first["requests"]) == 2
    assert first["requests"][1]["status"] == "available"
    assert first["requests"][0]["id"] == second["requests"][0]["id"]
    row = queue_db.execute("SELECT * FROM embedding_requests").fetchone()
    assert row["requested_by"] == "test-admin"
    assert row["attempts"] == 0
    assert row["kind"] == "library"
    assert first["worker"]["state"] == "unknown"


@pytest.mark.parametrize("tracks", [[], ["Artist - T0"] * 6, "Artist - T0", [5], ["x" * 501]])
def test_invalid_requests_cannot_write(queue_db, tracks):
    with pytest.raises(ValueError):
        requests.enqueue(tracks, requested_by="test-admin")
    assert queue_db.execute("SELECT COUNT(*) FROM embedding_requests").fetchone()[0] == 0


def test_unknown_or_fuzzy_identity_does_not_enqueue_partial_batch(queue_db):
    with pytest.raises(ValueError, match="no exact known"):
        enqueue("Artist - T0", "Artist - T")
    assert queue_db.execute("SELECT COUNT(*) FROM embedding_requests").fetchone()[0] == 0


def test_known_spotify_identity_and_candidate_share_the_queue(queue_db):
    track_id = "A" * 22
    queue_db.execute("UPDATE track_recording SET track_id=? WHERE track_id='tid-0'", (track_id,))
    queue_db.execute(
        "INSERT INTO candidate_recordings(candidate_key,artist,title) VALUES('new\tsong','New','Song')"
    )
    queue_db.commit()

    result = enqueue(f"https://open.spotify.com/track/{track_id}?si=test", "New - Song")

    assert [r["kind"] for r in result["requests"]] == ["library", "candidate"]
    assert result["outstanding"] == 2


def test_capacity_is_enforced_across_simultaneous_callers(queue_db):
    for start in (0, 5, 10, 15):
        enqueue(*(f"Artist - T{i}" for i in range(start, min(start + 5, 18))))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda start: enqueue(*(f"Artist - T{i}" for i in range(start, start + 5))),
                (18, 23),
            )
        )
    states = [r["status"] for result in results for r in result["requests"]]
    assert states.count("queued") == 2
    assert states.count("queue_full") == 8
    assert queue_db.execute("SELECT COUNT(*) FROM embedding_requests").fetchone()[0] == 20


def test_concurrent_duplicate_is_one_request(queue_db):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: enqueue("Artist - T0"), range(2)))
    assert results[0]["requests"][0]["id"] == results[1]["requests"][0]["id"]
    assert queue_db.execute("SELECT COUNT(*) FROM embedding_requests").fetchone()[0] == 1


def test_tool_authorization_precedes_cache_and_side_effects(queue_db):
    args = {"tracks": ["Artist - T0"]}
    seen = {}
    admin, _ = agent.exec_tool(queue_db, "request_embeddings", args, seen, can_admin=True)
    denied, _ = agent.exec_tool(queue_db, "request_embeddings", args, seen, can_admin=False)
    assert json.loads(admin)["requests"][0]["status"] == "queued"
    assert denied.startswith("ERROR:") and "administrator" in denied
    for name in ("request_embeddings", "embedding_request_status"):
        assert workflows.BY_TOOL[name].admin_only
        assert name not in [t["function"]["name"] for t in agent.tools_for(False)]
        assert name not in agent.system_prompt(queue_db, can_admin=False)


def test_turn_budget_cannot_be_bypassed_with_different_track_arguments(queue_db):
    seen = {}
    first, _ = agent.exec_tool(
        queue_db, "request_embeddings", {"tracks": ["Artist - T0"]}, seen, can_admin=True
    )
    second, _ = agent.exec_tool(
        queue_db, "request_embeddings", {"tracks": ["Artist - T1"]}, seen, can_admin=True
    )
    assert json.loads(first)["requests"][0]["status"] == "queued"
    assert second.startswith("ERROR:") and "budget" in second
    assert queue_db.execute("SELECT COUNT(*) FROM embedding_requests").fetchone()[0] == 1


def test_status_is_live_but_polling_is_bounded(queue_db):
    request_id = enqueue("Artist - T0")["requests"][0]["id"]
    seen, args = {}, {"request_ids": [request_id]}
    first, _ = agent.exec_tool(queue_db, "embedding_request_status", args, seen, can_admin=True)
    for model in requests.MODELS:
        add_embedding(queue_db, "artist\tt0", cluster_vector(0), model)
    queue_db.commit()
    second, _ = agent.exec_tool(queue_db, "embedding_request_status", args, seen, can_admin=True)
    third, _ = agent.exec_tool(queue_db, "embedding_request_status", args, seen, can_admin=True)
    assert json.loads(first)["requests"][0]["status"] == "queued"
    assert json.loads(second)["requests"][0]["status"] == "available"
    assert third.startswith("ERROR:") and "budget" in third


def test_preview_fetch_targets_one_recording_and_keeps_candidate_separate(queue_db, monkeypatch):
    queue_db.execute(
        "INSERT INTO candidate_recordings(candidate_key,artist,title) VALUES('new\tsong','New','Song')"
    )
    queue_db.commit()
    enqueue("New - Song")
    row = queue_db.execute("SELECT * FROM embedding_requests").fetchone()
    calls = []
    monkeypatch.setattr(
        audio,
        "_fetch_one",
        lambda rec, **_kwargs: (
            calls.append(rec) or ("fetched", "deezer_search", "123", 30.0, "audio/test.mp3")
        ),
    )

    requests.prepare_audio(queue_db, row)

    assert [c["recording_key"] for c in calls] == ["new\tsong"]
    assert queue_db.execute("SELECT COUNT(*) FROM audio_cache").fetchone()[0] == 0
    assert queue_db.execute("SELECT status FROM candidate_audio").fetchone()[0] == "fetched"
    assert queue_db.execute("SELECT status FROM candidate_recordings").fetchone()[0] == "preview"
    assert queue_db.execute("SELECT COUNT(*) FROM candidate_embedding_runs").fetchone()[0] == 3


def test_worker_runs_models_in_order_and_never_touches_other_tracks(queue_db, monkeypatch):
    loop = load_script("embed_loop")
    enqueue("Artist - T0", "Artist - T1")
    calls = []
    monkeypatch.setattr(requests, "prepare_audio", lambda conn, row: None)

    def run(command):
        calls.append(command)
        models = (
            requests.MODELS[:2] if command[1].endswith("embed_worker.py") else requests.MODELS[2:]
        )
        for model in models:
            add_embedding(queue_db, "artist\tt0", cluster_vector(0), model)
        queue_db.commit()
        return 0

    monkeypatch.setattr(loop, "run_stage", run)
    assert loop.process_one(queue_db)
    assert [Path(c[1]).name for c in calls] == ["embed_worker.py", "clap_worker.py"]
    assert all(c[c.index("--recording-key") + 1] == "artist\tt0" for c in calls)
    assert all("--batch=1" in c for c in calls)
    assert "--microbatch=1" in calls[1]
    assert [
        r[0] for r in queue_db.execute("SELECT status FROM embedding_requests ORDER BY id")
    ] == ["available", "queued"]


def test_worker_only_runs_missing_models(queue_db, monkeypatch):
    loop = load_script("embed_loop")
    original = cluster_vector(0)
    for model in requests.MODELS[:2]:
        add_embedding(queue_db, "artist\tt0", original, model)
    queue_db.commit()
    enqueue("Artist - T0")
    monkeypatch.setattr(requests, "prepare_audio", lambda conn, row: None)
    calls = []

    def run(command):
        calls.append(command)
        add_embedding(queue_db, "artist\tt0", cluster_vector(1), "clap-music")
        queue_db.commit()
        return 0

    monkeypatch.setattr(loop, "run_stage", run)
    loop.process_one(queue_db)
    assert len(calls) == 1 and Path(calls[0][1]).name == "clap_worker.py"
    assert (
        queue_db.execute("SELECT vector FROM embeddings WHERE model='discogs-effnet'").fetchone()[0]
        == original.tobytes()
    )


@pytest.mark.parametrize(
    "failure,expected", [(1, "error"), ("timeout", "error"), ("miss", "source_not_covered")]
)
def test_worker_failure_is_terminal_until_an_explicit_bounded_retry(
    queue_db, monkeypatch, failure, expected
):
    loop = load_script("embed_loop")
    enqueue("Artist - T0")

    def prepare(conn, row):
        if failure == "miss":
            raise requests.PreviewUnavailable("source_not_covered: no matching preview")

    def run(command):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, requests.STAGE_TIMEOUT)
        return failure

    monkeypatch.setattr(requests, "prepare_audio", prepare)
    monkeypatch.setattr(loop, "run_stage", run)
    for attempt in range(1, 4):
        assert loop.process_one(queue_db)
        row = queue_db.execute("SELECT * FROM embedding_requests").fetchone()
        assert (row["status"], row["attempts"]) == (expected, attempt)
        assert loop.process_one(queue_db) is False
        result = enqueue("Artist - T0")["requests"][0]
        assert result["status"] == ("queued" if attempt < 3 else "attempt_limit")


def test_busy_inference_yields_without_spending_a_retry(queue_db, monkeypatch):
    loop = load_script("embed_loop")
    enqueue("Artist - T0")
    monkeypatch.setattr(requests, "prepare_audio", lambda conn, row: None)
    monkeypatch.setattr(loop, "run_stage", lambda command: BUSY_EXIT)
    loop.process_one(queue_db)
    row = queue_db.execute("SELECT * FROM embedding_requests").fetchone()
    assert (row["status"], row["attempts"]) == ("queued", 0)


def test_inference_lock_excludes_another_process_and_releases_after_exit(tmp_path):
    code = (
        "from pathlib import Path; from acetate.embedding_guard import slot; "
        "ctx=slot(Path(__import__('sys').argv[1])); acquired=ctx.__enter__(); "
        "print(acquired, flush=True); ctx.__exit__(None,None,None)"
    )
    command = [sys.executable, "-c", code, str(tmp_path)]
    with slot(tmp_path) as acquired:
        assert acquired
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        assert result.stdout.strip() == "False"
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "True"


def test_both_worker_entrypoints_honor_the_shared_slot_before_loading_ml(queue_db, monkeypatch):
    for name in ("embed_worker", "clap_worker"):
        worker = load_script(name)
        monkeypatch.setattr(worker, "_run", lambda: pytest.fail("must not load or run a model"))
        with slot(db.db_path().parent):
            assert worker.main() == BUSY_EXIT


def test_targeted_clap_worker_selects_exactly_one_track(queue_db, monkeypatch):
    worker = load_script("clap_worker")
    for i in range(2):
        queue_db.execute(
            "INSERT INTO audio_cache(recording_key,status,path,updated_at) "
            "VALUES(?,'fetched',?,'now')",
            (f"artist\tt{i}", f"audio/{i}.mp3"),
        )
    queue_db.commit()
    files = []

    class Model:
        def get_audio_embedding_from_filelist(self, *, x, use_tensor):
            files.extend(x)
            return np.ones((len(x), 3), dtype=np.float32)

    monkeypatch.setattr(worker, "connect", lambda: queue_db)
    monkeypatch.setattr(worker, "load_model", Model)
    assert worker.cmd_embed(1, recording_key="artist\tt1") == 0
    assert files == [str(db.db_path().parent / "audio/1.mp3")]
    assert [r[0] for r in queue_db.execute("SELECT recording_key FROM embeddings")] == [
        "artist\tt1"
    ]


def test_targeted_effnet_selection_bypasses_exhausted_background_retry_only_for_named_track(
    queue_db,
):
    worker = load_script("embed_worker")
    for i in range(2):
        queue_db.execute(
            "INSERT INTO audio_cache(recording_key,status,path,updated_at) "
            "VALUES(?,'fetched',?,'now')",
            (f"artist\tt{i}", f"audio/{i}.mp3"),
        )
        for model in requests.MODELS[:2]:
            queue_db.execute(
                "INSERT INTO embedding_runs(recording_key,model,status,attempts,updated_at) "
                "VALUES(?,?,'error',3,'now')",
                (f"artist\tt{i}", model),
            )
    assert worker.todo_rows(queue_db, 1, False) == []
    targeted = worker.todo_rows(queue_db, 1, False, "artist\tt1")
    assert [r["recording_key"] for r in targeted] == ["artist\tt1"]


def test_track_report_shows_request_and_status_is_truthful_after_vector_loss(queue_db):
    enqueue("Artist - T0")
    report = track_report.report(queue_db, "Artist - T0")
    assert report["surfaces"]["embedding_request"]["state"] == "queued"
    queue_db.execute("UPDATE embedding_requests SET status='available'")
    queue_db.commit()
    assert requests.status(queue_db)["requests"][0]["status"] == "error"


@pytest.mark.parametrize("group,allowed", [("acetate-query", False), ("acetate-admin", True)])
def test_http_chat_captures_real_role_and_actor_even_for_a_forced_tool_call(
    queue_db, monkeypatch, group, allowed
):
    from fastapi.testclient import TestClient
    from acetate import auth

    monkeypatch.setattr(auth, "load_credentials", lambda: None)
    from acetate import web

    monkeypatch.setenv("ACETATE_AUTH_MODE", "authelia")
    monkeypatch.setenv("ACETATE_ADMIN_USERS", "")
    offered = []

    def chunk(*, content=None, calls=None):
        return SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=calls))],
        )

    def create(**kwargs):
        offered.append({t["function"]["name"] for t in kwargs["tools"]})
        if kwargs["messages"][-1]["role"] == "user":
            return iter(
                [
                    chunk(
                        calls=[
                            SimpleNamespace(
                                index=0,
                                id="embed-test",
                                function=SimpleNamespace(
                                    name="request_embeddings",
                                    arguments='{"tracks":["Artist - T0"]}',
                                ),
                            )
                        ]
                    )
                ]
            )
        return iter([chunk(content="Request inspected.")])

    monkeypatch.setattr(
        agent,
        "_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    response = TestClient(web.app).post(
        "/chat/stream",
        data={"q": "embed Artist - T0"},
        headers={"Remote-User": "test-operator", "Remote-Groups": group},
    )

    assert response.status_code == 200
    assert ("request_embeddings" in offered[0]) is allowed
    assert '"t": "error"' not in response.text
    rows = queue_db.execute("SELECT requested_by FROM embedding_requests").fetchall()
    assert [r[0] for r in rows] == (["test-operator"] if allowed else [])
    catalog = TestClient(web.app).get(
        "/tools", headers={"Remote-User": "test-operator", "Remote-Groups": group}
    )
    assert ("request_embeddings" in catalog.text) is allowed
    assert ("embedding_request_status" in catalog.text) is allowed


def test_controller_restart_does_not_silently_retry_interrupted_work(queue_db):
    enqueue("Artist - T0")
    queue_db.execute("UPDATE embedding_requests SET status='running',attempts=1")
    queue_db.commit()
    script = Path(__file__).parents[1] / "scripts" / "embed_loop.py"
    subprocess.run([sys.executable, str(script), "--once"], check=True, timeout=10)
    row = queue_db.execute("SELECT * FROM embedding_requests").fetchone()
    assert (row["status"], row["attempts"]) == ("error", 1)
    assert "interrupted" in row["error"]


def test_model_stage_timeout_reaps_the_child_process(tmp_path, monkeypatch):
    import os

    loop = load_script("embed_loop")
    monkeypatch.setattr(requests, "STAGE_TIMEOUT", 0.3)
    pidfile = tmp_path / "pid"
    code = (
        "import os,sys,time; from pathlib import Path; "
        "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        loop.run_stage([sys.executable, "-c", code, str(pidfile)])
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize("name", ["requests", "inference"])
def test_reconciliation_and_restore_refuse_to_swap_under_an_embedding_worker(
    queue_db, tmp_path, name
):
    from acetate import reconcile

    target = db.db_path()
    with slot(target.parent, name):
        with pytest.raises(RuntimeError, match="Stop the embedding"):
            reconcile.apply_bundle(target, tmp_path / "missing-bundle", tmp_path)
        with pytest.raises(RuntimeError, match="Stop the embedding"):
            reconcile.restore_backup(target, tmp_path / "missing-backup", tmp_path)


@pytest.fixture
def requested_candidate(queue_db):
    queue_db.execute(
        "INSERT INTO candidate_recordings(candidate_key,artist,title,status) "
        "VALUES('new\tsong','New','Song','resolved')"
    )
    queue_db.execute(
        "INSERT INTO candidate_resolutions(candidate_key,provider,status,external_id,"
        "expected_duration_s,matched_duration_s,updated_at) "
        "VALUES('new\tsong','deezer','matched','123',180,180,?)",
        (db.utcnow(),),
    )
    queue_db.commit()
    enqueue("New - Song")
    return queue_db.execute("SELECT * FROM embedding_requests").fetchone()


@pytest.mark.parametrize("provider", ["deezer", "itunes"])
@pytest.mark.parametrize(
    "artist,title,duration", [("Other", "Song", 180), ("New", "Other", 180), ("New", "Song", 200)]
)
def test_requested_candidate_rejects_wrong_preview_identity_or_duration(
    queue_db, requested_candidate, monkeypatch, provider, artist, title, duration
):
    hit = {
        "id": "wrong",
        "artist": {"name": artist},
        "title": title,
        "duration": duration,
        "preview": "https://example.test/wrong.mp3",
    }
    monkeypatch.setattr(audio, "deezer_search", lambda *_: [hit] if provider == "deezer" else [])
    monkeypatch.setattr(audio, "_itunes_search", lambda *_: [hit] if provider == "itunes" else [])
    monkeypatch.setattr(
        audio, "_download", lambda *_: pytest.fail("must not download a wrong song")
    )

    with pytest.raises(requests.PreviewUnavailable):
        requests.prepare_audio(queue_db, requested_candidate)

    assert (
        queue_db.execute("SELECT status FROM candidate_audio").fetchone()[0] == "source_not_covered"
    )
    assert queue_db.execute("SELECT status FROM candidate_recordings").fetchone()[0] == "resolved"
    assert queue_db.execute("SELECT COUNT(*) FROM candidate_embedding_runs").fetchone()[0] == 0


@pytest.mark.parametrize("provider", ["deezer", "itunes"])
def test_requested_candidate_accepts_verified_search_and_falls_back_after_wrong_hit(
    queue_db, requested_candidate, monkeypatch, provider
):
    good = {
        "id": "correct",
        "artist": {"name": "New"},
        "title": "Song",
        "duration": 180,
        "preview": "https://example.test/correct.mp3",
    }
    wrong = {**good, "id": "wrong", "artist": {"name": "Other"}}
    monkeypatch.setattr(
        audio, "deezer_search", lambda *_: [wrong, good] if provider == "deezer" else [wrong]
    )
    monkeypatch.setattr(audio, "_itunes_search", lambda *_: [wrong, good])
    downloads = []
    monkeypatch.setattr(
        audio,
        "_download",
        lambda url, key: downloads.append((url, key)) or ("audio/test.mp3", Path("unused")),
    )

    requests.prepare_audio(queue_db, requested_candidate)

    assert downloads == [(good["preview"], "new\tsong")]
    cached = queue_db.execute("SELECT status,source,source_id FROM candidate_audio").fetchone()
    assert tuple(cached) == ("fetched", f"{provider}_search", "correct")
    assert queue_db.execute("SELECT status FROM candidate_recordings").fetchone()[0] == "preview"


@pytest.mark.parametrize("kind", ["library", "candidate"])
@pytest.mark.parametrize("state", ["queued", "running", "available", "error", "source_not_covered"])
def test_bulk_workers_skip_request_owned_tracks_without_retrying_them(
    queue_db, monkeypatch, kind, state
):
    candidate = kind == "candidate"
    table, column, ledger = (
        ("candidate_audio", "candidate_key", "candidate_embedding_runs")
        if candidate
        else ("audio_cache", "recording_key", "embedding_runs")
    )
    keys = [f"new\tt{i}" if candidate else f"artist\tt{i}" for i in range(2)]
    if candidate:
        queue_db.executemany(
            "INSERT INTO candidate_recordings(candidate_key,artist,title,status) "
            "VALUES(?,'New',?,'preview')",
            [(key, f"T{i}") for i, key in enumerate(keys)],
        )
    for i, key in enumerate(keys):
        queue_db.execute(
            f"INSERT INTO {table}({column},status,path,updated_at) VALUES(?,'fetched',?,?)",
            (key, f"audio/{i}.mp3", db.utcnow()),
        )
        queue_db.executemany(
            f"INSERT INTO {ledger}({column},model,status,attempts,updated_at) "
            "VALUES(?,?,'error',1,?)",
            [(key, model, db.utcnow()) for model in requests.MODELS],
        )
    queue_db.commit()
    enqueue(keys[0])
    queue_db.execute("UPDATE embedding_requests SET status=?, attempts=1", (state,))
    queue_db.commit()
    effnet, clap = load_script("embed_worker"), load_script("clap_worker")

    assert [r["recording_key"] for r in effnet.todo_rows(queue_db, 10, candidate)] == [keys[1]]
    files = []

    class Model:
        def get_audio_embedding_from_filelist(self, *, x, use_tensor):
            files.extend(x)
            return np.ones((len(x), 3), dtype=np.float32)

    monkeypatch.setattr(clap, "connect", lambda: queue_db)
    monkeypatch.setattr(clap, "load_model", Model)
    assert clap.cmd_embed(10, candidates_only=candidate) == 0
    assert files == [str(db.db_path().parent / "audio/1.mp3")]
    assert (
        queue_db.execute(
            f"SELECT attempts FROM {ledger} WHERE {column}=? AND model='clap-music'", (keys[0],)
        ).fetchone()[0]
        == 1
    )


@pytest.mark.parametrize("state", ["duplicate", "dismissed"])
@pytest.mark.parametrize("when", ["before_fetch", "during_fetch", "cached"])
def test_candidate_exclusion_stops_request_and_cannot_be_overwritten_by_preview(
    queue_db, requested_candidate, monkeypatch, state, when
):
    loop = load_script("embed_loop")
    key = requested_candidate["recording_key"]
    if when == "cached":
        (db.db_path().parent / "cached.mp3").write_bytes(b"synthetic fixture")
        queue_db.execute(
            "INSERT INTO candidate_audio(candidate_key,status,path,updated_at) "
            "VALUES(?,'fetched','cached.mp3',?)",
            (key, db.utcnow()),
        )
    if when != "during_fetch":
        queue_db.execute("UPDATE candidate_recordings SET status=?", (state,))
    queue_db.commit()
    fetched = []

    def fetch(rec, **_kwargs):
        fetched.append(rec["recording_key"])
        other = db.connect()
        other.execute("UPDATE candidate_recordings SET status=?", (state,))
        other.commit()
        other.close()
        return "fetched", "deezer_search", "123", 180, "audio/test.mp3"

    monkeypatch.setattr(audio, "_fetch_one", fetch)
    monkeypatch.setattr(
        loop, "run_stage", lambda _: pytest.fail("excluded candidate must not run ML")
    )
    assert loop.process_one(queue_db)

    assert fetched == ([key] if when == "during_fetch" else [])
    assert queue_db.execute("SELECT status FROM candidate_recordings").fetchone()[0] == state
    result = queue_db.execute("SELECT status,error FROM embedding_requests").fetchone()
    assert result["status"] == "error" and state in result["error"]
    assert queue_db.execute("SELECT COUNT(*) FROM candidate_embedding_runs").fetchone()[0] == 0
    assert queue_db.execute("SELECT COUNT(*) FROM candidate_audio").fetchone()[0] == (
        when == "cached"
    )


@pytest.mark.parametrize("state", ["duplicate", "dismissed"])
def test_candidate_excluded_between_models_does_not_start_the_next_stage(
    queue_db, requested_candidate, monkeypatch, state
):
    loop = load_script("embed_loop")
    key = requested_candidate["recording_key"]
    monkeypatch.setattr(requests, "prepare_audio", lambda conn, row: None)
    commands = []

    def run(command):
        commands.append(command)
        for model in requests.MODELS[:2]:
            add_embedding(queue_db, key, cluster_vector(0), model)
        queue_db.execute("UPDATE candidate_recordings SET status=?", (state,))
        queue_db.commit()
        return 0

    monkeypatch.setattr(loop, "run_stage", run)
    assert loop.process_one(queue_db)

    assert [Path(c[1]).name for c in commands] == ["embed_worker.py"]
    result = queue_db.execute("SELECT status,error FROM embedding_requests").fetchone()
    assert result["status"] == "error" and state in result["error"]


@pytest.mark.parametrize("state", ["duplicate", "dismissed"])
def test_targeted_workers_recheck_candidate_exclusion_before_loading_models(
    queue_db, requested_candidate, monkeypatch, state
):
    key = requested_candidate["recording_key"]
    queue_db.execute(
        "INSERT INTO candidate_audio(candidate_key,status,path,updated_at) "
        "VALUES(?,'fetched','audio/test.mp3',?)",
        (key, db.utcnow()),
    )
    queue_db.execute("UPDATE candidate_recordings SET status=?", (state,))
    queue_db.commit()
    effnet, clap = load_script("embed_worker"), load_script("clap_worker")

    assert effnet.todo_rows(queue_db, 1, True, key) == []
    monkeypatch.setattr(clap, "connect", lambda: queue_db)
    monkeypatch.setattr(
        clap, "load_model", lambda: pytest.fail("excluded candidate must not load ML")
    )
    assert clap.cmd_embed(1, candidates_only=True, recording_key=key) == 0
