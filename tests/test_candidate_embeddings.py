from __future__ import annotations

import sqlite3

import numpy as np

from acetate import candidate_embeddings, db, similar


def _conn(tmp_path) -> sqlite3.Connection:
    conn = db.connect(tmp_path / "acetate-test.db")
    db.init_db(conn)
    return conn


def _embedding(conn: sqlite3.Connection, key: str, model: str, values: list[float]) -> None:
    vector = np.asarray(values, dtype=np.float32)
    conn.execute(
        "INSERT INTO embeddings(recording_key, model, dim, vector, updated_at) "
        "VALUES(?, ?, ?, ?, 'now')",
        (key, model, vector.size, vector.tobytes()),
    )


def _candidate(conn: sqlite3.Connection, key: str, artist: str, score: float) -> None:
    conn.execute(
        """
        INSERT INTO candidate_recordings(
            candidate_key, artist, title, status, score, source_count, updated_at
        )
        VALUES(?, ?, 'Track', 'preview', ?, 1, 'now')
        """,
        (key, artist, score),
    )


def test_finalize_and_rerank_candidate_embeddings(tmp_path) -> None:
    conn = _conn(tmp_path)
    conn.executemany(
        """
        INSERT INTO recording_stats(recording_key, artist, title, plays, liked, homed)
        VALUES(?, ?, ?, 20, 1, 1)
        """,
        [
            ("library-a", "Library A", "Track A"),
            ("library-b", "Library B", "Track B"),
        ],
    )
    _candidate(conn, "candidate-good", "Candidate Good", 10.0)
    _candidate(conn, "candidate-far", "Candidate Far", 10.0)
    for model in candidate_embeddings.REQUIRED_MODELS:
        _embedding(conn, "library-a", model, [1.0, 0.0])
        _embedding(conn, "library-b", model, [0.0, 1.0])
        _embedding(conn, "candidate-good", model, [1.0, 0.0])
        _embedding(conn, "candidate-far", model, [-1.0, 0.0])
    conn.commit()

    finalized = candidate_embeddings.finalize(conn)
    result = candidate_embeddings.rerank(conn, top_k=1)
    ranked = candidate_embeddings.ranked_candidates(conn)

    assert finalized == 2
    assert result == {"ranked": 2, "duplicates": 0}
    assert [row["candidate_key"] for row in ranked] == ["candidate-good", "candidate-far"]
    assert ranked[0]["combined_score"] > ranked[1]["combined_score"]
    assert ranked[0]["effnet_neighbor_artist"] == "Library A"
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM candidate_recordings WHERE status='embedded'"
        ).fetchone()[0]
        == 2
    )

    # With unchanged candidate, library, and scoring inputs, the next scheduled
    # pass has no expensive similarity work to repeat.
    assert candidate_embeddings.rerank(conn, top_k=1) == {
        "ranked": 0,
        "duplicates": 0,
    }


def test_rerank_updates_only_changed_candidate_until_global_input_changes(tmp_path) -> None:
    conn = _conn(tmp_path)
    conn.execute(
        "INSERT INTO recording_stats(recording_key, artist, title, plays, liked, homed) "
        "VALUES('library-a', 'Library A', 'Track A', 20, 1, 1)"
    )
    for key, score in (("candidate-low", 0.0), ("candidate-mid", 5.0), ("candidate-high", 10.0)):
        _candidate(conn, key, key, score)
        for model in candidate_embeddings.REQUIRED_MODELS:
            _embedding(conn, key, model, [1.0, 0.0])
    for model in candidate_embeddings.REQUIRED_MODELS:
        _embedding(conn, "library-a", model, [1.0, 0.0])
    conn.commit()
    assert candidate_embeddings.finalize(conn) == 3
    assert candidate_embeddings.rerank(conn, top_k=1)["ranked"] == 3

    # Changing a non-extreme score keeps the global normalization bounds fixed,
    # so only that candidate needs new similarity/upsert work.
    conn.execute("UPDATE candidate_recordings SET score=6 WHERE candidate_key='candidate-mid'")
    conn.commit()
    assert candidate_embeddings.rerank(conn, top_k=1)["ranked"] == 1
    assert candidate_embeddings.rerank(conn, top_k=1)["ranked"] == 0

    # A new high-signal reference-library vector can change every candidate's
    # nearest neighbours, so it deliberately invalidates the complete ranking.
    conn.execute(
        "INSERT INTO recording_stats(recording_key, artist, title, plays, liked, homed) "
        "VALUES('library-b', 'Library B', 'Track B', 20, 1, 1)"
    )
    for model in candidate_embeddings.REQUIRED_MODELS:
        _embedding(conn, "library-b", model, [0.0, 1.0])
    conn.commit()
    assert candidate_embeddings.rerank(conn, top_k=1)["ranked"] == 3


def test_rerank_full_rebuilds_for_scoring_change_or_explicit_request(tmp_path) -> None:
    conn = _conn(tmp_path)
    conn.execute(
        "INSERT INTO recording_stats(recording_key, artist, title, plays, liked, homed) "
        "VALUES('library', 'Library', 'Track', 20, 1, 1)"
    )
    _candidate(conn, "candidate", "Candidate", 1.0)
    for model in candidate_embeddings.REQUIRED_MODELS:
        _embedding(conn, "library", model, [1.0, 0.0])
        _embedding(conn, "candidate", model, [1.0, 0.0])
    conn.commit()
    assert candidate_embeddings.finalize(conn) == 1
    assert candidate_embeddings.rerank(conn, top_k=1)["ranked"] == 1
    assert candidate_embeddings.rerank(conn, top_k=2)["ranked"] == 1
    assert candidate_embeddings.rerank(conn, top_k=2)["ranked"] == 0
    assert candidate_embeddings.rerank(conn, top_k=2, force=True)["ranked"] == 1


def test_finalize_waits_for_every_candidate_model(tmp_path) -> None:
    conn = _conn(tmp_path)
    _candidate(conn, "candidate", "Candidate", 1.0)
    _embedding(conn, "candidate", candidate_embeddings.EFFNET_MODEL, [1.0, 0.0])
    _embedding(conn, "candidate", candidate_embeddings.STYLE_MODEL, [1.0, 0.0])
    conn.commit()

    assert candidate_embeddings.finalize(conn) == 0
    assert (
        conn.execute(
            "SELECT status FROM candidate_recordings WHERE candidate_key='candidate'"
        ).fetchone()[0]
        == "preview"
    )


def test_rerank_excludes_exact_isrc_library_duplicates(tmp_path) -> None:
    conn = _conn(tmp_path)
    conn.execute(
        "INSERT INTO recording_stats(recording_key, artist, title, plays, liked, homed) "
        "VALUES('library', 'Library Artist', 'Library Track', 5, 1, 1)"
    )
    conn.execute("INSERT INTO tracks(track_id, isrc) VALUES('track', 'USABC1234567')")
    conn.execute("INSERT INTO track_recording(track_id, recording_key) VALUES('track', 'library')")
    _candidate(conn, "candidate", "Candidate", 1.0)
    conn.execute(
        "UPDATE candidate_recordings SET status='embedded' WHERE candidate_key='candidate'"
    )
    conn.execute(
        "INSERT INTO candidate_external_ids(candidate_key, provider, external_id) "
        "VALUES('candidate', 'isrc', 'USABC1234567')"
    )
    conn.commit()

    result = candidate_embeddings.rerank(conn)

    assert result == {"ranked": 0, "duplicates": 1}
    assert (
        conn.execute(
            "SELECT status FROM candidate_recordings WHERE candidate_key='candidate'"
        ).fetchone()[0]
        == "duplicate"
    )
    duplicate = conn.execute(
        "SELECT library_recording_key, method FROM candidate_duplicates "
        "WHERE candidate_key='candidate'"
    ).fetchone()
    assert tuple(duplicate) == ("library", "isrc")


def test_library_similarity_matrix_excludes_candidate_vectors(tmp_path) -> None:
    conn = _conn(tmp_path)
    conn.execute(
        "INSERT INTO recording_stats(recording_key, artist, title, plays, liked, homed) "
        "VALUES('library', 'Library', 'Track', 5, 1, 1)"
    )
    _candidate(conn, "candidate", "Candidate", 1.0)
    _embedding(conn, "library", similar.EMBED_MODEL, [1.0, 0.0])
    _embedding(conn, "candidate", similar.EMBED_MODEL, [0.0, 1.0])
    conn.commit()
    similar._MAT_CACHE.clear()

    keys, matrix = similar._matrix(conn)

    assert keys == ["library"]
    assert matrix.shape == (1, 2)
    similar._MAT_CACHE.clear()
