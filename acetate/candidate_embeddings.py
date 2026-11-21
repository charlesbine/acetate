"""Finalize and rerank embedded outside-library recommendation candidates."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

import numpy as np

from . import candidates, db

EFFNET_MODEL = "discogs-effnet"
STYLE_MODEL = "discogs400"
CLAP_MODEL = "clap-music"
REQUIRED_MODELS = (EFFNET_MODEL, STYLE_MODEL, CLAP_MODEL)

PROVENANCE_WEIGHT = 0.45
EFFNET_WEIGHT = 0.55
CLAP_WEIGHT = 0.30
STYLE_WEIGHT = 0.15

# Bump when the ranking algorithm changes in a way not already represented by
# the models, weights, or top-k setting below. A changed signature forces one
# complete rebuild and then returns to incremental passes.
RERANK_ALGORITHM_VERSION = 2
RERANK_STATE_KEY = "candidate_rerank_state"


def finalize(conn: sqlite3.Connection) -> int:
    """Advance preview candidates only after all three model vectors exist."""
    now = db.utcnow()
    cursor = conn.execute(
        """
        UPDATE candidate_recordings
        SET status = 'embedded', updated_at = ?
        WHERE status = 'preview'
          AND EXISTS (SELECT 1 FROM embeddings e
                      WHERE e.recording_key = candidate_recordings.candidate_key
                        AND e.model = 'discogs-effnet')
          AND EXISTS (SELECT 1 FROM embeddings e
                      WHERE e.recording_key = candidate_recordings.candidate_key
                        AND e.model = 'discogs400')
          AND EXISTS (SELECT 1 FROM embeddings e
                      WHERE e.recording_key = candidate_recordings.candidate_key
                        AND e.model = 'clap-music')
        """,
        (now,),
    )
    conn.commit()
    return cursor.rowcount


def _normalized_vector(blob: bytes) -> np.ndarray:
    vector = np.frombuffer(blob, dtype=np.float32)
    return vector / (np.linalg.norm(vector) + 1e-9)


def _space_scores(
    conn: sqlite3.Connection,
    *,
    model: str,
    candidate_keys: list[str],
    top_k: int,
) -> dict[str, dict[str, Any]]:
    library_rows = conn.execute(
        """
        SELECT e.recording_key, e.vector
        FROM embeddings e
        JOIN recording_stats rs ON rs.recording_key = e.recording_key
        WHERE e.model = ? AND (rs.liked = 1 OR rs.homed = 1 OR rs.plays >= 5)
        """,
        (model,),
    ).fetchall()
    if not library_rows:
        return {}
    library_keys = [row["recording_key"] for row in library_rows]
    matrix = np.vstack([_normalized_vector(row["vector"]) for row in library_rows])

    out = {}
    for candidate_key in candidate_keys:
        row = conn.execute(
            "SELECT vector FROM embeddings WHERE recording_key=? AND model=?",
            (candidate_key, model),
        ).fetchone()
        if row is None:
            continue
        vector = _normalized_vector(row["vector"])
        if vector.size != matrix.shape[1]:
            continue
        similarities = matrix @ vector
        count = min(max(1, top_k), len(similarities))
        order = np.argsort(-similarities)[:count]
        out[candidate_key] = {
            "similarity": float(np.clip(similarities[order].mean(), 0.0, 1.0)),
            "neighbor_key": library_keys[int(order[0])],
        }
    return out


def mark_isrc_duplicates(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        """
        SELECT DISTINCT cr.candidate_key, ce.external_id AS isrc
        FROM candidate_recordings cr
        JOIN candidate_external_ids ce
          ON ce.candidate_key = cr.candidate_key AND ce.provider = 'isrc'
        WHERE cr.status = 'embedded'
        """
    ).fetchall()
    duplicates = 0
    for row in rows:
        library_key = candidates.library_recording_for_isrc(conn, row["isrc"])
        if not library_key:
            continue
        candidates.mark_library_duplicate(
            conn,
            candidate_key=row["candidate_key"],
            library_recording_key=library_key,
            method="isrc",
            external_id=row["isrc"],
        )
        conn.execute("DELETE FROM candidate_rerank WHERE candidate_key=?", (row["candidate_key"],))
        duplicates += 1
    conn.commit()
    return duplicates


def _hash_parts(parts: list[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _reference_fingerprint(conn: sqlite3.Connection) -> str:
    """Fingerprint only library inputs that can affect nearest neighbours.

    Play-count changes above the >=5 inclusion boundary do not matter, while a
    new qualifying recording or regenerated vector does. Workers update the
    embedding timestamp whenever they replace a vector, so hashing metadata is
    enough and avoids rereading tens of megabytes of vector blobs every pass.
    """
    rows = conn.execute(
        """
        SELECT rs.recording_key, e.model, e.dim, length(e.vector) AS vector_bytes,
               COALESCE(e.updated_at, '') AS embedding_updated_at
        FROM recording_stats rs
        JOIN embeddings e ON e.recording_key = rs.recording_key
        WHERE (rs.liked = 1 OR rs.homed = 1 OR rs.plays >= 5)
          AND e.model IN (?, ?, ?)
        ORDER BY rs.recording_key, e.model
        """,
        REQUIRED_MODELS,
    ).fetchall()
    return _hash_parts(
        [
            f"{row['recording_key']}|{row['model']}|{row['dim']}|"
            f"{row['vector_bytes']}|{row['embedding_updated_at']}"
            for row in rows
        ]
    )


def _scoring_fingerprint(top_k: int) -> str:
    return _hash_parts(
        [
            str(RERANK_ALGORITHM_VERSION),
            str(top_k),
            *REQUIRED_MODELS,
            *(
                float(weight).hex()
                for weight in (PROVENANCE_WEIGHT, EFFNET_WEIGHT, CLAP_WEIGHT, STYLE_WEIGHT)
            ),
        ]
    )


def _candidate_input_fingerprints(
    conn: sqlite3.Connection,
    rows: list[sqlite3.Row],
) -> dict[str, str]:
    parts = {row["candidate_key"]: [float(row["score"] or 0).hex()] for row in rows}
    embedding_rows = conn.execute(
        """
        SELECT e.recording_key, e.model, e.dim, length(e.vector) AS vector_bytes,
               COALESCE(e.updated_at, '') AS embedding_updated_at
        FROM embeddings e
        JOIN candidate_recordings cr ON cr.candidate_key = e.recording_key
        WHERE cr.status = 'embedded' AND e.model IN (?, ?, ?)
        ORDER BY e.recording_key, e.model
        """,
        REQUIRED_MODELS,
    ).fetchall()
    for row in embedding_rows:
        parts[row["recording_key"]].append(
            f"{row['model']}|{row['dim']}|{row['vector_bytes']}|{row['embedding_updated_at']}"
        )
    return {key: _hash_parts(value) for key, value in parts.items()}


def _load_rerank_state(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute("SELECT value FROM kv_state WHERE key=?", (RERANK_STATE_KEY,)).fetchone()
    if row is None:
        return {}
    try:
        value = json.loads(row["value"])
        return value if isinstance(value, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def rerank(
    conn: sqlite3.Connection,
    *,
    top_k: int = 10,
    force: bool = False,
) -> dict[str, int]:
    """Blend provenance with library affinity, rebuilding only stale rows.

    Existing candidates are recalculated when their score/vector inputs change.
    A reference-library, normalization-bound, top-k, model, weight, or algorithm
    change invalidates every row and automatically performs a full rebuild.
    """
    duplicates = mark_isrc_duplicates(conn)
    rows = conn.execute(
        """
        SELECT cr.candidate_key, cr.score, rr.input_fingerprint
        FROM candidate_recordings cr
        LEFT JOIN candidate_rerank rr ON rr.candidate_key = cr.candidate_key
        WHERE cr.status = 'embedded'
        ORDER BY cr.score DESC
        """
    ).fetchall()
    if not rows:
        return {"ranked": 0, "duplicates": duplicates}
    candidate_keys = [row["candidate_key"] for row in rows]
    scores = np.asarray([float(row["score"] or 0) for row in rows], dtype=np.float64)
    bounds = [float(scores.min()).hex(), float(scores.max()).hex()]
    next_state = {
        "reference": _reference_fingerprint(conn),
        "scoring": _scoring_fingerprint(top_k),
        "bounds": bounds,
    }
    full_rebuild = force or _load_rerank_state(conn) != next_state
    input_fingerprints = _candidate_input_fingerprints(conn, rows)
    selected_rows = [
        row
        for row in rows
        if full_rebuild or row["input_fingerprint"] != input_fingerprints[row["candidate_key"]]
    ]
    if not selected_rows:
        return {"ranked": 0, "duplicates": duplicates}
    selected_keys = [row["candidate_key"] for row in selected_rows]

    if float(scores.max()) == float(scores.min()):
        provenance_norm = np.ones_like(scores)
    else:
        provenance_norm = (scores - scores.min()) / (scores.max() - scores.min())
    normalized_by_key = {
        key: float(provenance_norm[index]) for index, key in enumerate(candidate_keys)
    }

    effnet = _space_scores(conn, model=EFFNET_MODEL, candidate_keys=selected_keys, top_k=top_k)
    clap = _space_scores(conn, model=CLAP_MODEL, candidate_keys=selected_keys, top_k=top_k)
    style = _space_scores(conn, model=STYLE_MODEL, candidate_keys=selected_keys, top_k=top_k)

    now = db.utcnow()
    ranked = 0
    for row in selected_rows:
        key = row["candidate_key"]
        if key not in effnet or key not in clap or key not in style:
            continue
        effnet_similarity = effnet[key]["similarity"]
        clap_similarity = clap[key]["similarity"]
        style_similarity = style[key]["similarity"]
        content_score = (
            EFFNET_WEIGHT * effnet_similarity
            + CLAP_WEIGHT * clap_similarity
            + STYLE_WEIGHT * style_similarity
        )
        combined_score = (
            PROVENANCE_WEIGHT * normalized_by_key[key] + (1.0 - PROVENANCE_WEIGHT) * content_score
        )
        conn.execute(
            """
            INSERT INTO candidate_rerank(
                candidate_key, provenance_score, provenance_norm,
                effnet_similarity, clap_similarity, style_similarity,
                content_score, combined_score, effnet_neighbor_key,
                clap_neighbor_key, style_neighbor_key, input_fingerprint, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(candidate_key) DO UPDATE SET
                provenance_score=excluded.provenance_score,
                provenance_norm=excluded.provenance_norm,
                effnet_similarity=excluded.effnet_similarity,
                clap_similarity=excluded.clap_similarity,
                style_similarity=excluded.style_similarity,
                content_score=excluded.content_score,
                combined_score=excluded.combined_score,
                effnet_neighbor_key=excluded.effnet_neighbor_key,
                clap_neighbor_key=excluded.clap_neighbor_key,
                style_neighbor_key=excluded.style_neighbor_key,
                input_fingerprint=excluded.input_fingerprint,
                updated_at=excluded.updated_at
            """,
            (
                key,
                float(row["score"] or 0),
                normalized_by_key[key],
                effnet_similarity,
                clap_similarity,
                style_similarity,
                content_score,
                combined_score,
                effnet[key]["neighbor_key"],
                clap[key]["neighbor_key"],
                style[key]["neighbor_key"],
                input_fingerprints[key],
                now,
            ),
        )
        ranked += 1
    conn.execute(
        """
        INSERT INTO kv_state(key, value, updated_at) VALUES(?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """,
        (RERANK_STATE_KEY, json.dumps(next_state, sort_keys=True), now),
    )
    conn.commit()
    return {"ranked": ranked, "duplicates": duplicates}


def ranked_candidates(conn: sqlite3.Connection, *, limit: int = 25) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT cr.candidate_key, cr.artist, cr.title,
               rr.provenance_score, rr.effnet_similarity, rr.clap_similarity,
               rr.style_similarity, rr.content_score, rr.combined_score,
               ers.artist AS effnet_neighbor_artist, ers.title AS effnet_neighbor_title,
               crs.artist AS clap_neighbor_artist, crs.title AS clap_neighbor_title,
               (SELECT style FROM recording_styles s
                WHERE s.recording_key = cr.candidate_key AND s.rank = 1) AS top_style
        FROM candidate_rerank rr
        JOIN candidate_recordings cr ON cr.candidate_key = rr.candidate_key
        LEFT JOIN recording_stats ers ON ers.recording_key = rr.effnet_neighbor_key
        LEFT JOIN recording_stats crs ON crs.recording_key = rr.clap_neighbor_key
        WHERE cr.status = 'embedded'
        ORDER BY rr.combined_score DESC, rr.content_score DESC, rr.provenance_score DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def status(conn: sqlite3.Connection) -> dict[str, int]:
    out = {
        f"candidate_embedding:{row['model']}:{row['status']}": row["n"]
        for row in conn.execute(
            "SELECT model, status, COUNT(*) AS n "
            "FROM candidate_embedding_runs GROUP BY model, status"
        )
    }
    out["candidate_reranked"] = conn.execute("SELECT COUNT(*) FROM candidate_rerank").fetchone()[0]
    return out
