"""Outside-library recommendation candidates.

The candidate ledger is the cheap frontier before audio work: every row is a
track-like thing not currently in the library, and every source row explains why
we found it. Expensive preview fetching and embedding should only run after this
metadata/provenance layer has narrowed the field.
"""

from __future__ import annotations

import csv
import io
import json
import math
import sqlite3
from typing import Any, Iterable

from . import db
from .canon import recording_key


LASTFM_SOURCE = "lastfm_track_similar"

_SOURCE_WEIGHT = {
    LASTFM_SOURCE: 1.0,
    "lastfm_artist_top_track": 0.55,
    "ytmusic_radio": 1.4,
    "deezer_related_artist": 0.7,
    "deezer_artist_top_track": 0.55,
    "deezer_same_artist": 0.3,
    "deezer_same_album": 0.45,
    "spotify_same_album": 0.35,
    "spotify_same_artist": 0.25,
    "spotify_artist_top_track": 0.45,
    "spotify_appears_on_album": 0.5,
    "spotify_compilation_copresence": 0.2,
}


def _library_keys(conn: sqlite3.Connection) -> set[str]:
    return {
        r["recording_key"]
        for r in conn.execute(
            "SELECT recording_key FROM recording_stats WHERE recording_key IS NOT NULL"
        )
    }


def add_external_id(
    conn: sqlite3.Connection,
    *,
    candidate_key: str,
    provider: str,
    external_id: str,
    id_type: str | None = None,
    confidence: float | None = None,
    raw: dict[str, Any] | None = None,
) -> None:
    now = db.utcnow()
    conn.execute(
        """
        INSERT INTO candidate_external_ids(
            candidate_key, provider, external_id, id_type, confidence, raw_json, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(candidate_key, provider, external_id) DO UPDATE SET
            id_type=COALESCE(excluded.id_type, candidate_external_ids.id_type),
            confidence=COALESCE(excluded.confidence, candidate_external_ids.confidence),
            raw_json=COALESCE(excluded.raw_json, candidate_external_ids.raw_json),
            updated_at=excluded.updated_at
        """,
        (
            candidate_key,
            provider,
            external_id,
            id_type,
            confidence,
            json.dumps(raw, sort_keys=True) if raw else None,
            now,
        ),
    )


def library_recording_for_isrc(conn: sqlite3.Connection, isrc: str) -> str | None:
    row = conn.execute(
        """
        SELECT tr.recording_key
        FROM tracks t
        JOIN track_recording tr ON tr.track_id = t.track_id
        JOIN recording_stats rs ON rs.recording_key = tr.recording_key
        WHERE t.isrc = ?
        ORDER BY rs.liked DESC, rs.plays DESC
        LIMIT 1
        """,
        (isrc,),
    ).fetchone()
    return row["recording_key"] if row else None


def mark_library_duplicate(
    conn: sqlite3.Connection,
    *,
    candidate_key: str,
    library_recording_key: str,
    method: str,
    external_id: str | None = None,
) -> None:
    now = db.utcnow()
    conn.execute(
        """
        INSERT INTO candidate_duplicates(
            candidate_key, library_recording_key, method, external_id, updated_at
        )
        VALUES(?, ?, ?, ?, ?)
        ON CONFLICT(candidate_key) DO UPDATE SET
            library_recording_key=excluded.library_recording_key,
            method=excluded.method,
            external_id=excluded.external_id,
            updated_at=excluded.updated_at
        """,
        (candidate_key, library_recording_key, method, external_id, now),
    )
    conn.execute(
        "UPDATE candidate_recordings SET status='duplicate', updated_at=? WHERE candidate_key=?",
        (now, candidate_key),
    )


def add_candidate_source(
    conn: sqlite3.Connection,
    *,
    artist: str,
    title: str,
    source: str,
    reason: str,
    seed_recording_key: str = "",
    seed_artist: str | None = None,
    seed_title: str | None = None,
    seed_external_id: str = "",
    source_rank: int | None = None,
    strength: float | None = None,
    raw: dict[str, Any] | None = None,
    external_ids: list[dict[str, Any]] | None = None,
    known: set[str] | None = None,
) -> str | None:
    candidate_key = recording_key(artist, title)
    if not candidate_key.strip():
        return None
    if known is None:
        known = _library_keys(conn)
    if candidate_key in known:
        return None
    now = db.utcnow()
    _upsert_candidate(
        conn,
        candidate_key=candidate_key,
        artist=artist,
        title=title,
        now=now,
    )
    _upsert_source(
        conn,
        candidate_key=candidate_key,
        source=source,
        reason=reason,
        seed_recording_key=seed_recording_key,
        seed_artist=seed_artist,
        seed_title=seed_title,
        seed_external_id=seed_external_id,
        source_rank=source_rank,
        strength=strength,
        raw=raw,
        now=now,
    )
    for external in external_ids or []:
        external_id = external.get("external_id")
        provider = external.get("provider")
        if provider and external_id:
            add_external_id(
                conn,
                candidate_key=candidate_key,
                provider=provider,
                external_id=str(external_id),
                id_type=external.get("id_type"),
                confidence=external.get("confidence"),
                raw=external.get("raw"),
            )
    return candidate_key


def _upsert_candidate(
    conn: sqlite3.Connection,
    *,
    candidate_key: str,
    artist: str,
    title: str,
    now: str,
) -> None:
    conn.execute(
        """
        INSERT INTO candidate_recordings(candidate_key, artist, title, first_seen, last_seen, updated_at)
        VALUES(?, ?, ?, ?, ?, ?)
        ON CONFLICT(candidate_key) DO UPDATE SET
            artist=excluded.artist,
            title=excluded.title,
            last_seen=excluded.last_seen,
            updated_at=excluded.updated_at
        """,
        (candidate_key, artist, title, now, now, now),
    )


def _upsert_source(
    conn: sqlite3.Connection,
    *,
    candidate_key: str,
    source: str,
    reason: str,
    seed_recording_key: str = "",
    seed_artist: str | None = None,
    seed_title: str | None = None,
    seed_external_id: str = "",
    source_rank: int | None = None,
    strength: float | None = None,
    raw: dict[str, Any] | None = None,
    now: str,
) -> None:
    conn.execute(
        """
        INSERT INTO candidate_sources(
            candidate_key, source, reason, seed_recording_key, seed_artist, seed_title,
            seed_external_id, source_rank, strength, raw_json, first_seen, last_seen, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(candidate_key, source, reason, seed_recording_key, seed_external_id)
        DO UPDATE SET
            source_rank=CASE
                WHEN candidate_sources.source_rank IS NULL THEN excluded.source_rank
                WHEN excluded.source_rank IS NULL THEN candidate_sources.source_rank
                WHEN excluded.source_rank < candidate_sources.source_rank THEN excluded.source_rank
                ELSE candidate_sources.source_rank
            END,
            strength=max(COALESCE(candidate_sources.strength, 0), COALESCE(excluded.strength, 0)),
            occurrences=max(candidate_sources.occurrences, excluded.occurrences),
            raw_json=COALESCE(excluded.raw_json, candidate_sources.raw_json),
            last_seen=excluded.last_seen,
            updated_at=excluded.updated_at
        """,
        (
            candidate_key,
            source,
            reason,
            seed_recording_key,
            seed_artist,
            seed_title,
            seed_external_id,
            source_rank,
            strength,
            json.dumps(raw, sort_keys=True) if raw else None,
            now,
            now,
            now,
        ),
    )


def promote_lastfm(
    conn: sqlite3.Connection,
    *,
    batch: int | None = None,
    min_match: float = 0.0,
    rescore: bool = True,
    progress=None,
) -> dict[str, int]:
    """Promote out-of-library Last.fm similar-track edges into candidates.

    Existing `lastfm_similar` rows are our largest ready-made outside-library
    corpus. This copies them into the generic provenance ledger, skipping anything
    that canonicalizes back into the library.
    """
    known = _library_keys(conn)
    params: list[Any] = [min_match]
    limit = ""
    if batch:
        limit = "LIMIT ?"
        params.append(batch)
    rows = conn.execute(
        f"""
        SELECT ls.recording_key AS seed_recording_key, rs.artist AS seed_artist,
               rs.title AS seed_title, ls.sim_artist, ls.sim_title, ls.match
        FROM lastfm_similar ls
        LEFT JOIN recording_stats rs ON rs.recording_key = ls.recording_key
        WHERE ls.sim_recording_key IS NULL
          AND ls.sim_artist IS NOT NULL AND ls.sim_artist != ''
          AND ls.sim_title IS NOT NULL AND ls.sim_title != ''
          AND COALESCE(ls.match, 0) >= ?
        ORDER BY COALESCE(ls.match, 0) DESC
        {limit}
        """,
        params,
    ).fetchall()

    counts = {"edges_seen": 0, "candidates": 0, "sources": 0, "skipped_library": 0}
    seen_candidates: set[str] = set()
    for index, row in enumerate(rows):
        counts["edges_seen"] += 1
        candidate_key = recording_key(row["sim_artist"], row["sim_title"])
        if not candidate_key.strip() or candidate_key in known:
            counts["skipped_library"] += 1
            continue
        inserted_key = add_candidate_source(
            conn,
            artist=row["sim_artist"],
            title=row["sim_title"],
            source=LASTFM_SOURCE,
            reason="track.getSimilar",
            seed_recording_key=row["seed_recording_key"] or "",
            seed_artist=row["seed_artist"],
            seed_title=row["seed_title"],
            strength=float(row["match"] or 0),
            raw={
                "sim_artist": row["sim_artist"],
                "sim_title": row["sim_title"],
                "match": row["match"],
            },
            known=known,
        )
        if inserted_key:
            if candidate_key not in seen_candidates:
                seen_candidates.add(candidate_key)
                counts["candidates"] += 1
            counts["sources"] += 1
        if index % 1000 == 0:
            conn.commit()
            if progress:
                progress(index, len(rows), counts)
    conn.commit()
    if rescore:
        scored = score_candidates(conn)
        counts["scored"] = scored["scored"]
    return counts


def _source_score(row: sqlite3.Row) -> float:
    source = row["source"] or ""
    strength = row["strength"] if row["strength"] is not None else 0.25
    strength = max(0.05, min(float(strength), 1.0))
    occurrences = max(1, int(row["occurrences"] or 1))
    seed_plays = max(0, int(row["seed_plays"] or 0))
    seed_liked = bool(row["seed_liked"])
    source_weight = _SOURCE_WEIGHT.get(source, 0.5)
    seed_weight = 1.0 + min(2.0, math.log1p(seed_plays) / 4.0)
    if seed_liked:
        seed_weight *= 1.2
    repeat_weight = 1.0 + min(1.5, math.log1p(occurrences) / 3.0)
    rank_weight = 1.0
    if row["source_rank"]:
        rank_weight = 1.0 / max(1.0, math.sqrt(float(row["source_rank"])))
    return source_weight * strength * seed_weight * repeat_weight * rank_weight


def score_candidates(
    conn: sqlite3.Connection,
    *,
    status: str = "metadata",
    candidate_keys: Iterable[str] | None = None,
) -> dict[str, int]:
    """Recompute provenance scores for candidate recordings."""
    scope_join = ""
    if candidate_keys is not None:
        keys = set(candidate_keys)
        if not keys:
            return {"scored": 0}
        conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS candidate_score_scope (candidate_key TEXT PRIMARY KEY)"
        )
        conn.execute("DELETE FROM candidate_score_scope")
        conn.executemany(
            "INSERT INTO candidate_score_scope(candidate_key) VALUES(?)",
            [(key,) for key in keys],
        )
        scope_join = "JOIN candidate_score_scope scope ON scope.candidate_key = cs.candidate_key"
    rows = conn.execute(
        f"""
        SELECT cs.candidate_key, cs.source, cs.strength, cs.occurrences, cs.source_rank,
               COALESCE(rs.plays, 0) AS seed_plays, COALESCE(rs.liked, 0) AS seed_liked
        FROM candidate_sources cs
        {scope_join}
        JOIN candidate_recordings cr ON cr.candidate_key = cs.candidate_key
        LEFT JOIN recording_stats rs ON rs.recording_key = cs.seed_recording_key
        WHERE cr.status = ?
        """,
        (status,),
    ).fetchall()
    scores: dict[str, float] = {}
    source_counts: dict[str, int] = {}
    for row in rows:
        key = row["candidate_key"]
        scores[key] = scores.get(key, 0.0) + _source_score(row)
        source_counts[key] = source_counts.get(key, 0) + 1

    now = db.utcnow()
    params = [(score, source_counts[key], now, key) for key, score in scores.items()]
    # Parallel harvesters score concurrently; keep the write transaction short
    # (executemany, not a Python loop) and wait out a competing writer instead
    # of surfacing "database is locked" after a completed harvest.
    for attempt in range(3):
        try:
            conn.executemany(
                """
                UPDATE candidate_recordings
                SET score = ?, source_count = ?, updated_at = ?
                WHERE candidate_key = ?
                """,
                params,
            )
            conn.commit()
            break
        except sqlite3.OperationalError as exc:
            conn.rollback()
            if "locked" not in str(exc) or attempt == 2:
                raise
    return {"scored": len(scores)}


def top_candidates(
    conn: sqlite3.Connection, *, limit: int = 25, status: str = "metadata"
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT candidate_key, artist, title, score, source_count, last_seen
        FROM candidate_recordings
        WHERE status = ?
        ORDER BY score DESC, source_count DESC, last_seen DESC
        LIMIT ?
        """,
        (status, limit),
    ).fetchall()
    out = []
    for row in rows:
        reasons = conn.execute(
            """
            SELECT cs.source, cs.reason, cs.strength, cs.occurrences,
                   cs.seed_artist, cs.seed_title, rs.plays AS seed_plays
            FROM candidate_sources cs
            LEFT JOIN recording_stats rs ON rs.recording_key = cs.seed_recording_key
            WHERE cs.candidate_key = ?
            ORDER BY COALESCE(cs.strength, 0) DESC, COALESCE(rs.plays, 0) DESC
            LIMIT 3
            """,
            (row["candidate_key"],),
        ).fetchall()
        out.append(
            {
                "candidate_key": row["candidate_key"],
                "artist": row["artist"],
                "title": row["title"],
                "score": float(row["score"] or 0),
                "source_count": int(row["source_count"] or 0),
                "last_seen": row["last_seen"],
                "reasons": [dict(reason) for reason in reasons],
            }
        )
    return out


def review_candidates(
    conn: sqlite3.Connection, *, limit: int = 100, status: str = "metadata"
) -> list[dict[str, Any]]:
    """Return top candidates with complete provenance and resolver details."""
    rows = conn.execute(
        """
        SELECT candidate_key, artist, title, status, score, source_count,
               first_seen, last_seen, updated_at
        FROM candidate_recordings
        WHERE status = ?
        ORDER BY score DESC, source_count DESC, last_seen DESC
        LIMIT ?
        """,
        (status, limit),
    ).fetchall()
    out = []
    for row in rows:
        provenance_rows = conn.execute(
            """
            SELECT cs.id, cs.source, cs.reason, cs.seed_recording_key,
                   cs.seed_artist, cs.seed_title, cs.seed_external_id,
                   cs.source_rank, cs.strength, cs.occurrences, cs.raw_json,
                   cs.first_seen, cs.last_seen, cs.updated_at,
                   COALESCE(rs.plays, 0) AS seed_plays,
                   COALESCE(rs.liked, 0) AS seed_liked
            FROM candidate_sources cs
            LEFT JOIN recording_stats rs ON rs.recording_key = cs.seed_recording_key
            WHERE cs.candidate_key = ?
            ORDER BY COALESCE(cs.strength, 0) DESC,
                     COALESCE(rs.plays, 0) DESC,
                     cs.source, cs.reason, cs.id
            """,
            (row["candidate_key"],),
        ).fetchall()
        external_id_rows = conn.execute(
            """
            SELECT provider, external_id, id_type, confidence, raw_json, updated_at
            FROM candidate_external_ids
            WHERE candidate_key = ?
            ORDER BY provider, id_type, external_id
            """,
            (row["candidate_key"],),
        ).fetchall()

        provenance = []
        for source_row in provenance_rows:
            source = dict(source_row)
            source["raw"] = _decode_json(source.pop("raw_json"))
            provenance.append(source)
        external_ids = []
        for external_id_row in external_id_rows:
            external_id = dict(external_id_row)
            external_id["raw"] = _decode_json(external_id.pop("raw_json"))
            external_ids.append(external_id)

        candidate = dict(row)
        candidate["score"] = float(candidate["score"] or 0)
        candidate["source_count"] = int(candidate["source_count"] or 0)
        candidate["provenance"] = provenance
        candidate["external_ids"] = external_ids
        out.append(candidate)
    return out


def _decode_json(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


_CSV_FIELDS = (
    "candidate_key",
    "artist",
    "title",
    "status",
    "score",
    "source_count",
    "candidate_first_seen",
    "candidate_last_seen",
    "candidate_updated_at",
    "source_id",
    "source",
    "reason",
    "seed_recording_key",
    "seed_artist",
    "seed_title",
    "seed_external_id",
    "source_rank",
    "strength",
    "occurrences",
    "seed_plays",
    "seed_liked",
    "source_first_seen",
    "source_last_seen",
    "source_updated_at",
    "source_raw_json",
    "external_ids_json",
)


def serialize_candidate_export(rows: list[dict[str, Any]], output_format: str) -> str:
    """Serialize review rows as nested JSON or provenance-flat CSV."""
    output_format = output_format.lower()
    if output_format == "json":
        return json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if output_format != "csv":
        raise ValueError("format must be 'json' or 'csv'")

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=_CSV_FIELDS)
    writer.writeheader()
    for candidate in rows:
        external_ids_json = json.dumps(
            candidate.get("external_ids") or [], ensure_ascii=False, sort_keys=True
        )
        sources = candidate.get("provenance") or [None]
        for source in sources:
            source = source or {}
            writer.writerow(
                {
                    "candidate_key": candidate["candidate_key"],
                    "artist": candidate["artist"],
                    "title": candidate["title"],
                    "status": candidate["status"],
                    "score": candidate["score"],
                    "source_count": candidate["source_count"],
                    "candidate_first_seen": candidate.get("first_seen"),
                    "candidate_last_seen": candidate.get("last_seen"),
                    "candidate_updated_at": candidate.get("updated_at"),
                    "source_id": source.get("id"),
                    "source": source.get("source"),
                    "reason": source.get("reason"),
                    "seed_recording_key": source.get("seed_recording_key"),
                    "seed_artist": source.get("seed_artist"),
                    "seed_title": source.get("seed_title"),
                    "seed_external_id": source.get("seed_external_id"),
                    "source_rank": source.get("source_rank"),
                    "strength": source.get("strength"),
                    "occurrences": source.get("occurrences"),
                    "seed_plays": source.get("seed_plays"),
                    "seed_liked": source.get("seed_liked"),
                    "source_first_seen": source.get("first_seen"),
                    "source_last_seen": source.get("last_seen"),
                    "source_updated_at": source.get("updated_at"),
                    "source_raw_json": json.dumps(
                        source.get("raw"), ensure_ascii=False, sort_keys=True
                    )
                    if source.get("raw") is not None
                    else None,
                    "external_ids_json": external_ids_json,
                }
            )
    return buffer.getvalue()


def status(conn: sqlite3.Connection) -> dict[str, int]:
    out = {
        f"candidate:{r['status']}": r["n"]
        for r in conn.execute(
            "SELECT status, COUNT(*) AS n FROM candidate_recordings GROUP BY status"
        )
    }
    for row in conn.execute("SELECT source, COUNT(*) AS n FROM candidate_sources GROUP BY source"):
        out[f"source:{row['source']}"] = row["n"]
    out["external_ids"] = conn.execute("SELECT COUNT(*) FROM candidate_external_ids").fetchone()[0]
    return out
