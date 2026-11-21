from __future__ import annotations

import csv
import io
import json
import sqlite3

from acetate import candidates, db


def _conn(tmp_path) -> sqlite3.Connection:
    conn = db.connect(tmp_path / "acetate-test.db")
    db.init_db(conn)
    return conn


def _recording(
    conn: sqlite3.Connection, key: str, artist: str, title: str, plays: int = 0, liked: int = 0
) -> None:
    conn.execute(
        """
        INSERT INTO recording_stats(recording_key, artist, title, plays, liked, homed)
        VALUES(?, ?, ?, ?, ?, 0)
        """,
        (key, artist, title, plays, liked),
    )


def _lastfm_edge(
    conn: sqlite3.Connection,
    seed_key: str,
    artist: str,
    title: str,
    match: float,
    sim_recording_key: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO lastfm_similar(
            recording_key, sim_artist, sim_title, sim_recording_key, match, updated_at
        )
        VALUES(?, ?, ?, ?, ?, '2026-07-10T00:00:00+00:00')
        """,
        (seed_key, artist, title, sim_recording_key, match),
    )


def test_promote_lastfm_creates_candidates_and_sources(tmp_path) -> None:
    conn = _conn(tmp_path)
    _recording(conn, "seed artist\tseed title", "Seed Artist", "Seed Title", plays=50, liked=1)
    _recording(conn, "other seed\tother title", "Other Seed", "Other Title", plays=10)
    _recording(conn, "known artist\tknown title", "Known Artist", "Known Title", plays=2)
    _lastfm_edge(conn, "seed artist\tseed title", "New Artist", "New Track", 0.9)
    _lastfm_edge(conn, "other seed\tother title", "New Artist", "New Track", 0.7)
    _lastfm_edge(conn, "seed artist\tseed title", "Known Artist", "Known Title", 1.0)
    _lastfm_edge(
        conn,
        "seed artist\tseed title",
        "Already Linked",
        "Library Track",
        1.0,
        sim_recording_key="already linked\tlibrary track",
    )
    conn.commit()

    result = candidates.promote_lastfm(conn)

    assert result["sources"] == 2
    assert result["skipped_library"] == 1
    row = conn.execute("SELECT * FROM candidate_recordings").fetchone()
    assert row["candidate_key"] == "new artist\tnew track"
    assert row["artist"] == "New Artist"
    assert row["title"] == "New Track"
    assert row["score"] > 0
    assert row["source_count"] == 2
    assert conn.execute("SELECT COUNT(*) FROM candidate_sources").fetchone()[0] == 2


def test_promote_lastfm_is_idempotent_for_cached_edges(tmp_path) -> None:
    conn = _conn(tmp_path)
    _recording(conn, "seed artist\tseed title", "Seed Artist", "Seed Title", plays=25, liked=1)
    _lastfm_edge(conn, "seed artist\tseed title", "New Artist", "New Track", 0.8)
    conn.commit()

    candidates.promote_lastfm(conn)
    candidates.promote_lastfm(conn)

    assert conn.execute("SELECT COUNT(*) FROM candidate_recordings").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM candidate_sources").fetchone()[0] == 1
    source = conn.execute("SELECT occurrences, strength FROM candidate_sources").fetchone()
    assert source["occurrences"] == 1
    assert source["strength"] == 0.8


def test_top_candidates_includes_provenance_reasons(tmp_path) -> None:
    conn = _conn(tmp_path)
    _recording(conn, "seed artist\tseed title", "Seed Artist", "Seed Title", plays=100, liked=1)
    _lastfm_edge(conn, "seed artist\tseed title", "New Artist", "New Track", 0.75)
    conn.commit()
    candidates.promote_lastfm(conn)

    rows = candidates.top_candidates(conn, limit=1)

    assert rows == [
        {
            "candidate_key": "new artist\tnew track",
            "artist": "New Artist",
            "title": "New Track",
            "score": rows[0]["score"],
            "source_count": 1,
            "last_seen": rows[0]["last_seen"],
            "reasons": rows[0]["reasons"],
        }
    ]
    assert rows[0]["score"] > 0
    assert rows[0]["reasons"][0]["source"] == candidates.LASTFM_SOURCE
    assert rows[0]["reasons"][0]["seed_artist"] == "Seed Artist"


def test_review_candidates_includes_all_provenance_and_external_ids(tmp_path) -> None:
    conn = _conn(tmp_path)
    for index in range(4):
        candidates.add_candidate_source(
            conn,
            artist="New Artist",
            title="New Track",
            source=f"source-{index}",
            reason=f"reason-{index}",
            seed_recording_key=f"seed-{index}",
            strength=1.0 - index / 10,
            raw={"index": index},
            external_ids=[
                {
                    "provider": "ytmusic",
                    "external_id": "video-1",
                    "id_type": "video_id",
                    "confidence": 0.9,
                    "raw": {"kind": "song"},
                }
            ]
            if index == 0
            else None,
            known=set(),
        )
    conn.commit()
    candidates.score_candidates(conn)

    rows = candidates.review_candidates(conn, limit=1)

    assert len(rows) == 1
    assert rows[0]["source_count"] == 4
    assert len(rows[0]["provenance"]) == 4
    assert rows[0]["provenance"][0]["raw"] == {"index": 0}
    assert len(rows[0]["external_ids"]) == 1
    assert rows[0]["external_ids"][0]["provider"] == "ytmusic"
    assert rows[0]["external_ids"][0]["external_id"] == "video-1"
    assert rows[0]["external_ids"][0]["raw"] == {"kind": "song"}


def test_candidate_export_json_and_csv_preserve_provenance(tmp_path) -> None:
    conn = _conn(tmp_path)
    candidates.add_candidate_source(
        conn,
        artist="New Artist",
        title="New Track",
        source="ytmusic_radio",
        reason="watch radio",
        seed_artist="Seed Artist",
        seed_title="Seed Track",
        source_rank=2,
        strength=0.8,
        raw={"videoId": "video-1"},
        known=set(),
    )
    conn.commit()
    candidates.score_candidates(conn)
    rows = candidates.review_candidates(conn)

    json_rows = json.loads(candidates.serialize_candidate_export(rows, "json"))
    csv_rows = list(csv.DictReader(io.StringIO(candidates.serialize_candidate_export(rows, "csv"))))

    assert json_rows[0]["provenance"][0]["raw"] == {"videoId": "video-1"}
    assert len(csv_rows) == 1
    assert csv_rows[0]["source"] == "ytmusic_radio"
    assert json.loads(csv_rows[0]["source_raw_json"]) == {"videoId": "video-1"}
