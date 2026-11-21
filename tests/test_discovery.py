from __future__ import annotations

import sqlite3

from acetate import discovery, proposals
from tests.conftest import FakeSpotify, add_embedding, cluster_vector


def _candidate(
    conn: sqlite3.Connection,
    key: str,
    artist: str,
    title: str,
    axis: int,
    seed: int,
    combined: float,
    spotify_id: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO candidate_recordings(candidate_key, artist, title, status, score) "
        "VALUES(?,?,?,'embedded',?)",
        (key, artist, title, combined),
    )
    add_embedding(conn, key, cluster_vector(axis, 0.06, seed=seed))
    conn.execute(
        "INSERT INTO candidate_rerank(candidate_key, provenance_score, provenance_norm, "
        "effnet_similarity, clap_similarity, style_similarity, content_score, "
        "combined_score) VALUES(?,?,?,?,?,?,?,?)",
        (key, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8, combined),
    )
    if spotify_id:
        conn.execute(
            "INSERT INTO candidate_external_ids(candidate_key, provider, external_id, "
            "id_type, confidence) VALUES(?,'spotify',?,'track_id',1.0)",
            (key, spotify_id),
        )


def test_for_playlist_prefers_matching_candidates(library) -> None:
    conn = library
    _candidate(
        conn,
        "cand\ttechno",
        "Cand",
        "Techno Cut",
        axis=0,
        seed=1,
        combined=0.7,
        spotify_id="sp-tech",
    )
    _candidate(conn, "cand\tfolk", "Cand", "Folk Cut", axis=1, seed=2, combined=0.7)
    conn.commit()
    res = discovery.for_playlist(conn, "P1", "Warehouse Techno", limit=1)
    assert res["proposed"] == 1
    (row,) = proposals.pending(conn, source="discovery")
    assert row["recording_key"] == "cand\ttechno"
    assert row["track_id"] == "sp-tech"
    assert row["kind"] == "candidate"
    # surfacing was logged for the feedback loop
    n = conn.execute(
        "SELECT COUNT(*) FROM recommendation_events WHERE surface='proposal:discovery'"
    ).fetchone()[0]
    assert n == 1


def test_duplicates_are_excluded(library) -> None:
    conn = library
    _candidate(
        conn, "cand\tdupe", "Cand", "Dupe", axis=0, seed=3, combined=0.9, spotify_id="sp-dupe"
    )
    conn.execute(
        "INSERT INTO candidate_duplicates(candidate_key, library_recording_key, "
        "method) VALUES('cand\tdupe','techno artist\tt0','isrc')"
    )
    conn.commit()
    res = discovery.for_playlist(conn, "P1", "Warehouse Techno", limit=5)
    assert res["proposed"] == 0


def test_build_discovery_playlist_rotates_and_logs(library) -> None:
    conn = library
    for i in range(3):
        _candidate(
            conn,
            f"cand\tc{i}",
            "Cand",
            f"C{i}",
            axis=0,
            seed=10 + i,
            combined=0.9 - i * 0.1,
            spotify_id=f"sp-{i}",
        )
    _candidate(conn, "cand\tnospotify", "Cand", "NoId", axis=0, seed=20, combined=0.95)
    conn.commit()
    sp = FakeSpotify()
    res = discovery.build_discovery_playlist(conn, sp, size=2)
    assert res["tracks"] == 2
    assert res["no_spotify_id"] == 1  # the id-less top candidate was skipped, visibly
    pid = res["playlist_id"]
    assert sp.replaced[pid] == ["spotify:track:sp-0", "spotify:track:sp-1"]
    # second run rotates: the never-surfaced candidate comes first now
    discovery.build_discovery_playlist(conn, sp, size=2)
    assert sp.replaced[pid][0] == "spotify:track:sp-2"
    events = conn.execute(
        "SELECT COUNT(*) FROM recommendation_events WHERE surface='discovery_playlist'"
    ).fetchone()[0]
    assert events == 4
