"""The Smart Playlist 'candidates_for' similarity source: rank library gap-fill
and outside-library candidates by taste membership against an anchor playlist."""

from __future__ import annotations


from acetate import smart
from tests.conftest import add_embedding


def _add_candidate(conn, key, vec, *, spotify_id, status="embedded") -> None:
    conn.execute(
        "INSERT INTO candidate_recordings(candidate_key, artist, title, status) VALUES(?,?,?,?)",
        (key, "Cand Artist", key, status),
    )
    add_embedding(conn, key, vec)
    if spotify_id:
        conn.execute(
            "INSERT INTO candidate_external_ids(candidate_key, provider, external_id, "
            "id_type, confidence) VALUES(?,?,?,?,?)",
            (key, "spotify", spotify_id, "track_id", 1.0),
        )


def test_library_pool_returns_gapfill_excludes_members_and_offcluster(library) -> None:
    # t9 is liked techno NOT in the Warehouse Techno playlist → the gap-fill target;
    # folk tracks are off-cluster; P1 members are excluded.
    ids = smart.evaluate(library, {"candidates_for": "Warehouse Techno", "pool": "library"})
    assert "tid-t9" in ids  # the techno gap-fill surfaces
    assert "tid-fodd" not in ids  # a current member is excluded
    assert all(i not in ids for i in [f"tid-f{i}" for i in range(8)])  # folk stays out


def test_outside_pool_ranks_candidates_by_membership(library) -> None:
    # a near-cluster techno candidate and a far folk-cluster candidate
    from tests.conftest import cluster_vector

    _add_candidate(library, "cand-techno", cluster_vector(0, 0.05, seed=7), spotify_id="sp-techno")
    _add_candidate(library, "cand-folk", cluster_vector(1, 0.05, seed=7), spotify_id="sp-folk")
    library.commit()
    ids = smart.evaluate(library, {"candidates_for": "Warehouse Techno", "pool": "outside"})
    assert "sp-techno" in ids
    assert "sp-folk" not in ids  # below the member-p10 floor


def test_both_pool_merges_library_and_outside(library) -> None:
    from tests.conftest import cluster_vector

    _add_candidate(library, "cand-techno", cluster_vector(0, 0.05, seed=7), spotify_id="sp-techno")
    library.commit()
    ids = smart.evaluate(library, {"candidates_for": "Warehouse Techno", "pool": "both"})
    assert "tid-t9" in ids and "sp-techno" in ids


def test_duplicate_candidate_is_skipped(library) -> None:
    from tests.conftest import cluster_vector

    _add_candidate(library, "cand-dup", cluster_vector(0, 0.05, seed=7), spotify_id="sp-dup")
    library.execute(
        "INSERT INTO candidate_duplicates(candidate_key, library_recording_key, method) "
        "VALUES('cand-dup','techno artist\tt0','isrc')"
    )
    library.commit()
    ids = smart.evaluate(library, {"candidates_for": "Warehouse Techno", "pool": "outside"})
    assert "sp-dup" not in ids


def test_min_membership_floor_filters(library) -> None:
    high = smart.evaluate(
        library, {"candidates_for": "Warehouse Techno", "pool": "library", "min_membership": 0.99}
    )
    assert high == []  # nothing clears a 0.99 floor


def test_unknown_anchor_returns_empty(library) -> None:
    assert smart.evaluate(library, {"candidates_for": "No Such Playlist"}) == []


def test_limit_caps_results(library) -> None:
    ids = smart.evaluate(
        library,
        {
            "candidates_for": "Warehouse Techno",
            "pool": "library",
            "min_membership": -1.0,
            "limit": 3,
        },
    )
    assert len(ids) <= 3


def test_describe_summarizes_candidate_spec() -> None:
    d = smart.describe({"candidates_for": "Melodic techno", "pool": "both", "limit": 40})
    assert "candidates for" in d and "Melodic techno" in d and "library + outside" in d


# --- explicit hand-picked track list (the chat-agent curation path) ----------


def test_explicit_track_ids_returned_verbatim_in_order(library) -> None:
    ids = ["tid-t3", "tid-t1", "tid-t7"]
    assert smart.evaluate(library, {"track_ids": ids}) == ids


def test_explicit_track_ids_ignore_filter_fields(library) -> None:
    # a track_ids spec must NOT fall through to the SQL top-N (the reported bug)
    got = smart.evaluate(library, {"track_ids": ["tid-t5"], "sort": "plays_desc", "limit": 50})
    assert got == ["tid-t5"]


def test_explicit_track_ids_dedup_preserve_order_and_limit(library) -> None:
    spec = {"track_ids": ["a", "b", "a", "c", "b"], "limit": 2}
    assert smart.evaluate(library, spec) == ["a", "b"]


def test_describe_explicit_tracks(library) -> None:
    assert smart.describe({"track_ids": ["a", "b", "c"]}) == "3 hand-picked tracks"
    assert smart.describe({"track_ids": ["a"]}) == "1 hand-picked track"
