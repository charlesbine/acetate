from __future__ import annotations

from acetate import health, proposals, taste
from tests.conftest import add_embedding, add_playlist, add_recording, cluster_vector


def test_report_metrics_and_misfit(library) -> None:
    rep = health.report(library, "P1", "Warehouse Techno")
    assert rep["members"] == 9  # 8 techno + the planted folk misfit
    assert rep["embedded"] == 9
    assert 0 < rep["sonic_cohesion"] <= 1
    misfit_keys = {m["recording_key"] for m in rep["misfits"]}
    assert "folk artist\tfodd" in misfit_keys


def test_colisten_density(library) -> None:
    keys = taste.playlist_recordings(library, "P1")
    assert health.colisten_density(library, keys) == 0.0
    library.execute(
        "INSERT INTO lastfm_similar(recording_key, sim_artist, sim_title, "
        "sim_recording_key, match, updated_at) "
        "VALUES('techno artist\tt0','Techno Artist','T1','techno artist\tt1',0.9,'now')"
    )
    library.commit()
    assert health.colisten_density(library, keys) > 0


def test_split_check_flags_bimodal_playlist(conn) -> None:
    ids = []
    for i in range(8):
        key, tid = f"a\ts{i}", f"tid-s{i}"
        add_recording(conn, key, "A", f"S{i}", tid)
        add_embedding(conn, key, cluster_vector(0, 0.05, seed=i))
        ids.append(tid)
    for i in range(8):
        key, tid = f"b\ts{i}", f"tid-b{i}"
        add_recording(conn, key, "B", f"S{i}", tid)
        add_embedding(conn, key, cluster_vector(1, 0.05, seed=50 + i))
        ids.append(tid)
    add_playlist(conn, "PX", "Two Genres Mashed", ids)
    conn.commit()
    region = taste.load_region(conn, "PX", "Two Genres Mashed")
    split = health.split_check(region)
    assert split is not None
    assert sorted(g["size"] for g in split["groups"]) == [8, 8]
    # a coherent playlist must NOT trigger the split
    coherent = taste.load_region(conn, "PX", "x")
    coherent.matrix = coherent.matrix[:8]
    coherent.member_keys = coherent.member_keys[:8]
    centroid = coherent.matrix.mean(axis=0)
    coherent.centroid = centroid / ((centroid**2).sum() ** 0.5)
    assert health.split_check(coherent) is None


def test_scan_files_proposals(library) -> None:
    res = health.scan(library, "P1", "Warehouse Techno")
    assert res["proposed"] >= 2  # at least the misfit removal + t9 gap-fill
    rows = proposals.pending(library, playlist_id="P1")
    actions = {(r["action"], r["recording_key"]) for r in rows}
    assert ("remove", "folk artist\tfodd") in actions
    assert ("add", "techno artist\tt9") in actions
    # re-scan is idempotent for pending rows
    health.scan(library, "P1", "Warehouse Techno")
    assert len(proposals.pending(library, playlist_id="P1")) == len(rows)


def test_matches_only_scan_queues_additions_without_removals(library) -> None:
    res = health.scan(library, "P1", "Warehouse Techno", include_misfits=False)
    assert res["additions"] >= 1
    assert res["removals"] == 0
    actions = {r["action"] for r in proposals.pending(library, playlist_id="P1")}
    assert actions == {"add"}
