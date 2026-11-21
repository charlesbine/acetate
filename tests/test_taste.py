from __future__ import annotations

from acetate import taste


def test_authored_playlists_finds_owner_and_excludes_managed(library) -> None:
    conn = library
    conn.execute(
        "INSERT INTO managed_playlists(kind, key, playlist_id, name) "
        "VALUES('smart','1','P2','Quiet Folk')"
    )
    pls = taste.authored_playlists(conn)
    assert [p["playlist_id"] for p in pls] == ["P1"]
    assert taste.user_owner_id(conn) == "alice"


def test_build_creates_regions_and_prunes_stale(library) -> None:
    conn = library
    res = taste.build(conn)
    assert res["regions_built"] == 2
    names = {r["name"] for r in taste.regions(conn)}
    assert names == {"Warehouse Techno", "Quiet Folk"}
    # a playlist that shrinks below MIN_EMBEDDED loses its region on rebuild
    conn.execute("DELETE FROM playlist_track_snapshots WHERE playlist_id='P2' AND position > 2")
    conn.commit()
    res = taste.build(conn)
    assert res["regions_built"] == 1
    assert {r["name"] for r in taste.regions(conn)} == {"Warehouse Techno"}


def test_membership_separates_clusters(library) -> None:
    region = taste.load_region(library, "P1", "Warehouse Techno")
    scores = taste.membership(library, region, ["techno artist\tt9", "folk artist\tf0"])
    assert scores["techno artist\tt9"] > scores["folk artist\tf0"] + 0.2


def test_misfits_flags_planted_outlier(library) -> None:
    region = taste.load_region(library, "P1", "Warehouse Techno")
    worst = taste.misfits(library, region, limit=1)[0]
    assert worst["recording_key"] == "folk artist\tfodd"
    assert worst["z"] < -1


def test_additions_surfaces_gap_fill_target(library) -> None:
    region = taste.load_region(library, "P1", "Warehouse Techno")
    adds = taste.additions(library, region, limit=3)
    assert adds[0]["recording_key"] == "techno artist\tt9"
    member_keys = set(taste.playlist_recordings(library, "P1"))
    assert all(a["recording_key"] not in member_keys for a in adds)


def test_evaluate_beats_chance(library) -> None:
    rows = taste.evaluate(library, holdout_frac=0.25, negatives_per_pos=5)
    assert rows, "expected at least one evaluable playlist"
    assert all(r["auc"] > 0.6 for r in rows)
