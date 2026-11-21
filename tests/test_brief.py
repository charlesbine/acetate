"""brief.py — the chat agent's playlist_report / playlist_fit backing."""

from __future__ import annotations

from acetate import brief, playlist_profile
from tests.conftest import add_playlist, add_recording


def test_norm_title_collapses_packaging_variants() -> None:
    same = [
        ("Sunset (feat. Marques Toliver)", "Sunset"),
        ("The Fifth Hand - Original Mix", "The Fifth Hand"),
        ("Sonnentanz - Original Version", "Sonnentanz"),
        ("What - Radio Edit", "What"),
        ("Still Can't Sleep - re-mastered", "Still Can't Sleep"),
        ("Crazy - Nôze Remix Extended Club Version", "Crazy - Nôze Remix - Extended Club Version"),
        ("6 Degrees (feat. Fiora) - Tale Of Us Remix", "6 Degrees - Tale Of Us Remix"),
    ]
    for a, b in same:
        assert brief._norm_title(a) == brief._norm_title(b), (a, b)
    # a remix is a different recording, not a packaging variant
    assert brief._norm_title("Crave") != brief._norm_title("Crave - ODESZA Remix")


def test_resolve_finds_playlist_and_family(library) -> None:
    add_playlist(library, "P1b", "Warehouse Techno (2)", ["tid-t0", "tid-t1"])
    library.commit()
    playlist_profile.rebuild(library)
    matches = brief.resolve(library, "warehouse techno")
    names = [m["name"] for m in matches]
    assert "Warehouse Techno" in names and "Warehouse Techno (2)" in names
    assert brief.resolve(library, "P1")[0]["name"] == "Warehouse Techno"  # id passthrough
    assert brief.resolve(library, "no such thing") == []


def test_report_misfits_gapfill_and_siblings(library) -> None:
    add_playlist(library, "P1b", "Warehouse Techno (2)", ["tid-t0"])
    library.commit()
    playlist_profile.rebuild(library)
    rep = brief.report(library, "warehouse techno")
    assert rep["playlist"]["playlist_id"] == "P1"
    assert any(m["playlist_id"] == "P1b" for m in rep["other_name_matches"])
    # the planted folk track is the worst-fitting member, with engagement fields
    worst = rep["misfits"][0]
    assert worst["artist"] == "Folk Artist"
    assert {"plays", "liked", "last_at", "added_at", "plays_180d"} <= set(worst)
    # the liked, un-playlisted techno track is the top gap-fill candidate
    gap = rep["gap_fill"][0]
    assert gap["title"] == "T9"
    assert "in_playlists" in gap
    assert rep["sonic"]["cohesion"] > 0


def test_report_unresolvable_playlist(library) -> None:
    playlist_profile.rebuild(library)
    assert "error" in brief.report(library, "zzz nothing")


def test_stale_links_spots_dead_variant(library) -> None:
    playlist_profile.rebuild(library)
    # the playlist member went quiet years ago...
    library.execute(
        "UPDATE recording_stats SET last_at='2017-04-15T00:00:00Z' "
        "WHERE recording_key='techno artist\tt0'"
    )
    # ...but the same song lives on as a re-linked variant recording
    add_recording(
        library, "techno artist\tt0v", "Techno Artist", "T0 - Original Mix", "tid-t0v", plays=67
    )
    library.execute(
        "UPDATE recording_stats SET last_at='2026-06-01T00:00:00Z' "
        "WHERE recording_key='techno artist\tt0v'"
    )
    library.commit()
    links = brief.stale_links(library, "P1")
    assert len(links) == 1
    assert links[0]["member"].endswith("T0")
    assert links[0]["live_variant"] == "T0 - Original Mix"


def test_fit_scores_and_percentiles(library) -> None:
    playlist_profile.rebuild(library)
    res = brief.fit(
        library, "warehouse techno", ["Techno Artist - T9", "Folk Artist - F1", "Nobody - Nothing"]
    )
    by_query = {t["query"]: t for t in res["tracks"]}
    techno, folk = by_query["Techno Artist - T9"], by_query["Folk Artist - F1"]
    assert techno["score"] > folk["score"]
    assert techno["percentile_vs_members"] > folk["percentile_vs_members"]
    assert not techno["in_playlist"]
    assert "error" in by_query["Nobody - Nothing"]
