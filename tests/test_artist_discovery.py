from __future__ import annotations

from acetate import artist_discovery, candidates


def _edge(conn, seed: int, artist: str, title: str, strength: float = 0.8) -> None:
    candidates.add_candidate_source(
        conn,
        artist=artist,
        title=title,
        source=candidates.LASTFM_SOURCE,
        reason="track.getSimilar",
        seed_recording_key=f"techno artist\tt{seed}",
        seed_artist="Techno Artist",
        seed_title=f"T{seed}",
        source_rank=1,
        strength=strength,
    )


def test_distinct_playlist_seeds_vote_for_artists(library) -> None:
    for seed in range(3):
        _edge(library, seed, "New Alpha", f"Alpha {seed}", 0.9 - seed * 0.05)
    for seed in range(2):
        _edge(library, seed, "New Beta", f"Beta {seed}", 0.82)
    # A large catalog around one seed still counts as one artist/seed vote.
    for index in range(5):
        _edge(library, 0, "Catalog Flood", f"Flood {index}", 0.99)
    # This artist is already in the library and is excluded in strict novelty mode.
    for seed in range(3):
        _edge(library, seed, "Folk Artist", f"Unheard Folk {seed}", 0.95)
    library.commit()

    result = artist_discovery.discover(
        library, "Warehouse Techno", novelty="outside_library", min_seed_coverage=2
    )

    assert [artist["artist"] for artist in result["artists"]] == ["New Alpha", "New Beta"]
    assert result["artists"][0]["seed_count"] == 3
    assert len(result["artists"][0]["connections"]) == 3
    assert "Catalog Flood" not in {artist["artist"] for artist in result["artists"]}
    assert "Folk Artist" not in {artist["artist"] for artist in result["artists"]}


def test_outside_playlist_can_include_familiar_artist(library) -> None:
    library.execute(
        "INSERT INTO recording_stats(recording_key, rep_track_id, artist, title, plays, liked, homed) "
        "VALUES('library resident\tknown', 'known-id', 'Library Resident', 'Known', 3, 1, 0)"
    )
    for seed in range(2):
        _edge(library, seed, "Library Resident", f"Unheard Track {seed}", 0.9)
    library.commit()

    result = artist_discovery.discover(
        library, "P1", novelty="outside_playlist", min_seed_coverage=2
    )

    assert result["artists"][0]["artist"] == "Library Resident"


def test_playlist_resolution_reports_ambiguity(library) -> None:
    result = artist_discovery.discover(library, "", min_seed_coverage=1)
    assert result["error"] == "playlist not found"
