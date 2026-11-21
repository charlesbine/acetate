from __future__ import annotations

import sqlite3

from acetate import genre_playlists, proposals
from tests.conftest import FakeSpotify


def _style(conn: sqlite3.Connection, key: str, style: str, score: float, rank: int = 1) -> None:
    conn.execute(
        "INSERT INTO recording_styles(recording_key, rank, style, score) VALUES(?,?,?,?)",
        (key, rank, style, score),
    )


def _tag(conn: sqlite3.Connection, track_id: str, genre: str, conf: float) -> None:
    conn.execute(
        "INSERT INTO track_genres(track_id, genre, source, confidence) VALUES(?,?,'lastfm',?)",
        (track_id, genre, conf),
    )


def test_anchor_playlists_match_by_name(library) -> None:
    anchors = genre_playlists.anchor_playlists(library, "techno")
    assert [a["name"] for a in anchors] == ["Warehouse Techno"]


def test_classify_three_signals_and_tiers(library) -> None:
    conn = library
    # t0: all three signals agree → auto tier
    _style(conn, "techno artist\tt0", "Electronic---Techno", 0.9)
    _tag(conn, "tid-t0", "techno", 95)
    # t9: region + style, no tag → still classifiable
    _style(conn, "techno artist\tt9", "Electronic---Techno", 0.8)
    # folk track: no techno signal at all → absent
    conn.commit()
    rows = genre_playlists.classify(conn, "techno")
    by_key = {r["recording_key"]: r for r in rows}
    assert by_key["techno artist\tt0"]["tier"] == "auto"
    assert by_key["techno artist\tt0"]["signals"] == 3
    assert "techno artist\tt9" in by_key
    assert "folk artist\tf0" not in by_key
    assert rows == sorted(rows, key=lambda r: -r["confidence"])


def test_sync_appends_auto_and_queues_review(library) -> None:
    conn = library
    _style(conn, "techno artist\tt0", "Electronic---Techno", 0.9)
    _tag(conn, "tid-t0", "techno", 95)
    _style(conn, "techno artist\tt1", "Electronic---Techno", 0.9)
    _tag(conn, "tid-t1", "techno", 90)
    conn.commit()
    sp = FakeSpotify()
    rows = genre_playlists.classify(conn, "techno")
    auto_ids = {r["track_id"] for r in rows if r["tier"] == "auto"}
    n_review = sum(1 for r in rows if r["tier"] == "review")
    res = genre_playlists.sync(conn, sp, "techno", classify_rows=rows)
    assert res["appended"] == len(auto_ids)
    assert res["review_proposed"] == n_review
    pid = res["playlist_id"]
    assert set(sp.replaced[pid]) == {f"spotify:track:{t}" for t in auto_ids}
    assert all(r["source"] == "genre" for r in proposals.pending(conn))

    # append-only: a second sync never removes previously added tracks
    conn.execute("DELETE FROM track_genres")  # signals vanish...
    conn.commit()
    res2 = genre_playlists.sync(conn, sp, "techno")
    assert res2["tracks"] >= len(auto_ids)  # ...but the playlist keeps them
    for uri in (f"spotify:track:{t}" for t in auto_ids):
        assert uri in sp.replaced[pid]
