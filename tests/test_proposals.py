from __future__ import annotations

from acetate import proposals


def _propose(conn, key="artist\tsong", action="add", source="health", score=0.9):
    return proposals.propose(
        conn,
        playlist_id="P1",
        playlist_name="Warehouse Techno",
        action=action,
        kind="library",
        recording_key=key,
        track_id="tid-1",
        artist="Artist",
        title="Song",
        score=score,
        reasons=["because"],
        source=source,
    )


def test_propose_dedupes_and_refreshes_score(conn) -> None:
    assert _propose(conn, score=0.5) is True
    assert _propose(conn, score=0.8) is True  # refresh while still pending
    rows = proposals.pending(conn)
    assert len(rows) == 1
    assert rows[0]["score"] == 0.8
    assert rows[0]["reasons"] == ["because"]


def test_rejection_is_durable_across_rescans(conn) -> None:
    _propose(conn)
    conn.commit()
    pid = proposals.pending(conn)[0]["id"]
    row = proposals.decide(conn, pid, "rejected")
    assert row["status"] == "rejected"
    # a re-scan proposing the same thing must NOT resurrect it
    assert _propose(conn) is False
    assert proposals.pending(conn) == []
    assert proposals.counts(conn) == {"rejected": 1}
    assert len(proposals.labels(conn)) == 1


def test_decide_only_flips_pending(conn) -> None:
    _propose(conn)
    conn.commit()
    pid = proposals.pending(conn)[0]["id"]
    proposals.decide(conn, pid, "accepted")
    row = proposals.decide(conn, pid, "rejected")  # second verdict is a no-op
    assert row["status"] == "accepted"


def test_feedback_counts_plays_after_recommendation(conn) -> None:
    conn.execute(
        "INSERT INTO track_recording(track_id, recording_key, rep_track_id, "
        "artist, title) VALUES('tid-1','a\tb','tid-1','A','B')"
    )
    proposals.log_event(
        conn,
        recording_key="a\tb",
        track_id="tid-1",
        surface="discovery_playlist",
        target="🔭 Discovery",
        score=0.9,
    )
    proposals.log_event(
        conn,
        recording_key="c\td",
        track_id="tid-2",
        surface="discovery_playlist",
        target="🔭 Discovery",
        score=0.4,
    )
    # a real play lands after the recommendation for tid-1 only
    conn.execute(
        "INSERT INTO play_events(track_id, played_at, ms_played, source) "
        "VALUES('tid-1', '2099-01-01T00:00:00+00:00', 60000, 'test')"
    )
    conn.commit()
    (report,) = proposals.feedback(conn)
    assert report["recommended"] == 2
    assert report["played_after"] == 1
    assert report["hit_rate"] == 0.5
