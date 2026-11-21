"""Filing inbox: surface liked-unplaylisted tracks with best-fit playlist
targets, file (append-only + ledger), and undo (scoped, reversible)."""

from __future__ import annotations

from acetate import filing, taste
from tests.conftest import FakeSpotify

T9 = "techno artist\tt9"  # liked techno, not in the techno playlist


def _regions(conn):
    taste.build(conn)
    return conn


def test_queue_surfaces_unfiled_with_best_fit_targets(library) -> None:
    conn = _regions(library)
    q = filing.queue(conn)
    item = next(i for i in q if i["recording_key"] == T9)
    assert item["track_id"] == "tid-t9"
    # its top target is the techno playlist, and it's a strong fit
    top = item["targets"][0]
    assert top["name"] == "Warehouse Techno"
    assert top["strong"] is True
    assert len(item["targets"]) <= 3


def test_file_appends_logs_and_leaves_queue(library) -> None:
    conn = _regions(library)
    sp = FakeSpotify()
    aid = filing.file_track(
        conn,
        sp,
        recording_key=T9,
        track_id="tid-t9",
        playlist_id="P1",
        playlist_name="Warehouse Techno",
        artist="Techno Artist",
        title="T9",
        score=0.94,
    )
    assert sp.added["P1"] == ["spotify:track:tid-t9"]  # append-only add
    row = conn.execute("SELECT * FROM filing_actions WHERE id=?", (aid,)).fetchone()
    assert row["status"] == "added" and row["snapshot_after"] is not None
    # filed track drops out of the queue immediately (no double-filing)
    assert all(i["recording_key"] != T9 for i in filing.queue(conn))


def test_undo_removes_exactly_and_restores_queue(library) -> None:
    conn = _regions(library)
    sp = FakeSpotify()
    aid = filing.file_track(
        conn,
        sp,
        recording_key=T9,
        track_id="tid-t9",
        playlist_id="P1",
        playlist_name="Warehouse Techno",
    )
    undone = filing.undo(conn, sp, aid)
    assert undone is not None
    assert sp.removed["P1"] == ["spotify:track:tid-t9"]  # exactly what we added
    assert (
        conn.execute("SELECT status FROM filing_actions WHERE id=?", (aid,)).fetchone()[0]
        == "undone"
    )
    # reappears in the queue for re-filing
    assert any(i["recording_key"] == T9 for i in filing.queue(conn))


def test_undo_refuses_unknown_action(library) -> None:
    conn = _regions(library)
    sp = FakeSpotify()
    assert filing.undo(conn, sp, 9999) is None
    assert sp.removed == {}  # never removes anything unledgered


def test_undo_is_idempotent(library) -> None:
    conn = _regions(library)
    sp = FakeSpotify()
    aid = filing.file_track(
        conn,
        sp,
        recording_key=T9,
        track_id="tid-t9",
        playlist_id="P1",
        playlist_name="Warehouse Techno",
    )
    assert filing.undo(conn, sp, aid) is not None
    assert filing.undo(conn, sp, aid) is None  # second undo is a no-op
    assert len(sp.removed["P1"]) == 1  # not removed twice


def test_history_newest_first_and_reflects_status(library) -> None:
    conn = _regions(library)
    sp = FakeSpotify()
    a1 = filing.file_track(
        conn,
        sp,
        recording_key=T9,
        track_id="tid-t9",
        playlist_id="P1",
        playlist_name="Warehouse Techno",
    )
    filing.undo(conn, sp, a1)
    hist = filing.history(conn)
    assert hist[0]["id"] == a1 and hist[0]["status"] == "undone"
