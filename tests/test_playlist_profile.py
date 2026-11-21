from __future__ import annotations

import sqlite3

from acetate import db as adb
from acetate import playlist_profile
from tests.conftest import add_recording


def _playlist(
    conn: sqlite3.Connection,
    pid: str,
    name: str,
    tracks: list[tuple[str, str]],
    owner: str = "alice",
    backup_time: str = "2026-07-01T00:00:00+00:00",
) -> None:
    """tracks = [(track_id, added_at)]"""
    conn.execute(
        "INSERT INTO playlist_snapshots(backup_time, playlist_id, name, owner_id, "
        "total_tracks) VALUES(?,?,?,?,?)",
        (backup_time, pid, name, owner, len(tracks)),
    )
    conn.execute(
        "INSERT OR IGNORE INTO playlist_seen(backup_time, playlist_id) VALUES(?,?)",
        (backup_time, pid),
    )
    for pos, (tid, added) in enumerate(tracks):
        conn.execute(
            "INSERT INTO playlist_track_snapshots(backup_time, playlist_id, position, "
            "track_id, added_at) VALUES(?,?,?,?,?)",
            (backup_time, pid, pos, tid, added),
        )


def _tracks(conn, prefix: str, n: int, plays: int = 0) -> list[str]:
    ids = []
    for i in range(n):
        tid = f"tid-{prefix}{i}"
        add_recording(
            conn, f"{prefix} artist\t{i}", f"{prefix.title()} Artist", f"S{i}", tid, plays=plays
        )
        ids.append(tid)
    return ids


def test_personas_from_add_cadence_and_usage(conn) -> None:
    now = adb.utcnow()
    # archive: 250 tracks all added the same day years ago
    arch = _tracks(conn, "arch", 250)
    _playlist(conn, "ARCH", "Songs as of 2019", [(t, "2019-01-05T10:00:00Z") for t in arch])
    # one-shot: 15 tracks added in one sitting, never played since
    once = _tracks(conn, "once", 15)
    _playlist(conn, "ONCE", "that warehouse show sets", [(t, "2022-03-14T02:00:00Z") for t in once])
    # rotation: added across many days, incl. recently
    rot = _tracks(conn, "rot", 10, plays=20)
    adds = [f"202{5 + (i > 5)}-0{(i % 6) + 1}-10T00:00:00Z" for i in range(10)]
    adds[-1] = now  # one add just now → alive
    _playlist(conn, "ROT", "Melodic techno", list(zip(rot, adds)))
    # dormant: curated over many days, but long ago and unplayed
    dorm = _tracks(conn, "dorm", 10)
    _playlist(
        conn,
        "DORM",
        "old slow burn",
        [(t, f"2020-{(i % 9) + 1:02d}-01T00:00:00Z") for i, t in enumerate(dorm)],
    )
    conn.commit()

    res = playlist_profile.rebuild(conn)
    assert res["playlists"] == 4
    personas = {
        r["playlist_id"]: r["persona"]
        for r in conn.execute("SELECT playlist_id, persona FROM playlist_stats")
    }
    assert personas == {"ARCH": "archive", "ONCE": "one-shot", "ROT": "rotation", "DORM": "dormant"}


def test_plays_from_context_and_recent_member_plays(conn) -> None:
    tids = _tracks(conn, "ctx", 3, plays=5)
    _playlist(conn, "CTX", "context test", [(t, "2026-01-01T00:00:00Z") for t in tids])
    conn.execute(
        "INSERT INTO play_events(played_at, track_id, ms_played, context_type, "
        "context_uri, source) VALUES(?,?,?,?,?,?)",
        (adb.utcnow(), tids[0], 60000, "playlist", "spotify:playlist:CTX", "recently_played"),
    )
    conn.execute(
        "INSERT INTO play_events(played_at, track_id, ms_played, source) VALUES(?,?,?,?)",
        (adb.utcnow(), tids[1], 60000, "recently_played"),
    )
    conn.commit()
    playlist_profile.rebuild(conn)
    row = conn.execute("SELECT * FROM playlist_stats WHERE playlist_id='CTX'").fetchone()
    assert row["plays_from"] == 1
    assert row["last_played_from"] is not None
    assert row["member_plays_180d"] == 2
    assert row["persona"] == "rotation"  # recently played from it


def test_deleted_playlist_detected(conn) -> None:
    tids = _tracks(conn, "del", 4)
    _playlist(
        conn,
        "GONE",
        "deleted one",
        [(t, "2024-01-01T00:00:00Z") for t in tids],
        backup_time="2026-06-01T00:00:00+00:00",
    )
    tids2 = _tracks(conn, "keep", 4)
    _playlist(
        conn,
        "KEEP",
        "kept one",
        [(t, "2024-01-01T00:00:00Z") for t in tids2],
        backup_time="2026-07-01T00:00:00+00:00",
    )
    conn.commit()
    playlist_profile.rebuild(conn)
    rows = {r["playlist_id"]: r for r in conn.execute("SELECT * FROM playlist_stats")}
    assert rows["GONE"]["is_deleted"] == 1
    assert rows["GONE"]["persona"] == "deleted"
    assert rows["KEEP"]["is_deleted"] == 0


def test_user_label_beats_heuristic(conn) -> None:
    tids = _tracks(conn, "lab", 250)
    _playlist(conn, "LAB", "big old dump", [(t, "2019-01-05T00:00:00Z") for t in tids])
    conn.commit()
    playlist_profile.rebuild(conn)
    assert (
        conn.execute("SELECT persona FROM playlist_stats WHERE playlist_id='LAB'").fetchone()[0]
        == "archive"
    )
    playlist_profile.set_label(conn, "LAB", "fave", note="actually my canon")
    row = conn.execute(
        "SELECT persona, user_label FROM playlist_stats WHERE playlist_id='LAB'"
    ).fetchone()
    assert row["user_label"] == "fave"
    assert row["persona"] == "rotation"
    # label survives a full rebuild
    playlist_profile.rebuild(conn)
    assert (
        conn.execute("SELECT persona FROM playlist_stats WHERE playlist_id='LAB'").fetchone()[0]
        == "rotation"
    )
    playlist_profile.clear_label(conn, "LAB")
    row = conn.execute(
        "SELECT persona, user_label FROM playlist_stats WHERE playlist_id='LAB'"
    ).fetchone()
    assert row["user_label"] is None
    assert row["persona"] == "archive"


def test_members_flattened_with_recording_keys(library) -> None:
    playlist_profile.rebuild(library)
    n = library.execute(
        "SELECT COUNT(*) FROM playlist_members WHERE recording_key IS NOT NULL"
    ).fetchone()[0]
    assert n == 17  # 9 in P1 + 8 in P2


def test_label_suggestions_use_title_and_activity_signals() -> None:
    archive = playlist_profile.suggest_label(
        {
            "name": "My top 2019",
            "n_tracks": 100,
            "member_plays_180d": 90,
            "add_days": 1,
            "persona": "rotation",
        }
    )
    assert (archive["label"], archive["confidence"]) == ("archive", "high")

    event = playlist_profile.suggest_label(
        {
            "name": "2024 Miami music week",
            "n_tracks": 42,
            "member_plays_180d": 26,
            "add_days": 9,
            "persona": "rotation",
        }
    )
    assert (event["label"], event["confidence"]) == ("one-shot", "high")

    rotation = playlist_profile.suggest_label(
        {
            "name": "Melodic techno",
            "n_tracks": 324,
            "member_plays_180d": 241,
            "add_days": 144,
            "persona": "rotation",
            "cohesion": 0.847,
        }
    )
    assert (rotation["label"], rotation["confidence"]) == ("rotation", "high")

    muted = playlist_profile.suggest_label(
        {
            "name": "untitled playlist",
            "n_tracks": 0,
            "member_plays_180d": 0,
            "add_days": 0,
            "persona": "other",
        }
    )
    assert (muted["label"], muted["confidence"]) == ("mute", "high")
