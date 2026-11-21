from __future__ import annotations

import sqlite3

import pytest

from acetate import db
from acetate.reconcile import (
    PROMOTED_TABLES,
    apply_bundle,
    export_bundle,
    inspect_database,
    read_manifest,
    restore_backup,
)


def _database(path):
    conn = db.connect(path)
    db.init_db(conn)
    return conn


def test_apply_promotes_local_enrichment_and_preserves_vps_state(tmp_path) -> None:
    local_path = tmp_path / "local.db"
    vps_path = tmp_path / "vps.db"
    bundle_path = tmp_path / "enrichment.db"
    backup_dir = tmp_path / "backups"

    local = _database(local_path)
    local.execute(
        "INSERT INTO embeddings VALUES(?,?,?,?,?)",
        ("local-recording", "discogs-effnet", 2, b"local-vector", "2026-07-11"),
    )
    local.execute(
        "INSERT INTO audio_cache(recording_key,status,path,updated_at) VALUES(?,?,?,?)",
        ("local-recording", "fetched", "audio/previews/aa/local.mp3", "2026-07-11"),
    )
    local.execute(
        "INSERT INTO candidate_recordings(candidate_key,status,updated_at) VALUES(?,?,?)",
        ("local-candidate", "embedded", "2026-07-11"),
    )
    local.commit()
    local.close()

    vps = _database(vps_path)
    vps.execute(
        "INSERT INTO play_events(played_at,track_id,source) VALUES(?,?,?)",
        ("2026-07-12T01:00:00Z", "new-vps-play", "recently_played"),
    )
    vps.execute(
        "INSERT INTO track_genres VALUES(?,?,?,?,?)",
        ("new-vps-play", "techno", "lastfm", 90.0, "2026-07-12"),
    )
    vps.execute(
        "INSERT INTO playlist_labels VALUES(?,?,?,?)",
        ("favorite-playlist", "fave", "VPS verdict", "2026-07-12"),
    )
    vps.execute(
        "INSERT INTO embeddings VALUES(?,?,?,?,?)",
        ("remote-only-old", "discogs-effnet", 2, b"old-vector", "2026-07-01"),
    )
    vps.execute(
        "INSERT INTO candidate_recordings(candidate_key,status,updated_at) VALUES(?,?,?)",
        ("remote-only-candidate", "metadata", "2026-07-01"),
    )
    vps.commit()
    vps.close()

    exported = export_bundle(local_path, bundle_path)
    assert set(exported["tables"]) == set(PROMOTED_TABLES)
    manifest = read_manifest(bundle_path)
    assert manifest["embeddings"]["row_count"] == 1

    applied = apply_bundle(vps_path, bundle_path, backup_dir, rebuild_derived=False)
    backup_path = applied["backup"]

    merged = sqlite3.connect(vps_path)
    assert merged.execute("SELECT COUNT(*) FROM play_events").fetchone()[0] == 1
    assert merged.execute("SELECT MAX(played_at) FROM play_events").fetchone()[0] == (
        "2026-07-12T01:00:00Z"
    )
    assert merged.execute("SELECT genre FROM track_genres").fetchone()[0] == "techno"
    assert merged.execute("SELECT label FROM playlist_labels").fetchone()[0] == "fave"
    assert merged.execute("SELECT recording_key FROM embeddings").fetchall() == [
        ("local-recording",)
    ]
    assert merged.execute("SELECT candidate_key FROM candidate_recordings").fetchall() == [
        ("local-candidate",)
    ]
    assert merged.execute("SELECT COUNT(*) FROM enrichment_sync").fetchone()[0] == 1
    assert merged.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    merged.close()

    restored = restore_backup(vps_path, backup_path, backup_dir)
    assert restored["safety_backup"]
    original = sqlite3.connect(vps_path)
    assert original.execute("SELECT recording_key FROM embeddings").fetchall() == [
        ("remote-only-old",)
    ]
    assert original.execute("SELECT candidate_key FROM candidate_recordings").fetchall() == [
        ("remote-only-candidate",)
    ]
    assert original.execute("SELECT COUNT(*) FROM play_events").fetchone()[0] == 1
    original.close()


def test_bundle_policy_is_exact_and_bundle_is_not_overwritten(tmp_path) -> None:
    source_path = tmp_path / "source.db"
    bundle_path = tmp_path / "bundle.db"
    source = _database(source_path)
    source.close()

    export_bundle(source_path, bundle_path)
    with pytest.raises(FileExistsError):
        export_bundle(source_path, bundle_path)

    conn = sqlite3.connect(bundle_path)
    conn.execute("DELETE FROM __reconcile_manifest WHERE table_name='embeddings'")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="policy mismatch"):
        read_manifest(bundle_path)


def test_inspect_reports_live_high_water_separately(tmp_path) -> None:
    path = tmp_path / "state.db"
    conn = _database(path)
    conn.execute(
        "INSERT INTO play_events(played_at,track_id,source) VALUES(?,?,?)",
        ("2026-07-13T02:00:00Z", "track", "recently_played"),
    )
    conn.execute(
        "INSERT INTO embeddings VALUES(?,?,?,?,?)",
        ("recording", "discogs-effnet", 1, b"vector", "2026-07-11"),
    )
    conn.commit()
    conn.close()

    report = inspect_database(path)
    assert report["high_water"]["latest_play"] == "2026-07-13T02:00:00Z"
    assert report["vps_owned"]["play_events"] == 1
    assert report["promoted"]["embeddings"] == 1
