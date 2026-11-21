"""Safely promote Mac-owned enrichment state into the VPS SQLite database.

The two databases deliberately have different owners:

* the VPS owns live/append-only state (plays, snapshots, chat, review verdicts,
  scheduler cursors, and managed-playlist state);
* the Mac owns expensive, reproducible enrichment (audio ledgers, embeddings,
  similarity caches, outside-library candidates, and provider harvests).

Reconciliation therefore starts from a consistent backup of the VPS database
and replaces only the explicit ``PROMOTED_TABLES`` set from a transport bundle.
Derived tables are rebuilt against the combined result before an atomic swap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from . import db, embedding_guard

BUNDLE_FORMAT_VERSION = 1
APPLY_CONFIRMATION = "PROMOTE_LOCAL_ENRICHMENT"
RESTORE_CONFIRMATION = "RESTORE_VPS_BACKUP"

# Order is parent-before-child. Apply deletes in reverse order and inserts in
# this order. Tables omitted here always remain exactly as they were on the VPS.
PROMOTED_TABLES = (
    "audio_cache",
    "embeddings",
    "recording_styles",
    "hyperspace_points",
    "lastfm_similar",
    "lastfm_info",
    "lastfm_track_runs",
    "lastfm_artist_cache",
    "lastfm_artist_similar",
    "lastfm_artist_top_tracks",
    "lastfm_artist_top_albums",
    "candidate_recordings",
    "candidate_sources",
    "candidate_external_ids",
    "candidate_duplicates",
    "candidate_resolutions",
    "candidate_audio",
    "candidate_embedding_runs",
    "candidate_rerank",
    "ytmusic_radio_seeds",
    "ytmusic_seed_matches",
    "deezer_track",
    "deezer_album",
    "deezer_artist",
    "deezer_metadata_runs",
    "deezer_artist_expansion",
    "deezer_artist_top_tracks",
    "deezer_artist_albums",
    "deezer_album_expansion",
    "deezer_album_tracks",
    "spotify_expansion_capabilities",
    "spotify_artist_expansion",
    "spotify_artist_top_tracks",
    "spotify_artist_albums",
    "spotify_album_expansion",
    "spotify_album_tracks",
    "spotify_features",
    "spotify_feature_coverage",
    "embedding_runs",
    "recording_mbid",
)

# These are never imported from the Mac. The report makes that boundary
# visible before an operator runs the apply command.
VPS_OWNED_TABLES = (
    "play_events",
    "playlist_snapshots",
    "playlist_seen",
    "playlist_track_snapshots",
    "saved_track_snapshots",
    "track_genres",
    "lastfm_tag_runs",
    "smart_playlists",
    "chat_threads",
    "chat_messages",
    "embedding_requests",
    "managed_playlists",
    "managed_playlist_tracks",
    "job_run",
    "kv_state",
    "playlist_proposals",
    "recommendation_events",
    "playlist_labels",
)

_MANIFEST_TABLE = "__reconcile_manifest"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _columns(conn: sqlite3.Connection, schema: str, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA {_quote(schema)}.table_info({_quote(table)})")]


def _row_count(conn: sqlite3.Connection, schema: str, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {_quote(schema)}.{_quote(table)}").fetchone()[0])


def _quick_check(path: Path) -> None:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        result = conn.execute("PRAGMA quick_check").fetchone()[0]
    finally:
        conn.close()
    if result != "ok":
        raise RuntimeError(f"SQLite quick_check failed for {path}: {result}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_bundle(
    source_db: Path, bundle_path: Path, tables: Iterable[str] = PROMOTED_TABLES
) -> dict:
    """Export the Mac-owned tables into a self-describing SQLite bundle."""
    source_db = source_db.resolve()
    bundle_path = bundle_path.resolve()
    tables = tuple(tables)
    if bundle_path.exists():
        raise FileExistsError(f"refusing to overwrite bundle: {bundle_path}")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    out = sqlite3.connect(bundle_path)
    try:
        out.execute("PRAGMA journal_mode=DELETE")
        out.execute("PRAGMA synchronous=OFF")
        out.execute("PRAGMA foreign_keys=OFF")
        out.execute("ATTACH DATABASE ? AS source", (str(source_db),))
        out.execute("""
            CREATE TABLE __reconcile_manifest (
                format_version INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                table_name TEXT PRIMARY KEY,
                row_count INTEGER NOT NULL,
                columns_json TEXT NOT NULL
            )""")
        created_at = datetime.now(timezone.utc).isoformat()
        counts: dict[str, int] = {}
        out.execute("BEGIN")
        for table in tables:
            row = out.execute(
                "SELECT sql FROM source.sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if not row or not row[0]:
                raise RuntimeError(f"source database is missing promoted table {table!r}")
            out.execute(row[0])
            columns = _columns(out, "source", table)
            column_sql = ", ".join(_quote(c) for c in columns)
            out.execute(
                f"INSERT INTO main.{_quote(table)} ({column_sql}) "
                f"SELECT {column_sql} FROM source.{_quote(table)}"
            )
            count = _row_count(out, "main", table)
            counts[table] = count
            out.execute(
                "INSERT INTO __reconcile_manifest VALUES(?,?,?,?,?)",
                (
                    BUNDLE_FORMAT_VERSION,
                    created_at,
                    table,
                    count,
                    json.dumps(columns, separators=(",", ":")),
                ),
            )
        out.commit()
        out.execute("DETACH DATABASE source")
    except Exception:
        out.close()
        bundle_path.unlink(missing_ok=True)
        raise
    else:
        out.close()

    _quick_check(bundle_path)
    return {
        "bundle": str(bundle_path),
        "bytes": bundle_path.stat().st_size,
        "sha256": sha256_file(bundle_path),
        "tables": counts,
    }


def read_manifest(bundle_path: Path) -> dict[str, dict]:
    _quick_check(bundle_path)
    conn = sqlite3.connect(f"file:{bundle_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT * FROM {_quote(_MANIFEST_TABLE)} ORDER BY table_name"
        ).fetchall()
    except sqlite3.OperationalError as exc:
        raise RuntimeError("not an Acetate reconciliation bundle") from exc
    finally:
        conn.close()
    manifest = {r["table_name"]: dict(r) for r in rows}
    if set(manifest) != set(PROMOTED_TABLES):
        missing = sorted(set(PROMOTED_TABLES) - set(manifest))
        extra = sorted(set(manifest) - set(PROMOTED_TABLES))
        raise RuntimeError(f"bundle table policy mismatch; missing={missing}, extra={extra}")
    versions = {r["format_version"] for r in manifest.values()}
    if versions != {BUNDLE_FORMAT_VERSION}:
        raise RuntimeError(f"unsupported bundle format versions: {sorted(versions)}")
    return manifest


def _online_backup(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite backup: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(source)
    dst = sqlite3.connect(destination)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    _quick_check(destination)


def _rebuild_derived(conn: sqlite3.Connection) -> dict:
    from . import canon, playlist_profile, taste

    canonical = canon.canonicalize(conn)
    profiles = playlist_profile.rebuild(conn)
    regions = taste.build(conn)
    return {"canonical": canonical, "profiles": profiles, "regions": regions}


def apply_bundle(
    target_db: Path, bundle_path: Path, backup_dir: Path, *, rebuild_derived: bool = True
) -> dict:
    with embedding_guard.quiescent(target_db.resolve().parent):
        return _apply_bundle(target_db, bundle_path, backup_dir, rebuild_derived=rebuild_derived)


def _apply_bundle(
    target_db: Path, bundle_path: Path, backup_dir: Path, *, rebuild_derived: bool = True
) -> dict:
    """Apply a verified bundle to an offline copy, then atomically swap it in."""
    target_db = target_db.resolve()
    bundle_path = bundle_path.resolve()
    backup_dir = backup_dir.resolve()
    if not target_db.exists():
        raise FileNotFoundError(target_db)
    manifest = read_manifest(bundle_path)

    stamp = _utc_stamp()
    backup_path = backup_dir / f"acetate-pre-reconcile-{stamp}.db"
    candidate_path = target_db.with_name(f".{target_db.name}.reconcile-{stamp}.tmp")
    _online_backup(target_db, backup_path)
    shutil.copy2(backup_path, candidate_path)

    before = inspect_database(target_db)
    conn = sqlite3.connect(candidate_path)
    conn.row_factory = sqlite3.Row
    derived: dict = {}
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA foreign_keys=OFF")
        db.init_db(conn)
        conn.execute("ATTACH DATABASE ? AS bundle", (str(bundle_path),))

        for table in PROMOTED_TABLES:
            target_columns = _columns(conn, "main", table)
            bundle_columns = _columns(conn, "bundle", table)
            manifest_columns = json.loads(manifest[table]["columns_json"])
            if target_columns != bundle_columns or bundle_columns != manifest_columns:
                raise RuntimeError(
                    f"schema mismatch for {table}: target={target_columns}, "
                    f"bundle={bundle_columns}, manifest={manifest_columns}"
                )

        conn.execute("BEGIN IMMEDIATE")
        for table in reversed(PROMOTED_TABLES):
            conn.execute(f"DELETE FROM main.{_quote(table)}")
        for table in PROMOTED_TABLES:
            columns = _columns(conn, "main", table)
            column_sql = ", ".join(_quote(c) for c in columns)
            conn.execute(
                f"INSERT INTO main.{_quote(table)} ({column_sql}) "
                f"SELECT {column_sql} FROM bundle.{_quote(table)}"
            )
            actual = _row_count(conn, "main", table)
            expected = int(manifest[table]["row_count"])
            if actual != expected:
                raise RuntimeError(
                    f"row-count mismatch for {table}: expected {expected}, got {actual}"
                )
        source_created_at = next(iter(manifest.values()))["created_at"]
        conn.execute(
            "INSERT INTO enrichment_sync(source_created_at, applied_at, bundle_name, tables_json) "
            "VALUES(?,?,?,?)",
            (
                source_created_at,
                datetime.now(timezone.utc).isoformat(),
                bundle_path.name,
                json.dumps(
                    {table: int(row["row_count"]) for table, row in manifest.items()},
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            ),
        )
        conn.commit()
        conn.execute("DETACH DATABASE bundle")

        if rebuild_derived:
            derived = _rebuild_derived(conn)

        foreign_key_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            raise RuntimeError(f"foreign_key_check failed with {len(foreign_key_errors)} row(s)")
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"integrity_check failed: {integrity}")
        conn.commit()
    except Exception:
        conn.close()
        candidate_path.unlink(missing_ok=True)
        raise
    else:
        conn.close()

    candidate_state = inspect_database(candidate_path)
    for table, expected in before["vps_owned"].items():
        actual = candidate_state["vps_owned"].get(table)
        if actual != expected:
            candidate_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"VPS-owned table changed during reconciliation: "
                f"{table} expected {expected}, got {actual}"
            )
    if candidate_state["high_water"] != before["high_water"]:
        candidate_path.unlink(missing_ok=True)
        raise RuntimeError(
            "VPS live-data high-water marks changed during reconciliation: "
            f"before={before['high_water']}, candidate={candidate_state['high_water']}"
        )

    # The candidate used DELETE journaling, so all committed bytes are in the
    # main file. Remove sidecars belonging to the old inode before the swap.
    Path(f"{target_db}-wal").unlink(missing_ok=True)
    Path(f"{target_db}-shm").unlink(missing_ok=True)
    os.replace(candidate_path, target_db)
    after = inspect_database(target_db)
    return {
        "target": str(target_db),
        "backup": str(backup_path),
        "bundle": str(bundle_path),
        "before": before,
        "after": after,
        "derived": derived,
    }


def restore_backup(target_db: Path, restore_from: Path, backup_dir: Path) -> dict:
    with embedding_guard.quiescent(Path(target_db).resolve().parent):
        return _restore_backup(target_db, restore_from, backup_dir)


def _restore_backup(target_db: Path, restore_from: Path, backup_dir: Path) -> dict:
    """Atomically restore a verified DB backup, preserving the current DB first."""
    target_db = Path(target_db).resolve()
    restore_from = Path(restore_from).resolve()
    backup_dir = Path(backup_dir).resolve()
    if not target_db.exists():
        raise FileNotFoundError(target_db)
    if not restore_from.exists():
        raise FileNotFoundError(restore_from)
    _quick_check(restore_from)

    stamp = _utc_stamp()
    safety_backup = backup_dir / f"acetate-pre-restore-{stamp}.db"
    candidate_path = target_db.with_name(f".{target_db.name}.restore-{stamp}.tmp")
    _online_backup(target_db, safety_backup)
    shutil.copy2(restore_from, candidate_path)
    _quick_check(candidate_path)

    Path(f"{target_db}-wal").unlink(missing_ok=True)
    Path(f"{target_db}-shm").unlink(missing_ok=True)
    os.replace(candidate_path, target_db)
    return {
        "target": str(target_db),
        "restored_from": str(restore_from),
        "safety_backup": str(safety_backup),
        "after": inspect_database(target_db),
    }


def inspect_database(path: Path) -> dict:
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        promoted = {
            table: _row_count(conn, "main", table) for table in PROMOTED_TABLES if table in existing
        }
        preserved = {
            table: _row_count(conn, "main", table)
            for table in VPS_OWNED_TABLES
            if table in existing
        }
        high_water = {
            "latest_play": conn.execute("SELECT MAX(played_at) FROM play_events").fetchone()[0]
            if "play_events" in existing
            else None,
            "latest_playlist_backup": conn.execute(
                "SELECT MAX(backup_time) FROM playlist_seen"
            ).fetchone()[0]
            if "playlist_seen" in existing
            else None,
            "latest_saved_backup": conn.execute(
                "SELECT MAX(backup_time) FROM saved_track_snapshots"
            ).fetchone()[0]
            if "saved_track_snapshots" in existing
            else None,
        }
    finally:
        conn.close()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "high_water": high_water,
        "vps_owned": preserved,
        "promoted": promoted,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="build a local enrichment bundle")
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--bundle", type=Path, required=True)

    inspect = sub.add_parser("inspect", help="print ownership-aware DB counts")
    inspect.add_argument("--target", type=Path, required=True)

    apply = sub.add_parser("apply", help="atomically apply a bundle to a stopped VPS DB")
    apply.add_argument("--target", type=Path, required=True)
    apply.add_argument("--bundle", type=Path, required=True)
    apply.add_argument("--backup-dir", type=Path, required=True)
    apply.add_argument("--confirm", required=True)

    restore = sub.add_parser("restore", help="atomically restore a prior VPS DB backup")
    restore.add_argument("--target", type=Path, required=True)
    restore.add_argument("--from-backup", type=Path, required=True)
    restore.add_argument("--backup-dir", type=Path, required=True)
    restore.add_argument("--confirm", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "export":
        result = export_bundle(args.source, args.bundle)
    elif args.command == "inspect":
        result = inspect_database(args.target)
    elif args.command == "apply":
        if args.confirm != APPLY_CONFIRMATION:
            raise SystemExit(f"--confirm must be exactly {APPLY_CONFIRMATION}")
        result = apply_bundle(args.target, args.bundle, args.backup_dir)
    else:
        if args.confirm != RESTORE_CONFIRMATION:
            raise SystemExit(f"--confirm must be exactly {RESTORE_CONFIRMATION}")
        result = restore_backup(args.target, args.from_backup, args.backup_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
