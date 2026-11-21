"""Migrate a legacy TinyDB backup (legacy-tinydb.json) into the SQLite schema.

The old store had two tables: `playlists` (full playlist objects, including
nested tracks) and `saved_tracks` (liked songs), each row stamped with a
`backup_time`. We replay those into append-only snapshot tables and populate
the track/album/artist dimensions.

Usage:  python -m acetate.migrate path/to/legacy-tinydb.json
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from . import db
from .ingest import upsert_track


def _migrate_playlists(conn: sqlite3.Connection, rows: dict, now: str) -> tuple[int, int]:
    n_playlists = n_tracks = 0
    for row in rows.values():
        backup_time = row.get("backup_time")
        tracks_obj = row.get("tracks") or {}
        items = tracks_obj.get("items") or []
        conn.execute(
            """
            INSERT INTO playlist_snapshots(
                backup_time, playlist_id, name, description, owner_id, owner_name,
                public, collaborative, spotify_snapshot_id, total_tracks, raw_json)
            VALUES(:bt, :pid, :name, :desc, :oid, :oname, :pub, :collab, :snap, :total, :raw)
            ON CONFLICT(backup_time, playlist_id) DO NOTHING
            """,
            {
                "bt": backup_time,
                "pid": row.get("id"),
                "name": row.get("name"),
                "desc": row.get("description"),
                "oid": (row.get("owner") or {}).get("id"),
                "oname": (row.get("owner") or {}).get("display_name"),
                "pub": int(bool(row.get("public"))) if row.get("public") is not None else None,
                "collab": int(bool(row.get("collaborative"))),
                "snap": row.get("snapshot_id"),
                "total": tracks_obj.get("total"),
                "raw": json.dumps({k: v for k, v in row.items() if k != "tracks"}),
            },
        )
        n_playlists += 1
        for position, item in enumerate(items):
            track = item.get("track") if isinstance(item, dict) else None
            track_id = upsert_track(conn, track, now)
            conn.execute(
                """
                INSERT INTO playlist_track_snapshots(
                    backup_time, playlist_id, position, track_id, added_at, added_by, raw_json)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    backup_time,
                    row.get("id"),
                    position,
                    track_id,
                    (item or {}).get("added_at"),
                    ((item or {}).get("added_by") or {}).get("id"),
                    json.dumps({k: v for k, v in (item or {}).items() if k != "track"}),
                ),
            )
            n_tracks += 1
    return n_playlists, n_tracks


def _migrate_saved_tracks(conn: sqlite3.Connection, rows: dict, now: str) -> int:
    n = 0
    for row in rows.values():
        track = row.get("track")
        track_id = upsert_track(conn, track, now)
        conn.execute(
            """
            INSERT INTO saved_track_snapshots(backup_time, track_id, added_at, raw_json)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(backup_time, track_id) DO NOTHING
            """,
            (
                row.get("backup_time"),
                track_id,
                row.get("added_at"),
                json.dumps({k: v for k, v in row.items() if k != "track"}),
            ),
        )
        n += 1
    return n


def migrate_tinydb(json_path: Path, conn: sqlite3.Connection) -> dict[str, int]:
    now = db.utcnow()
    data = json.loads(json_path.read_text())
    counts: dict[str, int] = {}
    if "playlists" in data:
        pl, pt = _migrate_playlists(conn, data["playlists"], now)
        counts["playlist_snapshots"], counts["playlist_track_snapshots"] = pl, pt
    if "saved_tracks" in data:
        counts["saved_track_snapshots"] = _migrate_saved_tracks(conn, data["saved_tracks"], now)
    conn.commit()
    return counts


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    json_path = Path(argv[0])
    if not json_path.is_file():
        print(f"No such file: {json_path}")
        return 1
    conn = db.connect()
    db.init_db(conn)
    counts = migrate_tinydb(json_path, conn)
    print(f"Migrated {json_path} -> {db.db_path()}")
    for table, n in counts.items():
        print(f"  {table}: {n}")
    # quick dimension tallies
    for table in ("tracks", "artists", "albums"):
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table} (deduped): {n}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
