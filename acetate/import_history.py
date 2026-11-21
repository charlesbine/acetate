"""Import Spotify's GDPR "Extended Streaming History" export into play_events.

The export is a folder of Streaming_History_Audio_<year>.json files; each record
is one stream with an end timestamp (`ts`), `ms_played`, a track URI, and
denormalized name/artist/album strings. We store every *music* stream raw (the
"is this a real play" threshold is applied at query time) plus a track label.

Usage:  python -m acetate.import_history "data/Spotify Extended Streaming History"
"""

from __future__ import annotations

import glob
import json
import sqlite3
import sys
from pathlib import Path

from . import db

SOURCE = "gdpr_export"


def _track_id_from_uri(uri: str | None) -> str | None:
    # "spotify:track:1tHjlmMkKYqB6IYOphpSFl" -> "1tHjlmMkKYqB6IYOphpSFl"
    if not uri or not uri.startswith("spotify:track:"):
        return None
    return uri.split(":")[-1]


def import_extended_history(export_dir: Path, conn: sqlite3.Connection) -> dict[str, int]:
    files = sorted(glob.glob(str(export_dir / "Streaming_History_Audio_*.json")))
    play_rows: list[tuple] = []
    labels: dict[str, tuple] = {}
    skipped_non_music = 0
    max_ts = ""
    now = db.utcnow()

    for path in files:
        for rec in json.loads(Path(path).read_text()):
            track_id = _track_id_from_uri(rec.get("spotify_track_uri"))
            if track_id is None:  # podcast / audiobook / local / unavailable
                skipped_non_music += 1
                continue
            ts = rec.get("ts")
            play_rows.append((ts, track_id, rec.get("ms_played"), SOURCE))
            if ts and ts > max_ts:
                max_ts = ts
            # last write wins for the label; good enough for a display string
            labels[track_id] = (
                track_id,
                rec.get("master_metadata_track_name"),
                rec.get("master_metadata_album_artist_name"),
                rec.get("master_metadata_album_album_name"),
                SOURCE,
                now,
            )

    cur = conn.cursor()
    before = conn.execute("SELECT COUNT(*) FROM play_events WHERE source=?", (SOURCE,)).fetchone()[
        0
    ]
    cur.executemany(
        "INSERT OR IGNORE INTO play_events(played_at, track_id, ms_played, source) VALUES(?,?,?,?)",
        play_rows,
    )
    cur.executemany(
        """INSERT INTO track_labels(track_id, track_name, artist_name, album_name, source, updated_at)
           VALUES(?,?,?,?,?,?)
           ON CONFLICT(track_id) DO UPDATE SET
             track_name=excluded.track_name, artist_name=excluded.artist_name,
             album_name=excluded.album_name, source=excluded.source, updated_at=excluded.updated_at""",
        list(labels.values()),
    )
    if max_ts:
        conn.execute(
            "INSERT INTO kv_state(key, value, updated_at) VALUES('gdpr_history_watermark', ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (max_ts, now),
        )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM play_events WHERE source=?", (SOURCE,)).fetchone()[0]

    return {
        "files": len(files),
        "music_streams_seen": len(play_rows),
        "play_events_inserted": after - before,
        "unique_tracks": len(labels),
        "skipped_non_music": skipped_non_music,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    export_dir = Path(argv[0])
    if not export_dir.is_dir():
        print(f"Not a directory: {export_dir}")
        return 1
    conn = db.connect()
    db.init_db(conn)
    stats = import_extended_history(export_dir, conn)
    print(f"Imported streaming history from {export_dir}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    wm = conn.execute("SELECT value FROM kv_state WHERE key='gdpr_history_watermark'").fetchone()
    print(f"  history watermark: {wm[0] if wm else None}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
