"""Read-only playlist workspace views over existing schedules and backup snapshots."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from urllib.parse import quote

from . import operations, smart

PLAYLIST_JOBS = {
    "smart": "Smart rules & first publications",
    "monthly": "Liked & Top Played collections",
    "on_loop": "On Loop",
    "gems": "Lost Gems",
}


def scheduled_updates(conn) -> dict:
    scheduler = operations.scheduler_status(conn)
    now = datetime.now(timezone.utc)
    healthy = scheduler["state"] == "running"
    jobs = {}
    for job in scheduler["jobs"]:
        if not isinstance(job, dict) or job.get("job") not in PLAYLIST_JOBS:
            continue
        next_at = job.get("next_run_at")
        try:
            parsed = datetime.fromisoformat(next_at.replace("Z", "+00:00")) if next_at else None
            if parsed and parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            parsed = None
        jobs[job["job"]] = {
            "job": job["job"],
            "name": PLAYLIST_JOBS[job["job"]],
            "next_run_at": next_at if healthy and parsed and parsed > now else None,
            "running": healthy and job["job"] in scheduler["active_jobs"],
        }
    jobs = sorted(jobs.values(), key=lambda item: (item["next_run_at"] or "9999", item["job"]))
    smart_job = next((item for item in jobs if item["job"] == "smart"), {})
    items = []
    for row in smart.list_all(conn):
        curated = smart.is_curated(row["spec"])
        if curated and row["published_at"]:
            continue
        enabled = bool(row["enabled"])
        items.append(
            {
                "name": row["name"],
                "curated": curated,
                "enabled": enabled,
                "status": ("First publication" if enabled else "Manual publication only")
                if curated
                else ("Daily refresh" if enabled else "Paused"),
                "next_run_at": smart_job.get("next_run_at") if enabled else None,
                "running": enabled and smart_job.get("running", False),
                "published_at": row["published_at"],
                "url": f"/playlists/saved/{row['id']}",
            }
        )
    return {"scheduler": scheduler, "jobs": jobs, "items": items}


def _track_ids(conn, playlist_id: str, at: str) -> list[str | None]:
    return [
        row["track_id"] or None
        for row in conn.execute(
            "SELECT track_id FROM playlist_track_snapshots WHERE playlist_id=? AND backup_time=? "
            "ORDER BY position, id",
            (playlist_id, at),
        )
    ]


def _track_labels(conn, ids: set[str]) -> dict[str, dict]:
    labels = {}
    ids = sorted(ids)
    for start in range(0, len(ids), 500):
        batch = ids[start : start + 500]
        marks = ",".join("?" for _ in batch)
        for row in conn.execute(
            f"SELECT t.track_id, t.name title, (SELECT group_concat(name, ', ') FROM "
            f"(SELECT a.name FROM track_artists ta JOIN artists a USING(artist_id) "
            f"WHERE ta.track_id=t.track_id ORDER BY ta.position)) artist "
            f"FROM tracks t WHERE t.track_id IN ({marks})",
            batch,
        ):
            labels[row["track_id"]] = {"title": row["title"], "artist": row["artist"]}
        for row in conn.execute(
            f"SELECT track_id, title, artist FROM track_recording WHERE track_id IN ({marks})",
            batch,
        ):
            labels[row["track_id"]] = {"title": row["title"], "artist": row["artist"]}
        for row in conn.execute(
            f"SELECT track_id, track_name title, artist_name artist FROM track_labels "
            f"WHERE track_id IN ({marks})",
            batch,
        ):
            previous = labels.get(row["track_id"], {})
            labels[row["track_id"]] = {
                "title": row["title"] or previous.get("title"),
                "artist": row["artist"] or previous.get("artist"),
            }
    return labels


def playlist_url(playlist_id: str) -> str:
    return f"/playlists/{quote(playlist_id, safe='')}"


def playlist_catalog(conn, *, query: str = "", page: int = 1, page_size: int = 60) -> dict:
    """Every archived, managed, or saved playlist, with one entry per Spotify playlist."""
    items = {}
    for row in conn.execute(
        "SELECT s.playlist_id, s.name, s.total_tracks FROM playlist_snapshots s JOIN "
        "(SELECT playlist_id, MAX(backup_time) at FROM playlist_snapshots GROUP BY playlist_id) l "
        "ON s.playlist_id=l.playlist_id AND s.backup_time=l.at"
    ):
        items[row["playlist_id"]] = {
            "name": row["name"] or row["playlist_id"],
            "kind": "Library backup",
            "count": row["total_tracks"],
            "url": playlist_url(row["playlist_id"]),
        }
    managed = list(conn.execute("SELECT playlist_id, name, kind, key FROM managed_playlists"))
    for row in managed:
        items.setdefault(
            row["playlist_id"],
            {
                "name": row["name"] or row["playlist_id"],
                "count": None,
                "url": playlist_url(row["playlist_id"]),
            },
        )["kind"] = row["kind"].replace("_", " ").capitalize()
    links = {row["key"]: row["playlist_id"] for row in managed if row["kind"] == "smart"}
    for row in smart.list_all(conn):
        key = links.get(str(row["id"])) or ("saved", row["id"])
        previous = items.get(key, {})
        items[key] = {
            "name": row["name"],
            "kind": "Curated · fixed selection"
            if smart.is_curated(row["spec"])
            else "Smart rule · daily"
            if row["enabled"]
            else "Smart rule · paused",
            "count": previous.get("count"),
            "url": f"/playlists/saved/{row['id']}",
        }
    query = query.strip()
    matches = sorted(
        (item for item in items.values() if query.casefold() in item["name"].casefold()),
        key=lambda item: item["name"].casefold(),
    )
    offset = (page - 1) * page_size
    return {
        "items": matches[offset : offset + page_size],
        "total": len(matches),
        "page": page,
        "has_more": len(matches) > offset + page_size,
        "query": query,
    }


def playlist_detail(conn, playlist_id: str) -> dict | None:
    """Use the latest metadata snapshot even when its track list is empty."""
    snapshot = conn.execute(
        "SELECT name, description, backup_time, total_tracks FROM playlist_snapshots "
        "WHERE playlist_id=? ORDER BY backup_time DESC LIMIT 1",
        (playlist_id,),
    ).fetchone()
    managed = conn.execute(
        "SELECT * FROM managed_playlists WHERE playlist_id=?",
        (playlist_id,),
    ).fetchone()
    if not snapshot and not managed:
        return None
    detail = {
        "name": (snapshot or managed)["name"] or playlist_id,
        "playlist_id": playlist_id,
        "description": "",
        "summary": "",
        "kind": "Managed · " + managed["kind"].replace("_", " ") if managed else "Library playlist",
        "observed_at": None,
        "tracks_known": True,
        "total": None,
        "entries": [],
    }
    if snapshot:
        detail.update(
            description=snapshot["description"] or "",
            observed_at=snapshot["backup_time"],
            total=snapshot["total_tracks"],
            source="Last recorded Spotify track list",
            entries=[
                dict(row)
                for row in conn.execute(
                    "SELECT track_id, position, raw_json FROM playlist_track_snapshots "
                    "WHERE playlist_id=? AND backup_time=? ORDER BY position, id",
                    (playlist_id, snapshot["backup_time"]),
                )
            ],
        )
    elif managed["kind"] == "smart" and (saved := smart.get(conn, managed["key"])):
        detail.update(selection_detail(conn, saved["spec"]))
    else:
        detail.update(
            source="Managed membership record · Spotify order is not recorded",
            entries=[
                {"track_id": row["track_id"]}
                for row in conn.execute(
                    "SELECT track_id FROM managed_playlist_tracks WHERE kind=? AND key=? "
                    "AND currently_in=1 ORDER BY first_added_at, track_id",
                    (managed["kind"], managed["key"]),
                )
            ],
        )
        detail["tracks_known"] = bool(detail["entries"])
    if detail["total"] is None and detail["tracks_known"]:
        detail["total"] = len(detail["entries"])
    return detail


def selection_detail(conn, spec: dict) -> dict:
    ids = smart.evaluate(conn, spec)
    return {
        "entries": [{"track_id": tid} for tid in ids],
        "total": len(ids),
        "tracks_known": True,
        "source": "Saved fixed selection"
        if smart.is_curated(spec)
        else "Current rule matches · may change at the next refresh",
    }


def detail_tracks(conn, detail: dict, *, page: int = 1, page_size: int = 100) -> dict:
    """Label one page, preserving duplicates, positions and unavailable items."""
    entries = detail.pop("entries")
    offset = (page - 1) * page_size
    entries_page = entries[offset : offset + page_size]
    labels = _track_labels(conn, {entry["track_id"] for entry in entries_page if entry["track_id"]})
    tracks = []
    for rank, entry in enumerate(entries_page, offset + 1):
        tid = entry["track_id"]
        try:
            raw = json.loads(entry.get("raw_json") or "{}") or {}
            raw = raw.get("track", raw) or {}
            raw_title = raw.get("name")
            raw_artist = ", ".join(a.get("name", "") for a in raw.get("artists", []))
        except (ValueError, TypeError, AttributeError):
            raw_title, raw_artist = None, None
        label = labels.get(tid, {})
        tracks.append(
            {
                "track_id": tid,
                "rank": entry["position"] + 1 if entry.get("position") is not None else rank,
                "title": label.get("title") or raw_title or tid or "Unavailable or local item",
                "artist": label.get("artist") or raw_artist or "Unknown artist",
            }
        )
    return {
        **detail,
        "tracks": tracks,
        "recorded_count": len(entries),
        "complete": detail["total"] is None or detail["total"] == len(entries),
        "page": page,
        "has_more": len(entries) > offset + page_size,
    }


def snapshot_history(
    conn, *, page: int = 1, page_size: int = 20, playlist_id: str | None = None
) -> dict:
    """Observed changes, with duplicate occurrences and order preserved in comparisons."""
    page = max(1, int(page))
    rows = conn.execute(
        "SELECT s.playlist_id, s.name, s.backup_time, s.total_tracks, "
        "(SELECT MAX(p.backup_time) FROM playlist_snapshots p "
        "WHERE p.playlist_id=s.playlist_id AND p.backup_time < s.backup_time) previous_at "
        "FROM playlist_snapshots s "
        + ("WHERE s.playlist_id=? " if playlist_id else "")
        + "ORDER BY s.backup_time DESC, s.id DESC LIMIT ? OFFSET ?",
        (*([playlist_id] if playlist_id else []), page_size + 1, (page - 1) * page_size),
    ).fetchall()
    items, ids = [], set()
    for row in rows[:page_size]:
        after = _track_ids(conn, row["playlist_id"], row["backup_time"])
        before = (
            _track_ids(conn, row["playlist_id"], row["previous_at"]) if row["previous_at"] else []
        )
        baseline = row["previous_at"] is None
        previous = (
            conn.execute(
                "SELECT total_tracks FROM playlist_snapshots WHERE playlist_id=? AND backup_time=?",
                (row["playlist_id"], row["previous_at"]),
            ).fetchone()
            if not baseline
            else None
        )
        complete = (row["total_tracks"] is None or len(after) == row["total_tracks"]) and (
            not previous
            or previous["total_tracks"] is None
            or len(before) == previous["total_tracks"]
        )
        added = Counter(tid for tid in after if tid) - Counter(tid for tid in before if tid)
        removed = Counter(tid for tid in before if tid) - Counter(tid for tid in after if tid)
        if baseline or not complete:
            added, removed = Counter(), Counter()
        unknown_before, unknown_after = before.count(None), after.count(None)
        item = {
            **dict(row),
            "baseline": baseline,
            "complete": complete,
            "count": row["total_tracks"] if row["total_tracks"] is not None else len(after),
            "added_counts": added,
            "removed_counts": removed,
            "added_count": sum(added.values()),
            "removed_count": sum(removed.values()),
            "unknown_before": unknown_before,
            "unknown_after": unknown_after,
            "reordered": complete
            and not baseline
            and not added
            and not removed
            and unknown_before == unknown_after
            and before != after,
        }
        ids.update(added)
        ids.update(removed)
        items.append(item)
    labels = _track_labels(conn, ids)
    for item in items:
        for kind in ("added", "removed"):
            item[kind] = [
                {
                    "track_id": tid,
                    "title": labels.get(tid, {}).get("title") or tid,
                    "artist": labels.get(tid, {}).get("artist") or "Unknown artist",
                    "count": count,
                }
                for tid, count in item.pop(f"{kind}_counts").items()
            ]
    return {"items": items, "page": page, "has_more": len(rows) > page_size}
