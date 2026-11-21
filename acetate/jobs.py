"""Job functions shared by the scheduler, the web triggers, and the CLI.

Each opens its own DB connection (SQLite connections aren't shareable across
threads) and builds a fresh authenticated client (refreshes the cached token).
"""

from __future__ import annotations

import ast
import json
import threading
import time
from contextlib import closing, contextmanager
from datetime import date

from loguru import logger

from . import auth, db, genre, playlists, snapshot


def _client():
    return auth.client(open_browser=False)


# --------------------------------------------------------------------------- #
#  Run guard — one job name at a time, per process
# --------------------------------------------------------------------------- #
# The web button, the chat agent, and the CLI all reach the same job functions,
# so the claim lives here rather than in any one caller. It is per-process by
# design: the scheduler runs in its own container and guards itself with
# APScheduler's max_instances=1. A manual run overlapping a scheduled one is
# safe anyway — every insert is idempotent — this just avoids the wasted work.

_run_lock = threading.Lock()
_running: set[str] = set()


def running_jobs() -> list[str]:
    with _run_lock:
        return sorted(_running)


def try_claim(name: str) -> bool:
    """Reserve a job name. False when it is already running here."""
    with _run_lock:
        if name in _running:
            return False
        _running.add(name)
        return True


def release(name: str) -> None:
    with _run_lock:
        _running.discard(name)


@contextmanager
def claimed(name: str):
    """Scoped `try_claim` — yields whether the claim was won, always releases."""
    acquired = try_claim(name)
    try:
        yield acquired
    finally:
        if acquired:
            release(name)


def run_poll() -> dict:
    """Pull recently-played into history (the frequent job)."""
    conn = db.connect()
    db.init_db(conn)
    res = snapshot.poll_recently_played(conn, _client())
    if res.get("inserted"):
        from . import canon

        canon.refresh_recording_stats(conn)  # keep all-time stats fresh after new plays
    logger.info("poll: fetched={} new={}", res["fetched"], res["inserted"])
    return res


def _plays_since(conn, watermark: str | None, limit: int = 10) -> list[dict]:
    """Polled plays newer than `watermark` — what this refresh actually learned."""
    rows = conn.execute(
        "SELECT p.played_at, l.artist_name artist, l.track_name title "
        "FROM play_events p LEFT JOIN track_labels l ON l.track_id=p.track_id "
        "WHERE p.source='recently_played' AND p.played_at > COALESCE(?, '') "
        "ORDER BY p.played_at DESC LIMIT ?",
        (watermark, limit),
    ).fetchall()
    return [{"played_at": r["played_at"], "artist": r["artist"], "title": r["title"]} for r in rows]


def refresh_plays() -> dict:
    """Poll recently-played right now and report what landed.

    The scheduler already polls every 15 minutes; this is the "I want acetate to
    know what I just heard" path — same job, same `job_run` audit trail. Runs
    synchronously because the answer (what came in) is the whole point.
    """
    if not try_claim("poll"):
        return {"status": "busy", "detail": "a poll is already running — try again in a moment"}
    started = time.monotonic()
    conn = db.connect()
    try:
        before = conn.execute(
            "SELECT MAX(played_at) FROM play_events WHERE source='recently_played'"
        ).fetchone()[0]
        try:
            result = record_run("poll", run_poll)
        except Exception as exc:
            logger.exception("manual play refresh failed")
            return {
                "status": "error",
                "error": str(exc)[:300],
                "duration_ms": round((time.monotonic() - started) * 1000),
            }
        new_plays = _plays_since(conn, before)
        return {
            "status": "ok",
            "fetched": result.get("fetched", 0),
            "inserted": result.get("inserted", 0),
            "new_plays": new_plays,
            "last_play_at": new_plays[0]["played_at"] if new_plays else before,
            "duration_ms": round((time.monotonic() - started) * 1000),
        }
    finally:
        conn.close()
        release("poll")


def run_backup(only_mine: bool = True) -> dict:
    """Snapshot playlists + saved tracks (the deletion trace)."""
    with closing(db.connect()) as conn:
        db.init_db(conn)
        res = snapshot.run_backup(conn, _client(), only_mine=only_mine)
        logger.info("backup: {}", res)
        # refresh playlist profiles off the fresh snapshot (chat-agent substrate)
        from . import playlist_profile

        res["profiled"] = playlist_profile.rebuild(conn)["playlists"]
        return res


def run_monthly(month: str | None = None, top_limit: int = 50) -> dict:
    """Build/refresh this month's 📆/🔥 playlists."""
    with closing(db.connect()) as conn:
        db.init_db(conn)
        sp = _client()
        ym = month or date.today().strftime("%Y-%m")
        out = {
            "liked": playlists.build_liked_month(conn, sp, ym),
            "top_played": playlists.build_top_played_month(conn, sp, ym, limit=top_limit),
            "liked_year": playlists.build_liked_year(conn, sp, ym[:4]),  # keep current year fresh
        }
        logger.info("monthly {}: {}", ym, out)
        return out


def run_backfill(batch: int = 2000) -> dict:
    """Tag a chunk of the library's genres via Last.fm (resumable)."""
    with closing(db.connect()) as conn:
        db.init_db(conn)
        res = genre.tag_backfill(conn, batch=batch)
        logger.info("backfill: {}", res)
        return res


def run_yt_radio(
    batch: int = 12, radio_limit: int = 50, pause_s: float = 1.5, min_plays: int = 5
) -> dict:
    """Advance the YouTube Music radio harvest by one polite batch. Drains the
    priority seed order (techno space + fave/hot playlists, then the rest of the
    liked/most-played universe); once every seed is fetched, re-harvests the
    oldest for drift. Resumable — each run just picks up the frontier."""
    from . import ytmusic

    if not ytmusic.is_available():
        logger.info("yt_radio skipped: optional dependency ytmusicapi is not installed")
        return {"skipped": "ytmusicapi_not_installed"}

    with closing(db.connect()) as conn:
        db.init_db(conn)
        seeds = ytmusic.background_seeds(conn, min_plays=min_plays)
        fetched = {
            r[0]
            for r in conn.execute(
                "SELECT recording_key FROM ytmusic_radio_seeds WHERE status='fetched'"
            )
        }
        unfetched = [k for k in seeds if k not in fetched]
        if unfetched:
            res = ytmusic.harvest_radio(
                conn, batch=batch, radio_limit=radio_limit, pause_s=pause_s, seed_keys=seeds
            )
            res["mode"] = "advance"
            res["remaining"] = max(0, len(unfetched) - res.get("fetched", 0))
        else:  # frontier covered — refresh the oldest seeds for drift
            res = ytmusic.harvest_radio(
                conn,
                batch=batch,
                radio_limit=radio_limit,
                pause_s=pause_s,
                include_fetched=True,
                seed_keys=ytmusic.stale_fetched_seeds(conn, batch),
            )
            res["mode"] = "refresh"
        logger.info("yt_radio: {}", res)
        return res


def run_candidate_prep(resolve_batch: int = 100, preview_batch: int = 60) -> dict:
    """The light half of the outside-library pipeline (runs in the slim app env):
    resolve top candidates to Deezer, download their previews, then finalize +
    rerank whatever the embed sidecar has since vectorized. The heavy EffNet/CLAP
    embedding runs in the separate acetate-embed container (Python 3.11 + torch);
    this job just keeps candidates preview-ready and folds finished vectors into
    the ranked outside-library pool. Each step is isolated so one flaky external
    API can't sink the others."""
    from . import candidate_audio, candidate_deezer, candidate_embeddings

    with closing(db.connect()) as conn:
        db.init_db(conn)
        out: dict = {}
        for name, step in (
            ("resolve", lambda: candidate_deezer.resolve(conn, batch=resolve_batch)),
            ("preview", lambda: candidate_audio.fetch_previews(conn, batch=preview_batch)),
            ("finalize", lambda: {"finalized": candidate_embeddings.finalize(conn)}),
            ("rerank", lambda: candidate_embeddings.rerank(conn)),
        ):
            try:
                out[name] = step()
            except Exception as exc:  # keep going; a flaky Deezer call shouldn't stop rerank
                logger.warning("candidate_prep {} failed: {}", name, exc)
                out[name] = {"error": str(exc)[:200]}
        logger.info("candidate_prep: {}", out)
        return out


def run_hydrate(limit: int | None = None) -> dict:
    """Fetch artist images + genres from Spotify for un-hydrated artists (resumable)."""
    with closing(db.connect()) as conn:
        db.init_db(conn)
        res = snapshot.hydrate_artists(conn, _client(), limit=limit)
        logger.info("hydrate artists: {}", res)
        return res


def run_gems(size: int = 50) -> dict:
    """Build/refresh the rotating 💎 Lost Gems playlist."""
    with closing(db.connect()) as conn:
        db.init_db(conn)
        res = playlists.build_lost_gems(conn, _client(), size=size)
        logger.info("lost gems: {}", res)
        return res


def run_on_loop(days: int = 21, size: int = 40) -> dict:
    """Rebuild the transient 🔁 On Loop playlist (current heavy rotation)."""
    with closing(db.connect()) as conn:
        db.init_db(conn)
        res = playlists.build_on_loop(conn, _client(), days=days, size=size)
        logger.info("on loop: {}", res)
        return res


def run_smart(smart_id: int | None = None) -> dict:
    """Refresh enabled rules and publish pending fixed selections once, or run one ID."""
    with closing(db.connect()) as conn:
        db.init_db(conn)
        from . import smart

        rows = [smart.get(conn, smart_id)] if smart_id is not None else smart.list_all(conn)
        rows = [
            r
            for r in rows
            if r
            and (
                smart_id is not None
                or (r["enabled"] and (not smart.is_curated(r["spec"]) or not r["published_at"]))
            )
        ]
        sp = _client() if rows else None
        out = [playlists.build_smart(conn, sp, r) for r in rows]
        logger.info("smart playlists: {}", out)
        return {"built": sum(1 for o in out if not o.get("skipped")), "results": out}


def run_canon() -> dict:
    """Rebuild the canonical recording map (collapse Spotify's duplicate track IDs)."""
    with closing(db.connect()) as conn:
        db.init_db(conn)
        from . import canon

        res = canon.canonicalize(conn)
        logger.info("canonicalize: {}", res)
        return res


def status() -> dict:
    """Snapshot of pipeline state for the dashboard."""
    with closing(db.connect()) as conn:

        def one(sql: str, *params):
            row = conn.execute(sql, params).fetchone()
            return row[0] if row else None

        universe = genre.universe_count(conn)
        tagged = len(genre._already_tagged(conn))
        latest_bt = one("SELECT MAX(backup_time) FROM saved_track_snapshots")
        managed = [
            {
                "kind": r["kind"],
                "key": r["key"],
                "name": r["name"],
                "url": f"https://open.spotify.com/playlist/{r['playlist_id']}",
                "updated_at": r["updated_at"],
            }
            for r in conn.execute(
                "SELECT kind, key, name, playlist_id, updated_at FROM managed_playlists "
                "ORDER BY updated_at DESC"
            )
        ]
        return {
            "play_events": one("SELECT COUNT(*) FROM play_events"),
            "plays_from_poll": one(
                "SELECT COUNT(*) FROM play_events WHERE source='recently_played'"
            ),
            "last_play_at": one("SELECT MAX(played_at) FROM play_events"),
            "last_poll_at": one(
                "SELECT MAX(played_at) FROM play_events WHERE source='recently_played'"
            ),
            "liked_tracks": one(
                "SELECT COUNT(*) FROM saved_track_snapshots WHERE backup_time=?", latest_bt
            ),
            "last_backup_at": latest_bt,
            "playlist_snapshots": one("SELECT COUNT(*) FROM playlist_snapshots"),
            "genre_universe": universe,
            "genre_tagged": tagged,
            "genre_pct": round(100 * tagged / universe) if universe else 0,
            "managed_playlists": managed,
        }


def record_run(job: str, fn) -> dict:
    """Run a job, logging start/finish/status/result to job_run (the audit trail)."""
    started = db.utcnow()
    status_, result_text, error = "error", None, None
    try:
        result = fn()
        status_, result_text = "ok", json.dumps(result, default=str)[:2000]
        return result
    except Exception as exc:
        error = str(exc)[:4000]  # keep the full raw error for the dashboard
        raise
    finally:
        with closing(db.connect()) as conn:
            conn.execute(
                "INSERT INTO job_run(job, started_at, finished_at, status, result, error) "
                "VALUES(?,?,?,?,?,?)",
                (job, started, db.utcnow(), status_, result_text, error),
            )
            conn.commit()


def _parse_result(text: str | None) -> dict:
    if not text:
        return {}
    for parse in (json.loads, ast.literal_eval):  # ast recovers older repr-stored rows
        try:
            v = parse(text)
            return v if isinstance(v, dict) else {"value": v}
        except Exception:
            continue
    return {}


def recent_runs(limit: int = 12) -> list[dict]:
    """Recent job runs, enriched for display: any created playlists become links,
    other results a compact summary, errors kept raw."""
    with closing(db.connect()) as conn:
        try:
            rows = conn.execute(
                "SELECT job, started_at, status, result, error FROM job_run "
                "ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        except Exception:
            return []
        out = []
        for r in rows:
            d = _parse_result(r["result"])
            pls: list[dict] = []

            def walk(x):
                if isinstance(x, dict):
                    if x.get("playlist_id"):
                        pls.append(
                            {
                                "name": x.get("name"),
                                "tracks": x.get("tracks"),
                                "url": f"https://open.spotify.com/playlist/{x['playlist_id']}",
                            }
                        )
                    for v in x.values():
                        walk(v)

            walk(d)
            summary = ", ".join(
                f"{k} {v}"
                for k, v in d.items()
                if k not in ("playlist_id",) and not isinstance(v, (dict, list))
            )
            out.append(
                {
                    "job": r["job"],
                    "started_at": r["started_at"],
                    "status": r["status"],
                    "error": r["error"],
                    "playlists": pls,
                    "summary": summary,
                }
            )
        return out


# Map trigger names → callables for the web layer.
TRIGGERS = {
    "poll": run_poll,
    "backup": run_backup,
    "monthly": run_monthly,
    "backfill": run_backfill,
    "hydrate": run_hydrate,
    "gems": run_gems,
    "on_loop": run_on_loop,
    "smart": run_smart,
    "canon": run_canon,
}
