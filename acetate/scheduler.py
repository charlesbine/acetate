"""Scheduled jobs for acetate (runs as its own container/process).

Run: python -m acetate.scheduler

Cadence (overridable via env):
  ACETATE_POLL_MINUTES   poll recently-played       (default 15 — sized to never drop plays)
  ACETATE_BACKUP_HOUR    daily library backup        (default 04:00)
  ACETATE_MONTHLY_HOUR   daily monthly-playlist build (default 05:00)
  ACETATE_BACKFILL_BATCH genre tags per hourly chunk  (default 2000; 0 disables)
  ACETATE_YTRADIO_BATCH  YT Music radio seeds per 2h chunk (default 12; 0 disables)
  ACETATE_CANDIDATE_PREP resolve+preview+rerank the candidate frontier (default 1; 0 disables)
"""

from __future__ import annotations

import os
import subprocess
import sys

from apscheduler.events import (
    EVENT_JOB_ERROR,
    EVENT_JOB_EXECUTED,
    EVENT_JOB_MISSED,
    EVENT_JOB_SUBMITTED,
    EVENT_SCHEDULER_SHUTDOWN,
    EVENT_SCHEDULER_STARTED,
)
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from . import operations


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _run_job(name: str, *args: int) -> None:
    # Exiting the child releases SQLite pages, NumPy arrays and native allocator
    # arenas even when a job fails. The scheduler keeps only its timing state.
    subprocess.run(
        [sys.executable, "-m", "acetate.scheduler_job", name, *map(str, args)],
        check=True,
    )


def main() -> None:
    poll_min = _int("ACETATE_POLL_MINUTES", 15)
    backup_hour = _int("ACETATE_BACKUP_HOUR", 4)
    monthly_hour = _int("ACETATE_MONTHLY_HOUR", 5)
    backfill_batch = _int("ACETATE_BACKFILL_BATCH", 2000)
    ytradio_batch = _int("ACETATE_YTRADIO_BATCH", 12)
    candidate_prep = _int("ACETATE_CANDIDATE_PREP", 1)

    sched = BlockingScheduler(timezone="UTC")
    # misfire_grace_time + coalesce: if the box was asleep, run once on wake, don't pile up.
    common = {"misfire_grace_time": 3600, "coalesce": True, "max_instances": 1}

    # Wrap every job in record_run so scheduled runs land in the job_run audit log.
    sched.add_job(_run_job, IntervalTrigger(minutes=poll_min), args=("poll",), id="poll", **common)
    sched.add_job(
        _run_job, CronTrigger(hour=backup_hour, minute=0), args=("backup",), id="backup", **common
    )
    # Rebuild the canonical recording map after the backup ingests new tracks,
    # before the playlist builders (which group through it).
    sched.add_job(
        _run_job, CronTrigger(hour=backup_hour, minute=30), args=("canon",), id="canon", **common
    )
    sched.add_job(
        _run_job,
        CronTrigger(hour=monthly_hour, minute=0),
        args=("monthly",),
        id="monthly",
        **common,
    )
    sched.add_job(
        _run_job,
        CronTrigger(day_of_week="sun", hour=6, minute=0),
        args=("gems",),
        id="gems",
        **common,
    )
    # Transient "current rotation" playlist — refresh daily so it stays "right now".
    sched.add_job(
        _run_job,
        CronTrigger(hour=monthly_hour, minute=30),
        args=("on_loop",),
        id="on_loop",
        **common,
    )
    # Refresh enabled Smart Playlist rules daily; publish new curated selections once.
    sched.add_job(
        _run_job, CronTrigger(hour=monthly_hour, minute=45), args=("smart",), id="smart", **common
    )
    sched.add_job(
        _run_job, CronTrigger(hour=3, minute=0), args=("hydrate", 3000), id="hydrate", **common
    )
    if backfill_batch > 0:
        sched.add_job(
            _run_job,
            CronTrigger(minute=30),
            args=("backfill", backfill_batch),
            id="backfill",
            **common,
        )
    # YT Music radio harvest — a polite trickle every 2h, priority seeds first,
    # broadening to the whole liked/most-played universe (unofficial API).
    if ytradio_batch > 0:
        sched.add_job(
            _run_job,
            CronTrigger(hour="*/2", minute=15),
            args=("yt_radio", ytradio_batch),
            id="yt_radio",
            **common,
        )
    # Candidate frontier prep — resolve/preview new candidates and fold freshly
    # embedded ones (from the acetate-embed sidecar) into the ranked pool.
    if candidate_prep > 0:
        sched.add_job(
            _run_job,
            CronTrigger(hour="*/2", minute=45),
            args=("candidate_prep",),
            id="candidate_prep",
            **common,
        )

    active_jobs: set[str] = set()

    def publish_state(event) -> None:
        if event.code == EVENT_JOB_SUBMITTED:
            active_jobs.add(event.job_id)
        elif event.code in (EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED):
            active_jobs.discard(event.job_id)
        state = "stopped" if event.code == EVENT_SCHEDULER_SHUTDOWN else "running"
        try:
            operations.record_scheduler_snapshot(
                sched,
                state=state,
                active_jobs=sorted(active_jobs),
                poll_minutes=poll_min,
            )
        except Exception as exc:
            logger.warning("could not publish scheduler state: {}", exc)

    sched.add_listener(
        publish_state,
        EVENT_SCHEDULER_STARTED
        | EVENT_SCHEDULER_SHUTDOWN
        | EVENT_JOB_SUBMITTED
        | EVENT_JOB_EXECUTED
        | EVENT_JOB_ERROR
        | EVENT_JOB_MISSED,
    )

    logger.info(
        "scheduler up: poll/{}min, backup @{:02d}:00, monthly @{:02d}:00, backfill batch={}, "
        "yt_radio batch={}",
        poll_min,
        backup_hour,
        monthly_hour,
        backfill_batch,
        ytradio_batch,
    )
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("scheduler stopping")


if __name__ == "__main__":
    main()
