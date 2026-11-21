"""One scheduled Acetate job, run in a disposable process."""

import argparse
from functools import partial

from . import jobs


def main() -> None:
    handlers = {
        "poll": jobs.run_poll,
        "backup": jobs.run_backup,
        "canon": jobs.run_canon,
        "monthly": jobs.run_monthly,
        "gems": jobs.run_gems,
        "on_loop": jobs.run_on_loop,
        "smart": jobs.run_smart,
        "hydrate": jobs.run_hydrate,
        "backfill": jobs.run_backfill,
        "yt_radio": jobs.run_yt_radio,
        "candidate_prep": jobs.run_candidate_prep,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=handlers)
    parser.add_argument("limit", type=int, nargs="?")
    args = parser.parse_args()
    call = handlers[args.job]
    if args.limit is not None:
        if args.job not in {"hydrate", "backfill", "yt_radio"}:
            parser.error("this job does not accept a limit")
        call = partial(call, args.limit)
    jobs.record_run(args.job, call)


if __name__ == "__main__":
    main()
