#!/usr/bin/env python3
"""Orchestrate the safe Mac-enrichment -> VPS database promotion."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from acetate.reconcile import (
    APPLY_CONFIRMATION,
    RESTORE_CONFIRMATION,
    export_bundle,
    inspect_database,
    sha256_file,
)


def _run(args: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print(f"+ {shlex.join(args)}", flush=True)
    try:
        return subprocess.run(
            args,
            check=True,
            text=True,
            capture_output=capture,
        )
    except subprocess.CalledProcessError as exc:
        if capture and exc.stdout:
            print(exc.stdout, file=sys.stdout, end="")
        if capture and exc.stderr:
            print(exc.stderr, file=sys.stderr, end="")
        raise


def _human_bytes(value: int) -> str:
    size = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if abs(size) < 1024 or unit == units[-1]:
            precision = 0 if unit == "B" else 1
            return f"{size:.{precision}f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def _humanize_sizes(value, key: str | None = None):
    if isinstance(value, dict):
        return {k: _humanize_sizes(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_humanize_sizes(v) for v in value]
    if isinstance(value, int) and (key == "bytes" or (key or "").endswith("_bytes")):
        return _human_bytes(value)
    return value


def _print_report(report: dict) -> None:
    print(json.dumps(_humanize_sizes(report), indent=2, sort_keys=True))


def _print_json_result(output: str) -> None:
    try:
        _print_report(json.loads(output))
    except json.JSONDecodeError:
        print(output)


def _remote_shell_path(path: str) -> str:
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    return shlex.quote(path)


def _ssh(host: str, command: str, *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return _run(["ssh", host, command], capture=capture)


def _compose_command(compose_dir: str, command: list[str]) -> str:
    return f"cd {_remote_shell_path(compose_dir)} && {shlex.join(command)}"


def _service_names(services: str) -> list[str]:
    return shlex.split(services) if services.strip() else ["app", "scheduler"]


def _app_service(services: str) -> str:
    return _service_names(services)[0]


def _remote_inspect(host: str, compose_dir: str, services: str) -> dict:
    result = _ssh(
        host,
        _compose_command(
            compose_dir,
            [
                "docker",
                "compose",
                "run",
                "--rm",
                "--no-deps",
                _app_service(services),
                "python",
                "-m",
                "acetate.reconcile",
                "inspect",
                "--target",
                "/data/acetate.db",
            ],
        ),
        capture=True,
    )
    return json.loads(result.stdout)


def _remote_free_bytes(host: str, compose_dir: str, services: str) -> int:
    result = _ssh(
        host,
        _compose_command(
            compose_dir,
            [
                "docker",
                "compose",
                "run",
                "--rm",
                "--no-deps",
                _app_service(services),
                "python",
                "-c",
                "import shutil; print(shutil.disk_usage('/data').free)",
            ],
        ),
        capture=True,
    )
    return int(result.stdout.strip())


def _remote_spec(host: str, remote_data_dir: str, relative_path: str) -> str:
    root = remote_data_dir.rstrip("/")
    return f"{host}:{root}/{relative_path}"


def _audio_sync_dry_run(args: argparse.Namespace) -> int:
    result = _run(
        [
            "rsync",
            "-a",
            "--dry-run",
            "--stats",
            "--ignore-existing",
            f"{args.audio_dir.resolve()}/",
            _remote_spec(args.host, args.remote_data_dir, "audio/"),
        ],
        capture=True,
    )
    match = re.search(r"Total transferred file size: ([0-9,]+) bytes", result.stdout)
    if match:
        return int(match.group(1).replace(",", ""))
    # Older rsync variants format stats differently. Falling back to the full
    # local tree is conservative and can only reject a run before it writes.
    return sum(p.stat().st_size for p in args.audio_dir.rglob("*") if p.is_file())


def plan(args: argparse.Namespace) -> None:
    print("\nLOCAL (promoted enrichment source)")
    _print_report(inspect_database(args.local_db))
    print("\nVPS (preserved live-data base)")
    _print_report(_remote_inspect(args.host, args.compose_dir, args.services))
    remote_free = _remote_free_bytes(args.host, args.compose_dir, args.services)
    print(f"VPS free space: {_human_bytes(remote_free)}")
    print("\nAUDIO SYNC DRY RUN (additive; no remote deletes or overwrites)")
    missing_audio = _audio_sync_dry_run(args)
    print(f"Estimated missing audio: {_human_bytes(missing_audio)}")
    print(
        "\nPlan only: VPS-owned rows remain untouched. "
        "Run `just db-reconcile-apply` to export, verify, upload, stop, "
        "atomically swap, and restart."
    )


def apply(args: argparse.Namespace) -> None:
    if args.confirm != APPLY_CONFIRMATION:
        raise SystemExit(f"--confirm must be exactly {APPLY_CONFIRMATION}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    bundle = work_dir / f"acetate-enrichment-{stamp}.db"
    remote_bundle_container = f"/data/reconcile/{bundle.name}"

    local_free = shutil.disk_usage(work_dir).free
    local_needed = int(args.local_db.stat().st_size * 1.10)
    if local_free < local_needed:
        raise RuntimeError(
            f"insufficient local free space: need at least {_human_bytes(local_needed)}, "
            f"have {_human_bytes(local_free)}"
        )

    remote_state = _remote_inspect(args.host, args.compose_dir, args.services)
    missing_audio = _audio_sync_dry_run(args)
    print("Exporting the explicit Mac-owned enrichment table set...")
    export = export_bundle(args.local_db, bundle)
    _print_report(export)

    # Before any remote writes, reserve space for missing assets + bundle + the
    # backup and candidate DB copies required during the offline apply.
    remote_free = _remote_free_bytes(args.host, args.compose_dir, args.services)
    apply_workspace = int(remote_state["bytes"] * 2.20) + 512 * 1024 * 1024
    remote_needed = missing_audio + int(export["bytes"]) + apply_workspace
    if remote_free < remote_needed:
        raise RuntimeError(
            f"insufficient VPS free space: need at least {_human_bytes(remote_needed)}, "
            f"have {_human_bytes(remote_free)}"
        )

    _ssh(
        args.host,
        "mkdir -p "
        + " ".join(
            [
                f"{_remote_shell_path(args.remote_data_dir)}/reconcile",
                f"{_remote_shell_path(args.remote_data_dir)}/backups",
                f"{_remote_shell_path(args.remote_data_dir)}/audio",
            ]
        ),
    )

    print("Syncing content-addressed audio assets additively...")
    _run(
        [
            "rsync",
            "-ah",
            "--partial",
            "--progress",
            "--ignore-existing",
            f"{args.audio_dir.resolve()}/",
            _remote_spec(args.host, args.remote_data_dir, "audio/"),
        ]
    )
    print("Uploading the verified enrichment bundle...")
    _run(
        [
            "rsync",
            "-ah",
            "--partial",
            "--progress",
            str(bundle),
            _remote_spec(args.host, args.remote_data_dir, f"reconcile/{bundle.name}"),
        ]
    )

    remote_hash_result = _ssh(
        args.host,
        f"sha256sum {_remote_shell_path(args.remote_data_dir)}/reconcile/"
        f"{shlex.quote(bundle.name)}",
        capture=True,
    )
    remote_hash = remote_hash_result.stdout.split()[0]
    local_hash = sha256_file(bundle)
    if remote_hash != local_hash:
        raise RuntimeError(f"bundle checksum mismatch: local={local_hash}, remote={remote_hash}")
    print(f"Bundle checksum verified: {local_hash}")

    # Bundle/assets now occupy their final space. Recheck the two-copy apply
    # workspace immediately before stopping any service.
    remote_free = _remote_free_bytes(args.host, args.compose_dir, args.services)
    if remote_free < apply_workspace:
        raise RuntimeError(
            f"insufficient VPS apply workspace: need {_human_bytes(apply_workspace)}, "
            f"have {_human_bytes(remote_free)}"
        )

    services = _service_names(args.services)
    app_service = _app_service(args.services)
    stopped = False
    try:
        print("Stopping database writers for the atomic reconciliation window...")
        _ssh(
            args.host, _compose_command(args.compose_dir, ["docker", "compose", "stop", *services])
        )
        stopped = True
        print("Applying bundle to an offline copy; the current DB is backed up first...")
        result = _ssh(
            args.host,
            _compose_command(
                args.compose_dir,
                [
                    "docker",
                    "compose",
                    "run",
                    "--rm",
                    "--no-deps",
                    app_service,
                    "python",
                    "-m",
                    "acetate.reconcile",
                    "apply",
                    "--target",
                    "/data/acetate.db",
                    "--bundle",
                    remote_bundle_container,
                    "--backup-dir",
                    "/data/backups",
                    "--confirm",
                    APPLY_CONFIRMATION,
                ],
            ),
            capture=True,
        )
        _print_json_result(result.stdout)
    finally:
        if stopped:
            print("Restarting services...")
            _ssh(
                args.host,
                _compose_command(args.compose_dir, ["docker", "compose", "up", "-d", *services]),
            )

    print("\nPOST-APPLY VPS STATE")
    _print_report(_remote_inspect(args.host, args.compose_dir, args.services))
    print(f"\nLocal bundle retained for audit: {bundle}")
    print("The remote pre-reconcile backup path is reported in the apply output above.")


def restore(args: argparse.Namespace) -> None:
    if args.confirm != RESTORE_CONFIRMATION:
        raise SystemExit(f"--confirm must be exactly {RESTORE_CONFIRMATION}")
    if not args.backup or Path(args.backup).name != args.backup:
        raise SystemExit("--backup must be one filename from the VPS data/backups directory")

    services = _service_names(args.services)
    app_service = _app_service(args.services)
    stopped = False
    try:
        print("Stopping database writers for rollback...")
        _ssh(
            args.host, _compose_command(args.compose_dir, ["docker", "compose", "stop", *services])
        )
        stopped = True
        result = _ssh(
            args.host,
            _compose_command(
                args.compose_dir,
                [
                    "docker",
                    "compose",
                    "run",
                    "--rm",
                    "--no-deps",
                    app_service,
                    "python",
                    "-m",
                    "acetate.reconcile",
                    "restore",
                    "--target",
                    "/data/acetate.db",
                    "--from-backup",
                    f"/data/backups/{args.backup}",
                    "--backup-dir",
                    "/data/backups",
                    "--confirm",
                    RESTORE_CONFIRMATION,
                ],
            ),
            capture=True,
        )
        _print_json_result(result.stdout)
    finally:
        if stopped:
            print("Restarting services...")
            _ssh(
                args.host,
                _compose_command(args.compose_dir, ["docker", "compose", "up", "-d", *services]),
            )

    print("\nPOST-RESTORE VPS STATE")
    _print_report(_remote_inspect(args.host, args.compose_dir, args.services))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "apply", "restore"))
    parser.add_argument("--host", required=True)
    parser.add_argument("--compose-dir", required=True)
    parser.add_argument("--remote-data-dir", required=True)
    parser.add_argument("--services", default="")
    parser.add_argument("--local-db", type=Path, default=Path("data/acetate.db"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/audio"))
    parser.add_argument("--work-dir", type=Path, default=Path("data/reconcile"))
    parser.add_argument("--confirm", default="")
    parser.add_argument("--backup", default="")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.mode == "plan":
        plan(args)
    elif args.mode == "apply":
        apply(args)
    else:
        restore(args)


if __name__ == "__main__":
    main()
