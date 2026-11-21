# /// script
# requires-python = ">=3.12"
# dependencies = ["playwright>=1.48"]
# ///
"""Capture retina (2x) showcase screenshots of Acetate from a disposable demo DB.

Usage:
    uv run scripts/capture_showcase.py
    uv run scripts/capture_showcase.py --out docs/showcase --offline

Spins up a loopback-only FastAPI/uvicorn server on a free ephemeral port against a
fresh disposable `/tmp` SQLite database seeded by `scripts/seed_demo.py`, captures
five curated screenshots, verifies zero broken images, and cleans up automatically.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parent.parent

# Candidate Chrome/Chromium paths when Playwright's bundled browser is not downloaded.
CHROME_CANDIDATES = [
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/usr/bin/google-chrome"),
    Path("/usr/bin/chromium"),
    Path("/usr/bin/chromium-browser"),
]

SHOTS: list[tuple[str, str, int, int | None]] = [
    ("home.png", "/", 800, None),
    ("wrapped.png", "/wrapped?kind=year&key=2026", 800, None),
    ("hyperspace-style.png", "/hyperspace", 2200, None),
    ("chat.png", "/chat", 1000, 1080),
    ("playlists.png", "/playlists", 800, None),
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_ready(port: int, proc: subprocess.Popen) -> None:
    url = f"http://127.0.0.1:{port}/healthz"
    for _ in range(80):
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            raise RuntimeError(f"uvicorn exited early ({proc.returncode}): {stderr[:600]}")
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for {url}")


def _launch_browser(playwright):
    args = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-webgl"]
    try:
        return playwright.chromium.launch(headless=True, args=args)
    except Exception:
        for candidate in CHROME_CANDIDATES:
            if candidate.is_file():
                return playwright.chromium.launch(
                    executable_path=str(candidate), headless=True, args=args
                )
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description="Capture 2x retina showcase screenshots.")
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO / "docs" / "showcase",
        help="Destination directory for PNGs (default: docs/showcase).",
    )
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Seed demo DB without external artwork lookups (SVG fallback covers).",
    )
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=920)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="acetate-showcase-"))
    demo_db = tmpdir / "acetate-demo.db"

    py_bin = REPO / ".venv" / "bin" / "python"
    python_exec = str(py_bin) if py_bin.is_file() else sys.executable

    seed_cmd = [python_exec, str(REPO / "scripts" / "seed_demo.py"), "--db", str(demo_db)]
    if args.offline:
        seed_cmd.append("--offline")
    subprocess.run(seed_cmd, cwd=str(REPO), check=True)

    port = _free_port()
    env = os.environ.copy()
    env["ACETATE_DB"] = str(demo_db)
    env["ACETATE_AUTH_MODE"] = "local"
    env["ACETATE_LOCAL_USER"] = "demo"
    env["ACETATE_TZ"] = "America/New_York"
    env["OPENROUTER_API_KEY"] = "demo-showcase-key"
    env["PYTHONPATH"] = str(REPO)
    # Keep subprocess isolated from any host 1Password CLI during headless capture.
    env["PATH"] = "/usr/bin:/bin"

    proc = subprocess.Popen(
        [
            python_exec,
            "-m",
            "uvicorn",
            "acetate.web:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        # Run from tmpdir so dotenv never picks up a developer's local .env file.
        cwd=str(tmpdir),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        _wait_ready(port, proc)
        with sync_playwright() as p:
            browser = _launch_browser(p)
            page = browser.new_page(
                viewport={"width": args.width, "height": args.height},
                device_scale_factor=2,
            )
            broken_total = 0
            for filename, route, settle_ms, custom_height in SHOTS:
                page.set_viewport_size(
                    {"width": args.width, "height": custom_height or args.height}
                )
                url = f"http://127.0.0.1:{port}{route}"
                page.goto(url, wait_until="networkidle", timeout=30_000)
                page.wait_for_timeout(settle_ms)
                dest = args.out / filename
                page.screenshot(path=str(dest), full_page=False)
                imgs = page.evaluate(
                    "() => Array.from(document.images).map(i => ({src: i.src.slice(0, 80), "
                    "ok: i.complete && i.naturalWidth > 0}))"
                )
                broken = [img for img in imgs if not img["ok"]]
                broken_total += len(broken)
                print(
                    f"captured {dest.relative_to(REPO)} ({len(imgs)} images, {len(broken)} broken)"
                )
                if broken:
                    for b in broken:
                        print("  BROKEN:", b["src"])
            browser.close()
            if broken_total:
                return 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(tmpdir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
