"""FastAPI + HTMX dashboard for acetate.

Pages: Home (overview + live status bar), Stats (deep analysis), Playlists,
Pipeline (status + job controls). Templates live in an in-memory DictLoader so
the whole UI ships in this one file (no template-dir wiring in Docker).

Run: uvicorn acetate.web:app --host 127.0.0.1 --port 8770
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, timedelta
from html import escape
from pathlib import Path
from urllib.parse import urlencode, urlsplit

from fastapi import BackgroundTasks, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from jinja2 import DictLoader, Environment
from loguru import logger
from markupsafe import Markup

from . import (
    access,
    agent,
    artist_discovery,
    auth,
    canon,
    chatstore,
    db,
    jobs,
    playlist_activity,
    smart,
    stats,
    timefmt,
    workflows,
)

auth.load_credentials()

app = FastAPI(title="acetate")


_MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_ALLOWED_FETCH_SITES = frozenset({"same-origin", "none"})


def _csrf_error(request: Request) -> str | None:
    """Reject cross-site browser mutations via Fetch Metadata and Origin checks.

    Non-browser callers (CLI scripts, TestClient) that send neither Sec-Fetch-Site
    nor Origin are allowed through.
    """
    if request.method.upper() not in _MUTATING_METHODS:
        return None
    sec_fetch_site = request.headers.get("sec-fetch-site")
    if sec_fetch_site is not None:
        if sec_fetch_site.strip().lower() not in _ALLOWED_FETCH_SITES:
            return "Cross-site request blocked"
    origin = request.headers.get("origin")
    if origin is not None:
        origin_value = origin.strip()
        if not origin_value or origin_value.lower() == "null":
            return "Cross-site origin blocked"
        parsed = urlsplit(origin_value)
        host = (request.headers.get("host") or "").strip().lower()
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or not host
            or parsed.netloc.lower() != host
        ):
            return "Cross-site origin mismatch"
    return None


@app.middleware("http")
async def resolve_identity(request: Request, call_next):
    # Container health checks intentionally do not traverse the auth proxy.
    if request.url.path != "/healthz":
        try:
            request.state.identity = access.identity_from_headers(request.headers)
        except HTTPException as exc:
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    if error := _csrf_error(request):
        return JSONResponse({"detail": error}, status_code=403)
    token = access.identity_context.set(getattr(request.state, "identity", None))
    try:
        return await call_next(request)
    finally:
        access.identity_context.reset(token)


# Repo assets (blog media etc.) — optional dir so deploys without it still boot.
_ASSETS = Path(__file__).parent.parent / "assets"
if _ASSETS.is_dir():
    app.mount("/assets", StaticFiles(directory=_ASSETS), name="assets")
_boot = db.connect()
db.init_db(_boot)  # ensure schema (incl. newer tables) exists at startup
canon.ensure_recording_stats(_boot)  # build the stats rollup if empty (first boot after deploy)

# The run guard lives in jobs.py — the chat agent triggers jobs too, and both
# callers have to see the same "already running" state.
_last_result: dict[str, str] = {}


def _usage_summary(usage: dict | None) -> str:
    """Compact billing summary shown after the model attribution."""
    if not isinstance(usage, dict):
        return ""
    parts: list[str] = []
    cost = usage.get("cost")
    if isinstance(cost, (int, float)):
        decimals = 6 if 0 < cost < 0.0001 else 4 if cost < 0.01 else 2
        parts.append(f"${cost:.{decimals}f} effective")
    parts.extend(
        [
            f"{int(usage.get('input_tokens') or 0):,} in",
            f"{int(usage.get('output_tokens') or 0):,} out",
            f"{int(usage.get('cached_tokens') or 0):,} cached",
        ]
    )
    return " · ".join(parts)


def _run_job(name: str) -> None:
    try:
        result = jobs.record_run(name, jobs.TRIGGERS[name])
        _last_result[name] = f"ok — {result}"[:200]
    except Exception as exc:
        logger.exception("job {} failed", name)
        _last_result[name] = f"error — {exc}"[:200]
    finally:
        jobs.release(name)


# --------------------------------------------------------------------------- #
#  Navigation
# --------------------------------------------------------------------------- #
# One source of truth for the bar. Fourteen top-level tabs never fit — not on a
# phone and not at 1080px either — so the curation and ops pages live in grouped
# menus and only the things you open for pleasure stay top-level. Each entry is
# (active_key, href, label); a None key is an external link with no active state.

NAV_PRIMARY = (
    ("home", "/", "Home"),
    ("stats", "/stats", "Stats"),
    ("wrapped", "/wrapped", "Wrapped"),
    ("hyperspace", "/hyperspace", "Hyperspace"),
    ("chat", "/chat", "Chat"),
)

NAV_GROUPS = (
    (
        "Playlists",
        (
            ("playlists", "/playlists", "All playlists"),
            ("smart", "/smart", "Smart rules"),
            ("curated", "/curated", "Curated playlists"),
            ("gems", "/gems", "Lost Gems"),
            ("health", "/health", "Health"),
            ("review", "/review", "Review queue"),
            ("filing", "/filing", "Filing"),
        ),
    ),
    (
        "System",
        (
            ("tools", "/tools", "Tools"),
            ("pipeline", "/pipeline", "Pipeline"),
            (None, "/how", "How it was built ↗"),
        ),
    ),
)


def nav_for(active: str | None) -> dict:
    """The nav bar with the current page resolved — including which group holds
    it, so a page inside a menu still lights its group up."""

    def item(entry: tuple) -> dict:
        key, href, label = entry
        return {
            "key": key,
            "href": href,
            "label": label,
            "on": key is not None and key == active,
            "external": key is None,
        }

    return {
        "primary": [item(entry) for entry in NAV_PRIMARY],
        "groups": [
            {
                "label": label,
                "items": [item(entry) for entry in entries],
                "on": any(key == active for key, _, _ in entries if key is not None),
            }
            for label, entries in NAV_GROUPS
        ],
    }


# --------------------------------------------------------------------------- #
#  Templates
# --------------------------------------------------------------------------- #

BASE = r"""<!doctype html><html lang=en><head>
<meta charset=utf-8><meta name=viewport content="width=device-width, initial-scale=1">
<title>acetate — {{ title|e }}</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text x='50' y='80' font-size='84' text-anchor='middle'>🎧</text></svg>">
<script src="https://unpkg.com/htmx.org@1.9.12/dist/htmx.min.js" integrity="sha384-ujb1lZYygJmzgSwoxRggbCHcjc0rB2XoQrxeTUQyRjrOnlCoYta87iKBWq3EsdM2" crossorigin="anonymous"></script>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
  :root{--bg:#0a0a0c;--panel:#121319;--panel2:#15161d;--line:#23252e;--fg:#f3f3ef;
    --muted:#888d99;--accent:#c8ff1f;--accent2:#ff5d8f;
    --display:"Bricolage Grotesque",sans-serif;--mono:"Space Mono",ui-monospace,monospace}
  *{box-sizing:border-box}html{scroll-behavior:smooth}
  body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--display);-webkit-font-smoothing:antialiased;overflow-x:hidden}
  body::before{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;opacity:.045;
    background-image:url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='120' height='120'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='3'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>")}
  body::after{content:"";position:fixed;top:-30vh;left:50%;transform:translateX(-50%);width:120vw;height:80vh;z-index:0;pointer-events:none;
    background:radial-gradient(50% 50% at 50% 50%,rgba(200,255,31,.10),transparent 70%)}
  @keyframes up{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
  .rise{animation:up .6s cubic-bezier(.2,.7,.3,1) both;animation-delay:calc(var(--i,0)*55ms)}

  nav{position:sticky;top:0;z-index:10;background:rgba(10,10,12,.72);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}
  nav .inner{max-width:1080px;margin:0 auto;padding:.85rem 1.5rem;display:flex;justify-content:space-between;align-items:center;gap:1rem}
  nav .brand{font-weight:800;font-size:1.05rem;letter-spacing:-.02em;color:var(--fg);text-decoration:none;white-space:nowrap;flex-shrink:0}.brand b{color:var(--accent)}
  nav .links{display:flex;gap:.2rem;font-family:var(--mono);font-size:.72rem;text-transform:uppercase;letter-spacing:.12em}
  nav .links a{color:var(--muted);text-decoration:none;padding:.4rem .7rem;border-radius:7px;transition:.15s}
  nav .links a:hover{color:var(--fg)}nav .links a.on{color:var(--accent)}
  nav .navgroup{position:relative;flex-shrink:0}
  nav .navgroup>summary{list-style:none;cursor:pointer;color:var(--muted);padding:.4rem .7rem;border-radius:7px;white-space:nowrap;transition:.15s}
  nav .navgroup>summary::-webkit-details-marker{display:none}
  nav .navgroup>summary::after{content:"▾";margin-left:.3rem;opacity:.55}
  nav .navgroup>summary:hover{color:var(--fg)}
  nav .navgroup.on>summary{color:var(--accent)}
  nav .navgroup[open]>summary{color:var(--fg);background:var(--panel2)}
  nav .navmenu{position:absolute;right:0;top:calc(100% + .5rem);min-width:11.5rem;display:flex;flex-direction:column;gap:.1rem;
    background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.4rem;
    box-shadow:0 18px 44px rgba(0,0,0,.55);z-index:20}
  nav .navmenu a{padding:.45rem .6rem;letter-spacing:.06em;text-transform:none;font-size:.74rem}
  nav .navmenu a:hover{background:var(--panel2);color:var(--fg)}
  .identity{font-family:var(--mono);font-size:.6rem;color:var(--muted);white-space:nowrap}
  .identity b{color:var(--fg)}.role{color:var(--accent);text-transform:uppercase}
  /* mobile: keep the whole bar on ONE line — brand left, links shrink to fit and
     scroll horizontally if they still can't (instead of wrapping or spilling).
     Home is dropped since the logo links home. */
  @media(max-width:640px){
    nav .inner{gap:.5rem;padding:.7rem 1rem}
    nav .links{min-width:0;overflow-x:auto;flex-wrap:nowrap;gap:.05rem;font-size:.62rem;letter-spacing:.04em;scrollbar-width:none}
    nav .links::-webkit-scrollbar{display:none}
    nav .links a{padding:.3rem .42rem;white-space:nowrap;flex-shrink:0}
    nav .links .home-link{display:none}
    nav .navgroup>summary{padding:.3rem .42rem}
    /* the scrolling links row is a clipping context, so an absolute menu would be
       cut off — pin it to the viewport under the bar instead */
    nav .navmenu{position:fixed;top:3rem;right:.75rem;left:auto;min-width:12rem;font-size:.7rem}
  }
  .wrap{position:relative;z-index:1;max-width:1080px;margin:0 auto;padding:2rem 1.5rem 6rem}

  h1{font-size:clamp(1.6rem,4vw,2.4rem);font-weight:800;letter-spacing:-.03em;margin:.4rem 0 1.8rem}
  h2{font-family:var(--mono);font-weight:700;font-size:.78rem;text-transform:uppercase;letter-spacing:.22em;color:var(--muted);margin:0 0 1.1rem;display:flex;align-items:center;gap:.7rem}
  h2::after{content:"";flex:1;height:1px;background:var(--line)}
  section{margin-bottom:3.2rem}
  .cols{display:grid;grid-template-columns:1fr 1fr;gap:2.4rem}@media(max-width:760px){.cols{grid-template-columns:1fr}}

  .statusbar{margin-bottom:2.4rem;padding-bottom:1rem;border-bottom:1px solid var(--line);min-height:1.4rem}
  .sb{font-family:var(--mono);font-size:.72rem;color:var(--muted);display:flex;gap:1.4rem;flex-wrap:wrap}
  .sb b{color:var(--fg)}.sb .acc{color:var(--accent)}

  .hero{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:3.4rem}
  .stat .n{font-size:clamp(2rem,5.5vw,3.6rem);font-weight:800;line-height:.9;letter-spacing:-.04em}
  .stat .n.accent{color:var(--accent)}
  .stat .l{font-family:var(--mono);font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.14em;margin-top:.55rem}
  @media(max-width:640px){.hero{grid-template-columns:repeat(2,1fr);gap:1.6rem}}

  .years{display:flex;align-items:flex-end;gap:.5rem;height:200px;padding-top:1rem}
  .yr{flex:1;display:flex;flex-direction:column;align-items:center;gap:.5rem;height:100%;justify-content:flex-end}
  .yr .bar{width:100%;background:linear-gradient(180deg,#2a2c36,#202229);border-radius:4px 4px 0 0;min-height:3px}
  .yr.peak .bar,.yr:hover .bar{background:linear-gradient(180deg,var(--accent),#7da00f)}
  .yr .v{font-family:var(--mono);font-size:.6rem;color:var(--muted)}.yr:hover .v{color:var(--accent)}
  .yr .y{font-family:var(--mono);font-size:.58rem;color:var(--muted);opacity:.7}
  .years.disc .yr .bar{background:linear-gradient(180deg,#3a2540,#241a2c)}
  .years.disc .yr.peak .bar,.years.disc .yr:hover .bar{background:linear-gradient(180deg,var(--accent2),#9c2f57)}

  ol{list-style:none;margin:0;padding:0}
  .row{position:relative;display:grid;grid-template-columns:1.6rem 42px 1fr auto;align-items:center;gap:.85rem;padding:.5rem .6rem;border-radius:8px;overflow:hidden}
  .row::before{content:"";position:absolute;inset:0;z-index:0;background:rgba(200,255,31,.07);width:var(--w,0);transition:width .8s ease}
  .row>*{position:relative;z-index:1}
  .row .rank{font-family:var(--mono);font-size:.8rem;color:var(--muted)}
  .row:hover{background:var(--panel)}
  .art{width:42px;height:42px;border-radius:6px;object-fit:cover;background:#1c1e26;display:flex;align-items:center;justify-content:center;color:#3a3d49;font-size:1rem}
  .ti{min-width:0}.ti .t{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .ti .a{font-size:.8rem;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .plays{font-family:var(--mono);font-size:.8rem;color:var(--accent);text-align:right}.plays span{color:var(--muted);font-size:.66rem}
  .artistrow{grid-template-columns:1.6rem 42px 1fr auto}
  .toy{grid-template-columns:3.4rem 42px 1fr auto}.toy .rank{color:var(--accent);font-weight:700;font-size:.9rem}

  .clock{display:flex;align-items:flex-end;gap:3px;height:130px}
  .clock .h{flex:1;display:flex;flex-direction:column;justify-content:flex-end;height:100%}
  .clock .h .b{background:#2a2c36;border-radius:2px 2px 0 0;min-height:2px}
  .clock .h.peak .b,.clock .h:hover .b{background:var(--accent)}
  .clockx{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.58rem;color:var(--muted);margin-top:.4rem}
  .hbars{display:grid;gap:.55rem}
  .hb{display:grid;grid-template-columns:2.6rem 1fr auto;align-items:center;gap:.7rem}
  .hb .hl{font-family:var(--mono);font-size:.74rem;color:var(--muted)}
  .hb .ht{height:10px;background:var(--line);border-radius:5px;overflow:hidden}
  .hb .ht>i{display:block;height:100%;width:var(--w);background:linear-gradient(90deg,var(--accent),var(--accent2))}
  .hb .hv{font-family:var(--mono);font-size:.68rem;color:var(--muted)}

  .gen{display:grid;gap:.55rem}.g .gl{display:flex;justify-content:space-between;font-size:.85rem}
  .g .gl b{font-weight:600}.g .gl span{font-family:var(--mono);font-size:.72rem;color:var(--muted)}
  .g .gb{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin-top:.25rem}
  .g .gb>i{display:block;height:100%;width:var(--w);background:linear-gradient(90deg,var(--accent),var(--accent2))}

  .pl{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1rem}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:1.2rem 1.3rem;text-decoration:none;color:var(--fg);transition:.2s;display:block}
  .card:hover{border-color:var(--accent);transform:translateY(-2px)}
  .card .pt{font-weight:700;font-size:1.1rem}.card .pm{font-family:var(--mono);font-size:.72rem;color:var(--muted);margin-top:.4rem;line-height:1.6}
  .lede{color:var(--muted);max-width:60ch;line-height:1.7;margin:-.6rem 0 2rem}

  .profile-list{display:flex;flex-direction:column;gap:.55rem}
  .profile-row{display:grid;grid-template-columns:minmax(190px,1.5fr) minmax(220px,1fr) auto;gap:1rem;align-items:center;
    background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:.8rem .9rem}
  .profile-row .pn{font-weight:700}.profile-row .pn a{color:inherit;text-decoration:none}.profile-row .pn a:hover{color:var(--accent)}.profile-row .pe{font-family:var(--mono);font-size:.66rem;color:var(--muted);margin-top:.2rem;line-height:1.5}
  .profile-signal{font-family:var(--mono);font-size:.66rem;color:var(--muted);line-height:1.5}
  .profile-signal b{color:var(--fg)}
  .profile-actions{display:flex;align-items:center;justify-content:flex-end;gap:.4rem;flex-wrap:wrap}
  .profile-actions form{display:flex;gap:.35rem;align-items:center}
  .profile-actions select{font-family:var(--mono);font-size:.64rem;background:var(--panel2);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:.35rem}
  .profile-actions a{font-family:var(--mono);font-size:.64rem;color:var(--accent);text-decoration:none;white-space:nowrap}
  .tag{display:inline-block;font-family:var(--mono);font-size:.58rem;text-transform:uppercase;letter-spacing:.08em;border:1px solid var(--line);border-radius:999px;padding:.18rem .42rem;color:var(--muted)}
  .tag.manual{color:var(--accent);border-color:rgba(200,255,31,.35)}
  .tag.suggested{color:#ffce6b;border-color:rgba(255,206,107,.35)}
  .profile-tabs{display:flex;gap:.35rem;flex-wrap:wrap;margin:0 0 1rem}
  .profile-tabs a{font-family:var(--mono);font-size:.64rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);text-decoration:none;border:1px solid var(--line);border-radius:6px;padding:.35rem .55rem}
  .profile-tabs a.on,.profile-tabs a:hover{color:var(--accent);border-color:var(--accent)}
  @media(max-width:760px){.profile-row{grid-template-columns:1fr}.profile-actions{justify-content:flex-start}}

  table.sched{width:100%;border-collapse:collapse}
  table.sched td,table.sched th{text-align:left;padding:.6rem .7rem;border-bottom:1px solid var(--line);font-size:.85rem;vertical-align:top}
  table.sched th{font-family:var(--mono);font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}
  table.sched .when{font-family:var(--mono);color:var(--accent);white-space:nowrap}
  .btns{display:flex;gap:.5rem;flex-wrap:wrap;margin:0 0 1rem}
  button{font-family:var(--mono);font-size:.74rem;text-transform:uppercase;letter-spacing:.08em;background:transparent;color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:.6rem .9rem;cursor:pointer;transition:.15s}
  button:hover{border-color:var(--accent);color:var(--accent)}
  #toast{font-family:var(--mono);font-size:.72rem;color:var(--accent);margin-left:.4rem}
  @keyframes toastfade{0%,55%{opacity:1}100%{opacity:0}}
  .toast{animation:toastfade 4.5s ease forwards}
  h2 button.refresh{font-size:.6rem;padding:.25rem .5rem;margin-left:.7rem;vertical-align:middle;color:var(--muted)}
  h2 button.refresh:hover{color:var(--fg)}
  h2 button.refresh:disabled{opacity:.45;cursor:default}
  .rcell{font-size:.78rem;max-width:440px}
  .rcell a{color:var(--accent);text-decoration:none;border-bottom:1px solid transparent;transition:.15s}
  .rcell a:hover{border-bottom-color:var(--accent)}
  .rerr{display:block;font-family:var(--mono);font-size:.66rem;color:var(--accent2);white-space:pre-wrap;word-break:break-word;max-height:8em;overflow:auto}
  .rsum{font-family:var(--mono);font-size:.7rem;color:var(--muted)}
  .auto{font-family:var(--mono);font-size:.58rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;opacity:.65}
  .smartbtns{display:flex;gap:.45rem;align-items:center;margin-top:.9rem;flex-wrap:wrap}
  .smartbtns a,.smartbtns button{font-family:var(--mono);font-size:.64rem;text-transform:uppercase;letter-spacing:.06em;
    color:var(--muted);background:transparent;border:1px solid var(--line);border-radius:6px;padding:.3rem .55rem;cursor:pointer;text-decoration:none;transition:.15s}
  .smartbtns a:hover,.smartbtns button:hover{color:var(--accent);border-color:var(--accent)}
  .smartbtns button.danger:hover{color:var(--accent2);border-color:var(--accent2)}
  .smartform{display:flex;flex-direction:column;gap:.85rem;max-width:660px}
  .smartform input,.smartform textarea{font-family:var(--mono);font-size:.8rem;background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:.7rem}
  .smartform textarea{resize:vertical;line-height:1.5}
  .smartform input:focus,.smartform textarea:focus{outline:none;border-color:var(--accent)}
  .presets{display:flex;gap:.4rem;flex-wrap:wrap;align-items:center;font-family:var(--mono);font-size:.64rem;color:var(--muted)}
  .presets button{background:transparent;color:var(--muted);border:1px solid var(--line);border-radius:6px;padding:.25rem .5rem;cursor:pointer}
  .presets button:hover{color:var(--accent);border-color:var(--accent)}
  .smartform button.primary{align-self:flex-start;background:var(--accent);color:var(--bg);border:none;border-radius:8px;font-family:var(--mono);font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;font-weight:700;padding:.6rem 1.1rem;cursor:pointer}
  .specdoc{font-family:var(--mono);font-size:.66rem;color:var(--muted);white-space:pre-wrap;line-height:1.55;background:var(--panel);padding:1rem;border-radius:8px;border:1px solid var(--line);margin-top:.8rem;overflow-x:auto}
  .spreview{list-style:none;margin:.7rem 0 .35rem;padding:0;display:flex;flex-direction:column;gap:.18rem}
  .spreview li{display:flex;justify-content:space-between;gap:.7rem;font-size:.8rem;white-space:nowrap;overflow:hidden}
  .spreview .t{overflow:hidden;text-overflow:ellipsis}.spreview .a{color:var(--muted);flex-shrink:0}
  .scount{font-family:var(--mono);font-size:.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
  .chat{display:flex;flex-direction:column;gap:1.4rem;margin-bottom:1.8rem}
  .turn{display:flex;flex-direction:column;gap:.7rem}
  .turn .q{align-self:flex-end;max-width:80%;background:var(--accent);color:var(--bg);font-weight:600;padding:.6rem .9rem;border-radius:12px 12px 2px 12px}
  .turn .a{max-width:88%;background:var(--panel);border:1px solid var(--line);padding:.8rem 1rem;border-radius:12px 12px 12px 2px;line-height:1.55}
  .err{color:var(--accent2);font-family:var(--mono);font-size:.78rem}
  .dots{display:inline-flex;gap:5px;padding:.25rem 0}
  .dots span{width:7px;height:7px;border-radius:50%;background:var(--muted);animation:blink 1.4s infinite both}
  .dots span:nth-child(2){animation-delay:.2s}.dots span:nth-child(3){animation-delay:.4s}
  @keyframes blink{0%,80%,100%{opacity:.25}40%{opacity:1}}
  .trace{display:flex;flex-direction:column;gap:.3rem}.trace:not(:empty){margin-bottom:.6rem}
  .tracestep{font-family:var(--mono);font-size:.68rem;color:var(--muted)}.tracestep summary{cursor:pointer}
  .tracestep .sql{background:var(--panel2);border:1px solid var(--line);border-radius:6px;padding:.5rem;margin:.3rem 0;white-space:pre-wrap;font-size:.66rem;overflow-x:auto;color:var(--muted)}
  .body>*:first-child{margin-top:0}.body>*:last-child{margin-bottom:0}
  .body p{margin:.5rem 0}.body ul,.body ol{margin:.4rem 0;padding-left:1.3rem}
  .body h1,.body h2,.body h3{font-size:1.05rem;margin:.7rem 0 .35rem}
  .body code{font-family:var(--mono);background:var(--panel2);padding:.08rem .3rem;border-radius:4px;font-size:.86em}
  .body pre{background:var(--panel2);padding:.7rem;border-radius:6px;overflow-x:auto}.body pre code{background:none;padding:0}
  .body table{border-collapse:collapse;margin:.6rem 0;font-size:.85rem;display:block;overflow-x:auto}
  .body th,.body td{border:1px solid var(--line);padding:.3rem .6rem;text-align:left;white-space:nowrap}.body th{color:var(--accent)}
  .body a{color:var(--accent);text-decoration:none;border-bottom:1px solid rgba(200,255,31,.3)}.body a:hover{border-bottom-color:var(--accent)}
  .chathead{display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap;margin-bottom:.4rem}
  .chathead h1{font-size:clamp(1.4rem,3.5vw,2rem)}
  .threadbtns{display:flex;gap:.5rem;flex-shrink:0}
  .threadbtns button{font-family:var(--mono);font-size:.66rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);background:transparent;border:1px solid var(--line);border-radius:7px;padding:.4rem .7rem;cursor:pointer;transition:.15s}
  .threadbtns button:hover{color:var(--accent);border-color:var(--accent)}
  .threadbtns button.primary{color:var(--bg);background:var(--accent);border-color:var(--accent);font-weight:700}
  .modal[hidden]{display:none}
  .modal{position:fixed;inset:0;z-index:50;display:flex;align-items:center;justify-content:center}
  .modalbg{position:absolute;inset:0;background:rgba(0,0,0,.6);backdrop-filter:blur(3px)}
  .modalcard{position:relative;z-index:1;width:min(440px,92vw);max-height:70vh;overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:1.1rem}
  .modalhead{display:flex;justify-content:space-between;align-items:center;margin-bottom:.8rem;font-family:var(--mono);text-transform:uppercase;letter-spacing:.1em;font-size:.8rem}
  .modalhead button{background:none;border:none;color:var(--muted);font-size:1rem;cursor:pointer}
  .threadlist{display:flex;flex-direction:column;gap:.3rem}
  .threadlist a{display:flex;justify-content:space-between;gap:.8rem;align-items:center;padding:.55rem .7rem;border-radius:8px;text-decoration:none;color:var(--fg);border:1px solid transparent}
  .threadlist a:hover{background:var(--panel2)}.threadlist a.on{border-color:var(--accent)}
  .threadlist .tt{font-size:.85rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .threadlist .td{font-family:var(--mono);font-size:.6rem;color:var(--muted);flex-shrink:0}
  .threadlist .arch{font-family:var(--mono);font-size:.55rem;color:var(--muted);border:1px solid var(--line);border-radius:4px;padding:0 .25rem;text-transform:uppercase}
  .steps{font-family:var(--mono);font-size:.7rem;color:var(--muted)}
  .steps summary{cursor:pointer}
  .steps .sql{background:var(--panel2);border:1px solid var(--line);border-radius:6px;padding:.6rem;margin:.4rem 0;white-space:pre-wrap;color:var(--muted);font-size:.68rem;overflow-x:auto}
  .proposal{border:1px solid var(--accent);border-radius:10px;padding:.9rem 1rem;margin:.9rem 0;background:rgba(200,255,31,.05);max-width:88%}
  .proposal .pt{font-weight:700;overflow-wrap:anywhere}.proposal .pm{font-family:var(--mono);font-size:.72rem;color:var(--muted);margin:.3rem 0 .7rem}
  .proposal-actions>a:not(.approve){color:var(--accent);font-family:var(--mono);font-size:.72rem;text-decoration:none}
  .proposal-actions{display:flex;align-items:center;gap:.8rem;flex-wrap:wrap;min-height:2rem}
  .proposal .approve{display:inline-flex;background:var(--accent);color:var(--bg);border:none;border-radius:7px;font-family:var(--mono);font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;font-weight:700;padding:.45rem .9rem;cursor:pointer;text-decoration:none;transition:background .15s,color .15s}
  .proposal .approve:disabled{opacity:.65;cursor:wait}.proposal .approve:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
  .proposal .open{background:transparent;color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent)}
  .proposal .open:hover{background:rgba(200,255,31,.1)}
  .proposal-status{font-family:var(--mono);font-size:.7rem;color:var(--accent)}
  .proposal-status:empty{display:none}.proposal-status.error{color:var(--accent2)}
  .playlist-type{font-family:var(--mono);font-size:.62rem;color:var(--accent);margin:.4rem 0;text-transform:uppercase;letter-spacing:.06em}
  .drafts .proposal{max-width:none;margin:0;height:100%}.proposal-source{font-family:var(--mono);font-size:.65rem;color:var(--muted);margin-top:.9rem}
  .proposal-source a,.activity a{color:var(--accent);text-decoration:none}.proposal-source time{display:block;margin-top:.3rem}
  .activity .card{margin-bottom:1rem}.activity .when{font-family:var(--mono);font-size:.75rem;margin:.7rem 0;color:var(--fg)}
  .history-entry{border:1px solid var(--line);border-radius:10px;background:var(--panel);margin:.8rem 0;padding:1rem}
  .history-entry summary{cursor:pointer;display:flex;gap:.5rem 1rem;align-items:center;flex-wrap:wrap}
  .history-entry summary::before{content:'▸';color:var(--accent)}.history-entry[open] summary::before{content:'▾'}
  .history-entry .event-name{font-weight:700;overflow-wrap:anywhere}.history-entry .event-time{margin-left:auto;font-family:var(--mono);font-size:.65rem;color:var(--muted)}
  .history-entry .change{font-family:var(--mono);font-size:.68rem;color:var(--muted)}.history-entry .added{color:var(--accent)}.history-entry .removed{color:var(--accent2)}
  .history-entry .event-note{font-family:var(--mono);font-size:.68rem;color:var(--muted);line-height:1.6}
  .change-lists{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1rem}.change-lists ul{padding-left:1.1rem;font-size:.8rem;line-height:1.7;overflow-wrap:anywhere}
  .change-lists span{color:var(--muted)}.history-pages{display:flex;gap:1rem;margin-top:1rem;font-family:var(--mono);font-size:.75rem}
  .playlist-search{display:flex;gap:.6rem;flex-wrap:wrap;margin-bottom:1.5rem}.playlist-search input{flex:1;min-width:10rem;background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:.6rem;font-family:var(--mono)}
  .playlist-search input:focus{outline:2px solid var(--accent);outline-offset:2px}
  .playlist-detail>a{font-family:var(--mono);font-size:.75rem;color:var(--muted);text-decoration:none}.playlist-detail>a:hover{color:var(--accent)}
  .playlist-detail h1{overflow-wrap:anywhere;margin-bottom:.6rem}.playlist-detail .description{white-space:pre-wrap;overflow-wrap:anywhere;color:var(--muted)}
  .playlist-tracks{margin-top:1.2rem}.playlist-track{display:grid;grid-template-columns:2.5rem minmax(0,1fr);gap:.8rem;padding:.75rem .5rem;border-bottom:1px solid var(--line)}
  .playlist-track .rank{font-family:var(--mono);font-size:.75rem;color:var(--muted);padding-top:.2rem}.playlist-track .track-name{font-weight:600;overflow-wrap:anywhere}
  .playlist-track .track-artist{color:var(--muted);font-size:.8rem;overflow-wrap:anywhere}.playlist-detail .smartbtns{margin:1.2rem 0;flex-wrap:wrap}
  .playlist-track a{color:var(--fg);text-decoration:none}.playlist-track a:hover{color:var(--accent)}
  .playlist-detail .lede a{color:var(--accent)}
  .playlist-track .smartbtns{grid-column:2;margin:.5rem 0 0;gap:.4rem}.playlist-track .smartbtns a{color:var(--muted);font-size:.6rem}
  .playlist-track .smartbtns a:hover{color:var(--accent)}.shortcut-note{color:var(--muted);font-family:var(--mono);font-size:.65rem;margin:0 0 1.5rem}
  .library-memory{border:1px solid var(--line);border-radius:10px;padding:.8rem 1rem;margin-bottom:1.5rem;background:var(--panel)}
  .library-memory summary{cursor:pointer;font-family:var(--mono);font-size:.72rem;color:var(--muted)}.library-memory summary:hover{color:var(--accent)}
  .library-memory p,.memory-status{font-size:.75rem;color:var(--muted);line-height:1.6}.library-memory textarea{box-sizing:border-box;width:100%;resize:vertical;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:.65rem;font: .75rem/1.6 var(--mono)}
  .library-memory pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.75rem}.library-memory .smartbtns{margin:.6rem 0 0}.memory-status.error{color:var(--accent2)}
  @media(max-width:600px){.change-lists{grid-template-columns:1fr}.history-entry .event-time{margin-left:0;width:100%}}
  /* thinking stream — transient by design: shown while it happens, never stored */
  .think{border-left:2px solid var(--line);padding-left:.7rem}
  .think summary{font-family:var(--mono);font-size:.66rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);cursor:pointer;list-style:none}
  .think summary::-webkit-details-marker{display:none}
  .think summary:hover{color:var(--fg)}
  .think[data-live] summary{color:var(--accent2);animation:thinkpulse 1.6s ease-in-out infinite}
  @keyframes thinkpulse{0%,100%{opacity:.55}50%{opacity:1}}
  .thoughts{font-family:var(--mono);font-size:.7rem;line-height:1.55;color:var(--muted);white-space:pre-wrap;word-break:break-word;
    max-height:11rem;overflow-y:auto;margin:.45rem 0 0;scrollbar-width:thin;scrollbar-color:var(--line) transparent;
    -webkit-mask-image:linear-gradient(180deg,transparent,#000 1.2rem);mask-image:linear-gradient(180deg,transparent,#000 1.2rem)}
  .thoughts::-webkit-scrollbar{width:6px}.thoughts::-webkit-scrollbar-thumb{background:var(--line);border-radius:3px}
  body.nothink .think{display:none}
  .thinktoggle{display:inline-flex;align-items:center;gap:.35rem;font-family:var(--mono);font-size:.66rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);cursor:pointer;user-select:none}
  .thinktoggle input{accent-color:var(--accent);cursor:pointer}
  .askform{display:flex;gap:.6rem;align-items:flex-end;flex-wrap:wrap;position:sticky;bottom:1rem;background:rgba(10,10,12,.85);backdrop-filter:blur(8px);padding:.6rem;border:1px solid var(--line);border-radius:12px}
  .via{font-family:var(--mono);font-size:.6rem;color:var(--muted);opacity:.6;text-transform:uppercase;letter-spacing:.1em}
  .chaterr{color:var(--accent2);font-family:var(--mono);font-size:.74rem;margin:.2rem 0 .5rem;animation:up .3s both}
  .askform textarea{flex:1;min-width:0;font-family:var(--display);font-size:.95rem;background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:9px;padding:.7rem .9rem;min-height:2.8rem;resize:none;line-height:1.4;max-height:160px;overflow-y:hidden;scrollbar-width:thin;scrollbar-color:var(--line) transparent}
  .askform .psel{height:2.8rem}
  .askform textarea:focus{outline:none;border-color:var(--accent)}
  .askform textarea::-webkit-scrollbar{width:8px}.askform textarea::-webkit-scrollbar-thumb{background:var(--line);border-radius:4px}.askform textarea::-webkit-scrollbar-track{background:transparent}
  .askform button{background:var(--accent);color:var(--bg);border:none;border-radius:9px;font-family:var(--mono);font-size:.74rem;text-transform:uppercase;letter-spacing:.06em;font-weight:700;padding:0 1.1rem;height:2.8rem;cursor:pointer}
  @media(max-width:600px){ #askform{display:grid;grid-template-columns:minmax(0,1fr) auto}#askform .psel{grid-column:1/-1;width:100%}}
  .htmx-indicator{display:none;font-family:var(--mono);font-size:.72rem;color:var(--muted)}.htmx-request .htmx-indicator,.htmx-indicator.htmx-request{display:inline}
  .results{font-family:var(--mono);font-size:.68rem;color:var(--muted);margin-top:1rem;line-height:1.7}
  .periodbar{display:flex;align-items:center;gap:.7rem;flex-wrap:wrap;margin-bottom:2.2rem}
  .tabs{display:flex;gap:.25rem;font-family:var(--mono);font-size:.7rem;text-transform:uppercase;letter-spacing:.1em}
  .tabs a{color:var(--muted);text-decoration:none;padding:.4rem .7rem;border:1px solid var(--line);border-radius:7px;transition:.15s}
  .tabs a:hover{color:var(--fg)}.tabs a.on{color:var(--bg);background:var(--accent);border-color:var(--accent)}
  .psel{appearance:none;-webkit-appearance:none;font-family:var(--mono);font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);background-color:transparent;border:1px solid var(--line);border-radius:7px;padding:.42rem 1.75rem .42rem .7rem;cursor:pointer;transition:.15s;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 10 10'%3E%3Cpath d='M2 3.5 5 6.5 8 3.5' stroke='%238a8a93' stroke-width='1.4' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right .62rem center}
  .psel:hover,.psel:focus{color:var(--accent);border-color:var(--accent);outline:none}
  .psel option{background:var(--panel);color:var(--fg);text-transform:none;letter-spacing:0}
  .nav-ar{font-family:var(--mono);font-size:.7rem;color:var(--muted);text-decoration:none;padding:.42rem .6rem;border:1px solid var(--line);border-radius:7px}
  .nav-ar:hover{color:var(--accent);border-color:var(--accent)}.nav-ar.off{opacity:.3;pointer-events:none}
  .delta{font-family:var(--mono);font-size:.78rem;padding:.25rem .6rem;border-radius:6px}
  .delta.up{color:var(--accent);background:rgba(200,255,31,.1)}.delta.down{color:var(--accent2);background:rgba(255,93,143,.1)}
  .spot{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:2.6rem}@media(max-width:640px){.spot{grid-template-columns:1fr}}
  .spot .hl{display:flex;gap:1rem;align-items:center;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:1.1rem}
  .spot .big{width:64px;height:64px;border-radius:8px;object-fit:cover;background:#1c1e26;flex:none;display:flex;align-items:center;justify-content:center;color:#3a3d49;font-size:1.4rem}
  .spot .meta{min-width:0}.spot .meta .k{font-family:var(--mono);font-size:.62rem;text-transform:uppercase;letter-spacing:.14em;color:var(--muted)}
  .spot .meta .t{font-weight:700;font-size:1.05rem;margin:.15rem 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .spot .meta .a{font-size:.82rem;color:var(--muted)}
  .callout{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:1.1rem 1.3rem;font-size:.92rem}.callout b{color:var(--accent)}
</style></head><body>
<nav><div class=inner>
  <a class=brand href="/"><b>●</b> acetate</a>
  <div class=links>
    {% for it in nav.primary %}<a href="{{ it.href }}" class="{{ 'home-link ' if it.key=='home' }}{{ 'on' if it.on }}">{{ it.label }}</a>
    {% endfor %}{% for g in nav.groups %}<details class="navgroup{{ ' on' if g.on }}">
      <summary>{{ g.label }}</summary>
      <div class=navmenu>{% for it in g["items"] %}<a href="{{ it.href }}" class="{{ 'on' if it.on }}"{% if it.external %} target=_blank rel=noopener{% endif %}>{{ it.label }}</a>{% endfor %}</div>
    </details>
    {% endfor %}
  </div>
  {% if identity %}<div class=identity><b>{{ identity.name }}</b> · <span class=role>{{ identity.role.name|lower }}</span></div>{% endif %}
</div></nav>
<div class=wrap>{% block content %}{% endblock %}</div>
<script>
// Nav menus work without JS (they're <details>); this just makes them behave like menus:
// one open at a time, and click-away or Escape closes them.
(function(){ const gs=[...document.querySelectorAll('nav details.navgroup')]; if(!gs.length) return;
  gs.forEach(d=>d.addEventListener('toggle',()=>{ if(d.open) gs.forEach(o=>{ if(o!==d) o.open=false; }); }));
  document.addEventListener('click',e=>{ gs.forEach(d=>{ if(d.open && !d.contains(e.target)) d.open=false; }); });
  document.addEventListener('keydown',e=>{ if(e.key==='Escape') gs.forEach(d=>{ d.open=false; }); });
})();
</script>
</body></html>"""

STATUSBAR = r"""<div class=sb>
  <span>last poll <b>{{ s.last_poll_at|localtime if s.last_poll_at else "—" }}</b></span>
  <span>plays <b>{{ "{:,}".format(s.play_events) }}</b></span>
  <span>liked <b>{{ "{:,}".format(s.liked_tracks or 0) }}</b></span>
  <span>last backup <b>{{ s.last_backup_at|localtime('%Y-%m-%d') if s.last_backup_at else "—" }}</b></span>
  <span>genres <b class=acc>{{ s.genre_pct }}%</b></span>
  {% if running %}<span style="color:#ffce6b">running: {{ running|join(", ") }}</span>{% endif %}
</div>"""

HOME = r"""{% extends "base" %}{% from "macros" import thumb, leaderboard, artist_thumb %}{% block content %}
<div class=statusbar hx-get=/status hx-trigger="load, every 8s"></div>
<h1 class=rise>{% if years|length > 1 %}{{ years|length }} years of listening, in numbers.{% else %}Your listening, in numbers.{% endif %}</h1>
<div class=hero>
  {% set hs=[("{:,}".format(t.plays),"real plays",true),("{:,}".format(t.hours),"hours",false),
             ("{:,}".format(t.liked),"liked songs",false),("{:,}".format(t.artists),"artists",false)] %}
  {% for n,l,acc in hs %}<div class="stat rise" style="--i:{{ loop.index }}"><div class="n{{ ' accent' if acc }}">{{ n }}</div><div class=l>{{ l }}</div></div>{% endfor %}
</div>
<section class=rise><h2>listening by year</h2>
  {% set ymax=(years|map(attribute='plays')|list|max) %}
  <div class=years>{% for y in years %}<div class="yr {{ 'peak' if y.plays==ymax }}" title="{{ y.year }}: {{ '{:,}'.format(y.plays) }} plays, {{ y.hours }}h">
    <div class=v>{{ y.hours }}h</div><div class=bar style="height:{{ (y.plays*100//ymax) }}%"></div><div class=y>'{{ y.year[2:] }}</div></div>{% endfor %}</div>
</section>
<div class=cols>
  <section class=rise><h2>top tracks · all time</h2>{{ leaderboard(top_tracks) }}</section>
  <section class=rise><h2>top artists · all time</h2>
    {% set amx=(top_artists|map(attribute='plays')|list|max) %}
    <ol>{% for x in top_artists %}<li class="row artistrow rise" style="--w:{{ x.plays*100//amx }}%;--i:{{ loop.index }}">
      <span class=rank>{{ "%02d"|format(loop.index) }}</span>{{ artist_thumb(x) }}
      <div class=ti><div class=t>{{ x.artist }}</div><div class=a>{{ x.tracks }} tracks</div></div>
      <div class=plays>{{ "{:,}".format(x.plays) }} <span>plays</span></div></li>{% endfor %}</ol>
  </section>
</div>
<section class=rise><h2>recently played
  {% if identity.can_admin %}<button class=refresh hx-post=/refresh-plays hx-target=#refreshed
    hx-indicator=#refreshing hx-disabled-elt=this>↻ refresh now</button>
  <span id=refreshing class=htmx-indicator>polling Spotify…</span>
  <span id=refreshed></span>{% endif %}</h2>
  <div hx-get=/recent hx-trigger="plays-refreshed from:body" hx-swap=innerHTML>
    {% include "_recent" %}</div>
</section>
{% endblock %}"""

# Its own fragment so a manual refresh can swap the list in place — the refresh
# endpoint fires `plays-refreshed` only when something actually landed.
RECENT_PLAYS = r"""{% from "macros" import thumb %}
<ol>{% for x in recent %}<li class="row rise" style="--i:{{ loop.index }}">
  <span class=rank>·</span>{{ thumb(x) }}
  <div class=ti><div class=t>{{ x.title }}</div><div class=a>{{ x.artist }}</div></div>
  <div class=plays><span>{{ x.played_at|localtime('%m-%d %H:%M') }}</span></div></li>{% endfor %}</ol>"""

STATS = r"""{% extends "base" %}{% from "macros" import thumb, leaderboard %}{% block content %}
<h1 class=rise>Stats</h1>
<section class=rise><h2>top tracks of the moment · last 90 days</h2>{{ leaderboard(moment) }}</section>
<section class=rise><h2>your #1 track, every year</h2>
  {% set mx=(toy|map(attribute='plays')|list|max) %}
  <ol>{% for x in toy %}<li class="row toy rise" style="--w:{{ x.plays*100//mx }}%;--i:{{ loop.index }}">
    <span class=rank>{{ x.year }}</span>{{ thumb(x) }}
    <div class=ti><div class=t>{{ x.title }}</div><div class=a>{{ x.artist }}</div></div>
    <div class=plays>{{ x.plays }} <span>plays</span></div></li>{% endfor %}</ol>
</section>
<div class=cols>
  <section class=rise><h2>when you listen · by hour (utc)</h2>
    {% set hmx=(hours|map(attribute='plays')|list|max) %}
    <div class=clock>{% for h in hours %}<div class="h {{ 'peak' if h.plays==hmx }}" title="{{ h.hour }}:00 — {{ '{:,}'.format(h.plays) }} plays">
      <div class=b style="height:{{ (h.plays*100//hmx) if hmx else 0 }}%"></div></div>{% endfor %}</div>
    <div class=clockx><span>00</span><span>06</span><span>12</span><span>18</span><span>23</span></div>
  </section>
  <section class=rise><h2>by day of week</h2>
    {% set dmx=(weekdays|map(attribute='plays')|list|max) %}
    <div class=hbars>{% for d in weekdays %}<div class=hb>
      <div class=hl>{{ d.day }}</div><div class=ht><i style="width:{{ d.plays*100//dmx }}%"></i></div>
      <div class=hv>{{ "{:,}".format(d.plays) }}</div></div>{% endfor %}</div>
  </section>
</div>
<section class=rise><h2>new tracks discovered · by year</h2>
  {% set nmx=(discovery|map(attribute='n')|list|max) %}
  <div class="years disc">{% for d in discovery %}<div class="yr {{ 'peak' if d.n==nmx }}" title="{{ d.year }}: {{ '{:,}'.format(d.n) }} new tracks">
    <div class=v>{{ d.n }}</div><div class=bar style="height:{{ d.n*100//nmx }}%"></div><div class=y>'{{ d.year[2:] }}</div></div>{% endfor %}</div>
</section>
<section class=rise><h2>top genres</h2>
  {% set gmx=(genres|map(attribute='plays')|list|max) if genres else 1 %}
  <div class=gen>{% for g in genres %}<div class="g rise" style="--i:{{ loop.index }}">
    <div class=gl><b>{{ g.genre }}</b><span>{{ "{:,}".format(g.plays) }} plays · {{ g.tracks }} tracks</span></div>
    <div class=gb><i style="width:{{ g.plays*100//gmx }}%"></i></div></div>{% endfor %}</div>
</section>
{% endblock %}"""

PLAYLIST_PROFILE_ROW = r"""<article class=profile-row id="profile-{{ profile.playlist_id }}">
  <div>
    <div class=pn><a href="/playlists/{{ profile.playlist_id|urlencode }}">{{ (profile.name or profile.playlist_id)|e }}</a></div>
    <div class=pe>{{ profile.n_tracks }} tracks · {{ profile.n_liked }} liked · {{ profile.add_days }} add days ·
      {{ profile.member_plays_180d }} member plays / 180d
      {% if profile.cohesion is not none %} · cohesion {{ '%.3f'|format(profile.cohesion) }}{% endif %}</div>
  </div>
  <div class=profile-signal>
    {% if profile.user_label %}<span class="tag manual">manual · {{ profile.user_label }}</span>
      <div>Your verdict overrides the automatic persona.</div>
    {% elif profile.suggestion.label %}<span class="tag suggested">suggested · {{ profile.suggestion.label }} · {{ profile.suggestion.confidence }}</span>
      <div>{{ profile.suggestion.reasons|join(' · ') }}</div>
    {% else %}<span class=tag>needs your call</span><div>{{ profile.suggestion.reasons|join(' · ') }}</div>{% endif %}
  </div>
  <div class=profile-actions>
    {% if profile.has_region %}<a href="/health/{{ profile.playlist_id }}">Find liked matches</a>{% else %}<span class=auto>too few embedded</span>{% endif %}
    {% if identity.can_admin %}
    <form hx-post="/playlists/{{ profile.playlist_id }}/label" hx-target="#profile-{{ profile.playlist_id }}" hx-swap=outerHTML>
      {% set chosen = profile.user_label or profile.suggestion.label or '' %}
      <select name=label aria-label="Category for {{ profile.name }}">
        <option value="" {{ 'selected' if not chosen }}>automatic</option>
        {% for value in ['fave','rotation','archive','one-shot','dormant','mute'] %}
          <option value="{{ value }}" {{ 'selected' if chosen==value }}>{{ value }}</option>
        {% endfor %}
      </select>
      <button type=submit>Save</button>
    </form>
    {% endif %}
  </div>
</article>"""

PLAYLISTS = r"""{% extends "base" %}{% block content %}
<h1 class=rise>Your playlists</h1>
<p class="lede rise">Open any library playlist or saved selection to see its tracks. Backups keep older playlists viewable too.
Use the category filters to decide which of your authored playlists express your taste.</p>
<div class="smartbtns rise"><a href=/curated?view=drafts>Drafts →</a><a href=/curated?view=scheduled>Scheduled updates →</a><a href=/curated?view=history>History →</a></div>
<div class="profile-tabs rise">
  {% for key, label in [('all','All playlists'),('unlabeled','Unlabeled'),('rotation','Rotation'),('dormant','Dormant'),('archive','Archive'),('one-shot','One-shot'),('mute','Mute')] %}
    <a href="/playlists?show={{ key }}" class="{{ 'on' if show==key }}">{{ label }}</a>
  {% endfor %}
</div>
{% if show=='all' %}
<form class=playlist-search method=get action=/playlists role=search>
  <input type=search name=q value="{{ catalog.query|e }}" placeholder="Find a playlist" aria-label="Find a playlist"><button type=submit>Search</button>
</form>
<p class=lede>{{ catalog.total }} playlist{{ 's' if catalog.total!=1 else '' }}{% if catalog.query %} matching “{{ catalog.query|e }}”{% endif %} · page {{ catalog.page }}</p>
<div class=pl>{% for it in catalog['items'] %}<a class=card href="{{ it.url }}">
  <div class=pt>{{ it.name|e }}</div><div class=playlist-type>{{ it.kind }}</div>
  <div class=pm>{% if it.count is not none %}{{ it.count }} tracks · {% endif %}View playlist →</div>
</a>{% else %}<p class=lede>No playlists in this view.</p>{% endfor %}</div>
<div class=history-pages>{% if catalog.page>1 %}<a href="/playlists?q={{ catalog.query|urlencode }}&amp;page={{ catalog.page-1 }}">← Previous</a>{% endif %}
  {% if catalog.has_more %}<a href="/playlists?q={{ catalog.query|urlencode }}&amp;page={{ catalog.page+1 }}">Next →</a>{% endif %}</div>
{% else %}
<div class="sb rise" style="margin-bottom:1.2rem">
  <span>owned <b>{{ totals.owned }}</b></span><span>manually labeled <b class=acc>{{ totals.labeled }}</b></span>
  <span>still automatic <b>{{ totals.unlabeled }}</b></span><span>showing <b>{{ profiles|length }}</b></span>
</div>
<div class="btns rise">
  {% if identity.can_admin %}
  <button hx-post=/playlists/profile hx-target=#playlist-toast>Refresh signals</button>
  <form method=post action=/playlists/label-suggested onsubmit="return confirm('Apply every high-confidence title/activity suggestion? Nothing is deleted, and every label can be changed here.')">
    <button type=submit>Apply obvious suggestions</button>
  </form>
  {% endif %}
  <a href=/health style="color:var(--accent);font-family:var(--mono);font-size:.72rem;align-self:center">Playlist health →</a>
  <span id=playlist-toast></span>
</div>
<section class=rise>
  <div class=profile-list>{% for profile in profiles %}{% include "playlist_profile_row" %}
    {% else %}<p class=lede>No playlists in this view.</p>{% endfor %}</div>
</section>

<section class=rise><h2>acetate-generated playlists</h2>
<p class=lede>These are managed safely by acetate and are never used as evidence about which of your authored playlists matter.</p>
<div class=pl>{% for p in items %}
  <a class="card" href="{{ p.url }}">
    <div class=pt>{{ p.name|e }}</div>
    <div class=pm>{% if p.total is not none %}{{ p.total }} tracks · {% endif %}{{ p.kind.replace('_',' ') }}<br>updated {{ p.updated_at|localtime('%Y-%m-%d') if p.updated_at }}</div>
  </a>
{% else %}<p class=lede>None yet — use <a href="/pipeline" style="color:var(--accent)">Pipeline</a> to build one.</p>{% endfor %}</div>
</section>
{% endif %}
{% endblock %}"""

PLAYLIST_DETAIL = r"""{% extends "base" %}{% block content %}
<article class="playlist-detail rise">
<a href=/playlists>← All playlists</a>
<h1>{{ playlist.name|e }}</h1>
<div class=playlist-type>{{ playlist.kind|e }}</div>
{% if playlist.summary %}<p class=lede>{{ playlist.summary|e }}</p>{% endif %}
{% if playlist.description %}<p class=description>{{ playlist.description|e }}</p>{% endif %}
<div class=smartbtns>
  {% if identity.can_query %}
  {% if playlist.more_url %}<a href="{{ playlist.more_url|e }}" target=_blank rel=noopener>More like this ↗</a>{% endif %}
  <a href="{{ playlist.ask_url|e }}" target=_blank rel=noopener>Ask about playlist ↗</a>
  {% endif %}
  {% if playlist.history_url %}<a href="{{ playlist.history_url }}">View history →</a>{% endif %}
  {% if playlist.manage_url %}<a href="{{ playlist.manage_url }}">Manage {{ 'selection' if playlist.curated else 'rule' }} →</a>{% endif %}
  {% if playlist.playlist_id %}<a href="https://open.spotify.com/playlist/{{ playlist.playlist_id|urlencode }}" target=_blank rel=noopener>Open in Spotify ↗</a>{% endif %}
  {% if playlist.thread_id %}<a href="/chat?thread={{ playlist.thread_id }}">Source conversation →</a>{% endif %}
</div>
{% if identity.can_query %}<p class=shortcut-note>Chat shortcuts open a draft question in a new tab.</p>{% endif %}
{% if playlist.save_url %}<div class=proposal><div class=proposal-actions>
  <span class=proposal-status role=status></span>{% if identity.can_admin %}<form method=post action="{{ playlist.save_url }}" data-approve><button type=submit class=approve>Approve &amp; save →</button></form>{% endif %}
</div></div>{% endif %}
<section><h2>tracks{% if playlist.total is not none %} · {{ playlist.total }}{% endif %}</h2>
<p class=lede>{{ playlist.source|e }}{% if playlist.observed_at %} · observed {{ playlist.observed_at|localtime }}{% endif %}.</p>
{% if not playlist.playlist_id and not playlist.save_url %}<p class=lede>History starts when a published playlist is captured in a Spotify backup.</p>{% endif %}
{% if not playlist.tracks_known %}<div class=callout>No track list has been recorded yet. A library backup will make it available here.</div>
{% elif not playlist.complete %}<div class=callout>{{ playlist.recorded_count }} of {{ playlist.total }} track details were recorded. Missing details are not shown as removals.</div>{% endif %}
<ol class=playlist-tracks>{% for track in playlist.tracks %}<li class=playlist-track>
  <span class=rank>{{ track.rank }}</span><div><div class=track-name>{{ track.title|e }}</div>
  <div class=track-artist>{{ track.artist|e }}{% if not track.track_id %} · unavailable or local{% endif %}</div></div>
  {% if track.track_id %}<div class="smartbtns track-actions">
    {% if identity.can_query %}<a href="{{ track.more_url|e }}" target=_blank rel=noopener aria-label="More like {{ track.title|e }}">More like this ↗</a>
    <a href="{{ track.ask_url|e }}" target=_blank rel=noopener aria-label="Ask about {{ track.title|e }}">About this track ↗</a>{% endif %}
    <a href="https://open.spotify.com/track/{{ track.track_id|urlencode }}" target=_blank rel=noopener aria-label="Open {{ track.title|e }} in Spotify">Spotify ↗</a>
  </div>{% endif %}
</li>{% else %}{% if playlist.tracks_known and playlist.complete %}<p class=lede>{{ 'No tracks on this page.' if playlist.page>1 else 'This playlist has no tracks right now.' }}</p>{% endif %}{% endfor %}</ol>
<div class=history-pages>{% if playlist.page>1 %}<a href="?page={{ playlist.page-1 }}">← Previous tracks</a>{% endif %}{% if playlist.has_more %}<a href="?page={{ playlist.page+1 }}">More tracks →</a>{% endif %}</div>
</section></article>
{% if playlist.save_url %}{% include "proposal_script" %}<script>bindPlaylistApprovals(document.querySelector('.playlist-detail'));</script>{% endif %}
{% endblock %}"""

PIPELINE = r"""{% extends "base" %}{% block content %}
<h1 class=rise>Pipeline</h1>
<div class="statusbar rise" hx-get=/status hx-trigger="load, every 5s"></div>
<section class=rise><h2>run a job</h2>
  {% if identity.can_admin %}
  <div class=btns>
    <button hx-post=/refresh-plays hx-target=#toast hx-indicator=#polling
      hx-disabled-elt=this>↻ Refresh plays now</button>
    <span id=polling class=htmx-indicator>polling Spotify…</span>
    <button hx-post=/run/poll hx-target=#toast>Poll plays (background)</button>
    <button hx-post=/run/backup hx-target=#toast>Backup library</button>
    <button hx-post=/run/monthly hx-target=#toast>Build monthly</button>
    <button hx-post=/run/backfill hx-target=#toast>Tag genres</button>
    <button hx-post=/run/hydrate hx-target=#toast>Hydrate artists</button>
    <button hx-post=/run/gems hx-target=#toast>Build Lost Gems</button>
    <button hx-post=/run/on_loop hx-target=#toast>Build On Loop</button>
    <span id=toast></span>
  </div>
  {% else %}<p class=lede>Job controls are available to administrators.</p>{% endif %}
</section>
<section class=rise><h2>schedule</h2>
  <table class=sched><tr><th>job</th><th>when</th><th>what it does</th></tr>
    <tr><td>poll</td><td class=when>every 15 min</td><td>Pull the last ≤50 recently-played tracks into history (dedup-safe). Sized so it never overflows Spotify's 50-track window.</td></tr>
    <tr><td>backup</td><td class=when>daily 04:00</td><td>Snapshot liked songs + owned playlists — store-on-change (content diff) + a deletion trace.</td></tr>
    <tr><td>monthly</td><td class=when>daily 05:00</td><td>Rebuild 📆 Liked · / 🔥 Top Played · for the current month + current year. Idempotent.</td></tr>
    <tr><td>gems</td><td class=when>weekly</td><td>Rotate a fresh batch of faded 💎 Lost Gems into the playlist.</td></tr>
    <tr><td>on_loop</td><td class=when>daily 05:30</td><td>Rebuild 🔁 On Loop — your current heavy rotation (last 21 days, recency-weighted).</td></tr>
    <tr><td>hydrate</td><td class=when>daily 03:00</td><td>Fetch artist images + genres (one at a time; the bulk endpoint is blocked).</td></tr>
    <tr><td>backfill</td><td class=when>hourly :30</td><td>Tag up to 2,000 untagged tracks' genres via Last.fm, then idle on only new tracks.</td></tr>
  </table>
</section>
<section class=rise><h2>genre tagging</h2>
  {% set gp = status.genre_pct %}
  <div class=gb style="height:10px;background:var(--line);border-radius:5px;overflow:hidden"><i style="display:block;height:100%;width:{{ gp }}%;background:linear-gradient(90deg,var(--accent),var(--accent2))"></i></div>
  <p class=results>{{ "{:,}".format(status.genre_tagged) }} / {{ "{:,}".format(status.genre_universe) }} tracks tagged ({{ gp }}%)</p>
</section>
<section class=rise><h2>recent runs <span class=auto>· live</span></h2>
  <div hx-get=/runs hx-trigger="load, every 5s" hx-swap=innerHTML></div>
</section>
{% endblock %}"""

WRAPPED = r"""{% extends "base" %}{% from "macros" import thumb, leaderboard, artist_thumb %}{% block content %}
<h1 class=rise>Wrapped · {{ label }}</h1>
<div class="periodbar rise">
  <div class=tabs>{% for k in ['year','quarter','month','week'] %}<a href="/wrapped?kind={{k}}" class="{{ 'on' if kind==k }}">{{k}}</a>{% endfor %}</div>
  <a class="nav-ar {{ 'off' if not prev_key }}" href="/wrapped?kind={{kind}}&key={{prev_key}}">‹ older</a>
  <select class=psel onchange="location.href='/wrapped?kind={{kind}}&key='+encodeURIComponent(this.value)">
    {% for p in periods %}<option value="{{p}}" {{ 'selected' if p==key }}>{{ p }}</option>{% endfor %}</select>
  <a class="nav-ar {{ 'off' if not next_key }}" href="/wrapped?kind={{kind}}&key={{next_key}}">newer ›</a>
</div>
{% if s.plays %}
<div class=hero>
  {% set hs=[("{:,}".format(s.plays),"plays",true),("{:,}".format(s.hours),"hours",false),
             ("{:,}".format(s.tracks),"tracks",false),("{:,}".format(s.artists),"artists",false),
             ("{:,}".format(s.new_count),"new finds",false)] %}
  {% for n,l,acc in hs %}<div class="stat rise" style="--i:{{ loop.index }}"><div class="n{{ ' accent' if acc }}">{{ n }}</div><div class=l>{{ l }}</div></div>{% endfor %}
</div>
{% if s.delta is not none %}<p class=rise style="margin:-2rem 0 2.4rem"><span class="delta {{ 'up' if s.delta>=0 else 'down' }}">{{ '+' if s.delta>=0 }}{{ s.delta }}% plays vs prior {{ kind }}</span></p>{% endif %}
<div class="spot rise">
  {% set t1=s.top_tracks[0] if s.top_tracks else none %}{% set a1=s.top_artists[0] if s.top_artists else none %}
  {% if t1 %}<div class=hl>{% if t1.art %}<img class=big src="{{ t1.art }}" alt="">{% else %}<div class=big>♪</div>{% endif %}
    <div class=meta><div class=k>#1 track</div><div class=t>{{ t1.title }}</div><div class=a>{{ t1.artist }} · {{ t1.plays }} plays</div></div></div>{% endif %}
  {% if a1 %}<div class=hl>{% if a1.art %}<img class=big style="border-radius:50%" src="{{ a1.art }}" alt="">{% else %}<div class=big>★</div>{% endif %}
    <div class=meta><div class=k>#1 artist</div><div class=t>{{ a1.artist }}</div><div class=a>{{ a1.plays }} plays · {{ a1.tracks }} tracks</div></div></div>{% endif %}
</div>
<div class=cols>
  <section class=rise><h2>top tracks</h2>{{ leaderboard(s.top_tracks) }}</section>
  <section class=rise><h2>top artists</h2>
    {% set amx=(s.top_artists|map(attribute='plays')|list|max) if s.top_artists else 1 %}
    <ol>{% for x in s.top_artists %}<li class="row artistrow rise" style="--w:{{ x.plays*100//amx }}%;--i:{{loop.index}}">
      <span class=rank>{{ "%02d"|format(loop.index) }}</span>{{ artist_thumb(x) }}
      <div class=ti><div class=t>{{ x.artist }}</div><div class=a>{{ x.tracks }} tracks</div></div>
      <div class=plays>{{ x.plays }} <span>plays</span></div></li>{% endfor %}</ol></section>
</div>
<div class=cols>
  <section class=rise><h2>top genres</h2>
    {% if s.genres %}{% set gmx=(s.genres|map(attribute='plays')|list|max) %}
    <div class=gen>{% for g in s.genres %}<div class="g rise" style="--i:{{loop.index}}">
      <div class=gl><b>{{ g.genre }}</b><span>{{ g.plays }} plays</span></div>
      <div class=gb><i style="width:{{ g.plays*100//gmx }}%"></i></div></div>{% endfor %}</div>
    {% else %}<p class=lede>not enough tagged tracks here yet</p>{% endif %}</section>
  <section class=rise><h2>new to you</h2>
    {% if s.new_top %}{{ leaderboard(s.new_top) }}{% else %}<p class=lede>no first-time discoveries this period</p>{% endif %}</section>
</div>
<div class=cols>
  <section class=rise><h2>when you listened · by hour</h2>
    {% set hmx=(s.hours_clock|map(attribute='plays')|list|max) %}
    <div class=clock>{% for h in s.hours_clock %}<div class="h {{ 'peak' if hmx and h.plays==hmx }}"><div class=b style="height:{{ (h.plays*100//hmx) if hmx else 0 }}%"></div></div>{% endfor %}</div>
    <div class=clockx><span>00</span><span>06</span><span>12</span><span>18</span><span>23</span></div></section>
  <section class=rise><h2>busiest day</h2>
    {% if s.busiest %}<div class=callout><b>{{ s.busiest.date }}</b> — {{ s.busiest.n }} plays in one day</div>{% else %}<p class=lede>—</p>{% endif %}</section>
</div>
{% else %}<p class="lede rise">No plays in this period.</p>{% endif %}
{% endblock %}"""

GEMS = r"""{% extends "base" %}{% from "macros" import thumb %}{% block content %}
<h1 class=rise>Lost Gems</h1>
<p class="lede rise">Liked songs that never made it into a real playlist — buried in a {{ "{:,}".format(s.liked) }}-track Liked pile. Different ways they got lost:</p>
<div class=hero>
  <div class="stat rise" style="--i:1"><div class=n>{{ "{:,}".format(s.orphan) }}</div><div class=l>liked, never playlisted</div></div>
  <div class="stat rise" style="--i:2"><div class="n accent">{{ "{:,}".format(s.faded) }}</div><div class=l>faded · silent 6mo+</div></div>
  <div class="stat rise" style="--i:3"><div class=n>{{ "{:,}".format(s.liked) }}</div><div class=l>liked songs total</div></div>
</div>
<div class="periodbar rise">
  <div class=tabs>{% for c in cats %}<a href="/gems?cat={{c.key}}" class="{{ 'on' if cat==c.key }}">{{ c.label }} <span style="opacity:.55">{{ c.count }}</span></a>{% endfor %}</div>
</div>
<p class="lede rise" style="margin-top:-1.4rem">{{ desc }}</p>
<section class=rise><h2>{{ heading }}</h2>
  {% if gems %}{% set mx=(gems|map(attribute='plays')|list|max) or 1 %}
  <ol>{% for x in gems %}<li class="row rise" style="--w:{{ (x.plays*100//mx) if x.plays else 0 }}%;--i:{{ loop.index }}">
    <span class=rank>{{ "%02d"|format(loop.index) }}</span>{{ thumb(x) }}
    <div class=ti><div class=t>{{ x.title }}</div><div class=a>{{ x.artist }}</div></div>
    <div class=plays>{% if x.plays %}{{ x.plays }} <span>plays{% if x.last %} · last {{ x.last[:7] }}{% endif %}</span>{% else %}<span>never played</span>{% endif %}</div></li>{% endfor %}</ol>
  {% else %}<p class=lede>nothing here — no favorites lost this way.</p>{% endif %}
</section>
<p class="lede rise">💎 The <b>Faded</b> set rotates into a Lost Gems playlist on Spotify — refresh it from <a href="/pipeline" style="color:var(--accent)">Pipeline</a>. (Per-category playlists coming next.)</p>
{% endblock %}"""

STATUS_FRAGMENT = r"""{% include "_sb" %}"""

# Recent job-run audit table (auto-refreshed on the Pipeline page). Errors show raw;
# successes that created playlists become links; everything else a compact summary.
RUNS = r"""<table class=sched><tr><th>job</th><th>when</th><th>status</th><th>result</th></tr>
{% for r in runs %}<tr><td>{{ r.job }}</td>
  <td class=when>{{ r.started_at|localtime if r.started_at else '' }}</td>
  <td style="color:{{ 'var(--accent)' if r.status=='ok' else 'var(--accent2)' }}">{{ r.status }}</td>
  <td class=rcell>{% if r.status=='error' %}<span class=rerr>{{ r.error }}</span>
  {%- elif r.playlists %}{% for p in r.playlists %}<a href="{{ p.url }}" target=_blank rel=noopener>{{ p.name or 'playlist' }}</a>{% if p.tracks is not none %} · {{ p.tracks }} tracks{% endif %}{% if not loop.last %} · {% endif %}{% endfor %}
  {%- elif r.summary %}<span class=rsum>{{ r.summary }}</span>
  {%- else %}<span class=rsum>—</span>{% endif %}</td></tr>
{% else %}<tr><td colspan=4 style="color:var(--muted)">no runs logged yet</td></tr>{% endfor %}</table>"""

# Macros available to every template (leaderboard + thumb).
MACROS = r"""
{% macro thumb(x) %}{% if x.art %}<img class=art src="{{ x.art }}" alt="">{% else %}<div class=art>♪</div>{% endif %}{% endmacro %}
{% macro artist_thumb(x) %}{% if x.art %}<img class=art style="border-radius:50%" src="{{ x.art }}" alt="">{% else %}<div class=art style="border-radius:50%">{{ (x.artist[0]|upper) if x.artist else '★' }}</div>{% endif %}{% endmacro %}
{% macro leaderboard(items) %}{% set mx=(items|map(attribute='plays')|list|max) if items else 1 %}
<ol>{% for x in items %}<li class="row rise" style="--w:{{ x.plays*100//mx }}%;--i:{{ loop.index }}">
  <span class=rank>{{ "%02d"|format(loop.index) }}</span>{{ thumb(x) }}
  <div class=ti><div class=t>{{ x.title }}</div><div class=a>{{ x.artist }}</div></div>
  <div class=plays>{{ x.plays }} <span>plays</span></div></li>{% endfor %}</ol>{% endmacro %}
{% macro playlist_proposal(p, identity) %}
<div class="proposal{{ ' saved' if p.url else '' }}">
  <div class=pt>🎧 {{ p.name|e }}</div>
  <div class=playlist-type>{{ 'Fixed selection · no automatic updates' if p.kind=='curated' else 'Smart rule · refreshes daily' }}</div>
  <div class=pm>{{ p.summary|e }}{% if p.matches is not none %} · ~{{ p.matches|e }} tracks{% endif %}</div>
  {% if p.preview %}<ol class=spreview>{% for t in p.preview %}<li><span class=t>{{ t.title|e }}</span><span class=a>{{ t.artist|e }}</span></li>{% endfor %}</ol>{% endif %}
  <div class=proposal-actions>
    <span class=proposal-status role=status>{{ '✓ Saved' if p.url else '' }}</span>
    {% if p.url %}<a class="approve open" href="{{ p.url }}" target=_blank rel=noopener>Open ↗</a>
    {% elif identity.can_admin %}<form method=post action="{{ p.save_url }}" data-approve>
      <button type=submit class=approve>Approve &amp; save →</button></form>{% endif %}
    {% if p.view_url and not p.url %}<a href="{{ p.view_url }}" target=_blank rel=noopener>View full playlist ↗</a>{% endif %}
  </div>
  {% if p.thread_id is defined %}<div class=proposal-source>From <a href="/chat?thread={{ p.thread_id }}" target=_blank rel=noopener>{{ (p.thread_title or 'New chat')|e }}</a>
    <time>{{ p.created_at|localtime }}</time></div>{% endif %}
</div>
{% endmacro %}
"""

PROPOSAL_SCRIPT = r"""<script>
function bindPlaylistApprovals(root){
  const E=(t,c,x)=>{const e=document.createElement(t); if(c)e.className=c; if(x!=null)e.textContent=x; return e;};
  function savedProp(card,url){
    const actions=card.querySelector('.proposal-actions'), status=actions.querySelector('[role=status]');
    const link=E('a','approve open','Open ↗'); link.href=url; link.target='_blank'; link.rel='noopener';
    const f=actions.querySelector('form'); if(f) f.replaceWith(link); else actions.appendChild(link);
    status.classList.remove('error'); status.textContent='✓ Saved'; card.classList.add('saved'); return link;
  }
  root.addEventListener('submit', async e=>{
    const f=e.target.closest('form[data-approve]'); if(!f) return; e.preventDefault();
    const button=f.querySelector('button'), card=f.closest('.proposal'), status=card.querySelector('[role=status]');
    if(button.disabled) return;
    const focused=f.contains(document.activeElement);
    button.disabled=true; button.textContent='Saving…'; card.setAttribute('aria-busy','true');
    status.textContent=''; status.classList.remove('error');
    try{
      const r=await fetch(f.action,{method:'POST',headers:{'Accept':'application/json'}});
      if(!r.ok) throw new Error('Save failed');
      const result=await r.json(); if(!result.url) throw new Error('Missing saved playlist');
      const restoreFocus=focused&&(document.activeElement===document.body||f.contains(document.activeElement));
      const link=savedProp(card,result.url); if(restoreFocus) link.focus({preventScroll:true});
      root.dispatchEvent(new CustomEvent('playlist-approved'));
    }catch(err){
      button.disabled=false; button.textContent='Retry save →';
      status.classList.add('error'); status.textContent='Couldn’t save. Try again.';
      if(focused&&document.activeElement===document.body) button.focus({preventScroll:true});
    }finally{ card.removeAttribute('aria-busy'); }
  });
  return savedProp;
}
</script>"""

SMART = r"""{% extends "base" %}{% from "macros" import playlist_proposal %}{% block content %}
<h1 class=rise>{% if view=='drafts' %}Playlist Drafts{% elif view=='scheduled' %}Scheduled Playlist Updates{% elif view=='history' %}Playlist History{% else %}{{ 'Curated Playlists' if curated else 'Smart Playlists' }}{% endif %}</h1>
{% if view=='drafts' %}<p class="lede rise">Proposals from your conversations, ready for review. Drafts are not scheduled until you approve them.</p>
{% elif view=='scheduled' %}<p class="lede rise">Upcoming refreshes and first publications, across your playlist collections. Times in {{ display_tz|e }}.</p>
{% elif view=='history' %}<p class="lede rise">Track changes observed in Spotify backups. These are observation times, not exact edit times. Times in {{ display_tz|e }}.</p>
{% elif curated %}<p class="lede rise">Hand-picked selections for a moment or a mood. Published once, with a fixed track list —
your Spotify edits stay yours.</p>
{% else %}<p class="lede rise">Saved rules that choose tracks from your listening data and refresh on Spotify daily.
Pause a rule whenever you want to keep its current selection.</p>{% endif %}
<div class="tabs rise" style="margin-bottom:1.8rem;flex-wrap:wrap">
  <a href=/smart class="{{ 'on' if view=='saved' and not curated }}">Smart · refreshes daily</a>
  <a href=/curated class="{{ 'on' if view=='saved' and curated }}">Curated · fixed selections</a>
  <a href="{{ base_path }}?view=drafts" class="{{ 'on' if view=='drafts' }}">Drafts <span data-draft-count>{{ draft_count }}</span></a>
  <a href="{{ base_path }}?view=scheduled" class="{{ 'on' if view=='scheduled' }}">Scheduled</a>
  <a href="{{ base_path }}?view=history" class="{{ 'on' if view=='history' }}">History</a>
</div>

{% if view=='drafts' %}
<section class="rise drafts"><h2>awaiting approval</h2>
  {% if drafts %}<div id=playlist-drafts class=pl>{% for p in drafts %}{{ playlist_proposal(p, identity) }}{% endfor %}</div>
  {% else %}<p class=lede>No playlist drafts awaiting approval. <a href=/chat>Ask in Chat</a> for your next selection or smart rule.</p>{% endif %}
</section>
{% if track_suggestions %}<p class=lede>{{ track_suggestions }} track suggestion{{ 's' if track_suggestions!=1 else '' }} also await review. <a href=/review>Open the review queue →</a></p>{% endif %}
{% elif view=='scheduled' %}
<div class=activity>
  {% if schedule.scheduler.state!='running' %}<div class=callout>Scheduler timing is {{ schedule.scheduler.state|e }}.
    {% if schedule.scheduler.state=='stale' %}Its last report is too old to confirm upcoming runs.{% else %}No current next-run times are confirmed.{% endif %}
    <a href=/pipeline>Check Pipeline →</a></div>{% endif %}
  {% if schedule.scheduler.recorded_at %}<p class=lede>Scheduler last reported {{ schedule.scheduler.recorded_at|localtime }}.</p>{% endif %}
  <section class=rise><h2>automatic update batches</h2>
    {% if schedule.jobs %}<div class=pl>{% for job in schedule.jobs %}<div class=card>
      <div class=pt>{{ job.name|e }}</div>
      <div class=when>{% if job.running %}Update batch running{% elif job.next_run_at %}Next · {{ job.next_run_at|localtime }}{% else %}Next run not confirmed{% endif %}</div>
    </div>{% endfor %}</div>{% else %}<p class=lede>No playlist update batches have been reported by the scheduler.</p>{% endif %}
  </section>
  <section class=rise><h2>rules &amp; pending publications</h2>
    {% if schedule['items'] %}<div class=pl>{% for it in schedule['items'] %}<div class=card>
      <div class=pt>{{ it.name|e }}</div><div class=playlist-type>{{ it.status }}</div>
      <div class=when>{% if it.running %}Update batch running{% elif it.next_run_at %}Next · {{ it.next_run_at|localtime }}{% elif it.enabled %}Next run not confirmed{% else %}No automatic run{% endif %}</div>
      {% if it.published_at %}<div class=pm>Last published {{ it.published_at|localtime }}</div>{% endif %}
      <div class=smartbtns><a href="{{ it.url }}">manage {{ 'selection' if it.curated else 'rule' }} →</a></div>
    </div>{% endfor %}</div>{% else %}<p class=lede>No saved rules or first publications are waiting for updates.</p>{% endif %}
  </section>
</div>
{% elif view=='history' %}
<section class="rise activity"><h2>changes from backups</h2>
  {% if history_filter %}<p class=lede>History for <a href="{{ history_filter.url }}">{{ history_filter.name|e }}</a> · <a href="{{ base_path }}?view=history">All playlist history →</a></p>{% endif %}
  {% for event in history['items'] %}<details class=history-entry>
    <summary><span class=event-name>{{ (event.name or event.playlist_id)|e }}</span>
      {% if not event.complete %}<span class=change>Snapshot detail incomplete</span>
      {% elif event.baseline %}<span class=change>First recorded snapshot · {{ event.count }} tracks</span>
      {% else %}<span class="change added">+{{ event.added_count }} added</span><span class="change removed">−{{ event.removed_count }} removed</span>
        {% if event.reordered %}<span class=change>Track order changed</span>{% endif %}{% endif %}
      <time class=event-time>{{ event.backup_time|localtime }}</time>
    </summary>
    <p class=event-note>{% if not event.complete %}The recorded track details are incomplete, so additions and removals cannot be determined.
      {% elif event.baseline %}This is a baseline. Earlier membership is unknown; these tracks are not counted as new additions.
      {% else %}Compared with the previous snapshot from {{ event.previous_at|localtime }}.
        {% if not event.added_count and not event.removed_count and not event.reordered and event.unknown_before==event.unknown_after %}No track-list change in these recorded snapshots.{% endif %}
      {% endif %}</p>
    {% if event.unknown_after or event.unknown_before %}<p class=event-note>Unavailable or local items: {% if not event.baseline %}{{ event.unknown_before }} → {% endif %}{{ event.unknown_after }}. Their identities cannot be compared.</p>{% endif %}
    {% if event.added or event.removed %}<div class=change-lists>
      {% for key, title in [('added','Added'),('removed','Removed')] %}<div><b class="change {{ key }}">{{ title }}</b>
        <ul>{% for track in event[key] %}<li><a href="https://open.spotify.com/track/{{ track.track_id|urlencode }}" target=_blank rel=noopener>{{ track.title|e }}</a> <span>· {{ track.artist|e }}{% if track.count>1 %} · ×{{ track.count }}{% endif %}</span></li>{% else %}<li>None</li>{% endfor %}</ul>
      </div>{% endfor %}
    </div>{% endif %}
    <p class=event-note><a href="/playlists/{{ event.playlist_id|urlencode }}">View playlist →</a></p>
  </details>{% else %}<p class=lede>No playlist snapshots on this page. Changes appear after library backups record them.</p>{% endfor %}
  <div class=history-pages>{% if history.page>1 %}<a href="{{ history_base }}&amp;page={{ history.page-1 }}">← Newer</a>{% endif %}
    {% if history.has_more %}<a href="{{ history_base }}&amp;page={{ history.page+1 }}">Older →</a>{% endif %}</div>
</section>
{% else %}
<section class=rise><h2>{{ 'your selections' if curated else 'your rules' }}</h2>
  {% if items %}<div class=pl>{% for it in items %}
    <div class=card id="{{ 'curated' if curated else 'smart' }}-{{ it.id }}" style="--i:{{ loop.index }};scroll-margin-top:5rem">
      <div class=pt>{{ it.name|e }}</div>
      <div class=playlist-type aria-live=polite>{% if curated %}Fixed selection · {{ 'published' if it.published_at else 'ready to publish' }}{% else %}{{ 'Refreshes daily' if it.enabled else 'Automatic refresh paused' }}{% endif %}</div>
      <div class=pm>{{ it.summary|e }}</div>
      {% if it.preview %}<ol class=spreview>{% for t in it.preview %}<li><span class=t>{{ t.title|e }}</span><span class=a>{{ t.artist|e }}</span></li>{% endfor %}</ol>
      {% else %}<div class=pm style="opacity:.55">{{ 'Track details are not available yet.' if it.count else 'No tracks right now.' }}</div>{% endif %}
      <div class=scount>{{ it.count }} track{{ '' if it.count==1 else 's' }}{% if it.preview and it.count > it.preview|length %} · preview of {{ it.preview|length }}{% endif %}</div>
      <div class=smartbtns>
        <a href="{{ it.url }}">view playlist →</a>
        {% if it.spotify_id %}<a href="https://open.spotify.com/playlist/{{ it.spotify_id|urlencode }}" target=_blank rel=noopener>Spotify ↗</a>{% endif %}
        {% if identity.can_admin %}
        {% if curated %}{% if not it.published_at and it.count %}<button type=button data-publish="/curated/{{ it.id }}/publish">publish to Spotify</button>{% endif %}
        {% else %}<button hx-post="/smart/{{ it.id }}/run" hx-target=#toast hx-disabled-elt=this>refresh now</button>
        <form method=post action="/smart/{{ it.id }}/toggle" style="display:contents"><button>{{ 'enable' if not it.enabled else 'pause' }}</button></form>
        {% endif %}
        <form method=post action="/smart/{{ it.id }}/delete" style="display:contents" onsubmit="return confirm('Remove this saved {{ 'selection' if curated else 'rule' }}? Your Spotify playlist is left untouched.')"><button class=danger>remove</button></form>
        {% endif %}
      </div>
    </div>
  {% endfor %}</div><span id=toast></span>
  {% else %}<p class=lede>{% if curated %}No curated playlists yet — <a href=/chat>ask in Chat</a> for a selection to save.{% else %}No rules yet — make one below.{% endif %}</p><span id=toast></span>{% endif %}
</section>

{% if identity.can_admin and not curated %}<section class=rise><h2>new rule</h2>
  <form method=post action=/smart/create class=smartform>
    <input name=name placeholder="🎧 Playlist name" required>
    <div class=presets><span>presets:</span>{% for k in examples %}<button type=button onclick="fillEx('{{ k }}')">{{ k }}</button>{% endfor %}</div>
    <textarea name=spec id=spec rows=9 spellcheck=false>{{ template_spec|e }}</textarea>
    <button type=submit class=primary>Create rule</button>
  </form>
  <details style="margin-top:1.4rem"><summary class=lede style="cursor:pointer">spec fields</summary>
    <pre class=specdoc>{{ field_docs|e }}</pre></details>
</section>
{% endif %}
{% endif %}
{% include "proposal_script" %}
<script>
  const draftRoot=document.getElementById('playlist-drafts');
  if(draftRoot){ bindPlaylistApprovals(draftRoot); draftRoot.addEventListener('playlist-approved',()=>{
    const count=document.querySelector('[data-draft-count]'); if(count) count.textContent=String(Math.max(0,Number(count.textContent)-1));
  }); }
  document.querySelectorAll('button[data-publish]').forEach(button=>button.addEventListener('click',async()=>{
    if(button.disabled) return;
    const card=button.closest('.card'), status=card.querySelector('.playlist-type');
    const focused=document.activeElement===button;
    button.disabled=true; button.textContent='publishing…'; status.textContent='Publishing fixed selection…';
    try{
      const response=await fetch(button.dataset.publish,{method:'POST'});
      if(!response.ok) throw new Error('Publish failed');
      const result=await response.json(); if(!result.url) throw new Error('Missing playlist');
      const restoreFocus=focused&&document.activeElement===document.body;
      const link=document.createElement('a'); link.href=result.url; link.target='_blank';
      link.rel='noopener'; link.textContent='open ↗'; button.replaceWith(link);
      status.textContent='Fixed selection · published';
      if(restoreFocus) link.focus({preventScroll:true});
    }catch(err){
      button.disabled=false; button.textContent='retry publish';
      status.textContent='Couldn’t publish. Try again.';
      if(focused&&document.activeElement===document.body) button.focus({preventScroll:true});
    }
  }));
  const SMART_EX = {{ examples_json }};
  function fillEx(k){ document.getElementById('spec').value = JSON.stringify(SMART_EX[k], null, 2);
    const n = document.querySelector('input[name=name]'); if(!n.value) n.value = k; }
</script>
{% endblock %}"""

CHAT = r"""{% extends "base" %}{% from "macros" import playlist_proposal %}{% block content %}
<div class="chathead rise">
  <h1 style="margin:0">{{ (thread.title or 'Chat with your library')|e }}</h1>
  <div class=threadbtns>
    {% if identity.can_query %}<label class=thinktoggle title="Show the model's reasoning as it streams. Thinking is never saved with the conversation.">
      <input type=checkbox id=thinkopt checked> 💭 thinking</label>{% endif %}
    <button type=button onclick="document.getElementById('threadmodal').hidden=false">☰ threads</button>
    {% if identity.can_query %}<form method=post action=/chat/new style="display:contents"><button class=primary>+ new</button></form>{% endif %}
  </div>
</div>
<p class="lede rise">Explore your listening history and propose playlists for approval.
{% if identity.can_admin %}Refresh recent plays or request missing track embeddings in chat.{% else %}Library queries are read-only.{% endif %}
{% if not configured %}<br><span style="color:var(--accent2)">⚠ Set <code>OPENROUTER_API_KEY</code> in <code>.env</code> or <code>.env.local</code> to enable chat.</span>{% endif %}</p>

<details class="library-memory rise" id=library-memory>
  <summary>Library memory · {{ 'saved note' if memory.text else 'no notes yet' }}</summary>
  <p>One shared Markdown note across all chats: naming conventions, playlist roles, confirmed decisions, and work still to do.
  {% if identity.can_admin %}Ask Chat to remember a decision, or edit the note here. Save an empty note to clear it.{% endif %}</p>
  {% if identity.can_admin %}<form id=memory-editor action=/chat/memory method=post>
    <input type=hidden name=revision value="{{ memory.revision|e }}">
    <textarea name=text rows=6 maxlength=6000 aria-label="Library memory" placeholder="e.g. Keep fixed mood selections separate from daily rules. Next: organize ambient.">{{ memory.text|e }}</textarea>
    <div class=smartbtns><button type=submit>Save memory</button><span class=memory-status role=status></span></div>
  </form>{% else %}<pre>{{ (memory.text or 'No organization notes saved yet.')|e }}</pre>{% endif %}
</details>

<div id=threadmodal class=modal hidden>
  <div class=modalbg onclick="document.getElementById('threadmodal').hidden=true"></div>
  <div class=modalcard>
    <div class=modalhead><b>Threads</b><button type=button onclick="document.getElementById('threadmodal').hidden=true">✕</button></div>
    <div class=threadlist>
      {% for t in threads %}<a href="/chat?thread={{ t.id }}" class="{{ 'on' if t.id==thread.id }}">
        <span class=tt>{{ (t.title or 'New chat')|e }}{% if t.archived %} <span class=arch>archived</span>{% endif %}</span>
        <span class=td>{{ t.updated_at|localtime }}</span></a>
      {% else %}<p class=lede>no threads yet</p>{% endfor %}
    </div>
  </div>
</div>

<div id=transcript class=chat>
  {% for m in msgs %}{% if m.role=='user' %}<div class=turn><div class=q>{{ m.content|e }}</div></div>
  {% else %}<div class=turn><div class=a>
    {% if m.meta and m.meta.steps %}<div class=trace>{% for s in m.meta.steps %}<details class=tracestep>
      {% if s.tool=='track_report' %}<summary>🎚 inspected track data</summary><pre class=sql>{{ s.query|e }}</pre>
      {% elif s.tool=='query_db' %}<summary>🔎 ran a query</summary><pre class=sql>{{ s.sql|e }}</pre>
      {% elif s.tool=='similar_tracks' %}<summary>🎵 looked up similar tracks</summary><pre class=sql>seed: {{ s.query|e }}</pre>
      {% elif s.tool=='search_by_sound' %}<summary>👂 searched by sound</summary><pre class=sql>“{{ s.query|e }}”</pre>
      {% elif s.tool=='discover_artists' %}<summary>🧭 discovered artists</summary><pre class=sql>playlist: {{ s.playlist|e }}</pre>
      {% elif s.tool=='refresh_plays' %}<summary>↻ refreshed play history</summary>
      {% elif s.tool=='update_library_memory' %}<summary>✎ organization memory</summary>
      {% elif s.tool=='propose_playlist' %}<summary>📝 prepared a playlist proposal</summary>
      {% else %}<summary>ran {{ s.tool|e }}</summary>{% endif %}
      {% if s.ok is defined %}<pre class=sql>{{ 'ok' if s.ok else 'error' }} · {{ s.duration_ms }}ms{% if s.row_count is defined %} · {{ s.row_count }} row(s){% endif %}{% if s.surface_states %}\n{{ s.surface_states|tojson }}{% endif %}</pre>{% endif %}
    </details>{% endfor %}</div>{% endif %}
    <div class=body data-md="{{ m.content|e }}"></div>
    {% if m.meta %}{% for p in m.meta.proposals %}{{ playlist_proposal(p, identity) }}{% endfor %}{% if m.meta.model %}<div class=via>via {{ m.meta.model }}{% if m.meta.usage %} · {{ usage_summary(m.meta.usage) }}{% endif %}{% if m.meta.context and m.meta.context.summarized %} · earlier messages summarized{% endif %}</div>{% endif %}{% endif %}
  </div></div>{% endif %}{% endfor %}
</div>
{% if identity.can_query %}<form id=askform class="askform rise">
  <input type=hidden name=thread value="{{ thread.id }}">
  <select name=model class=psel title="model">{% for group in model_groups %}<optgroup label="{{ group.label }}">{% for m in group.models %}{% set selection = m.id or m.slug %}<option value="{{ selection }}" {{ 'selected' if selection==default }}>{{ m.label }}</option>{% endfor %}</optgroup>{% endfor %}</select>
  <textarea name=q rows=1 autocomplete=off required aria-label="Your question" placeholder="ask about your listening, or ask for a playlist…  (Enter to send, Shift+Enter for newline)">{{ draft_query|default('')|e }}</textarea>
  <button type=submit>ask</button>
</form>
{% else %}<div class=callout>Query access is not enabled for this account.</div>{% endif %}
{% include "proposal_script" %}
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js" integrity="sha384-/TQbtLCAerC3jgaim+N78RZSDYV7ryeoBCVqTuzRrFec2akfBkHS7ACQ3PQhvMVi" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.4.18/dist/purify.min.js" integrity="sha384-lkArbWjiZKuE+bzKbj1vOEF7DKYBDNvgwZMCt5OH8bdRB92zpzyNska9Q/XKPHQD" crossorigin="anonymous"></script>
<script>
(function(){
  const transcript=document.getElementById('transcript'), form=document.getElementById('askform');
  if(!form) return;
  const canAdmin={{ 'true' if identity.can_admin else 'false' }};
  const memoryEditor=document.getElementById('memory-editor');
  let savedMemory=memoryEditor?memoryEditor.elements.text.value:'';
  function syncMemory(memory){
    if(!memoryEditor||!memory) return;
    const status=memoryEditor.querySelector('[role=status]');
    if(memoryEditor.elements.text.value!==savedMemory){
      status.classList.add('error'); status.textContent='Memory changed elsewhere. Reload before saving your edits.'; return;
    }
    memoryEditor.elements.text.value=memory.text; memoryEditor.elements.revision.value=memory.revision;
    savedMemory=memory.text; status.classList.remove('error'); status.textContent='Memory updated from chat.';
    document.querySelector('#library-memory summary').textContent='Library memory · '+(memory.text?'saved note':'no notes yet');
  }
  if(memoryEditor) memoryEditor.addEventListener('submit',async e=>{
    e.preventDefault(); const button=memoryEditor.querySelector('button'), status=memoryEditor.querySelector('[role=status]');
    button.disabled=true; status.classList.remove('error'); status.textContent='Saving…';
    const sentText=memoryEditor.elements.text.value;
    try{
      const response=await fetch(memoryEditor.action,{method:'POST',body:new FormData(memoryEditor)});
      const result=await response.json(); if(!response.ok) throw new Error(result.detail||'Couldn’t save memory. Try again.');
      savedMemory=sentText; memoryEditor.elements.revision.value=result.revision;
      status.textContent='Saved. Used from the next question onward.';
      document.querySelector('#library-memory summary').textContent='Library memory · '+(result.text?'saved note':'no notes yet');
    }catch(err){ status.classList.add('error'); status.textContent=err.message||'Couldn’t save memory. Try again.'; }
    finally{ button.disabled=false; }
  });
  const E=(t,c,x)=>{const e=document.createElement(t); if(c)e.className=c; if(x!=null)e.textContent=x; return e;};
  const savedProp=bindPlaylistApprovals(transcript);
  DOMPurify.addHook('afterSanitizeAttributes', n=>{ if(n.tagName==='A'){ n.setAttribute('target','_blank'); n.setAttribute('rel','noopener'); } });
  const md=t=>DOMPurify.sanitize(marked.parse(t||''));
  const scroll=()=>{transcript.scrollTop=transcript.scrollHeight;};
  document.querySelectorAll('.body[data-md]').forEach(el=>{ el.innerHTML=md(el.dataset.md); });  // persisted history
  scroll();
  document.addEventListener('keydown', e=>{ if(e.key==='Escape'){ const m=document.getElementById('threadmodal'); if(m&&!m.hidden) m.hidden=true; } });
  const qbox=form.q, grow=()=>{ qbox.style.height='auto'; const sh=qbox.scrollHeight; qbox.style.height=Math.min(sh,160)+'px'; qbox.style.overflowY=sh>160?'auto':'hidden'; };
  grow();
  qbox.addEventListener('input', grow);
  qbox.addEventListener('keydown', e=>{ if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); form.requestSubmit(); } });
  function flashErr(msg){ const e=E('div','chaterr','⚠ '+msg); form.parentNode.insertBefore(e, form); setTimeout(()=>e.remove(), 6000); }
  function dots(){ const d=E('div','dots'); d.appendChild(E('span')); d.appendChild(E('span')); d.appendChild(E('span')); return d; }
  // Thinking: a live panel that folds itself away once the real answer starts.
  // Purely client-side — the toggle hides panels instantly and never touches the stream.
  const thinkOpt=document.getElementById('thinkopt');
  const showThink=()=>localStorage.getItem('acetate.think')!=='0';
  function applyThink(){ document.body.classList.toggle('nothink', !showThink()); }
  if(thinkOpt){ thinkOpt.checked=showThink(); applyThink();
    thinkOpt.addEventListener('change',()=>{ localStorage.setItem('acetate.think', thinkOpt.checked?'1':'0'); applyThink(); }); }
  function thinkPanel(trace){ const d=E('details','think'); d.open=true; d.dataset.live='1';
    d.appendChild(E('summary',null,'💭 thinking…')); const pre=E('pre','thoughts');
    d.appendChild(pre); trace.appendChild(d);          // in the trace, so it interleaves with tool calls
    return {d:d,pre:pre,t0:Date.now()}; }
  function endThink(th){ if(!th||!th.d.dataset.live) return; delete th.d.dataset.live; th.d.open=false;
    const secs=((Date.now()-th.t0)/1000).toFixed(1);
    th.d.querySelector('summary').textContent='💭 thought for '+secs+'s'; }
  function addStep(trace,s){ const d=E('details','tracestep'); d.dataset.callId=s.call_id||'';
    if(s.tool==='track_report'){ d.appendChild(E('summary',null,'🎚 inspected track data')); d.appendChild(E('pre','sql',s.query||'')); }
    else if(s.tool==='query_db'){ d.appendChild(E('summary',null,'🔎 ran a query')); d.appendChild(E('pre','sql',s.sql)); }
    else if(s.tool==='similar_tracks'){ d.appendChild(E('summary',null,'🎵 looked up similar tracks')); d.appendChild(E('pre','sql','seed: '+(s.query||''))); }
    else if(s.tool==='search_by_sound'){ d.appendChild(E('summary',null,'👂 searched by sound (~15s)')); d.appendChild(E('pre','sql','“'+(s.query||'')+'”')); }
    else if(s.tool==='discover_artists'){ d.appendChild(E('summary',null,'🧭 discovered artists')); d.appendChild(E('pre','sql','playlist: '+(s.playlist||''))); }
    else if(s.tool==='refresh_plays'){ d.appendChild(E('summary',null,'↻ refreshed play history')); d.appendChild(E('pre','sql','polling Spotify…')); }
    else if(s.tool==='update_library_memory'){ d.appendChild(E('summary',null,'✎ organization memory')); }
    else if(s.tool==='propose_playlist'){ d.appendChild(E('summary',null,'📝 prepared a playlist proposal')); }
    else { d.appendChild(E('summary',null,'ran '+s.tool)); }
    trace.appendChild(d); }
  function finishStep(trace,s){ const d=[...trace.querySelectorAll('.tracestep')].find(x=>x.dataset.callId===s.call_id); if(!d)return;
    let line=(s.ok?'ok':'error')+' · '+(s.duration_ms||0)+'ms'; if(s.row_count!=null)line+=' · '+s.row_count+' row(s)';
    if(s.surface_states)line+='\n'+JSON.stringify(s.surface_states); d.appendChild(E('pre','sql',line)); }
  function addProps(turn,props){ props.forEach(p=>{ const b=E('div','proposal');
    b.appendChild(E('div','pt','🎧 '+(p.name||'Smart Playlist')));
    b.appendChild(E('div','playlist-type',p.kind==='curated'?'Fixed selection · no automatic updates':'Smart rule · refreshes daily'));
    b.appendChild(E('div','pm',(p.summary||'')+(p.matches!=null?' · ~'+p.matches+' tracks':'')));
    const actions=E('div','proposal-actions'), status=E('span','proposal-status'); status.setAttribute('role','status');
    actions.appendChild(status); b.appendChild(actions);
    if(p.url) savedProp(b,p.url);
    else if(canAdmin&&p.save_url){ const f=E('form'); f.method='post'; f.action=p.save_url; f.dataset.approve='';
      f.appendChild(E('button','approve','Approve & save →')); actions.appendChild(f); }
    if(p.view_url&&!p.url){ const link=E('a',null,'View full playlist ↗'); link.href=p.view_url; link.target='_blank'; link.rel='noopener'; actions.appendChild(link); }
    turn.appendChild(b); }); }
  function usageSummary(u){ if(!u) return ''; const p=[];
    if(typeof u.cost==='number'){ const n=u.cost, d=n>0&&n<.0001?6:n<.01?4:2; p.push('$'+n.toFixed(d)+' effective'); }
    p.push(Number(u.input_tokens||0).toLocaleString()+' in');
    p.push(Number(u.output_tokens||0).toLocaleString()+' out');
    p.push(Number(u.cached_tokens||0).toLocaleString()+' cached'); return p.join(' · '); }
  form.addEventListener('submit', async e=>{
    e.preventDefault();
    const q=form.q.value.trim(); if(!q) return;
    const model=form.model.value, thread=form.thread.value; form.q.value=''; grow();
    const turn=E('div','turn'); turn.appendChild(E('div','q',q));                 // user bubble, immediately
    const ans=E('div','a'), trace=E('div','trace'), body=E('div','body');
    body.appendChild(dots());                                                     // typing dots until tokens arrive
    ans.appendChild(trace); ans.appendChild(body);
    turn.appendChild(ans); transcript.appendChild(turn); scroll();
    let text='', props=[], started=false, think=null;
    const fail=msg=>{ if(started){ body.appendChild(E('div','err',msg)); } else { turn.remove(); form.q.value=q; grow(); flashErr(msg); } };
    try{
      const fd=new FormData(); fd.append('q',q); fd.append('model',model); fd.append('thread',thread);
      const r=await fetch('/chat/stream',{method:'POST',body:fd});
      const rd=r.body.getReader(), dec=new TextDecoder(); let buf='';
      for(;;){
        const {value,done}=await rd.read(); if(done) break;
        buf+=dec.decode(value,{stream:true}); let nl;
        while((nl=buf.indexOf('\n'))>=0){
          const ln=buf.slice(0,nl); buf=buf.slice(nl+1); if(!ln.trim()) continue;
          let ev; try{ev=JSON.parse(ln);}catch(_){continue;}
          if(ev.t==='token'){ endThink(think); if(!started){body.innerHTML='';started=true;} text+=ev.v; body.innerHTML=md(text); }
          else if(ev.t==='think'){ if(!think||!think.d.dataset.live) think=thinkPanel(trace);  // a fresh burst gets its own panel
            think.pre.textContent+=ev.v; think.pre.scrollTop=think.pre.scrollHeight; }
          else if(ev.t==='step'){ endThink(think); think=null; addStep(trace,ev.v); }  // tool calls stream in live
          else if(ev.t==='step_result'){ finishStep(trace,ev.v); }
          else if(ev.t==='status'){ if(!started) body.textContent=ev.v; }
          else if(ev.t==='proposal'){ props.push(ev.v); }
          else if(ev.t==='error'){ endThink(think); fail(ev.v); return; }
          else if(ev.t==='done'){ endThink(think); syncMemory(ev.v.memory); if(!started) body.innerHTML=''; addProps(turn,ev.v.proposals||props);
            if(ev.v&&ev.v.model){ const u=usageSummary(ev.v.usage);
              const c=ev.v.context&&ev.v.context.summarized?' · earlier messages summarized':'';
              turn.appendChild(E('div','via','via '+ev.v.model+(u?' · '+u:'')+c)); } }
          scroll();
        }
      }
    }catch(err){ fail(String(err)); return; }
    const d=body.querySelector('.dots'); if(d) d.remove();
  });
})();
</script>
{% endblock %}"""

CHAT_TURN = r"""{% from "macros" import playlist_proposal %}<div class=turn>
  <div class=q>{{ q|e }}</div>
  {% if error %}<div class="a err">{{ error }}</div>
  {% else %}<div class=a>{{ reply|e }}</div>
    {% if steps %}<details class=steps><summary>ran {{ steps|length }} workflow step(s)</summary>
      {% for s in steps %}<pre class=sql>{{ s.tool|e }}{% if s.sql %}: {{ s.sql|e }}{% elif s.query %}: {{ s.query|e }}{% elif s.playlist %}: {{ s.playlist|e }}{% endif %}</pre>{% endfor %}</details>{% endif %}
    {% for p in proposals %}{{ playlist_proposal(p, identity) }}{% endfor %}
    {% if model %}<div class=via>via {{ model }}</div>{% endif %}
  {% endif %}
</div>"""

HYPERSPACE = r"""{% extends "base" %}{% block content %}
{% if n_points == 0 %}
<h1 class=rise>Hyperspace</h1>
<div class=callout>No map yet — <b>{{ n_embedded }}</b> tracks are embedded so far.
The pipeline is: <code>acetate embed fetch</code> → <code>acetate embed run</code> →
<code>acetate embed reduce</code>. Once the reduce step has written projection points,
this page becomes a 3D flight through your library.</div>
{% else %}
<style>
  .hyper{position:relative;width:100vw;margin-left:calc(50% - 50vw);height:calc(100dvh - 118px);min-height:480px}
  .hyper canvas{display:block}
  .hud,.hudl{position:absolute;top:1rem;width:266px;display:flex;flex-direction:column;gap:.7rem;z-index:5}
  .hud{right:1rem} .hudl{left:1rem;width:286px}
  .hud .box,.hudl .box{background:rgba(18,19,25,.85);backdrop-filter:blur(10px);border:1px solid var(--line);border-radius:12px;padding:.85rem .95rem}
  .hud h3,.hudl h3{font-family:var(--mono);font-size:.62rem;text-transform:uppercase;letter-spacing:.18em;color:var(--muted);margin:0 0 .55rem}
  .modes{display:flex;gap:.3rem;flex-wrap:wrap}
  .modes button{flex:1;font-family:var(--mono);font-size:.64rem;text-transform:uppercase;letter-spacing:.06em;
    background:none;border:1px solid var(--line);border-radius:7px;color:var(--muted);padding:.36rem .2rem;cursor:pointer;transition:.15s;white-space:nowrap}
  .modes button:hover{color:var(--fg)}.modes button.on{color:#0a0a0c;background:var(--accent);border-color:var(--accent);font-weight:700}
  .hin{width:100%;background:rgba(10,10,12,.6);border:1px solid var(--line);border-radius:8px;color:var(--fg);
    font-family:var(--mono);font-size:.72rem;padding:.48rem .6rem;outline:none;box-sizing:border-box}
  .hin:focus{border-color:var(--accent)}
  .claprow{display:flex;gap:.35rem}
  .claprow .go{background:var(--accent);border:none;border-radius:8px;cursor:pointer;padding:0 .6rem;font-size:.85rem;flex:none}
  .note{font-family:var(--mono);font-size:.6rem;color:var(--muted);margin-top:.4rem;min-height:.8rem}
  .drop{display:flex;flex-direction:column;margin-top:.35rem;max-height:32vh;overflow-y:auto}
  .drop .row2{padding:.32rem .4rem;border-radius:6px;cursor:pointer;font-size:.75rem;line-height:1.3;color:var(--fg)}
  .drop .row2:hover{background:rgba(255,255,255,.06)}
  .drop .row2 em{font-style:normal;color:var(--muted);font-size:.63rem;display:block}
  .drop,.lg,.hud .box,.hudl .box{scrollbar-width:thin;scrollbar-color:#2a2c36 transparent}
  .drop::-webkit-scrollbar,.lg::-webkit-scrollbar,.hud .box::-webkit-scrollbar,.hudl .box::-webkit-scrollbar{width:8px;height:8px}
  .drop::-webkit-scrollbar-thumb,.lg::-webkit-scrollbar-thumb,.hud .box::-webkit-scrollbar-thumb,.hudl .box::-webkit-scrollbar-thumb{background:#2a2c36;border-radius:4px}
  .drop::-webkit-scrollbar-thumb:hover,.lg::-webkit-scrollbar-thumb:hover{background:#3a3d49}
  .drop::-webkit-scrollbar-track,.lg::-webkit-scrollbar-track,.hud .box::-webkit-scrollbar-track,.hudl .box::-webkit-scrollbar-track{background:transparent}
  .likedrow{display:flex;gap:.45rem;align-items:center;font-family:var(--mono);font-size:.66rem;color:var(--fg);margin-top:.55rem;cursor:pointer}
  .likedrow input{accent-color:var(--accent);color-scheme:dark;width:13px;height:13px;cursor:pointer;margin:0}
  .lg{display:flex;flex-direction:column;gap:.28rem;max-height:26vh;overflow-y:auto}
  .lg .it{display:flex;align-items:center;gap:.55rem;font-size:.78rem;color:var(--fg);cursor:pointer;
    padding:.14rem .3rem;border-radius:6px;user-select:none}
  .lg .it:hover{background:rgba(255,255,255,.05)}.lg .it.off{opacity:.32}
  .lg .sw{width:10px;height:10px;border-radius:50%;flex:none}
  .lg .n{margin-left:auto;font-family:var(--mono);font-size:.62rem;color:var(--muted)}
  .lg .grad{height:10px;border-radius:5px;margin:.2rem 0 .3rem}
  .lg .ends{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.62rem;color:var(--muted)}
  .sel .t{font-weight:700;font-size:.95rem;line-height:1.25}
  .sel .a{font-size:.8rem;color:var(--muted);margin:.15rem 0 .5rem}
  .sel .chips{display:flex;flex-wrap:wrap;gap:.3rem;margin-bottom:.6rem}
  .sel .chip{font-family:var(--mono);font-size:.6rem;color:var(--muted);border:1px solid var(--line);border-radius:99px;padding:.14rem .5rem}
  .sel .chip b{color:var(--fg);font-weight:400}
  .sel .acts{display:flex;gap:.4rem}
  .sel .acts a,.sel .acts button{font-family:var(--mono);font-size:.66rem;text-transform:uppercase;letter-spacing:.08em;
    border:1px solid var(--line);border-radius:7px;background:none;color:var(--fg);padding:.4rem .7rem;cursor:pointer;text-decoration:none}
  .sel .acts .play{background:var(--accent);border-color:var(--accent);color:#0a0a0c;font-weight:700}
  .sel .acts a:hover,.sel .acts button:hover{border-color:var(--accent)}
  .nbrs{margin-top:.6rem;border-top:1px solid var(--line);padding-top:.5rem}
  .nbrs .nh{font-family:var(--mono);font-size:.58rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:.25rem}
  .tip{position:absolute;pointer-events:none;z-index:6;background:rgba(10,10,12,.92);border:1px solid var(--line);
    border-radius:8px;padding:.45rem .65rem;font-size:.76rem;max-width:280px;display:none}
  .tip .ta{color:var(--muted);font-size:.68rem}
  .hcount{position:absolute;left:1.2rem;bottom:1rem;font-family:var(--mono);font-size:.64rem;color:var(--muted);z-index:5}
  @media(max-width:900px){
    .hyper{height:calc(100dvh - 108px)}
    .hudl{left:.5rem;right:.5rem;width:auto}
    .hud{top:auto;bottom:0;right:0;left:0;width:auto;flex-direction:row;overflow-x:auto;padding:.6rem;gap:.5rem}
    .hud .box{flex:none;width:230px;max-height:34vh;overflow-y:auto}
  }
</style>
<div class=hyper id=hyper>
  <div class=tip id=tip></div>
  <div class=hudl>
    <div class=box>
      <h3>Find a track</h3>
      <input class=hin id=q placeholder="search artist / title…" autocomplete=off>
      <div id=qdrop class=drop></div>
    </div>
    <div class=box>
      <h3>Describe a sound</h3>
      {% if identity.can_query %}
      <form id=clapform class=claprow>
        <input class=hin id=claptext placeholder="hypnotic melodic techno…" autocomplete=off>
        <button class=go type=submit>👂</button>
      </form>
      <div id=clapnote class=note></div>
      <div id=clapres class=drop></div>
      {% else %}<div class=note>Query access is not enabled for this account.</div>{% endif %}
    </div>
    <div class="box sel" id=selbox style="display:none">
      <h3>Selected</h3>
      <div class=t id=selt></div><div class=a id=sela></div>
      <div class=chips id=selchips></div>
      <div class=acts>
        <button class=play id=selplay>▶ Play</button>
        <a id=sellink target=_blank rel=noopener>Spotify ↗</a>
      </div>
      <div class=nbrs id=selnbrs style="display:none"><div class=nh>sounds nearest to</div><div class=drop id=nbrlist></div></div>
    </div>
  </div>
  <div class=hud>
    <div class=box>
      <h3>Space</h3>
      <div class=modes id=spaces>
        <button data-s=effnet class=on title="genre-shaped (discogs-effnet)">effnet</button><button data-s=clap title="description-shaped (CLAP)">clap</button>
      </div>
    </div>
    <div class=box>
      <h3>Color by</h3>
      <div class=modes id=modes>
        <button data-m=style class=on>Style</button><button data-m=era>Era</button><button data-m=cluster>Cluster</button><button data-m=plays>Plays</button><button data-m=rarity>Rarity</button><button data-m=label>Label</button>
      </div>
    </div>
    <div class=box>
      <h3>Filter</h3>
      <div class=modes id=plays>
        <button data-mp=0 class=on>all</button><button data-mp=5>5+</button><button data-mp=20>20+</button><button data-mp=50>50+</button>
      </div>
      <label class=likedrow><input type=checkbox id=likedonly> liked only</label>
      <label class=likedrow><input type=checkbox id=autoplay> autoplay on click</label>
    </div>
    <div class=box><h3 id=lgtitle>Styles</h3><div class=lg id=legend></div></div>
  </div>
  <div class=hcount id=hcount></div>
</div>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/* Categorical palette: validated (CVD worst-adjacent ΔE 26, ≥3:1 on #0a0a0c,
   lightness spread ≤0.10 for equal salience). Slot 9 = neutral "Other". */
const PAL = ['#b2d916','#ff70a0','#38d6f5','#ffb224','#b39dff','#2fe6a8','#ff8a4d','#8fa9ff','#5b5f6b'];
const ERA_RAMP = ['#4d5a0c','#6d8210','#8fa913','#b2d916','#d3ef52','#f2ffa8'];
const PINK_RAMP = ['#3d1120','#6b1e3a','#992b54','#c73f72','#f0568e','#ff9dc0'];
const NOISE = '#3a3d47';
const AUTOPLAY_KEY = 'acetate_hs_autoplay';
let autoplay = localStorage.getItem(AUTOPLAY_KEY) !== '0';

const el = (id) => document.getElementById(id);
const esc = (s) => (s || '').replace(/[&<>"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

/* ---------- scene ---------- */
const holder = el('hyper');
const scene = new THREE.Scene();
scene.background = new THREE.Color('#0a0a0c');
const camera = new THREE.PerspectiveCamera(55, holder.clientWidth/holder.clientHeight, 0.1, 4000);
camera.position.set(0, 40, 260);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(holder.clientWidth, holder.clientHeight);
holder.prepend(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = .06;
controls.autoRotate = true; controls.autoRotateSpeed = .5;
controls.addEventListener('start', () => controls.autoRotate = false);

const mat = new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  uniforms: { uPx: { value: Math.min(devicePixelRatio, 2) } },
  vertexShader: 'uniform float uPx; attribute float size; attribute float dim; varying vec3 vC; varying float vD;\n' +
    'void main(){ vC = color; vD = dim; vec4 mv = modelViewMatrix * vec4(position, 1.0);\n' +
    'gl_PointSize = size * uPx * (150.0 / -mv.z); gl_Position = projectionMatrix * mv; }',
  fragmentShader: 'varying vec3 vC; varying float vD;\n' +
    'void main(){ float d = length(gl_PointCoord - vec2(0.5)); if (d > 0.5) discard;\n' +
    'float a = smoothstep(0.5, 0.1, d) * (0.95 - 0.85 * vD); gl_FragColor = vec4(vC, a); }',
  vertexColors: true,
});
const ring = new THREE.Mesh(new THREE.RingGeometry(2.2, 2.6, 40),
  new THREE.MeshBasicMaterial({color:'#c8ff1f', side:THREE.DoubleSide, transparent:true, opacity:.9}));
ring.visible = false; scene.add(ring);

/* ---------- state ---------- */
let data = null, pts = [], N = 0, geo = null, cloud = null, byId = new Map();
let mode = 'style', space = 'effnet', hidden = new Set();
let likedOnly = false, minPlays = 0, clapSet = null, searchSet = null;
let hover = -1, sel = -1, fly = null;
let col = null, dim = null, maxPlays = 1, maxLi = 1;

/* ---------- coloring ---------- */
const c3 = (hex) => new THREE.Color(hex);
const eraCol = (e) => {
  if (!e) return c3(NOISE);
  const t = Math.max(0, Math.min(1, (e - data.years[0]) / Math.max(1, data.years[1] - data.years[0])));
  const f = t * (ERA_RAMP.length - 1), i = Math.floor(f);
  return c3(ERA_RAMP[i]).lerp(c3(ERA_RAMP[Math.min(i+1, ERA_RAMP.length-1)]), f - i);
};
const clusterCol = (c) => {
  if (c < 0) return c3(NOISE);
  return c3(PAL[c % 8]).multiplyScalar(1 - .16 * (Math.floor(c / 8) % 3));
};
const rampCol = (ramp, t) => {
  const f = Math.max(0, Math.min(1, t)) * (ramp.length - 1), i = Math.floor(f);
  return c3(ramp[i]).lerp(c3(ramp[Math.min(i+1, ramp.length-1)]), f - i);
};
const playsCol = (p) => rampCol(ERA_RAMP, Math.log(1 + p.p) / Math.log(1 + maxPlays));
const rarityCol = (p) => p.li == null ? c3(NOISE)
  : rampCol(PINK_RAMP, 1 - Math.log(1 + p.li) / Math.log(1 + maxLi));
const labelCol = (p) => c3(PAL[p.lb]);
function dimOf(p) {
  if (mode === 'style' && hidden.size && hidden.has(p.g)) return 1;
  if (mode === 'label' && hidden.size && hidden.has(p.lb)) return 1;
  if (likedOnly && !p.l) return 1;
  if (p.p < minPlays) return 1;
  if (clapSet && !clapSet.has(p.i)) return 1;
  if (searchSet && !searchSet.has(p.i)) return 1;
  return 0;
}
function paint() {
  if (!geo) return;
  for (let i = 0; i < N; i++) {
    const p = pts[i];
    const c = mode === 'style' ? c3(PAL[p.g]) : mode === 'era' ? eraCol(p.e)
      : mode === 'plays' ? playsCol(p) : mode === 'rarity' ? rarityCol(p)
      : mode === 'label' ? labelCol(p) : clusterCol(p.c);
    col[i*3] = c.r; col[i*3+1] = c.g; col[i*3+2] = c.b;
    dim[i] = dimOf(p);
  }
  geo.attributes.color.needsUpdate = true;
  geo.attributes.dim.needsUpdate = true;
}

/* ---------- build / space switch ---------- */
function buildCloud(d) {
  data = d; pts = d.points; N = pts.length;
  byId = new Map(); pts.forEach((p, i) => byId.set(p.i, i));
  maxPlays = 1; maxLi = 1;
  pts.forEach(p => { if (p.p > maxPlays) maxPlays = p.p;
                     if (p.li != null && p.li > maxLi) maxLi = p.li; });
  searchSet = null; el('qdrop').innerHTML = '';
  if (cloud) { scene.remove(cloud); geo.dispose(); }
  const pos = new Float32Array(N*3), size = new Float32Array(N);
  col = new Float32Array(N*3); dim = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const p = pts[i];
    pos[i*3] = p.x; pos[i*3+1] = p.y; pos[i*3+2] = p.z;
    size[i] = Math.min(9, 1.7 + Math.sqrt(p.p || 1) * .55);
  }
  geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.setAttribute('size', new THREE.BufferAttribute(size, 1));
  geo.setAttribute('dim', new THREE.BufferAttribute(dim, 1));
  cloud = new THREE.Points(geo, mat);
  scene.add(cloud);
  hidden.clear(); clapSet = null;
  el('clapres').innerHTML = ''; el('clapnote').textContent = '';
  selHide();
  legend(); paint();
  el('hcount').textContent = N.toLocaleString() + ' tracks · ' + d.clusters +
    ' clusters · ' + space + ' space · drag to fly, click a star to listen';
}
async function loadSpace(s) {
  space = s;
  el('spaces').querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.s === s));
  const d = await (await fetch('/api/hyperspace?space=' + s)).json();
  buildCloud(d);
}

/* ---------- legend ---------- */
function legend() {
  const box = el('legend');
  if (mode === 'style') {
    el('lgtitle').textContent = 'Styles';
    const counts = Array(data.styles.length).fill(0);
    pts.forEach(p => counts[p.g]++);
    box.innerHTML = data.styles.map((s, i) =>
      '<div class="it' + (hidden.has(i) ? ' off' : '') + '" data-g="' + i + '"><span class=sw style="background:' +
      PAL[i] + '"></span>' + esc(s) + '<span class=n>' + counts[i].toLocaleString() + '</span></div>').join('');
    box.querySelectorAll('.it').forEach(it => it.onclick = () => {
      const g = +it.dataset.g;
      hidden.has(g) ? hidden.delete(g) : hidden.add(g);
      legend(); paint();
    });
  } else if (mode === 'era') {
    el('lgtitle').textContent = 'First played';
    box.innerHTML = '<div class=grad style="background:linear-gradient(90deg,' + ERA_RAMP.join(',') +
      ')"></div><div class=ends><span>' + data.years[0] + '</span><span>' + data.years[1] + '</span></div>';
  } else if (mode === 'plays') {
    el('lgtitle').textContent = 'Play count';
    box.innerHTML = '<div class=grad style="background:linear-gradient(90deg,' + ERA_RAMP.join(',') +
      ')"></div><div class=ends><span>1 play</span><span>' + maxPlays.toLocaleString() + '</span></div>';
  } else if (mode === 'rarity') {
    el('lgtitle').textContent = 'Rarity (global Last.fm listeners)';
    box.innerHTML = '<div class=grad style="background:linear-gradient(90deg,' + PINK_RAMP.join(',') +
      ')"></div><div class=ends><span>mainstream</span><span>deep cut</span></div>' +
      '<div class=it style="color:var(--muted);font-size:.7rem">gray = no listener data</div>';
  } else if (mode === 'label') {
    el('lgtitle').textContent = 'Record label';
    const counts = Array(data.labels.length).fill(0);
    pts.forEach(p => counts[p.lb]++);
    box.innerHTML = data.labels.map((s, i) =>
      '<div class="it' + (hidden.has(i) ? ' off' : '') + '" data-g="' + i + '"><span class=sw style="background:' +
      PAL[i] + '"></span>' + esc(s) + '<span class=n>' + counts[i].toLocaleString() + '</span></div>').join('');
    box.querySelectorAll('.it').forEach(it => it.onclick = () => {
      const g = +it.dataset.g;
      hidden.has(g) ? hidden.delete(g) : hidden.add(g);
      legend(); paint();
    });
  } else {
    el('lgtitle').textContent = 'Clusters';
    box.innerHTML = '<div class=it><span class=sw style="background:' + NOISE + '"></span>unclustered' +
      '</div><div class=it style="color:var(--muted);font-size:.72rem">' + data.clusters +
      ' HDBSCAN clusters in this space — hover for ids</div>';
  }
}
el('modes').querySelectorAll('button').forEach(b => b.onclick = () => {
  mode = b.dataset.m; hidden.clear();
  el('modes').querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
  legend(); paint();
});
el('spaces').querySelectorAll('button').forEach(b => b.onclick = () => {
  if (b.dataset.s !== space) loadSpace(b.dataset.s);
});
el('plays').querySelectorAll('button').forEach(b => b.onclick = () => {
  minPlays = +b.dataset.mp;
  el('plays').querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
  paint();
});
el('likedonly').onchange = (ev) => { likedOnly = ev.target.checked; paint(); };
el('autoplay').checked = autoplay;
el('autoplay').onchange = (ev) => {
  autoplay = ev.target.checked;
  localStorage.setItem(AUTOPLAY_KEY, autoplay ? '1' : '0');
};

/* ---------- fly-to ---------- */
function flyTo(i) {
  const p = pts[i];
  const target = new THREE.Vector3(p.x, p.y, p.z);
  const dir = camera.position.clone().sub(controls.target).normalize();
  fly = { t: 0, fromT: controls.target.clone(), toT: target,
          fromP: camera.position.clone(), toP: target.clone().add(dir.multiplyScalar(46)) };
  controls.autoRotate = false;
}

/* ---------- selection + detail sidebar ---------- */
const audio = new Audio(); let playing = false;
audio.onended = () => { playing = false; el('selplay').textContent = '▶ Play'; };
function selHide() { sel = -1; ring.visible = false; el('selbox').style.display = 'none'; }
function select(i, doFly) {
  sel = i; const p = pts[i];
  el('selbox').style.display = 'block';
  el('selt').textContent = p.t; el('sela').textContent = p.a;
  el('selchips').innerHTML = '<span class=chip>' + esc((p.s || 'unknown')) + '</span>' +
    '<span class=chip>' + p.p + ' plays</span>' + (p.l ? '<span class=chip>♥ liked</span>' : '');
  el('sellink').href = 'https://open.spotify.com/track/' + p.i;
  el('selnbrs').style.display = 'none';
  ring.position.set(p.x, p.y, p.z); ring.visible = true;
  el('selplay').style.display = p.u ? '' : 'none';
  if (p.u) {
    audio.src = '/preview/' + p.u.slice(0, 2) + '/' + p.u + '.mp3';
    if (autoplay) { audio.play().catch(() => 0); playing = true; el('selplay').textContent = '⏸ Pause'; }
    else { audio.pause(); playing = false; el('selplay').textContent = '▶ Play'; }
  }
  if (doFly) flyTo(i);
  fetchDetail(p);
}
async function fetchDetail(p) {
  try {
    const d = await (await fetch('/api/hyperspace/detail?id=' + encodeURIComponent(p.i) +
                                 '&space=' + space)).json();
    if (sel < 0 || pts[sel].i !== p.i) return;  // stale response
    const chips = [];
    if (d.styles && d.styles.length) chips.push('<span class=chip>' + esc(d.styles[0].style) + ' <b>' + d.styles[0].score + '</b></span>');
    if (d.first) chips.push('<span class=chip>first <b>' + d.first + '</b></span>');
    chips.push('<span class=chip><b>' + d.plays + '</b> plays</span>');
    if (d.bpm) chips.push('<span class=chip><b>' + d.bpm + '</b> bpm</span>');
    if (d.label) chips.push('<span class=chip>' + esc(d.label) + '</span>');
    if (d.listeners != null) chips.push('<span class=chip><b>' + d.listeners.toLocaleString() + '</b> listeners</span>');
    if (d.released) chips.push('<span class=chip>rel <b>' + d.released + '</b></span>');
    if (d.liked) chips.push('<span class=chip>♥ liked</span>');
    el('selchips').innerHTML = chips.join('');
    const nl = el('nbrlist'); nl.innerHTML = '';
    (d.neighbors || []).forEach(nb => {
      const r = document.createElement('div'); r.className = 'row2';
      r.innerHTML = esc(nb.title) + '<em>' + esc(nb.artist) + ' · cos ' + nb.score.toFixed(2) + '</em>';
      r.onclick = () => { const j = byId.get(nb.i); if (j != null) select(j, true); };
      nl.appendChild(r);
    });
    el('selnbrs').style.display = (d.neighbors || []).length ? '' : 'none';
  } catch (e) { /* sidebar stays basic */ }
}
el('selplay').onclick = () => {
  if (playing) { audio.pause(); playing = false; el('selplay').textContent = '▶ Play'; }
  else { audio.play().catch(() => 0); playing = true; el('selplay').textContent = '⏸ Pause'; }
};

/* ---------- track search (client-side, instant) ----------
   typing lists matches (click one → fly there); ENTER lights up the whole
   matching group in the cloud and dims everything else. */
function clearSearchSet() { searchSet = null; paint(); }
el('q').addEventListener('input', () => {
  const q = el('q').value.trim().toLowerCase();
  const box = el('qdrop'); box.innerHTML = '';
  if (searchSet) clearSearchSet();
  if (q.length < 2) return;
  let n = 0;
  for (let i = 0; i < N && n < 12; i++) {
    const p = pts[i];
    if ((p.a + ' ' + p.t).toLowerCase().includes(q)) {
      n++;
      const r = document.createElement('div'); r.className = 'row2';
      r.innerHTML = esc(p.t) + '<em>' + esc(p.a) + ' · ' + p.p + ' plays</em>';
      r.onclick = () => { box.innerHTML = ''; select(i, true); };
      box.appendChild(r);
    }
  }
  if (!n) box.innerHTML = '<div class=row2><em>no match in this space (yet — the backfill grows it)</em></div>';
});
el('q').addEventListener('keydown', (ev) => {
  if (ev.key !== 'Enter') return;
  ev.preventDefault();
  const q = el('q').value.trim().toLowerCase();
  const box = el('qdrop'); box.innerHTML = '';
  if (q.length < 2) return;
  searchSet = new Set();
  pts.forEach(p => { if ((p.a + ' ' + p.t).toLowerCase().includes(q)) searchSet.add(p.i); });
  if (!searchSet.size) { searchSet = null; box.innerHTML = '<div class=row2><em>nothing matches</em></div>'; return; }
  paint();
  const clear = document.createElement('div'); clear.className = 'row2';
  clear.innerHTML = '<b>' + searchSet.size.toLocaleString() + '</b> lit in the cloud <em>✕ click to clear (or edit the search)</em>';
  clear.onclick = () => { clearSearchSet(); box.innerHTML = ''; el('q').value = ''; };
  box.appendChild(clear);
});

/* ---------- CLAP text query ---------- */
if (el('clapform')) el('clapform').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const text = el('claptext').value.trim();
  if (!text) return;
  el('clapnote').textContent = 'listening… (~15s: the model has to wake up)';
  el('clapres').innerHTML = '';
  try {
    const fd = new FormData(); fd.append('text', text);
    const d = await (await fetch('/api/hyperspace/clapquery', { method: 'POST', body: fd })).json();
    if (d.error) { el('clapnote').textContent = d.error; return; }
    clapSet = new Set(d.results.map(x => x.i));
    paint();
    el('clapnote').textContent = 'top ' + d.results.length + ' matches lit, everything else dimmed';
    const box = el('clapres');
    const clear = document.createElement('div'); clear.className = 'row2';
    clear.innerHTML = '<em>✕ clear this search</em>';
    clear.onclick = () => { clapSet = null; paint(); box.innerHTML = ''; el('clapnote').textContent = ''; };
    box.appendChild(clear);
    d.results.slice(0, 10).forEach(x => {
      const r = document.createElement('div'); r.className = 'row2';
      r.innerHTML = esc(x.title) + '<em>' + esc(x.artist) + ' · ' + x.s.toFixed(2) + '</em>';
      r.onclick = () => { const j = byId.get(x.i); if (j != null) select(j, true); };
      box.appendChild(r);
    });
  } catch (e) { el('clapnote').textContent = 'query failed'; }
});

/* ---------- picking / tooltip ---------- */
const ray = new THREE.Raycaster(); ray.params.Points.threshold = 1.6;
const ndc = new THREE.Vector2();
const tip = el('tip');
renderer.domElement.addEventListener('pointermove', (ev) => {
  if (!cloud) return;
  const r = renderer.domElement.getBoundingClientRect();
  ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const hit = ray.intersectObject(cloud)[0];
  hover = hit ? hit.index : -1;
  if (hover >= 0) {
    const p = pts[hover];
    tip.style.display = 'block';
    tip.style.left = (ev.clientX - r.left + 14) + 'px';
    tip.style.top = (ev.clientY - r.top + 10) + 'px';
    tip.innerHTML = '<b>' + esc(p.t) + '</b><div class=ta>' + esc(p.a) + ' · ' + esc(p.s || '?') +
      ' · ' + (p.e || '?') + ' · ' + p.p + ' plays' + (p.l ? ' · ♥' : '') + '</div>';
    renderer.domElement.style.cursor = 'pointer';
  } else { tip.style.display = 'none'; renderer.domElement.style.cursor = 'grab'; }
});
renderer.domElement.addEventListener('click', () => { if (hover >= 0) select(hover, false); });

/* ---------- loop ---------- */
addEventListener('resize', () => {
  camera.aspect = holder.clientWidth / holder.clientHeight; camera.updateProjectionMatrix();
  renderer.setSize(holder.clientWidth, holder.clientHeight);
});
renderer.setAnimationLoop(() => {
  if (fly) {
    fly.t += 0.03;
    const e = 1 - Math.pow(1 - Math.min(fly.t, 1), 3);
    controls.target.lerpVectors(fly.fromT, fly.toT, e);
    camera.position.lerpVectors(fly.fromP, fly.toP, e);
    if (fly.t >= 1) fly = null;
  }
  controls.update();
  if (ring.visible) ring.lookAt(camera.position);
  renderer.render(scene, camera);
});

await loadSpace('effnet');
</script>
{% endif %}
{% endblock %}"""

REVIEW = r"""{% extends "base" %}{% block content %}
<h1 class=rise>Review queue</h1>
<p class="lede rise">Every consumer — playlist health, discovery, genre classification — files its
suggestions here. Accept or reject; each verdict is also a training label for the taste model.
Accepted changes to <b>your</b> playlists are yours to apply (acetate never edits a playlist it
didn't create); accepted genre/discovery adds flow into the next build.</p>
<div class="sb rise" style="margin-bottom:2rem">
  <span>pending <b class=acc>{{ counts.get('proposed', 0) }}</b></span>
  <span>accepted <b>{{ counts.get('accepted', 0) }}</b></span>
  <span>rejected <b>{{ counts.get('rejected', 0) }}</b></span>
</div>
{% for group in groups %}
<section class=rise><h2>{{ group.name }}</h2>
  <ol>{% for p in group.rows %}
    <li class="row proposalrow" id="prop-{{ p.id }}" style="grid-template-columns:auto 1fr auto auto">
      <span class="pact {{ p.action }}">{{ '+' if p.action=='add' else '−' }}</span>
      <div class=ti>
        <div class=t>{{ p.artist }} — {{ p.title }}
          {% if p.kind=='candidate' %}<span class=auto>outside library</span>{% endif %}</div>
        <div class=a>{{ p.reasons|join(' · ') }} <span class=auto>via {{ p.source }}</span></div>
      </div>
      {% if p.preview_url %}<audio controls preload=none src="{{ p.preview_url }}" style="height:28px;max-width:200px"></audio>
      {% else %}<span></span>{% endif %}
      <span class=smartbtns style="margin:0">
        {% if identity.can_admin %}
        <button hx-post="/review/{{ p.id }}/accept" hx-target="#prop-{{ p.id }}" hx-swap="outerHTML">✓</button>
        <button class=danger hx-post="/review/{{ p.id }}/reject" hx-target="#prop-{{ p.id }}" hx-swap="outerHTML">✕</button>
        {% endif %}
      </span>
    </li>{% endfor %}</ol>
</section>
{% else %}<p class="lede rise">Queue is empty — run a health scan, discovery pass, or genre
classify to fill it.</p>{% endfor %}
{% if feedback %}
<section class=rise><h2>self-grading — surfaced vs played after</h2>
  <table class=sched><tr><th>surface</th><th>recommended</th><th>played after</th><th>hit rate</th></tr>
  {% for f in feedback %}<tr><td>{{ f.surface }}</td><td>{{ f.recommended }}</td>
    <td>{{ f.played_after }}</td><td class=when>{{ '%.0f%%'|format(f.hit_rate*100) }}</td></tr>{% endfor %}
  </table>
</section>{% endif %}
<style>
  .pact{font-family:var(--mono);font-weight:700;width:1.6rem;height:1.6rem;border-radius:6px;display:flex;align-items:center;justify-content:center}
  .pact.add{color:var(--accent);border:1px solid var(--accent)}
  .pact.remove{color:var(--accent2);border:1px solid var(--accent2)}
  .proposalrow{padding:.6rem}
</style>
{% endblock %}"""

FILING = r"""{% extends "base" %}{% block content %}
<h1 class=rise>Filing</h1>
<p class="lede rise">Songs you liked but never put in a playlist — most-played first — with the three
playlists each one fits best (sonic membership). One click files it; every add is append-only,
logged, and reversible. <b>This is the only place acetate writes to a playlist you made</b> — it
never removes or reorders anything except an add you undo here.</p>
<div class="sb rise" style="margin-bottom:1.6rem">
  <span>unfiled likes <b class=acc>{{ unfiled_total }}</b></span>
  <span>shown <b>{{ queue|length }}</b></span>
  <span>filed <b>{{ filed_count }}</b></span>
</div>
{% if queue %}
<section class=rise><ol id=filequeue>
  {% for it in queue %}
  <li class="row filerow" style="grid-template-columns:auto 1fr auto">
    <span class=plays title="all-time plays">{{ it.plays }}▶</span>
    <div class=ti>
      <div class=t>{{ it.artist }} — {{ it.title }}</div>
      <div class=a>files into:
        {% if identity.can_admin %}{% for t in it.targets %}
        <button class="chip {{ 'strong' if t.strong }}"
          hx-post="/filing/file"
          hx-vals='{{ {"track_id": it.track_id, "playlist_id": t.playlist_id, "score": t.score}|tojson }}'
          hx-target="closest li" hx-swap="outerHTML"
          title="membership {{ t.score }}{{ ' · strong fit' if t.strong }}">{{ t.name }}<span class=sc>{{ t.score }}</span></button>
        {% endfor %}{% else %}{% for t in it.targets %}<span class=chip>{{ t.name }} {{ t.score }}</span>{% endfor %}{% endif %}
      </div>
    </div>
    {% if it.preview_url %}<audio controls preload=none src="{{ it.preview_url }}" style="height:28px;max-width:180px"></audio>{% else %}<span></span>{% endif %}
  </li>
  {% endfor %}
</ol></section>
{% else %}<p class="lede rise">Nothing to file — every liked song with a sonic match is already in a playlist. 🎉</p>{% endif %}

<section class=rise style="margin-top:2.4rem"><h2>Filing ledger</h2>
<p class=lede>Every add acetate made to your playlists, newest first. Undo removes exactly that track.</p>
{% if ledger %}<ol id=ledger>
  {% for a in ledger %}
  <li class="row ledgerrow {{ 'undone' if a.status=='undone' }}" style="grid-template-columns:auto 1fr auto">
    <span class="pact {{ 'remove' if a.status=='undone' else 'add' }}">{{ '↩' if a.status=='undone' else '+' }}</span>
    <div class=ti>
      <div class=t>{{ a.artist }} — {{ a.title }}</div>
      <div class=a>{{ '→ ' if a.status=='added' else 'was in ' }}<b>{{ a.playlist_name }}</b>
        <span class=auto>{{ a.added_at|localtime }}{% if a.status=='undone' %} · undone {{ a.undone_at|localtime }}{% endif %}</span></div>
    </div>
    {% if a.status=='added' and identity.can_admin %}
    <button class="chip danger" hx-post="/filing/undo/{{ a.id }}" hx-target="closest li" hx-swap="outerHTML">undo</button>
    {% else %}<span class=auto>{{ a.status }}</span>{% endif %}
  </li>
  {% endfor %}
</ol>{% else %}<p class=lede>No filings yet.</p>{% endif %}
</section>
<style>
  .filerow,.ledgerrow{padding:.55rem .7rem}
  .plays{font-family:var(--mono);font-size:.72rem;color:var(--muted);min-width:2.6rem}
  .chip{display:inline-flex;align-items:center;gap:.4rem;background:transparent;color:var(--fg);
    border:1px solid var(--muted);border-radius:20px;font-family:var(--mono);font-size:.72rem;
    padding:.22rem .6rem;margin:.1rem .2rem .1rem 0;cursor:pointer}
  .chip.strong{border-color:var(--accent);color:var(--accent)}
  .chip .sc{color:var(--muted);font-size:.66rem}
  .chip.danger{border-color:var(--accent2);color:var(--accent2)}
  .chip:hover{background:rgba(200,255,31,.08)}
  .ledgerrow.undone{opacity:.55}
</style>
{% endblock %}"""

HEALTH = r"""{% extends "base" %}{% block content %}
<h1 class=rise>Playlist health</h1>
<p class="lede rise">Your authored playlists as regions of the embedding space. Cohesion is the mean
member similarity to the playlist's sonic center — click through for misfits, liked-song matches, and split checks.
Choose which playlists matter first on <a href="/playlists" style="color:var(--accent)">Your playlists</a>.</p>
<div class="btns rise">
  {% if identity.can_admin %}<button hx-post=/health/build hx-target=#toast>Rebuild regions</button><span id=toast></span>{% endif %}
</div>
<section class=rise>
  <table class=sched><tr><th>playlist</th><th>tracks</th><th>embedded</th><th>cohesion</th><th>built</th></tr>
  {% for r in regions %}
    <tr><td><a href="/health/{{ r.playlist_id }}" style="color:var(--accent);text-decoration:none">{{ r.name }}</a></td>
    <td>{{ r.n_members }}</td><td>{{ r.n_embedded }}</td>
    <td class=when>{{ '%.3f'|format(r.spread) if r.spread else '—' }}</td>
    <td>{{ r.built_at|localtime('%Y-%m-%d') }}</td></tr>
  {% else %}<tr><td colspan=5>No regions yet — hit “Rebuild regions” (needs embedded playlists).</td></tr>{% endfor %}
  </table>
</section>
{% endblock %}"""

TOOLS_PAGE = r"""{% extends "base" %}{% block content %}
<h1 class=rise>Tools</h1>
<p class="lede rise">The same capabilities power this website and the chat agent. Every workflow
has one shared contract, one deterministic implementation, and one documentation entry.</p>
<section class=rise>
  <div class=pl>{% for workflow in workflows %}
    <a class=card id="{{ workflow.id }}" href="{{ workflow.web_path }}">
      <div class=pt>{{ workflow.title }}</div>
      <div class=pm>{{ workflow.summary }}<br><span class=acc>{{ workflow.access }}</span>
        · agent: {{ workflow.tool_name }}</div>
    </a>
  {% endfor %}</div>
</section>
<section class=rise id=discover-artists><h2>discover artists from a playlist</h2>
  <p class=lede>Each playlist track votes for artists in its cached neighbors. Artists win by
  matching several distinct seeds, not by flooding one seed with a large catalog.</p>
  {% if identity.can_query %}<form class=askform hx-post=/tools/discover-artists hx-target=#artist-results hx-indicator=#artist-spin>
    <select class=psel name=playlist required aria-label="Playlist">
      <option value="">Choose a playlist…</option>
      {% for playlist in playlists %}<option value="{{ playlist.playlist_id }}">{{ playlist.name }}</option>{% endfor %}
    </select>
    <select class=psel name=novelty aria-label="Novelty">
      <option value=outside_library>New to my library</option>
      <option value=outside_playlist>Not in this playlist</option>
    </select>
    <select class=psel name=limit aria-label="Result count">
      {% for n in [10,20,30,50] %}<option value="{{ n }}" {{ 'selected' if n==20 }}>{{ n }} artists</option>{% endfor %}
    </select>
    <button type=submit>Discover</button><span id=artist-spin class=htmx-indicator>scanning neighbors…</span>
  </form>
  {% else %}<div class=callout>Query access is not enabled for this account.</div>{% endif %}
  <div id=artist-results></div>
</section>
{% endblock %}"""

ARTIST_DISCOVERY_RESULTS = r"""{% if result.error %}
<div class=callout style="margin-top:1rem"><b>{{ result.error }}</b>{% if result.matches %} —
  {{ result.matches|join(', ') }}{% endif %}</div>
{% elif not result.artists %}
<div class=callout style="margin-top:1rem">{{ result.reason or 'No matching artists yet.' }}</div>
{% else %}
<div class="sb" style="margin:1.2rem 0">
  <span>playlist tracks <b>{{ result.playlist_recordings }}</b></span>
  <span>artists considered <b>{{ result.artists_considered }}</b></span>
  <span>novelty <b class=acc>{{ result.novelty.replace('_', ' ') }}</b></span>
</div>
<ol>{% for artist in result.artists %}<li class=row style="grid-template-columns:auto minmax(0,1fr) auto">
  <span class=rank>{{ loop.index }}</span>
  <div class=ti><div class=t>{{ artist.artist }}</div>
    <div class=a>{{ artist.seed_count }} playlist tracks voted · {{ '%.1f'|format(artist.coverage*100) }}% coverage ·
      mean match {{ '%.3f'|format(artist.mean_similarity) }}</div>
    <details class=steps><summary>why this artist</summary>
      {% for edge in artist.connections %}<div class=sql>{{ edge.seed_artist }} — {{ edge.seed_title }}
        → {{ artist.artist }} — {{ edge.candidate_title }} · {{ '%.3f'|format(edge.strength) }}</div>{% endfor %}
    </details>
  </div>
  <div class=plays>{{ '%.3f'|format(artist.score) }}</div>
</li>{% endfor %}</ol>
{% endif %}"""

HEALTH_DETAIL = r"""{% extends "base" %}{% block content %}
<h1 class=rise>{{ rep.name }}</h1>
<div class="sb rise" style="margin-bottom:2rem">
  <span>tracks <b>{{ rep.members }}</b></span>
  <span>embedded <b>{{ rep.embedded }}</b></span>
  <span>sonic cohesion <b class=acc>{{ rep.sonic_cohesion }}</b></span>
  <span>style entropy <b>{{ rep.style_entropy }}</b></span>
  <span>co-listen density <b>{{ rep.colisten_density }}</b></span>
</div>
<div class="btns rise">
  {% if identity.can_admin %}
  <button hx-post="/health/{{ rep.playlist_id }}/matches" hx-target=#toast>Queue liked matches</button>
  <button hx-post="/health/{{ rep.playlist_id }}/scan" hx-target=#toast>Queue full health pass</button>
  {% endif %}
  <a href="/playlists" style="color:var(--accent);font-family:var(--mono);font-size:.72rem;align-self:center">Categorize playlists</a>
  <span id=toast></span>
</div>
<div class=cols>
<section class=rise><h2>sound profile</h2>
  {% set smx = rep.styles[0][1] if rep.styles else 1 %}
  <div class=gen>{% for s, n in rep.styles %}<div class=g>
    <div class=gl><b>{{ s }}</b><span>{{ n }} tracks</span></div>
    <div class=gb><i style="width:{{ n*100//smx }}%"></i></div></div>{% endfor %}</div>
</section>
<section class=rise><h2>misfits — fit this playlist worst</h2>
  <ol>{% for m in rep.misfits %}<li class=row style="grid-template-columns:auto 1fr auto">
    <span class=rank>{{ m.z }}</span>
    <div class=ti><div class=t>{{ m.artist }} — {{ m.title }}</div></div>
    <div class=plays>{{ '%.3f'|format(m.score) }}</div></li>
  {% else %}<li class=lede>nothing sticks out — coherent playlist.</li>{% endfor %}</ol>
</section>
</div>
<section class=rise><h2>belongs here (liked, not in the playlist)</h2>
  <ol>{% for a in additions %}<li class=row style="grid-template-columns:auto 1fr auto auto">
    <span class=rank>{{ loop.index }}</span>
    <div class=ti><div class=t>{{ a.artist }} — {{ a.title }}</div><div class=a>{{ a.plays }} plays</div></div>
    {% if a.preview_url %}<audio controls preload=none src="{{ a.preview_url }}" style="height:28px;max-width:200px"></audio>{% else %}<span></span>{% endif %}
    <div class=plays>{{ '%.3f'|format(a.score) }}</div></li>{% endfor %}</ol>
</section>
{% if rep.split %}
<section class=rise><h2>split check</h2>
  <div class=callout>This looks like <b>two playlists</b> — clusters of {{ rep.split.groups[0].size }}
  and {{ rep.split.groups[1].size }} tracks would each be tighter (cohesion gain +{{ rep.split.gain }}).</div>
</section>
{% endif %}
{% endblock %}"""

env = Environment(
    loader=DictLoader(
        {
            "base": BASE,
            "macros": MACROS,
            "home": HOME,
            "stats": STATS,
            "wrapped": WRAPPED,
            "gems": GEMS,
            "playlists": PLAYLISTS,
            "pipeline": PIPELINE,
            "playlist_profile_row": PLAYLIST_PROFILE_ROW,
            "playlist_detail": PLAYLIST_DETAIL,
            "proposal_script": PROPOSAL_SCRIPT,
            "smart": SMART,
            "chat": CHAT,
            "chat_turn": CHAT_TURN,
            "hyperspace": HYPERSPACE,
            "review": REVIEW,
            "filing": FILING,
            "health": HEALTH,
            "health_detail": HEALTH_DETAIL,
            "tools": TOOLS_PAGE,
            "artist_discovery_results": ARTIST_DISCOVERY_RESULTS,
            "_sb": STATUSBAR,
            "_status": STATUS_FRAGMENT,
            "runs": RUNS,
            "_recent": RECENT_PLAYS,
        }
    ),
    autoescape=True,
)


def _localtime(value, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Render a stored UTC ISO timestamp in the display tz (ACETATE_TZ, default
    America/New_York). All times in the DB are UTC; this is display-only.

    Shares `timefmt` with the chat agent so the page and the model can never
    disagree about what time it is for the user."""
    if not value:
        return ""
    rendered = timefmt.to_local(str(value), fmt)
    return rendered or str(value)[:16].replace("T", " ")


env.filters["localtime"] = _localtime


def _render(name: str, **ctx) -> str:
    ctx.setdefault("identity", access.identity_context.get())
    ctx.setdefault("nav", nav_for(ctx.get("active")))
    return env.get_template(name).render(**ctx)


# --------------------------------------------------------------------------- #
#  Routes
# --------------------------------------------------------------------------- #


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    conn = db.connect()
    return _render(
        "home",
        title="listening",
        active="home",
        t=stats.totals(conn),
        years=stats.by_year(conn),
        top_tracks=stats.top_tracks(conn, limit=10),
        top_artists=stats.top_artists(conn, limit=10),
        recent=stats.recent_plays(conn, limit=10),
    )


@app.get("/stats", response_class=HTMLResponse)
def stats_page() -> str:
    conn = db.connect()
    since = (date.today() - timedelta(days=90)).isoformat()
    return _render(
        "stats",
        title="stats",
        active="stats",
        moment=stats.top_tracks(conn, since=since, limit=10),
        toy=stats.tracks_of_the_year(conn),
        hours=stats.by_hour(conn),
        weekdays=stats.by_weekday(conn),
        discovery=stats.discovery_by_year(conn),
        genres=stats.top_genres(conn, limit=14),
    )


def _decorate_playlist_profile(row) -> dict:
    from . import playlist_profile

    profile = dict(row)
    profile["has_region"] = bool(profile.get("has_region"))
    profile["suggestion"] = playlist_profile.suggest_label(profile)
    return profile


def _playlist_profile_row(conn, playlist_id: str) -> dict | None:
    row = conn.execute(
        """
        SELECT ps.*, tr.spread AS cohesion,
               CASE WHEN tr.playlist_id IS NULL THEN 0 ELSE 1 END AS has_region
        FROM playlist_stats ps LEFT JOIN taste_regions tr USING(playlist_id)
        WHERE ps.playlist_id=? AND ps.is_own=1 AND ps.is_managed=0""",
        (playlist_id,),
    ).fetchone()
    return _decorate_playlist_profile(row) if row else None


@app.get("/playlists", response_class=HTMLResponse)
def playlists_page(show: str = "all", q: str = "", page: int = 1) -> str:
    from . import playlist_profile

    conn = db.connect()
    if page < 1:
        conn.close()
        raise HTTPException(status_code=404, detail="Playlist page not found")
    if show == "all":
        try:
            return _render(
                "playlists",
                title="playlists",
                active="playlists",
                show=show,
                catalog=playlist_activity.playlist_catalog(conn, query=q, page=page),
            )
        finally:
            conn.close()
    if not conn.execute("SELECT 1 FROM playlist_stats LIMIT 1").fetchone():
        playlist_profile.rebuild(conn)
    valid_views = {"unlabeled", "all", "rotation", "archive", "one-shot", "dormant", "mute"}
    if show not in valid_views:
        show = "unlabeled"
    where, params = ["ps.is_own=1", "ps.is_managed=0"], []
    if show == "unlabeled":
        where.append("ps.user_label IS NULL")
    elif show != "all":
        where.append("ps.persona=?")
        params.append(show)
    profile_rows = conn.execute(
        f"""
        SELECT ps.*, tr.spread AS cohesion,
               CASE WHEN tr.playlist_id IS NULL THEN 0 ELSE 1 END AS has_region
        FROM playlist_stats ps LEFT JOIN taste_regions tr USING(playlist_id)
        WHERE {" AND ".join(where)}
        ORDER BY ps.member_plays_180d DESC, ps.member_plays DESC, lower(ps.name)
        LIMIT 500""",
        params,
    ).fetchall()
    profiles = [_decorate_playlist_profile(r) for r in profile_rows]
    totals_row = conn.execute("""
        SELECT COUNT(*) owned, SUM(user_label IS NOT NULL) labeled,
               SUM(user_label IS NULL) unlabeled
        FROM playlist_stats WHERE is_own=1 AND is_managed=0""").fetchone()
    totals = dict(totals_row)
    rows = conn.execute(
        "SELECT kind, key, name, playlist_id, updated_at FROM managed_playlists "
        "ORDER BY updated_at DESC"
    ).fetchall()
    items = [dict(r) for r in rows]
    for it in items:
        it["url"] = playlist_activity.playlist_url(it["playlist_id"])
        snap = conn.execute(
            "SELECT total_tracks FROM playlist_snapshots WHERE playlist_id=? "
            "ORDER BY backup_time DESC LIMIT 1",
            (it["playlist_id"],),
        ).fetchone()
        if snap and snap["total_tracks"] is not None:
            it["total"] = snap["total_tracks"]
        else:
            it["total"] = conn.execute(
                "SELECT COUNT(*) FROM managed_playlist_tracks "
                "WHERE kind=? AND key=? AND currently_in=1",
                (it["kind"], it["key"]),
            ).fetchone()[0]
    return _render(
        "playlists",
        title="playlists",
        active="playlists",
        items=items,
        profiles=profiles,
        totals=totals,
        show=show,
    )


def _playlist_detail_page(conn, detail: dict | None, page: int) -> str:
    if detail is None or page < 1:
        raise HTTPException(status_code=404, detail="Playlist not found")
    pid = detail.get("playlist_id")
    detail["history_url"] = (
        "/curated?" + urlencode({"view": "history", "playlist_id": pid}) if pid else None
    )
    playlist = playlist_activity.detail_tracks(conn, detail, page=page)
    context = f"Playlist: {json.dumps(playlist['name'], ensure_ascii=False)}"
    if pid:
        context += f"\nSpotify playlist: https://open.spotify.com/playlist/{pid}"
    if playlist.get("saved_id"):
        context += f"\nSaved playlist ID: {playlist['saved_id']}"
    if playlist.get("summary"):
        context += f"\nSelection: {playlist['summary']}"
    if playlist["tracks"]:
        context += "\nExample tracks from this page:\n" + "\n".join(
            f"- {json.dumps(track['artist'] + ' — ' + track['title'], ensure_ascii=False)} "
            f"(https://open.spotify.com/track/{track['track_id']})"
            for track in playlist["tracks"][:6]
            if track["track_id"]
        )
    playlist["ask_url"] = _chat_draft_url(
        "Tell me about this playlist: its sound, listening patterns, and tracks that stand out.\n\n"
        + context
    )
    playlist["more_url"] = (
        _chat_draft_url(
            "Find more music like this playlist. Recommend tracks that are not already in it "
            "and explain why they fit.\n\n" + context
        )
        if playlist["recorded_count"]
        else None
    )
    for track in playlist["tracks"]:
        if not track["track_id"]:
            continue
        seed = (
            f"Track: {json.dumps(track['artist'] + ' — ' + track['title'], ensure_ascii=False)}\n"
            f"Spotify track: https://open.spotify.com/track/{track['track_id']}"
        )
        track["more_url"] = _chat_draft_url(
            "Find more tracks like this one. Compare sound and listening overlap where available, "
            "and explain the fit.\n\n" + seed
        )
        track["ask_url"] = _chat_draft_url(
            "Tell me about this track, using my listening history and the available track data.\n\n"
            + seed
        )
    return _render("playlist_detail", title=detail["name"], active="playlists", playlist=playlist)


def _chat_draft_url(question: str) -> str:
    return "/chat?" + urlencode({"q": question})


@app.get("/playlists/saved/{sid}", response_class=HTMLResponse)
def saved_playlist_page(sid: int, page: int = 1) -> str:
    conn = db.connect()
    try:
        saved = smart.get(conn, sid)
        if not saved:
            raise HTTPException(status_code=404, detail="Saved playlist not found")
        link = conn.execute(
            "SELECT playlist_id FROM managed_playlists WHERE kind='smart' AND key=?", (str(sid),)
        ).fetchone()
        pid = link["playlist_id"] if link else None
        detail = (
            playlist_activity.playlist_detail(conn, pid)
            if pid
            else {
                **playlist_activity.selection_detail(conn, saved["spec"]),
                "playlist_id": None,
                "observed_at": None,
            }
        )
        curated = smart.is_curated(saved["spec"])
        collection = "curated" if curated else "smart"
        summary = smart.describe(saved["spec"])
        if curated and detail.get("observed_at"):
            summary = "Originally selected · " + summary
        detail.update(
            name=saved["name"],
            summary=summary,
            curated=curated,
            saved_id=sid,
            kind=(
                "Curated · fixed selection"
                if curated
                else "Smart rule · refreshes daily"
                if saved["enabled"]
                else "Smart rule · paused"
            ),
            manage_url=f"/{collection}#{collection}-{sid}",
        )
        return _playlist_detail_page(conn, detail, page)
    finally:
        conn.close()


@app.get("/playlists/{playlist_id}", response_class=HTMLResponse)
def playlist_page(playlist_id: str, page: int = 1) -> str:
    conn = db.connect()
    try:
        saved = conn.execute(
            "SELECT s.id FROM managed_playlists m JOIN smart_playlists s "
            "ON m.key=CAST(s.id AS TEXT) WHERE m.kind='smart' AND m.playlist_id=?",
            (playlist_id,),
        ).fetchone()
        if saved and page > 0:
            return RedirectResponse(f"/playlists/saved/{saved['id']}?page={page}", status_code=303)
        return _playlist_detail_page(
            conn, playlist_activity.playlist_detail(conn, playlist_id), page
        )
    finally:
        conn.close()


@app.get("/chat/messages/{message_id}/proposals/{index}", response_class=HTMLResponse)
def draft_playlist_page(message_id: int, index: int, page: int = 1):
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT meta, thread_id FROM chat_messages WHERE id=? AND role='assistant'",
            (message_id,),
        ).fetchone()
        try:
            proposals = json.loads(row["meta"])["proposals"] if row else []
            proposal = proposals[index] if 0 <= index < len(proposals) else None
        except (ValueError, TypeError, KeyError):
            proposal = None
        if not isinstance(proposal, dict) or not isinstance(proposal.get("spec"), dict):
            raise HTTPException(status_code=404, detail="Playlist draft not found")
        sid = chatstore.saved_playlist_id(conn, proposal)
        if sid is not None:
            return RedirectResponse(f"/playlists/saved/{sid}", status_code=303)
        curated = smart.is_curated(proposal["spec"])
        detail = {
            **playlist_activity.selection_detail(conn, proposal["spec"]),
            "name": proposal.get("name") or "Playlist draft",
            "playlist_id": None,
            "summary": proposal.get("summary") or smart.describe(proposal["spec"]),
            "kind": "Draft · fixed selection" if curated else "Draft · smart rule",
            "source": "Draft fixed selection" if curated else "Current draft rule matches",
            "thread_id": row["thread_id"],
            "save_url": f"/chat/messages/{message_id}/proposals/{index}/approve",
        }
        return _playlist_detail_page(conn, detail, page)
    finally:
        conn.close()


@app.post("/playlists/profile", response_class=HTMLResponse)
def playlists_profile_route(
    _identity: access.Identity = Depends(access.require_admin),
) -> HTMLResponse:
    from . import playlist_profile

    res = playlist_profile.rebuild(db.connect())
    return HTMLResponse(
        f"<span id=playlist-toast class=toast>refreshed {res['playlists']} playlists — "
        f'<a href=/playlists style="color:var(--accent)">reload</a></span>'
    )


@app.post("/playlists/label-suggested")
def playlists_label_suggested(
    _identity: access.Identity = Depends(access.require_admin),
) -> RedirectResponse:
    from . import playlist_profile

    conn = db.connect()
    rows = conn.execute("""
        SELECT ps.*, tr.spread AS cohesion
        FROM playlist_stats ps LEFT JOIN taste_regions tr USING(playlist_id)
        WHERE ps.is_own=1 AND ps.is_managed=0 AND ps.user_label IS NULL""").fetchall()
    for row in rows:
        suggestion = playlist_profile.suggest_label(dict(row))
        if suggestion["label"] and suggestion["confidence"] == "high":
            playlist_profile.set_label(
                conn,
                row["playlist_id"],
                suggestion["label"],
                note="Web suggestion: " + "; ".join(suggestion["reasons"]),
            )
    return RedirectResponse("/playlists?show=unlabeled", status_code=303)


@app.post("/playlists/{playlist_id}/label", response_class=HTMLResponse)
def playlists_label_route(
    playlist_id: str,
    label: str = Form(""),
    _identity: access.Identity = Depends(access.require_admin),
) -> HTMLResponse:
    from . import playlist_profile

    conn = db.connect()
    if not _playlist_profile_row(conn, playlist_id):
        raise HTTPException(status_code=404, detail="playlist not found")
    if label:
        if label not in playlist_profile.KNOWN_LABELS:
            raise HTTPException(status_code=422, detail="invalid playlist label")
        playlist_profile.set_label(conn, playlist_id, label)
    else:
        playlist_profile.clear_label(conn, playlist_id)
    profile = _playlist_profile_row(conn, playlist_id)
    return HTMLResponse(_render("playlist_profile_row", profile=profile))


@app.get("/gems", response_class=HTMLResponse)
def gems_page(cat: str = "faded") -> str:
    conn = db.connect()
    if cat not in stats.GEM_CATEGORIES:
        cat = "faded"
    counts = stats.gem_counts(conn)
    cats = [
        {"key": k, "label": m["label"], "count": counts.get(k, 0)}
        for k, m in stats.GEM_CATEGORIES.items()
    ]
    meta = stats.GEM_CATEGORIES[cat]
    return _render(
        "gems",
        title="lost gems",
        active="gems",
        s=stats.lost_gems_summary(conn),
        cats=cats,
        cat=cat,
        desc=meta["desc"],
        heading=meta["heading"],
        gems=stats.gems(conn, cat, limit=80),
    )


@app.get("/pipeline", response_class=HTMLResponse)
def pipeline_page() -> str:
    return _render("pipeline", title="pipeline", active="pipeline", status=jobs.status())


_MONTHS = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def _period_label(kind: str, key: str) -> str:
    if kind == "quarter":
        y, q = key.split("-Q")
        return f"Q{q} {y}"
    if kind == "month":
        y, m = key.split("-")
        return f"{_MONTHS[int(m)]} {y}"
    if kind == "week":
        y, w = key.split("-W")
        return f"Week {int(w)}, {y}"
    return key


@app.get("/wrapped", response_class=HTMLResponse)
def wrapped(kind: str = "year", key: str | None = None) -> str:
    conn = db.connect()
    if kind not in ("year", "quarter", "month", "week"):
        kind = "year"
    periods = stats.available_periods(conn, kind)
    if not periods:
        return _render(
            "wrapped",
            title="wrapped",
            active="wrapped",
            kind=kind,
            key=None,
            label="—",
            periods=[],
            prev_key=None,
            next_key=None,
            s={"plays": 0},
        )
    if key not in periods:
        key = periods[0]
    start, end = stats.period_range(kind, key)
    length = end - start
    s = stats.period_summary(
        conn, start.isoformat(), end.isoformat(), (start - length).isoformat(), start.isoformat()
    )
    i = periods.index(key)
    return _render(
        "wrapped",
        title="wrapped",
        active="wrapped",
        kind=kind,
        key=key,
        label=_period_label(kind, key),
        periods=periods,
        prev_key=(periods[i + 1] if i + 1 < len(periods) else None),
        next_key=(periods[i - 1] if i > 0 else None),
        s=s,
    )


@app.get("/status", response_class=HTMLResponse)
def status_fragment() -> str:
    return _render("_status", s=jobs.status(), running=jobs.running_jobs())


@app.get("/runs", response_class=HTMLResponse)
def runs_fragment() -> str:
    return _render("runs", runs=jobs.recent_runs())


@app.get("/recent", response_class=HTMLResponse)
def recent_fragment() -> str:
    return _render("_recent", recent=stats.recent_plays(db.connect()))


@app.get("/api/status")
def api_status() -> JSONResponse:
    return JSONResponse(jobs.status())


@app.get("/healthz")
def healthz() -> JSONResponse:
    """Liveness + readiness probe: app is up and the SQLite DB is reachable.

    Returns 200 when healthy, 503 otherwise. Intentionally checks no external
    services (Spotify/Last.fm) so it stays a true container-health signal.
    """
    try:
        conn = db.connect()
        try:
            conn.execute("SELECT 1").fetchone()
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("healthz: db check failed: {}", exc)
        return JSONResponse(
            {"status": "unhealthy", "db": "down", "error": str(exc)[:200]},
            status_code=503,
        )
    return JSONResponse({"status": "ok", "db": "up", "time": db.utcnow()})


# --------------------------------------------------------------------------- #
#  Hyperspace: the embedding-space point cloud
# --------------------------------------------------------------------------- #


@app.get("/hyperspace", response_class=HTMLResponse)
def hyperspace_page() -> str:
    conn = db.connect()
    n = conn.execute("SELECT COUNT(*) FROM hyperspace_points").fetchone()[0]
    emb = conn.execute(
        "SELECT COUNT(*) FROM embeddings e JOIN recording_stats rs USING(recording_key) "
        "WHERE e.model='discogs-effnet'"
    ).fetchone()[0]
    return _render(
        "hyperspace", title="hyperspace", active="hyperspace", n_points=n, n_embedded=emb
    )


_SPACE_MODELS = {"effnet": "discogs-effnet", "clap": "clap-music"}


@app.get("/api/hyperspace")
def api_hyperspace(space: str = "effnet") -> JSONResponse:
    """The full point cloud in one payload (~1-4 MB). Style groups are assigned
    server-side: top-8 sub-styles by frequency + 'Other' (slot 9), so the
    client's categorical palette maps by fixed index. `space` picks which
    embedding space's projection to serve (effnet | clap)."""
    if space not in _SPACE_MODELS:
        raise HTTPException(status_code=400)
    conn = db.connect()
    rows = conn.execute(
        """
        SELECT hp.x, hp.y, hp.z, hp.cluster,
               rs.rep_track_id AS tid, rs.artist, rs.title, rs.plays, rs.liked,
               CAST(substr(rs.first_at, 1, 4) AS INTEGER) AS era,
               (SELECT style FROM recording_styles s
                 WHERE s.recording_key = hp.recording_key AND s.rank = 1) AS style,
               (SELECT substr(a.path, 19, 40) FROM audio_cache a
                 WHERE a.recording_key = hp.recording_key AND a.status = 'fetched') AS prev,
               (SELECT li.listeners FROM lastfm_info li
                 WHERE li.recording_key = hp.recording_key) AS listeners,
               (SELECT da.label FROM deezer_track dt LEFT JOIN deezer_album da USING(album_id)
                 WHERE dt.recording_key = hp.recording_key) AS label
        FROM hyperspace_points hp
        JOIN recording_stats rs ON rs.recording_key = hp.recording_key
        WHERE hp.space = ?
    """,
        (space,),
    ).fetchall()
    subs = [(r["style"] or "").split("---")[-1] or None for r in rows]
    from collections import Counter

    top = [s for s, _ in Counter(s for s in subs if s).most_common(8)]
    gidx = {s: i for i, s in enumerate(top)}
    top_labels = [s for s, _ in Counter(r["label"] for r in rows if r["label"]).most_common(8)]
    lidx = {s: i for i, s in enumerate(top_labels)}
    years = [e for r in rows if (e := r["era"])]
    points = [
        {
            "x": round(r["x"], 2),
            "y": round(r["y"], 2),
            "z": round(r["z"], 2),
            "i": r["tid"],
            "a": r["artist"],
            "t": r["title"],
            "p": r["plays"],
            "e": r["era"],
            "s": sub,
            "g": gidx.get(sub, 8),
            "c": r["cluster"],
            "l": r["liked"],
            "u": r["prev"],
            "li": r["listeners"],
            "lb": lidx.get(r["label"], 8),
        }
        for r, sub in zip(rows, subs)
    ]
    n_clusters = max((r["cluster"] for r in rows), default=-1) + 1
    return JSONResponse(
        {
            "points": points,
            "styles": top + ["Other"],
            "space": space,
            "labels": top_labels + ["Other / unknown"],
            "years": [min(years, default=2012), max(years, default=2026)],
            "clusters": n_clusters,
        }
    )


@app.get("/api/hyperspace/detail")
def hyperspace_detail(id: str, space: str = "effnet") -> JSONResponse:
    """Everything we hold on one recording (sidebar payload): stats, styles,
    Last.fm global counts, Deezer bpm/label, MBID, plus nearest neighbors in
    the requested embedding space (cosine over a cached matrix)."""
    conn = db.connect()
    tr = conn.execute(
        "SELECT recording_key FROM track_recording WHERE track_id = ?", (id,)
    ).fetchone()
    if not tr:
        raise HTTPException(status_code=404)
    rk = tr["recording_key"]
    rs = conn.execute("SELECT * FROM recording_stats WHERE recording_key = ?", (rk,)).fetchone()
    styles = [
        {"style": r["style"].split("---")[-1], "score": round(r["score"], 2)}
        for r in conn.execute(
            "SELECT style, score FROM recording_styles "
            "WHERE recording_key = ? ORDER BY rank LIMIT 5",
            (rk,),
        )
    ]
    lf = conn.execute(
        "SELECT listeners, playcount FROM lastfm_info WHERE recording_key = ?", (rk,)
    ).fetchone()
    dz = conn.execute(
        "SELECT dt.bpm, dt.gain, dt.rank, dt.release_date, da.label FROM deezer_track dt "
        "LEFT JOIN deezer_album da USING(album_id) WHERE dt.recording_key = ?",
        (rk,),
    ).fetchone()
    mb = conn.execute(
        "SELECT mbid FROM recording_mbid WHERE recording_key = ? AND status = 'resolved'", (rk,)
    ).fetchone()
    neighbors = []
    try:
        from . import similar as _similar

        for nb in _similar.sonic_neighbors(
            conn, rk, k=8, model=_SPACE_MODELS.get(space, "discogs-effnet")
        ):
            neighbors.append(
                {
                    "i": nb["track_id"],
                    "artist": nb["artist"],
                    "title": nb["title"],
                    "score": nb["score"],
                }
            )
    except Exception as exc:
        logger.warning("hyperspace detail neighbors failed: {}", exc)
    return JSONResponse(
        {
            "artist": rs["artist"],
            "title": rs["title"],
            "plays": rs["plays"],
            "liked": bool(rs["liked"]),
            "first": (rs["first_at"] or "")[:10],
            "last": (rs["last_at"] or "")[:10],
            "styles": styles,
            "listeners": lf["listeners"] if lf else None,
            "scrobbles": lf["playcount"] if lf else None,
            "bpm": (round(dz["bpm"]) if dz and dz["bpm"] else None),
            "label": dz["label"] if dz else None,
            "released": dz["release_date"] if dz else None,
            "mbid": mb["mbid"] if mb else None,
            "neighbors": neighbors,
        }
    )


@app.post("/api/hyperspace/clapquery")
def hyperspace_clapquery(
    text: str = Form(...), _identity: access.Identity = Depends(access.require_query)
) -> JSONResponse:
    """Free-text CLAP search for the map: shells into the CLAP worker env
    (~15s: model load + text encode + cosine). Degrades to a clear error on
    hosts without the local uv/torch env (e.g. the slim deploy image)."""
    import subprocess

    script = Path(__file__).parent.parent / "scripts" / "clap_worker.py"
    try:
        proc = subprocess.run(
            ["uv", "run", str(script), "query", "--text", text, "--k=80"],
            capture_output=True,
            text=True,
            timeout=240,
        )
        line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        d = json.loads(line)
    except Exception as exc:
        return JSONResponse({"error": f"CLAP search unavailable here: {str(exc)[:120]}"})
    if d.get("error"):
        return JSONResponse({"error": d["error"]})
    return JSONResponse(
        {
            "results": [
                {"i": r["track_id"], "s": r["score"], "artist": r["artist"], "title": r["title"]}
                for r in d["results"]
            ]
        }
    )


@app.get("/how")
def how_page() -> RedirectResponse:
    """The build journey lives on the public site now — one source of truth."""
    return RedirectResponse("https://acetate.dev/journey/")


_PREVIEW_RE = re.compile(r"^[0-9a-f]{2}$")
_PREVIEW_NAME_RE = re.compile(r"^[0-9a-f]{40}\.mp3$")


@app.get("/preview/{h2}/{name}")
def preview_clip(h2: str, name: str) -> FileResponse:
    """Serve a cached 30s preview clip (click-to-play in Hyperspace)."""
    if not (_PREVIEW_RE.match(h2) and _PREVIEW_NAME_RE.match(name)):
        raise HTTPException(status_code=404)
    path = db.db_path().parent / "audio" / "previews" / h2 / name
    if not path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="audio/mpeg")


@app.post("/run/{name}", response_class=HTMLResponse)
def run(
    name: str, bg: BackgroundTasks, _identity: access.Identity = Depends(access.require_admin)
) -> HTMLResponse:
    if name not in jobs.TRIGGERS:
        return HTMLResponse(f"unknown job: {escape(name)}", status_code=404)
    if not jobs.try_claim(name):
        return HTMLResponse(
            '<span id=toast class=toast style="color:#ffce6b">already running…</span>'
        )
    bg.add_task(_run_job, name)
    return HTMLResponse(
        f"<span id=toast class=toast>started {escape(name)} · check recent runs ↓</span>"
    )


@app.post("/refresh-plays", response_class=HTMLResponse)
def refresh_plays(_identity: access.Identity = Depends(access.require_admin)) -> HTMLResponse:
    """Poll recently-played now and report what landed, inline.

    Deliberately synchronous, unlike /run/{name}: the point of pressing this is
    to find out whether the thing you just listened to made it in.
    """
    res = jobs.refresh_plays()
    if res.get("status") == "busy":
        return HTMLResponse(
            '<span class=toast style="color:#ffce6b">a poll is already '
            "running — try again in a moment</span>"
        )
    if res.get("status") == "error":
        return HTMLResponse(
            '<span class=toast style="color:#ff8a8a">refresh failed — '
            f"{escape(str(res.get('error'))[:160])}</span>"
        )
    new = res.get("new_plays") or []
    if not new:
        return HTMLResponse(
            "<span class=toast>up to date · nothing new since the last "
            f"poll ({res.get('fetched', 0)} checked)</span>"
        )
    latest = new[0]
    label = escape(
        " — ".join(p for p in (latest.get("title"), latest.get("artist")) if p) or "a new track"
    )
    plural = "play" if len(new) == 1 else "plays"
    return HTMLResponse(
        f"<span class=toast>+{len(new)} {plural} · latest <b>{label}</b> at "
        f"{escape(_localtime(latest['played_at'], '%H:%M'))}</span>",
        headers={"HX-Trigger": "plays-refreshed"},
    )  # swap the list in behind the toast


# --------------------------------------------------------------------------- #
#  Capability tools
# --------------------------------------------------------------------------- #


@app.get("/tools", response_class=HTMLResponse)
def tools_page(request: Request) -> str:
    from . import taste

    conn = db.connect()
    return _render(
        "tools",
        title="tools",
        active="tools",
        workflows=workflows.visible(can_admin=access.current_identity(request).can_admin),
        playlists=taste.authored_playlists(conn),
    )


@app.post("/tools/discover-artists", response_class=HTMLResponse)
def tools_discover_artists(
    playlist: str = Form(...),
    novelty: str = Form("outside_library"),
    limit: int = Form(20),
    _identity: access.Identity = Depends(access.require_query),
) -> str:
    conn = db.connect()
    try:
        result = artist_discovery.discover(
            conn, playlist, novelty=novelty, limit=limit, min_seed_coverage=2
        )
    except ValueError as exc:
        result = {"error": str(exc), "artists": []}
    return _render("artist_discovery_results", result=result)


# --------------------------------------------------------------------------- #
#  Smart Playlists
# --------------------------------------------------------------------------- #


def _playlist_collection(
    curated: bool, view: str = "saved", page: int = 1, playlist_id: str | None = None
) -> dict:
    if view not in {"saved", "drafts", "scheduled", "history"} or page < 1:
        raise HTTPException(status_code=404, detail="Playlist view not found")
    conn = db.connect()
    managed = {
        r["key"]: r["playlist_id"]
        for r in conn.execute("SELECT key, playlist_id FROM managed_playlists WHERE kind='smart'")
    }
    items = []
    for r in smart.list_all(conn) if view == "saved" else []:
        if smart.is_curated(r["spec"]) != curated:
            continue
        items.append(
            {
                "id": r["id"],
                "name": r["name"],
                "enabled": r["enabled"],
                "published_at": r["published_at"],
                "summary": smart.describe(r["spec"]),
                "count": len(smart.evaluate(conn, r["spec"])),
                "preview": smart.preview(conn, r["spec"], limit=6),
                "url": f"/playlists/saved/{r['id']}",
                "spotify_id": (
                    managed[str(r["id"])]
                    if str(r["id"]) in managed and (not curated or r["published_at"])
                    else None
                ),
            }
        )
    raw_drafts = chatstore.playlist_drafts(conn)
    ctx = {
        "curated": curated,
        "items": items,
        "view": view,
        "display_tz": timefmt.tz_name(),
        "base_path": "/curated" if curated else "/smart",
        "draft_count": len(raw_drafts),
        "examples": list(smart.EXAMPLES),
        "examples_json": Markup(json.dumps(smart.EXAMPLES).replace("</", "<\\/")),
        "template_spec": json.dumps(
            {
                "liked": True,
                "homed": False,
                "not_played_days": 183,
                "sort": "plays_desc",
                "limit": 50,
            },
            indent=2,
        ),
        "field_docs": (smart.__doc__ or "").strip(),
    }
    if view == "drafts":
        from . import proposals

        ctx["drafts"] = [
            {
                **p,
                "kind": "curated" if smart.is_curated(p["spec"]) else "smart",
                "url": None,
                "view_url": f"/chat/messages/{p['message_id']}/proposals/{p['index']}",
                "save_url": f"/chat/messages/{p['message_id']}/proposals/{p['index']}/approve",
                "summary": p.get("summary") or smart.describe(p["spec"]),
                "preview": smart.preview(conn, p["spec"]) if smart.is_curated(p["spec"]) else [],
            }
            for p in raw_drafts
        ]
        ctx["track_suggestions"] = proposals.counts(conn).get("proposed", 0)
    elif view == "scheduled":
        ctx["schedule"] = playlist_activity.scheduled_updates(conn)
    elif view == "history":
        ctx["history"] = playlist_activity.snapshot_history(
            conn, page=page, playlist_id=playlist_id
        )
        ctx["history_base"] = (
            ctx["base_path"]
            + "?"
            + urlencode(
                {"view": "history", **({"playlist_id": playlist_id} if playlist_id else {})}
            )
        )
        if playlist_id:
            detail = playlist_activity.playlist_detail(conn, playlist_id)
            ctx["history_filter"] = {
                "name": detail["name"] if detail else playlist_id,
                "url": playlist_activity.playlist_url(playlist_id),
            }
    conn.close()
    return ctx


@app.get("/smart", response_class=HTMLResponse)
def smart_page(view: str = "saved", page: int = 1, playlist_id: str | None = None) -> str:
    return _render(
        "smart",
        title="smart" if view == "saved" else view,
        active="smart",
        **_playlist_collection(curated=False, view=view, page=page, playlist_id=playlist_id),
    )


@app.get("/curated", response_class=HTMLResponse)
def curated_page(view: str = "saved", page: int = 1, playlist_id: str | None = None) -> str:
    return _render(
        "smart",
        title="curated" if view == "saved" else view,
        active="curated",
        **_playlist_collection(curated=True, view=view, page=page, playlist_id=playlist_id),
    )


@app.post("/curated/{sid}/publish")
def curated_publish(sid: int, _identity: access.Identity = Depends(access.require_admin)) -> dict:
    conn = db.connect()
    try:
        row = smart.get(conn, sid)
        if not row or not smart.is_curated(row["spec"]):
            raise HTTPException(status_code=404, detail="Curated playlist not found")
        if not row["published_at"]:
            try:
                jobs.record_run(f"smart:{sid}", lambda: jobs.run_smart(sid))
            except Exception as exc:
                logger.exception("curated playlist publication failed")
                raise HTTPException(status_code=502, detail="Couldn't publish. Try again.") from exc
        if not smart.get(conn, sid)["published_at"]:
            raise HTTPException(status_code=400, detail="This selection has no tracks to publish")
        return {"url": _saved_playlist_url(sid)}
    finally:
        conn.close()


@app.post("/smart/create")
def smart_create(
    name: str = Form(...),
    spec: str = Form(...),
    _identity: access.Identity = Depends(access.require_admin),
) -> RedirectResponse:
    conn = db.connect()
    try:
        try:
            spec_obj = json.loads(spec)
            if not name.strip() or not isinstance(spec_obj, dict):
                raise ValueError("A playlist name and rule object are required")
            smart.evaluate(conn, spec_obj)
        except (TypeError, ValueError, KeyError) as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid playlist specification: {exc}"
            ) from exc
        smart.create(conn, name.strip(), spec_obj)
        return RedirectResponse(
            "/curated" if smart.is_curated(spec_obj) else "/smart", status_code=303
        )
    finally:
        conn.close()


@app.post("/smart/{sid}/run", response_class=HTMLResponse)
def smart_run(
    sid: int, bg: BackgroundTasks, _identity: access.Identity = Depends(access.require_admin)
) -> HTMLResponse:
    conn = db.connect()
    row = smart.get(conn, sid)
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Saved playlist not found")
    if smart.is_curated(row["spec"]) and row["published_at"]:
        return HTMLResponse(
            "<span id=toast class=toast>already published — open it on Spotify</span>"
        )
    bg.add_task(lambda: jobs.record_run(f"smart:{sid}", lambda: jobs.run_smart(sid)))
    return HTMLResponse(
        "<span id=toast class=toast>building… refresh this collection in a moment</span>"
    )


@app.post("/smart/{sid}/toggle")
def smart_toggle(
    sid: int, _identity: access.Identity = Depends(access.require_admin)
) -> RedirectResponse:
    conn = db.connect()
    row = smart.get(conn, sid)
    if row and smart.is_curated(row["spec"]):
        raise HTTPException(
            status_code=400, detail="Fixed selections do not have automatic updates"
        )
    if row:
        smart.update(conn, sid, enabled=not row["enabled"])
    return RedirectResponse("/smart", status_code=303)


@app.post("/smart/{sid}/delete")
def smart_delete(
    sid: int, _identity: access.Identity = Depends(access.require_admin)
) -> RedirectResponse:
    conn = db.connect()
    row = smart.get(conn, sid)
    smart.delete(conn, sid)  # removes local tracking only — never touches the Spotify playlist
    conn.close()
    destination = "/curated" if row and smart.is_curated(row["spec"]) else "/smart"
    return RedirectResponse(destination, status_code=303)


# --------------------------------------------------------------------------- #
#  Chat with your library (read-only agent)
# --------------------------------------------------------------------------- #

# --- review queue + playlist health ---------------------------------------- #


def _preview_url(conn, recording_key: str, kind: str) -> str | None:
    """Map a recording/candidate to its cached 30s clip URL, if fetched."""
    if kind == "candidate":
        row = conn.execute(
            "SELECT path FROM candidate_audio WHERE candidate_key=? AND status='fetched'",
            (recording_key,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT path FROM audio_cache WHERE recording_key=? AND status='fetched'",
            (recording_key,),
        ).fetchone()
    if not row or not row["path"]:
        return None
    parts = Path(row["path"]).parts  # e.g. audio/previews/ab/<sha1>.mp3
    if len(parts) < 2:
        return None
    prefix = "candidate-preview" if kind == "candidate" else "preview"
    return f"/{prefix}/{parts[-2]}/{parts[-1]}"


@app.get("/review", response_class=HTMLResponse)
def review_page() -> str:
    from . import proposals as _proposals

    conn = db.connect()
    rows = _proposals.pending(conn, limit=300)
    groups: dict[str, list[dict]] = {}
    for p in rows:
        p["preview_url"] = _preview_url(conn, p["recording_key"], p["kind"])
        groups.setdefault(p["playlist_name"] or p["playlist_id"], []).append(p)
    return _render(
        "review",
        title="review",
        active="review",
        counts=_proposals.counts(conn),
        groups=[{"name": k, "rows": v} for k, v in groups.items()],
        feedback=_proposals.feedback(conn),
    )


@app.post("/review/{pid}/{verdict}", response_class=HTMLResponse)
def review_decide(
    pid: int,
    verdict: str,
    _identity: access.Identity = Depends(access.require_admin),
) -> HTMLResponse:
    from . import proposals as _proposals

    if verdict not in ("accept", "reject"):
        raise HTTPException(status_code=404)
    conn = db.connect()
    row = _proposals.decide(conn, pid, "accepted" if verdict == "accept" else "rejected")
    if not row:
        return HTMLResponse("")
    color = "var(--accent)" if verdict == "accept" else "var(--accent2)"
    return HTMLResponse(
        f'<li class="row" style="color:{color};font-family:var(--mono);font-size:.72rem">'
        f"{verdict}ed — {escape(row['artist'] or '')} · {escape(row['title'] or '')}</li>"
    )


@app.get("/filing", response_class=HTMLResponse)
def filing_page() -> str:
    from . import filing as _filing

    conn = db.connect()
    q = _filing.queue(conn, min_plays=1, limit=100, top_n=3)
    for it in q:
        it["preview_url"] = _preview_url(conn, it["recording_key"], "library")
    filed = conn.execute("SELECT COUNT(*) FROM filing_actions WHERE status='added'").fetchone()[0]
    return _render(
        "filing",
        title="filing",
        active="filing",
        queue=q,
        ledger=_filing.history(conn, limit=50),
        unfiled_total=_filing.unfiled_count(conn),
        filed_count=filed,
    )


def _filing_li(a: dict) -> str:
    """The ledger/confirmation row returned to HTMX after a file or undo."""
    import html

    art, ttl = html.escape(a.get("artist") or ""), html.escape(a.get("title") or "")
    pln = html.escape(a.get("playlist_name") or "")
    if a["status"] == "undone":
        return (
            f'<li class="row ledgerrow undone" style="grid-template-columns:auto 1fr auto">'
            f'<span class="pact remove">↩</span><div class=ti><div class=t>{art} — {ttl}</div>'
            f"<div class=a>removed from <b>{pln}</b></div></div>"
            f"<span class=auto>undone</span></li>"
        )
    return (
        f'<li class="row filerow" style="grid-template-columns:auto 1fr auto">'
        f'<span class="pact add">✓</span><div class=ti><div class=t>{art} — {ttl}</div>'
        f"<div class=a>filed → <b>{pln}</b></div></div>"
        f'<button class="chip danger" hx-post="/filing/undo/{a["id"]}" '
        f'hx-target="closest li" hx-swap="outerHTML">undo</button></li>'
    )


@app.post("/filing/file", response_class=HTMLResponse)
def filing_file(
    track_id: str = Form(...),
    playlist_id: str = Form(...),
    score: float = Form(None),
    _identity: access.Identity = Depends(access.require_admin),
) -> HTMLResponse:
    from . import auth, filing as _filing

    conn = db.connect()
    rec = conn.execute(
        "SELECT tr.recording_key, rs.artist, rs.title FROM track_recording tr "
        "JOIN recording_stats rs ON rs.recording_key=tr.recording_key "
        "WHERE tr.track_id=?",
        (track_id,),
    ).fetchone()
    if not rec:
        raise HTTPException(status_code=404, detail="unknown track")
    pl = conn.execute(
        "SELECT name FROM taste_regions WHERE playlist_id=?", (playlist_id,)
    ).fetchone()
    pname = pl["name"] if pl else None
    if pname is None:
        r2 = conn.execute(
            "SELECT name FROM playlist_snapshots WHERE playlist_id=? "
            "ORDER BY backup_time DESC LIMIT 1",
            (playlist_id,),
        ).fetchone()
        pname = r2["name"] if r2 else playlist_id
    sp = auth.client(open_browser=False)
    aid = _filing.file_track(
        conn,
        sp,
        recording_key=rec["recording_key"],
        track_id=track_id,
        playlist_id=playlist_id,
        playlist_name=pname,
        artist=rec["artist"],
        title=rec["title"],
        score=score,
    )
    return HTMLResponse(
        _filing_li(
            {
                "id": aid,
                "status": "added",
                "artist": rec["artist"],
                "title": rec["title"],
                "playlist_name": pname,
            }
        )
    )


@app.post("/filing/undo/{action_id}", response_class=HTMLResponse)
def filing_undo(
    action_id: int,
    _identity: access.Identity = Depends(access.require_admin),
) -> HTMLResponse:
    from . import auth, filing as _filing

    conn = db.connect()
    sp = auth.client(open_browser=False)
    row = _filing.undo(conn, sp, action_id)
    if not row:
        return HTMLResponse("")
    return HTMLResponse(_filing_li({**row, "status": "undone"}))


@app.get("/health", response_class=HTMLResponse)
def health_page() -> str:
    from . import taste as _taste

    conn = db.connect()
    return _render("health", title="playlist health", active="health", regions=_taste.regions(conn))


@app.post("/health/build", response_class=HTMLResponse)
def health_build(
    _identity: access.Identity = Depends(access.require_admin),
) -> HTMLResponse:
    from . import taste as _taste

    conn = db.connect()
    res = _taste.build(conn)
    return HTMLResponse(
        f"<span id=toast class=toast>built {res['regions_built']} regions "
        f'({res["too_small"]} too small) — <a href=/health style="color:var(--accent)">refresh</a></span>'
    )


@app.get("/health/{playlist_id}", response_class=HTMLResponse)
def health_detail(playlist_id: str) -> str:
    from . import health as _health, taste as _taste

    conn = db.connect()
    named = conn.execute(
        "SELECT name FROM taste_regions WHERE playlist_id=?", (playlist_id,)
    ).fetchone()
    rep = _health.report(conn, playlist_id, named["name"] if named else None)
    if rep is None:
        raise HTTPException(status_code=404, detail="playlist not modelable")
    region = _taste.load_region(conn, playlist_id, rep["name"])
    additions = _taste.additions(conn, region, limit=15)
    for a in additions:
        a["preview_url"] = _preview_url(conn, a["recording_key"], "library")
    return _render(
        "health_detail", title=rep["name"], active="health", rep=rep, additions=additions
    )


@app.post("/health/{playlist_id}/scan", response_class=HTMLResponse)
def health_scan_route(
    playlist_id: str,
    _identity: access.Identity = Depends(access.require_admin),
) -> HTMLResponse:
    from . import health as _health

    conn = db.connect()
    named = conn.execute(
        "SELECT name FROM taste_regions WHERE playlist_id=?", (playlist_id,)
    ).fetchone()
    res = _health.scan(conn, playlist_id, named["name"] if named else None)
    return HTMLResponse(
        f"<span id=toast class=toast>{res.get('additions', 0)} liked matches + "
        f"{res.get('removals', 0)} misfits queued — "
        f'<a href=/review style="color:var(--accent)">review</a></span>'
    )


@app.post("/health/{playlist_id}/matches", response_class=HTMLResponse)
def health_matches_route(
    playlist_id: str,
    _identity: access.Identity = Depends(access.require_admin),
) -> HTMLResponse:
    """Queue liked-song additions without mixing in removal suggestions."""
    from . import health as _health

    conn = db.connect()
    named = conn.execute(
        "SELECT name FROM taste_regions WHERE playlist_id=?", (playlist_id,)
    ).fetchone()
    res = _health.scan(conn, playlist_id, named["name"] if named else None, include_misfits=False)
    return HTMLResponse(
        f"<span id=toast class=toast>{res.get('additions', 0)} liked matches queued — "
        f'<a href=/review style="color:var(--accent)">review</a></span>'
    )


_CAND_PREVIEW_NAME_RE = re.compile(r"^[0-9a-f]{40}\.mp3$")


@app.get("/candidate-preview/{h2}/{name}")
def candidate_preview_clip(h2: str, name: str) -> FileResponse:
    """Serve a cached candidate preview clip (review-queue listening)."""
    if not (_PREVIEW_RE.match(h2) and _CAND_PREVIEW_NAME_RE.match(name)):
        raise HTTPException(status_code=404)
    path = db.db_path().parent / "audio" / "candidates" / h2 / name
    if not path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="audio/mpeg")


def _saved_playlist_url(sid: int) -> str:
    return f"/playlists/saved/{sid}"


def _chat_proposals(conn, message_id: int, proposals: list[dict]) -> list[dict]:
    out = []
    for index, proposal in enumerate(proposals):
        sid = chatstore.saved_playlist_id(conn, proposal)
        out.append(
            {
                **proposal,
                "kind": "curated" if smart.is_curated(proposal["spec"]) else "smart",
                "view_url": f"/chat/messages/{message_id}/proposals/{index}",
                "save_url": f"/chat/messages/{message_id}/proposals/{index}/approve",
                "url": _saved_playlist_url(sid) if sid is not None else None,
            }
        )
    return out


@app.get("/chat", response_class=HTMLResponse)
def chat_page(thread: int | None = None, q: str = "") -> str:
    conn = db.connect()
    th = chatstore.open_thread(conn, thread) if thread else chatstore.get_active_thread(conn)
    msgs = chatstore.messages(conn, th["id"])
    for message in msgs:
        if message.get("meta") and message["meta"].get("proposals"):
            message["meta"]["proposals"] = _chat_proposals(
                conn, message["id"], message["meta"]["proposals"]
            )
    return _render(
        "chat",
        title="chat",
        active="chat",
        configured=bool(os.environ.get("OPENROUTER_API_KEY")),
        model_groups=agent.MODEL_GROUPS,
        default=agent.default_model(),
        usage_summary=_usage_summary,
        draft_query=q,
        memory=chatstore.library_memory(conn),
        thread=th,
        msgs=msgs,
        threads=chatstore.list_threads(conn),
    )


@app.post("/chat/new")
def chat_new(_identity: access.Identity = Depends(access.require_query)) -> RedirectResponse:
    th = chatstore.create_thread(db.connect())
    return RedirectResponse(f"/chat?thread={th['id']}", status_code=303)


@app.post("/chat/memory")
def chat_save_memory(
    text: str = Form(""),
    revision: str = Form(""),
    _identity: access.Identity = Depends(access.require_admin),
) -> dict:
    conn = db.connect()
    try:
        return chatstore.save_library_memory(conn, text, revision)
    except LookupError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        conn.close()


@app.post("/chat/messages/{message_id}/proposals/{index}/approve")
def chat_approve_playlist(
    message_id: int, index: int, _identity: access.Identity = Depends(access.require_admin)
) -> dict:
    conn = db.connect()
    try:
        sid = chatstore.approve_playlist_proposal(conn, message_id, index)
        return {"url": _saved_playlist_url(sid)}
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        conn.close()


@app.post("/chat/stream")
def chat_stream_endpoint(
    q: str = Form(...),
    model: str = Form(None),
    thread: int | None = Form(None),
    identity: access.Identity = Depends(access.require_query),
) -> StreamingResponse:
    # Capture the role here, not inside the generator: Starlette consumes the
    # stream after the request scope (and its identity contextvar) is gone.
    can_admin = identity.can_admin

    def gen():
        # Open the DB inside the generator: Starlette iterates this in a threadpool thread,
        # and SQLite connections are bound to the thread that created them.
        conn = db.connect()
        try:
            th = (
                chatstore.get_thread(conn, thread) if thread else None
            ) or chatstore.get_active_thread(conn)
            memory_revision = chatstore.library_memory(conn)["revision"]
            checkpoint = th["summary_through_id"]
            history = chatstore.history_for_agent(conn, th["id"], through_id=checkpoint)

            def save_summary(summary, through_id):
                nonlocal checkpoint
                through_id = through_id if through_id is not None else checkpoint
                if chatstore.save_summary(conn, th["id"], summary, through_id, checkpoint):
                    checkpoint = through_id

            reply, steps, props, mdl, usage, errored = [], [], [], None, None, False
            context_meta = None
            done = {}
            for ev in agent.chat_stream(
                history + [{"role": "user", "content": q}],
                model=model,
                can_admin=can_admin,
                requested_by=identity.username,
                summary=th["summary"] or "",
                on_summary=save_summary,
            ):
                t = ev.get("t")
                if t == "token":
                    reply.append(ev["v"])
                elif t == "step":
                    steps.append(ev["v"])
                elif t == "step_result":
                    result = ev["v"]
                    for step in reversed(steps):
                        if step.get("call_id") == result.get("call_id"):
                            step.update({k: v for k, v in result.items() if k != "call_id"})
                            break
                elif t == "proposal":
                    props.append(ev["v"])
                elif t == "error":
                    errored = True
                elif t == "done":
                    done = ev.get("v") or {}
                    mdl, usage = done.get("model"), done.get("usage")
                    context_meta = done.get("context")
                    continue  # Show actionable cards only after their message is committed.
                yield json.dumps(ev) + "\n"
            # persist only a real exchange — an errored turn never strands a saved message
            if not errored and ("".join(reply).strip() or props):
                chatstore.add_message(conn, th["id"], "user", q)
                message_id = chatstore.add_message(
                    conn,
                    th["id"],
                    "assistant",
                    "".join(reply),
                    meta={
                        "steps": steps,
                        "proposals": props,
                        "model": mdl,
                        "usage": usage,
                        "context": context_meta,
                    },
                )
                done["proposals"] = _chat_proposals(conn, message_id, props)
            if not errored:
                memory = chatstore.library_memory(conn)
                if memory["revision"] != memory_revision:
                    done["memory"] = memory
                yield json.dumps({"t": "done", "v": done}) + "\n"
        except Exception as exc:
            logger.exception("chat stream failed")
            yield json.dumps({"t": "error", "v": str(exc)[:300]}) + "\n"
        finally:
            conn.close()

    return StreamingResponse(
        gen(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
