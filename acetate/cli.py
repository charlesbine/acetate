"""acetate command-line interface.

Commands are grouped by what they touch, so the direction of data is obvious:
  pull *     read FROM Spotify into the local DB (read-only; never changes your account)
  import *   load local files into the DB (offline)
  publish *  write playlists TO your Spotify account
  (top-level auth/whoami/probe/stats are setup + inspection)
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import auth, db

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Personal Spotify backup, history, and playlist tools.",
)
pull_app = typer.Typer(
    no_args_is_help=True, help="Pull data FROM Spotify into the local DB (read-only; safe)."
)
import_app = typer.Typer(
    no_args_is_help=True, help="Import data from local files into the DB (offline)."
)
publish_app = typer.Typer(
    no_args_is_help=True, help="Publish playlists TO your Spotify account (writes your account)."
)
genre_app = typer.Typer(
    no_args_is_help=True, help="Classify favorites by genre (Last.fm) and explore them (local DB)."
)
embed_app = typer.Typer(
    no_args_is_help=True, help="Audio previews + embeddings for the Hyperspace map (local compute)."
)
similar_app = typer.Typer(
    no_args_is_help=True, help="Similar tracks: sonic (embeddings) + collaborative (Last.fm)."
)
candidates_app = typer.Typer(
    no_args_is_help=True, help="Outside-library recommendation candidates and provenance."
)
taste_app = typer.Typer(
    no_args_is_help=True, help="Playlist-as-region taste model: build, evaluate, query membership."
)
health_app = typer.Typer(
    no_args_is_help=True, help="Playlist health: coherence, misfits, gap-fill, split checks."
)
review_app = typer.Typer(
    no_args_is_help=True, help="The proposals queue: list, accept, reject, feedback."
)
discover_app = typer.Typer(
    no_args_is_help=True,
    help="Outside-library discovery: per-playlist recs + the Discovery playlist.",
)
filing_app = typer.Typer(
    no_args_is_help=True,
    help="Filing inbox: liked-unplaylisted songs → best-fit playlists, with undo.",
)
playlists_app = typer.Typer(
    no_args_is_help=True,
    help="Playlist profiles: personas (archive/one-shot/rotation), usage, labels.",
)
ledger_app = typer.Typer(
    no_args_is_help=True, help="Audit and conservatively repair enrichment provenance ledgers."
)
app.add_typer(pull_app, name="pull")
app.add_typer(import_app, name="import")
app.add_typer(publish_app, name="publish")
app.add_typer(genre_app, name="genre")
app.add_typer(embed_app, name="embed")
app.add_typer(similar_app, name="similar")
app.add_typer(candidates_app, name="candidates")
app.add_typer(taste_app, name="taste")
app.add_typer(health_app, name="health")
app.add_typer(review_app, name="review")
app.add_typer(discover_app, name="discover")
app.add_typer(filing_app, name="filing")
app.add_typer(playlists_app, name="playlists")
app.add_typer(ledger_app, name="ledger")

console = Console()


@ledger_app.command("audit")
def ledger_audit() -> None:
    """Show legacy rows whose provider/backfill meaning is missing or recoverable."""
    from . import ledger

    conn = db.connect()
    db.init_db(conn)
    for key, value in ledger.audit(conn).items():
        console.print(f"{key}: {value}")


@ledger_app.command("repair")
def ledger_repair(
    apply: bool = typer.Option(False, "--apply", help="Apply safe repairs; default is dry-run."),
) -> None:
    """Recover only evidence-backed states; ambiguous provider empties stay unknown."""
    from . import ledger

    conn = db.connect()
    db.init_db(conn)
    result = ledger.repair_legacy(conn, apply=apply)
    console.print(result)


def _require_creds() -> None:
    missing = auth.missing_env()
    if missing:
        console.print(f"[red]Missing credentials in env/.env:[/] {', '.join(missing)}")
        console.print("Set them in .env (from your Spotify developer app) and retry.")
        raise typer.Exit(1)


# --- setup + inspection (top level) ----------------------------------------


@app.command("auth")
def auth_login() -> None:
    """Run the one-time browser OAuth and cache a refresh token."""
    _require_creds()
    me = auth.client(open_browser=True).me()
    console.print(f"[green]Authenticated as[/] {me['display_name']} ({me['id']})")
    console.print(f"Token cached. Scopes: {', '.join(auth.SCOPES)}")


@app.command()
def whoami() -> None:
    """Show the currently authenticated Spotify user (read-only)."""
    _require_creds()
    me = auth.client(open_browser=False).me()
    console.print(f"{me['display_name']} ({me['id']}) — {me.get('product', '?')} account")


@app.command()
def probe() -> None:
    """Check whether this app still has access to audio-features (read-only)."""
    _require_creds()
    import spotipy

    conn = db.connect()
    row = conn.execute(
        "SELECT track_id FROM play_events WHERE track_id IS NOT NULL LIMIT 1"
    ).fetchone()
    if not row:
        console.print("[yellow]No track in DB to probe with.[/]")
        raise typer.Exit(1)
    sp = auth.client(open_browser=False)
    try:
        feats = sp.audio_features([row["track_id"]])
        ok = feats and feats[0]
        console.print(
            f"[green]audio-features AVAILABLE[/] — sample energy={ok.get('energy')}"
            if ok
            else "[yellow]audio-features returned empty[/]"
        )
    except spotipy.SpotifyException as e:
        console.print(
            f"[red]audio-features blocked[/] (HTTP {e.http_status}). "
            "Expected for post-2024 apps — Acetate uses local audio embeddings + Deezer/Last.fm metadata instead."
        )


@publish_app.command("liked-year")
def publish_liked_year(
    year: int = typer.Option(None, help="Year (YYYY); default current year."),
    all_years: bool = typer.Option(
        False, "--all", help="Build one for every year you have liked songs."
    ),
) -> None:
    """Build/refresh '📆 Liked YYYY' — every song you added to Liked that year."""
    _require_creds()
    from datetime import date
    from . import playlists

    conn = db.connect()
    db.init_db(conn)
    sp = auth.client(open_browser=False)
    years = playlists.liked_years(conn) if all_years else [str(year or date.today().year)]
    for y in years:
        res = playlists.build_liked_year(conn, sp, y)
        if res.get("skipped"):
            console.print(f"[yellow]{res['name']}[/]: 0 tracks — skipped")
        else:
            console.print(
                f"[green]{res['name']}[/]: {res['tracks']} tracks "
                f"(https://open.spotify.com/playlist/{res['playlist_id']})"
            )


@publish_app.command("gems")
def publish_gems(
    size: int = typer.Option(50, help="How many gems to surface this run."),
) -> None:
    """Build/refresh the rotating '💎 Lost Gems' playlist — faded liked favorites."""
    _require_creds()
    from . import playlists

    conn = db.connect()
    db.init_db(conn)
    res = playlists.build_lost_gems(conn, auth.client(open_browser=False), size=size)
    if res.get("skipped"):
        console.print("[yellow]💎 Lost Gems[/]: no candidates")
    else:
        console.print(
            f"[green]{res['name']}[/]: {res['tracks']} of {res['pool']} gems "
            f"(https://open.spotify.com/playlist/{res['playlist_id']})"
        )


@publish_app.command("on-loop")
def publish_on_loop(
    days: int = typer.Option(21, help="Window of days that counts as 'right now'."),
    size: int = typer.Option(40, help="How many tracks in the playlist."),
) -> None:
    """Rebuild the transient '🔁 On Loop' playlist — your current heavy rotation."""
    _require_creds()
    from . import playlists

    conn = db.connect()
    db.init_db(conn)
    res = playlists.build_on_loop(conn, auth.client(open_browser=False), days=days, size=size)
    if res.get("skipped"):
        console.print("[yellow]🔁 On Loop[/]: no recent plays")
    else:
        console.print(
            f"[green]{res['name']}[/]: {res['tracks']} tracks "
            f"(https://open.spotify.com/playlist/{res['playlist_id']})"
        )


@app.command()
def stats() -> None:
    """Quick offline summary of what's in the local DB."""
    conn = db.connect()
    t = Table(title="acetate DB")
    t.add_column("table")
    t.add_column("rows", justify="right")
    for name in (
        "play_events",
        "saved_track_snapshots",
        "playlist_snapshots",
        "playlist_track_snapshots",
        "tracks",
        "artists",
        "track_labels",
        "track_genres",
        "managed_playlists",
    ):
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        except Exception:
            n = "—"
        t.add_row(name, str(n))
    console.print(t)


# --- pull: Spotify -> local DB (read-only) ---------------------------------


@pull_app.command("plays")
def pull_plays() -> None:
    """Fetch recently-played and append new plays to the local history."""
    _require_creds()
    from . import snapshot

    conn = db.connect()
    db.init_db(conn)
    res = snapshot.poll_recently_played(conn, auth.client(open_browser=False))
    console.print(f"Pulled recently-played: fetched {res['fetched']}, new {res['inserted']}")


@pull_app.command("library")
def pull_library(
    only_mine: bool = typer.Option(False, help="Only back up playlists you own."),
) -> None:
    """Snapshot all playlists + saved tracks into the DB (the deletion trace)."""
    _require_creds()
    from . import snapshot

    conn = db.connect()
    db.init_db(conn)
    res = snapshot.run_backup(conn, auth.client(open_browser=False), only_mine=only_mine)
    liked = "changed" if res["saved_changed"] else "unchanged"
    console.print(
        f"Backup @ {res['backup_time'][:19]}: liked songs {liked} ({res['saved_tracks']}); "
        f"{res['playlists']} playlists, {res['changed']} changed "
        f"→ {res['tracks']} membership rows written"
    )


# --- import: local files -> DB (offline) -----------------------------------


@import_app.command("history")
def import_history(
    export_dir: Path = typer.Argument(..., help="GDPR 'Extended Streaming History' dir"),
) -> None:
    """Import the GDPR extended streaming-history export."""
    from .import_history import import_extended_history

    conn = db.connect()
    db.init_db(conn)
    for k, v in import_extended_history(export_dir, conn).items():
        console.print(f"  {k}: {v}")


@import_app.command("legacy")
def import_legacy(
    tinydb_json: Path = typer.Argument(..., help="Path to legacy TinyDB JSON"),
) -> None:
    """Import a legacy TinyDB backup (e.g. legacy-tinydb.json) into SQLite."""
    from .migrate import migrate_tinydb

    conn = db.connect()
    db.init_db(conn)
    for table, n in migrate_tinydb(tinydb_json, conn).items():
        console.print(f"  {table}: {n}")


# --- publish: local DB -> Spotify (writes your account) --------------------


@publish_app.command("monthly")
def publish_monthly(
    month: str = typer.Option(None, "--month", help="YYYY-MM (default: current month)"),
    top_limit: int = typer.Option(50, help="Max tracks in the Top Played playlist."),
) -> None:
    """Build/refresh this month's '📆 Liked ·' and '🔥 Top Played ·' playlists on Spotify."""
    _require_creds()
    from datetime import date
    from . import playlists

    ym = month or date.today().strftime("%Y-%m")
    conn = db.connect()
    db.init_db(conn)
    sp = auth.client(open_browser=False)
    for res in (
        playlists.build_liked_month(conn, sp, ym),
        playlists.build_top_played_month(conn, sp, ym, limit=top_limit),
    ):
        if res.get("skipped"):
            console.print(f"[yellow]{res['name']}[/]: 0 tracks — skipped")
        else:
            console.print(
                f"[green]{res['name']}[/]: {res['tracks']} tracks "
                f"(https://open.spotify.com/playlist/{res['playlist_id']})"
            )


# --- genre: classify favorites + discovery (local DB; reads Last.fm) -------


@genre_app.command("tag")
def genre_tag(
    min_plays: int = typer.Option(10, help="Only tag tracks with at least this many plays."),
    limit: int = typer.Option(None, help="Cap how many (most-played first); omit for all."),
) -> None:
    """Fetch Last.fm genre tags for your favorite tracks into the DB (resumable)."""
    auth.load_credentials()  # resolves LASTFM_API_KEY from .env/1Password
    from . import genre

    conn = db.connect()
    db.init_db(conn)
    res = genre.tag_favorites(
        conn,
        min_plays=min_plays,
        limit=limit,
        progress=lambda i, n, t: console.print(f"  …{i}/{n} ({t} tagged)"),
    )
    console.print(
        f"[green]Tagged {res['newly_tagged']} new tracks[/] "
        f"({res['universe']} candidates, {res['already_tagged']} already done)."
    )


@genre_app.command("backfill")
def genre_backfill(
    batch: int = typer.Option(
        None,
        help="Max tracks to tag this run (for chunked/scheduled "
        "backfill); omit to do the whole remaining library.",
    ),
) -> None:
    """Slowly tag your ENTIRE library's genres (resumable, favorites first).
    Re-run to continue; once caught up it only tags newly-ingested tracks."""
    auth.load_credentials()
    from . import genre

    conn = db.connect()
    db.init_db(conn)
    res = genre.tag_backfill(
        conn, batch=batch, progress=lambda i, n, t: console.print(f"  …{i}/{n} ({t} tagged)")
    )
    untagged = max(0, res["universe"] - res["already_tagged"] - res["newly_tagged"])
    console.print(
        f"[green]Tagged {res['newly_tagged']} tracks this run.[/] "
        f"Library: {res['universe']} total, ~{untagged} still untagged."
    )


@genre_app.command("top")
def genre_top(
    min_plays: int = typer.Option(10, help="Favorite threshold for the play-weighting."),
    limit: int = typer.Option(40, help="How many genres to show."),
) -> None:
    """Show your top genres — the menu to pick 🎧 playlists from."""
    from . import genre

    conn = db.connect()
    t = Table(title="Your top genres (play-weighted)")
    t.add_column("genre")
    t.add_column("tracks", justify="right")
    t.add_column("plays", justify="right")
    for g, tracks, plays in genre.top_genres(conn, min_plays=min_plays, limit=limit):
        t.add_row(g, str(tracks), str(plays))
    console.print(t)


# --- embed: preview audio + embeddings for Hyperspace (local compute) ------


@embed_app.command("fetch")
def embed_fetch(
    batch: int = typer.Option(
        None, help="Max recordings to fetch this run; omit for the whole queue."
    ),
    min_plays: int = typer.Option(
        5, help="Universe = liked ∪ recordings with at least this many plays."
    ),
) -> None:
    """Fetch 30s preview clips (Deezer by ISRC, then search) into data/audio/. Resumable."""
    from . import audio

    conn = db.connect()
    db.init_db(conn)
    res = audio.fetch_pending(
        conn,
        batch=batch,
        min_plays=min_plays,
        progress=lambda i, n, c: console.print(
            f"  …{i}/{n} (fetched {c['fetched']}, miss {c['miss']}, err {c['error']})"
        ),
    )
    console.print(
        f"[green]Fetched {res['fetched']}[/] "
        f"(new in queue {res['queued_new']}, miss {res['miss']}, error {res['error']}, "
        f"remaining {res['remaining']})"
    )


@embed_app.command("status")
def embed_status() -> None:
    """Audio-cache and embedding coverage counts."""
    from . import audio

    conn = db.connect()
    db.init_db(conn)
    t = Table(title="Hyperspace pipeline")
    t.add_column("stage")
    t.add_column("count", justify="right")
    for k, v in sorted(audio.status(conn).items()):
        t.add_row(k, str(v))
    console.print(t)


def _run_worker(script: str, *args: str) -> None:
    """The ML workers run in their own uv script env (heavy deps stay out of the
    app env / Docker image); this just streams the subprocess."""
    import subprocess

    script_path = Path(__file__).parent.parent / "scripts" / script
    cmd = ["uv", "run", str(script_path), *args]
    console.print(f"[dim]$ {' '.join(cmd)}[/]")
    raise typer.Exit(subprocess.run(cmd).returncode)


@embed_app.command("run")
def embed_run(
    batch: int = typer.Option(None, help="Max recordings to embed this run; omit for all fetched."),
) -> None:
    """Embed fetched clips (discogs-effnet + style head) — isolated Python 3.14 env."""
    _run_worker("embed_worker.py", *([f"--batch={batch}"] if batch else []))


@embed_app.command("reduce")
def embed_reduce(
    space: str = typer.Option("all", help="Which projection to rebuild: effnet, clap, or all."),
) -> None:
    """Rebuild the 3D UMAP projection(s) + clusters for the Hyperspace page."""
    import subprocess

    db.init_db(db.connect())  # ensure schema migrations ran (worker connects raw)
    script = Path(__file__).parent.parent / "scripts" / "reduce_worker.py"
    spaces = ["effnet", "clap"] if space == "all" else [space]
    rc = 0
    for s in spaces:
        console.print(f"[dim]reducing space {s}…[/]")
        rc |= subprocess.run(["uv", "run", str(script), f"--space={s}"]).returncode
    raise typer.Exit(rc)


@embed_app.command("enrich")
def embed_enrich(
    batch: int = typer.Option(None, help="Max tracks this run; omit for all un-enriched."),
) -> None:
    """Harvest Deezer metadata for fetched tracks: bpm/gain/rank, album label+genres,
    artist fans + related-artists graph. Run after a fetch pass (shared rate budget)."""
    from . import audio

    conn = db.connect()
    db.init_db(conn)
    res = audio.enrich(
        conn, batch=batch, progress=lambda i, n, c: console.print(f"  …{i}/{n} tracks")
    )
    console.print(
        f"[green]Enriched[/] {res['tracks']} tracks, {res['albums']} albums, "
        f"{res['artists']} artists (provider misses {res['provider_miss']}, "
        f"errors {res['errors']})"
    )


@embed_app.command("clap")
def embed_clap(
    batch: int = typer.Option(None, help="Max clips to embed this run; omit for all fetched."),
) -> None:
    """Embed cached clips into the CLAP text↔audio space (pinned music checkpoint)."""
    _run_worker("clap_worker.py", "embed", *([f"--batch={batch}"] if batch else []))


@app.command("search")
def search(
    prompt: str = typer.Argument(..., help="Describe a sound: 'hypnotic dub techno, no vocals'."),
    k: int = typer.Option(20, help="How many tracks to return."),
) -> None:
    """Semantic search over the library's AUDIO via CLAP text↔audio embeddings."""
    import json
    import subprocess

    script = Path(__file__).parent.parent / "scripts" / "clap_worker.py"
    proc = subprocess.run(
        ["uv", "run", str(script), "query", "--text", prompt, f"--k={k}"],
        capture_output=True,
        text=True,
    )
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        console.print(f"[red]search failed[/]: {(proc.stderr or proc.stdout)[-400:]}")
        raise typer.Exit(1)
    if data.get("error"):
        console.print(f"[yellow]{data['error']}[/]")
        raise typer.Exit(1)
    t = Table(title=f"“{prompt}” — searched {data['searched']:,} tracks ({data['model']})")
    t.add_column("artist — title")
    t.add_column("style")
    t.add_column("plays", justify="right")
    t.add_column("score", justify="right")
    for r in data["results"]:
        t.add_row(
            f"{r['artist']} — {r['title']}" + (" ♥" if r["liked"] else ""),
            (r["style"] or "").split("---")[-1],
            str(r["plays"]),
            f"{r['score']:.3f}",
        )
    console.print(t)


# --- similar: recommendations from two spaces (local DB + Last.fm/MusicBrainz) --


@similar_app.command("to")
def similar_to(
    query: str = typer.Argument(..., help="'artist - title' or a fragment of either."),
    k: int = typer.Option(15, help="Neighbors per space."),
) -> None:
    """Show a track's neighbors: sonic (embedding cosine) and collaborative (Last.fm)."""
    auth.load_credentials()
    from . import similar

    conn = db.connect()
    db.init_db(conn)
    seed = similar.find_recording(conn, query)
    if not seed:
        console.print(f"[red]No recording matching[/] {query!r}")
        raise typer.Exit(1)
    console.print(f"Seed: [bold]{seed['artist']} — {seed['title']}[/] ({seed['plays']} plays)\n")

    sonic = similar.sonic_neighbors(conn, seed["recording_key"], k=k)
    t = Table(title="Sonic neighbors (discogs-effnet cosine)")
    t.add_column("artist — title")
    t.add_column("style")
    t.add_column("cos", justify="right")
    t.add_column("plays", justify="right")
    for s in sonic:
        t.add_row(
            f"{s['artist']} — {s['title']}" + (" ♥" if s["liked"] else ""),
            (s["style"] or "").split("---")[-1],
            f"{s['score']:.3f}",
            str(s["plays"]),
        )
    console.print(t if sonic else "[yellow]Seed not embedded yet — run the embed pipeline.[/]")

    collab = similar.collaborative(conn, seed["artist"], seed["title"], limit=k)
    t = Table(title="Collaborative neighbors (Last.fm co-listening)")
    t.add_column("artist — title")
    t.add_column("match", justify="right")
    t.add_column("in library")
    for s in collab:
        mark = (
            "♥" if s["liked"] else ("✓ " + str(s["plays"]) + " plays" if s["in_library"] else "—")
        )
        t.add_row(f"{s['artist']} — {s['title']}", f"{s['score']:.2f}", mark)
    console.print(t if collab else "[yellow]Last.fm returned no similar tracks.[/]")


@similar_app.command("cache")
def similar_cache(
    batch: int = typer.Option(None, help="Max recordings to fetch this run; omit for all."),
    min_plays: int = typer.Option(
        5, help="Universe = liked ∪ recordings with at least this many plays."
    ),
) -> None:
    """Cache the Last.fm co-listening graph + listeners/playcount over the library
    (resumable, most-played first, ~1.5h full run). The collaborative substrate for
    genre propagation, gems, and the autoplaylist retrieve stage."""
    auth.load_credentials()
    from . import similar

    conn = db.connect()
    db.init_db(conn)
    res = similar.cache_lastfm(
        conn,
        batch=batch,
        min_plays=min_plays,
        progress=lambda i, n, c: console.print(
            f"  …{i}/{n} ({c['edges']} edges, {c['in_library']} in-library)"
        ),
    )
    console.print(
        f"[green]Cached {res['fetched']} recordings[/] — {res['edges']} edges "
        f"({res['in_library']} point back into the library)"
    )


@similar_app.command("resolve")
def similar_resolve(
    batch: int = typer.Option(None, help="Max recordings to resolve this run; omit for all."),
    min_plays: int = typer.Option(
        5, help="Universe = liked ∪ recordings with at least this many plays."
    ),
) -> None:
    """Backfill ISRC → MusicBrainz MBIDs (1 req/s; resumable; ~2.5h full run).
    MBIDs are the join key for ListenBrainz-style open datasets."""
    from . import mbid

    conn = db.connect()
    db.init_db(conn)
    res = mbid.resolve_pending(
        conn,
        batch=batch,
        min_plays=min_plays,
        progress=lambda i, n, c: console.print(
            f"  …{i}/{n} (resolved {c['resolved']}, miss {c['miss']})"
        ),
    )
    console.print(
        f"[green]Resolved {res['resolved']}[/] (miss {res['miss']}, "
        f"error {res['error']}, remaining {res['remaining']})"
    )


# --- candidates: outside-library recommendation frontier -------------------


@candidates_app.command("status")
def candidates_status() -> None:
    """Show candidate frontier counts by status/source."""
    from . import candidates

    conn = db.connect()
    db.init_db(conn)
    t = Table(title="Recommendation candidates")
    t.add_column("bucket")
    t.add_column("count", justify="right")
    for k, v in sorted(candidates.status(conn).items()):
        t.add_row(k, str(v))
    try:
        from . import ytmusic

        for k, v in sorted(ytmusic.status(conn).items()):
            t.add_row(k, str(v))
    except Exception:
        pass
    try:
        from . import candidate_deezer

        for k, v in sorted(candidate_deezer.status(conn).items()):
            t.add_row(k, str(v))
    except Exception:
        pass
    try:
        from . import candidate_audio

        for k, v in sorted(candidate_audio.status(conn).items()):
            t.add_row(k, str(v))
    except Exception:
        pass
    try:
        from . import candidate_embeddings

        for k, v in sorted(candidate_embeddings.status(conn).items()):
            t.add_row(k, str(v))
    except Exception:
        pass
    try:
        from . import lastfm_artists

        for k, v in sorted(lastfm_artists.status(conn).items()):
            t.add_row(k, str(v))
    except Exception:
        pass
    try:
        from . import deezer_expansion

        for k, v in sorted(deezer_expansion.status(conn).items()):
            t.add_row(k, str(v))
    except Exception:
        pass
    try:
        from . import spotify_expansion

        for k, v in sorted(spotify_expansion.status(conn).items()):
            t.add_row(k, str(v))
    except Exception:
        pass
    console.print(t)


@candidates_app.command("promote-lastfm")
def candidates_promote_lastfm(
    batch: int = typer.Option(None, help="Max Last.fm edges to promote; omit for all."),
    min_match: float = typer.Option(
        0.0, help="Only promote edges with at least this Last.fm match."
    ),
    no_rescore: bool = typer.Option(
        False, "--no-rescore", help="Skip recomputing scores after import."
    ),
) -> None:
    """Promote out-of-library Last.fm similar-track edges into the candidate ledger."""
    from . import candidates

    conn = db.connect()
    db.init_db(conn)
    res = candidates.promote_lastfm(
        conn,
        batch=batch,
        min_match=min_match,
        rescore=not no_rescore,
        progress=lambda i, n, c: console.print(
            f"  …{i}/{n} ({c['candidates']} candidates, {c['sources']} sources)"
        ),
    )
    console.print(
        f"[green]Promoted {res['sources']} Last.fm edges[/] into "
        f"{res['candidates']} candidate rows "
        f"(skipped library {res['skipped_library']}, scored {res.get('scored', 0)})"
    )


@candidates_app.command("harvest-lastfm-artists")
def candidates_harvest_lastfm_artists(
    batch: int = typer.Option(10, help="How many high-signal library artists to cache."),
    min_plays: int = typer.Option(
        5, help="Artist universe = liked or tracks with this many plays."
    ),
    limit: int = typer.Option(25, help="Similar artists, top tracks, and top albums per seed."),
    include_fetched: bool = typer.Option(
        False, "--include-fetched", help="Refresh artists already fetched."
    ),
    pause_s: float = typer.Option(1.0, help="Minimum elapsed seconds per four-call artist batch."),
) -> None:
    """Cache Last.fm artist info/similar/top tracks/top albums and promote candidates."""
    auth.load_credentials()
    from . import lastfm_artists

    conn = db.connect()
    db.init_db(conn)
    res = lastfm_artists.harvest(
        conn,
        batch=batch,
        min_plays=min_plays,
        limit=limit,
        include_fetched=include_fetched,
        pause_s=pause_s,
        progress=lambda i, n, c: console.print(
            f"  ...{i}/{n} (fetched {c['fetched']}, candidates {c['candidates']}, "
            f"miss {c['miss']}, error {c['error']})"
        ),
    )
    console.print(
        f"[green]Last.fm artists[/]: checked {res['seen']}, "
        f"fetched {res['fetched']}, candidates {res['candidates']}, "
        f"similar {res['similar']}, tracks {res['tracks']}, albums {res['albums']}, "
        f"miss {res['miss']}, error {res['error']}, remaining {res['remaining']}"
    )


@candidates_app.command("harvest-deezer-artists")
def candidates_harvest_deezer_artists(
    artist_batch: int = typer.Option(5, help="How many high-signal Deezer artists to cache."),
    album_batch: int = typer.Option(25, help="How many queued album tracklists to cache."),
    min_plays: int = typer.Option(5, help="Seed universe = liked or tracks with this many plays."),
    top_limit: int = typer.Option(25, help="Artist top tracks retained per seed."),
    album_limit: int = typer.Option(10, help="Artist albums retained per seed."),
    track_limit: int = typer.Option(100, help="Tracks retained per album."),
    include_fetched: bool = typer.Option(
        False, "--include-fetched", help="Refresh artist and album rows already fetched."
    ),
) -> None:
    """Cache Deezer artist top/albums and expand bounded album tracklists."""
    from . import deezer_expansion

    conn = db.connect()
    db.init_db(conn)

    def progress(phase: str, index: int, total: int, counts: dict[str, int]) -> None:
        console.print(
            f"  ...{phase} {index}/{total} "
            f"(artists {counts['artist_fetched']}, albums {counts['album_fetched']}, "
            f"candidates {counts['top_tracks'] + counts['source_edges']})"
        )

    res = deezer_expansion.harvest(
        conn,
        artist_batch=artist_batch,
        album_batch=album_batch,
        min_plays=min_plays,
        top_limit=top_limit,
        album_limit=album_limit,
        track_limit=track_limit,
        include_fetched=include_fetched,
        progress=progress,
    )
    console.print(
        f"[green]Deezer expansion[/]: artists {res['artist_fetched']}/{res['artist_seen']}, "
        f"albums {res['album_fetched']}/{res['album_seen']}, top tracks {res['top_tracks']}, "
        f"album tracks {res['album_tracks']}, candidates {res['candidates']}, "
        f"duplicates {res['duplicates']}, errors "
        f"{res['artist_error'] + res['album_error']}, remaining "
        f"{res['artist_remaining']} artists/{res['album_remaining']} albums"
    )


@candidates_app.command("harvest-spotify-artists")
def candidates_harvest_spotify_artists(
    artist_batch: int = typer.Option(3, help="How many high-signal Spotify artists to cache."),
    album_batch: int = typer.Option(20, help="How many queued album tracklists to cache."),
    min_plays: int = typer.Option(5, help="Seed universe = liked or tracks with this many plays."),
    album_limit: int = typer.Option(10, help="Albums retained per artist group."),
    track_limit: int = typer.Option(100, help="Tracks retained per album."),
    include_fetched: bool = typer.Option(
        False, "--include-fetched", help="Refresh artist and album rows already fetched."
    ),
) -> None:
    """Cache Spotify artist albums and bounded album tracklists for candidates."""
    from . import spotify_expansion

    conn = db.connect()
    db.init_db(conn)
    sp = auth.client(open_browser=False)

    def progress(phase: str, index: int, total: int, counts: dict[str, int]) -> None:
        console.print(
            f"  ...{phase} {index}/{total} "
            f"(artists {counts['artist_fetched']}, albums {counts['album_fetched']}, "
            f"candidate edges {counts['source_edges']})"
        )

    res = spotify_expansion.harvest(
        conn,
        sp=sp,
        artist_batch=artist_batch,
        album_batch=album_batch,
        min_plays=min_plays,
        album_limit=album_limit,
        track_limit=track_limit,
        include_fetched=include_fetched,
        progress=progress,
    )
    console.print(
        f"[green]Spotify expansion[/]: artists {res['artist_fetched']}/{res['artist_seen']}, "
        f"albums {res['album_fetched']}/{res['album_seen']}, top tracks {res['top_tracks']}, "
        f"artist albums {res['artist_albums']}, album tracks {res['album_tracks']}, "
        f"candidates {res['candidates']}, errors {res['artist_error'] + res['album_error']}, "
        f"remaining {res['artist_remaining']} artists/{res['album_remaining']} albums"
    )
    if res["top_unavailable"]:
        console.print(
            "[yellow]Spotify artist top tracks are unavailable to this app; cached once and skipped.[/]"
        )


@candidates_app.command("rescore")
def candidates_rescore() -> None:
    """Recompute candidate scores from provenance edges."""
    from . import candidates

    conn = db.connect()
    db.init_db(conn)
    res = candidates.score_candidates(conn)
    console.print(f"[green]Scored {res['scored']} candidates[/]")


@candidates_app.command("top")
def candidates_top(
    limit: int = typer.Option(25, help="How many candidates to show."),
    status: str = typer.Option("metadata", help="Candidate status bucket."),
) -> None:
    """Show the highest-scoring outside-library candidates and why they surfaced."""
    from . import candidates

    conn = db.connect()
    db.init_db(conn)
    rows = candidates.top_candidates(conn, limit=limit, status=status)
    t = Table(title=f"Top {status} candidates")
    t.add_column("artist — title")
    t.add_column("score", justify="right")
    t.add_column("sources", justify="right")
    t.add_column("top provenance")
    for row in rows:
        reasons = []
        for reason in row["reasons"]:
            seed = " — ".join(p for p in (reason.get("seed_artist"), reason.get("seed_title")) if p)
            match = reason.get("strength")
            reasons.append(
                f"{reason['source']} {match:.2f}" + (f" via {seed}" if seed else "")
                if match is not None
                else reason["source"]
            )
        t.add_row(
            f"{row['artist']} — {row['title']}",
            f"{row['score']:.2f}",
            str(row["source_count"]),
            " | ".join(reasons),
        )
    console.print(t if rows else "[yellow]No candidates yet.[/]")


@candidates_app.command("export")
def candidates_export(
    limit: int = typer.Option(100, help="How many top candidates to export."),
    status: str = typer.Option("metadata", help="Candidate status bucket."),
    output_format: str = typer.Option("json", "--format", "-f", help="Output format: json or csv."),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write to this file; default stdout."
    ),
) -> None:
    """Export top candidates with complete provenance and external ids."""
    from . import candidates

    conn = db.connect()
    db.init_db(conn)
    rows = candidates.review_candidates(conn, limit=limit, status=status)
    try:
        rendered = candidates.serialize_candidate_export(rows, output_format)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(2) from exc
    if output is None:
        typer.echo(rendered, nl=False)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    console.print(f"[green]Exported {len(rows)} candidates[/] to {output}")


@candidates_app.command("harvest-yt-radio")
def candidates_harvest_yt_radio(
    batch: int = typer.Option(25, help="How many library seed recordings to try."),
    radio_limit: int = typer.Option(50, help="Radio tracks to request per seed."),
    min_plays: int = typer.Option(
        20, help="Seed universe = liked ∪ recordings with this many plays."
    ),
    include_fetched: bool = typer.Option(
        False, "--include-fetched", help="Re-run seeds already fetched."
    ),
    pause_s: float = typer.Option(1.0, help="Seconds to pause between seeds."),
    priority: bool = typer.Option(
        False,
        "--priority",
        help="Seed from the techno/melodic-techno "
        "genre space + fave/most-played playlists, most-played first.",
    ),
) -> None:
    """Harvest unauthenticated YT Music radio queues into the candidate ledger."""
    from . import ytmusic

    if not ytmusic.is_available():
        console.print(f"[yellow]{ytmusic._MISSING_YTMUSIC_MSG}[/]")
        raise typer.Exit(1)

    conn = db.connect()
    db.init_db(conn)
    seed_keys = None
    if priority:
        seed_keys = ytmusic.priority_seeds(conn, min_plays=min_plays)
        console.print(
            f"[dim]priority seed pool: {len(seed_keys)} recordings "
            f"(techno space + fave/hot playlists), harvesting up to {batch}[/]"
        )
    res = ytmusic.harvest_radio(
        conn,
        batch=batch,
        radio_limit=radio_limit,
        min_plays=min_plays,
        include_fetched=include_fetched,
        pause_s=pause_s,
        seed_keys=seed_keys,
        progress=lambda i, n, c: console.print(
            f"  …{i}/{n} (fetched {c['fetched']}, candidates {c['candidates']}, "
            f"miss {c['miss']}, err {c['error']})"
        ),
    )
    console.print(
        f"[green]YT radio[/]: {res['fetched']} seeds fetched, "
        f"{res['candidates']} candidate edges, {res['miss']} miss, "
        f"{res['error']} error, scored {res['scored']}"
    )


@candidates_app.command("resolve-deezer")
def candidates_resolve_deezer(
    batch: int = typer.Option(100, help="How many top metadata candidates to search."),
    min_score: float = typer.Option(0.0, help="Only resolve candidates at or above this score."),
    include_attempted: bool = typer.Option(
        False, "--include-attempted", help="Re-run cached misses and prior attempts."
    ),
) -> None:
    """Resolve top candidates to Deezer ids and preview availability."""
    from . import candidate_deezer

    conn = db.connect()
    db.init_db(conn)
    res = candidate_deezer.resolve(
        conn,
        batch=batch,
        min_score=min_score,
        include_attempted=include_attempted,
        progress=lambda i, n, c: console.print(
            f"  ...{i}/{n} (matched {c['matched']}, preview {c['preview']}, "
            f"provider miss {c['provider_miss']}, error {c['error']})"
        ),
    )
    console.print(
        f"[green]Deezer resolve[/]: checked {res['seen']}, "
        f"matched {res['matched']} ({res['preview']} with preview), "
        f"duplicate {res['duplicate']}, provider miss {res['provider_miss']}, "
        f"error {res['error']}, remaining {res['remaining']}"
    )


@candidates_app.command("fetch-previews")
def candidates_fetch_previews(
    batch: int = typer.Option(25, help="How many top resolved candidates to download."),
    min_score: float = typer.Option(0.0, help="Only fetch candidates at or above this score."),
    include_attempted: bool = typer.Option(
        False, "--include-attempted", help="Retry parked preview-download errors."
    ),
) -> None:
    """Download validated preview clips for top resolved candidates."""
    from . import candidate_audio

    conn = db.connect()
    db.init_db(conn)
    res = candidate_audio.fetch_previews(
        conn,
        batch=batch,
        min_score=min_score,
        include_attempted=include_attempted,
        progress=lambda i, n, c: console.print(
            f"  ...{i}/{n} (fetched {c['fetched']}, error {c['error']})"
        ),
    )
    console.print(
        f"[green]Candidate previews[/]: checked {res['seen']}, "
        f"fetched {res['fetched']}, error {res['error']}, "
        f"remaining {res['remaining']}"
    )


@candidates_app.command("repair-preview-403s")
def candidates_repair_preview_403s(
    batch: int = typer.Option(60, help="Maximum parked 403 rows to revalidate."),
    min_score: float = typer.Option(0.0, help="Only repair candidates at or above this score."),
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Revalidate and update rows; default is dry-run.",
    ),
) -> None:
    """Revalidate previews parked after expired signed Deezer URLs returned 403."""
    from . import candidate_audio

    conn = db.connect()
    db.init_db(conn)
    res = candidate_audio.repair_parked_403s(
        conn,
        apply=apply,
        batch=batch,
        min_score=min_score,
        progress=lambda i, n, c: console.print(
            f"  ...{i}/{n} (fetched {c['fetched']}, provider miss "
            f"{c['provider_miss']}, no preview {c['source_not_covered']}, "
            f"error {c['error']})"
        ),
    )
    console.print(res)


@candidates_app.command("embed")
def candidates_embed(
    batch: int = typer.Option(25, help="Maximum candidate clips per model worker."),
    top_k: int = typer.Option(10, help="Library neighbors averaged for content reranking."),
) -> None:
    """Embed candidate previews with EffNet/CLAP, then rerank against the library."""
    import subprocess
    from . import candidate_embeddings

    conn = db.connect()
    db.init_db(conn)
    conn.close()
    scripts = Path(__file__).parent.parent / "scripts"
    commands = [
        ["uv", "run", str(scripts / "embed_worker.py"), "--candidates-only", f"--batch={batch}"],
        [
            "uv",
            "run",
            str(scripts / "clap_worker.py"),
            "embed",
            "--candidates-only",
            f"--batch={batch}",
        ],
    ]
    for command in commands:
        console.print(f"[dim]$ {' '.join(command)}[/]")
        result = subprocess.run(command)
        if result.returncode:
            raise typer.Exit(result.returncode)

    conn = db.connect()
    db.init_db(conn)
    finalized = candidate_embeddings.finalize(conn)
    reranked = candidate_embeddings.rerank(conn, top_k=top_k)
    console.print(
        f"[green]Candidate embeddings[/]: finalized {finalized}, "
        f"reranked {reranked['ranked']}, duplicates {reranked['duplicates']}"
    )


@candidates_app.command("rerank")
def candidates_rerank(
    top_k: int = typer.Option(10, help="Library neighbors averaged per embedding space."),
    full: bool = typer.Option(False, "--full", help="Force a complete rebuild."),
) -> None:
    """Refresh changed content scores and exact-ISRC duplicate exclusions."""
    from . import candidate_embeddings

    conn = db.connect()
    db.init_db(conn)
    result = candidate_embeddings.rerank(conn, top_k=top_k, force=full)
    console.print(
        f"[green]Candidate rerank[/]: ranked {result['ranked']}, duplicates {result['duplicates']}"
    )


@candidates_app.command("ranked")
def candidates_ranked(
    limit: int = typer.Option(25, help="How many embedded candidates to show."),
) -> None:
    """Show the provenance plus EffNet/CLAP/style candidate rerank."""
    from . import candidate_embeddings

    conn = db.connect()
    db.init_db(conn)
    rows = candidate_embeddings.ranked_candidates(conn, limit=limit)
    table = Table(title="Embedded recommendation candidates")
    table.add_column("artist - title")
    table.add_column("combined", justify="right")
    table.add_column("prov", justify="right")
    table.add_column("content", justify="right")
    table.add_column("EffNet neighbor")
    table.add_column("top style")
    for row in rows:
        neighbor = " - ".join(
            value
            for value in (row["effnet_neighbor_artist"], row["effnet_neighbor_title"])
            if value
        )
        table.add_row(
            f"{row['artist']} - {row['title']}",
            f"{row['combined_score']:.3f}",
            f"{row['provenance_score']:.1f}",
            f"{row['content_score']:.3f}",
            neighbor,
            (row["top_style"] or "").split("---")[-1],
        )
    console.print(table if rows else "[yellow]No embedded candidates yet.[/]")


@candidates_app.command("resolve-yt-laion")
def candidates_resolve_yt_laion(
    batch: int = typer.Option(100, help="How many library seed recordings to try."),
    min_plays: int = typer.Option(
        20, help="Seed universe = liked ∪ recordings with this many plays."
    ),
    include_matched: bool = typer.Option(
        False, "--include-matched", help="Re-check already cached rows."
    ),
) -> None:
    """Resolve library seeds to YT Music ids via the local LAION-DISCO-12M catalog."""
    from . import ytmusic

    conn = db.connect()
    db.init_db(conn)
    res = ytmusic.resolve_laion_seeds(
        conn,
        batch=batch,
        min_plays=min_plays,
        include_matched=include_matched,
        progress=lambda i, n, c: console.print(
            f"  …{i}/{n} (matched {c['matched']}, miss {c['miss']})"
        ),
    )
    console.print(
        f"[green]LAION YT resolve[/]: checked {res['seen']}, "
        f"matched {res['matched']}, miss {res['miss']}"
    )


# --- taste: the playlist-as-region membership engine -----------------------


def _find_playlist(conn, query: str) -> dict | None:
    """Resolve a name fragment (or exact id) to one authored playlist."""
    from . import taste

    pls = taste.authored_playlists(conn)
    exact = [
        p for p in pls if p["playlist_id"] == query or (p["name"] or "").lower() == query.lower()
    ]
    if exact:
        return exact[0]
    matches = [p for p in pls if query.lower() in (p["name"] or "").lower()]
    if len(matches) > 1:
        console.print(
            "[yellow]Ambiguous[/] — matches: " + ", ".join(p["name"] for p in matches[:8])
        )
        return None
    return matches[0] if matches else None


@taste_app.command("build")
def taste_build() -> None:
    """(Re)build taste regions from your authored playlists' latest snapshots."""
    from . import taste

    conn = db.connect()
    db.init_db(conn)
    res = taste.build(conn)
    console.print(
        f"[green]taste regions[/]: {res['regions_built']} built from "
        f"{res['playlists']} authored playlists "
        f"({res['too_small']} too small to model)"
    )


@taste_app.command("list")
def taste_list() -> None:
    """Show every taste region with its size and tightness."""
    from . import taste

    conn = db.connect()
    db.init_db(conn)
    table = Table(title="Taste regions (authored playlists as embedding-space labels)")
    table.add_column("playlist")
    table.add_column("members", justify="right")
    table.add_column("embedded", justify="right")
    table.add_column("cohesion", justify="right")
    for r in taste.regions(conn):
        table.add_row(
            r["name"],
            str(r["n_members"]),
            str(r["n_embedded"]),
            f"{r['spread']:.3f}" if r["spread"] else "—",
        )
    console.print(table)


@taste_app.command("eval")
def taste_eval(
    holdout: float = typer.Option(0.2, help="Fraction of members held out per playlist."),
    negatives: int = typer.Option(20, help="Sampled library negatives per held-out positive."),
) -> None:
    """Grade the membership engine: held-out members vs library negatives."""
    from . import taste

    conn = db.connect()
    db.init_db(conn)
    rows = taste.evaluate(conn, holdout_frac=holdout, negatives_per_pos=negatives)
    if not rows:
        console.print("[yellow]No playlist has enough embedded members to evaluate.[/]")
        return
    table = Table(title="Taste model eval (centroid + k-NN baseline)")
    table.add_column("playlist")
    table.add_column("members", justify="right")
    table.add_column("held out", justify="right")
    table.add_column("P@10", justify="right")
    table.add_column("AUC", justify="right")
    for r in rows:
        table.add_row(
            r["playlist"],
            str(r["members"]),
            str(r["held_out"]),
            f"{r['precision_at_10']:.2f}",
            f"{r['auc']:.3f}",
        )
    mean_auc = sum(r["auc"] for r in rows) / len(rows)
    mean_p10 = sum(r["precision_at_10"] for r in rows) / len(rows)
    console.print(table)
    console.print(
        f"macro avg — P@10 [green]{mean_p10:.2f}[/], AUC [green]{mean_auc:.3f}[/] "
        f"over {len(rows)} playlists"
    )


@taste_app.command("fit")
def taste_fit(
    playlist: str = typer.Argument(..., help="Playlist name fragment or id."),
    limit: int = typer.Option(15, help="How many suggestions to show."),
) -> None:
    """Top library tracks that belong in a playlist but aren't in it (preview only)."""
    from . import taste

    conn = db.connect()
    db.init_db(conn)
    pl = _find_playlist(conn, playlist)
    if not pl:
        console.print(f"[red]No authored playlist matches[/] {playlist!r}")
        raise typer.Exit(1)
    region = taste.load_region(conn, pl["playlist_id"], pl["name"])
    if region is None:
        console.print("[yellow]Too few embedded members to model this playlist.[/]")
        raise typer.Exit(1)
    table = Table(title=f"Belongs in “{pl['name']}” (not in it today)")
    table.add_column("artist - title")
    table.add_column("membership", justify="right")
    table.add_column("plays", justify="right")
    for a in taste.additions(conn, region, limit=limit):
        table.add_row(f"{a['artist']} - {a['title']}", f"{a['score']:.3f}", str(a["plays"]))
    console.print(table)


# --- health: curation assist over authored playlists -----------------------


@health_app.command("report")
def health_report(
    playlist: str = typer.Argument(..., help="Playlist name fragment or id."),
) -> None:
    """Full health readout: coherence lenses, misfits, split check."""
    from . import health

    conn = db.connect()
    db.init_db(conn)
    pl = _find_playlist(conn, playlist)
    if not pl:
        console.print(f"[red]No authored playlist matches[/] {playlist!r}")
        raise typer.Exit(1)
    rep = health.report(conn, pl["playlist_id"], pl["name"])
    if rep is None:
        console.print("[yellow]Too few embedded members to analyze.[/]")
        raise typer.Exit(1)
    console.print(f"[bold]{rep['name']}[/] — {rep['members']} tracks ({rep['embedded']} embedded)")
    console.print(f"  sonic cohesion   [green]{rep['sonic_cohesion']}[/]")
    console.print(
        f"  style entropy    [green]{rep['style_entropy']}[/]  "
        + ", ".join(f"{s} ×{n}" for s, n in rep["styles"][:5])
    )
    console.print(f"  co-listen density [green]{rep['colisten_density']}[/]")
    if rep["misfits"]:
        console.print("  [bold]misfits[/] (fit the playlist worst):")
        for m in rep["misfits"]:
            console.print(f"    {m['artist']} - {m['title']}  (z={m['z']})")
    if rep["split"]:
        g = rep["split"]["groups"]
        console.print(
            f"  [bold]split suggestion[/]: two clusters of {g[0]['size']} + {g[1]['size']} "
            f"tracks would each be tighter (+{rep['split']['gain']})"
        )


@health_app.command("scan")
def health_scan(
    playlist: str = typer.Argument(None, help="Playlist name fragment/id (default: all regions)."),
    gapfill: int = typer.Option(15, help="Max gap-fill additions to propose per playlist."),
) -> None:
    """File misfit-removal and gap-fill-addition proposals into the review queue."""
    from . import health, taste

    conn = db.connect()
    db.init_db(conn)
    targets = []
    if playlist:
        pl = _find_playlist(conn, playlist)
        if not pl:
            console.print(f"[red]No authored playlist matches[/] {playlist!r}")
            raise typer.Exit(1)
        targets = [pl]
    else:
        region_ids = {r["playlist_id"]: r["name"] for r in taste.regions(conn)}
        targets = [{"playlist_id": pid, "name": name} for pid, name in region_ids.items()]
    total = 0
    for t in targets:
        res = health.scan(conn, t["playlist_id"], t["name"], gapfill_limit=gapfill)
        total += res.get("proposed", 0)
        console.print(
            f"  {t['name']}: {res.get('proposed', 0)} proposals"
            + (f" ({res['skipped']})" if res.get("skipped") else "")
        )
    console.print(
        f"[green]{total}[/] proposals queued — review at /review or `acetate review list`"
    )


# --- review: the proposals queue --------------------------------------------


@review_app.command("list")
def review_list(
    limit: int = typer.Option(30, help="Max proposals to show."),
) -> None:
    """Pending proposals, best score first."""
    from . import proposals

    conn = db.connect()
    db.init_db(conn)
    rows = proposals.pending(conn, limit=limit)
    if not rows:
        console.print("[yellow]Queue is empty.[/]")
        return
    table = Table(title="Pending proposals")
    table.add_column("id", justify="right")
    table.add_column("playlist")
    table.add_column("action")
    table.add_column("artist - title")
    table.add_column("score", justify="right")
    table.add_column("source")
    for r in rows:
        table.add_row(
            str(r["id"]),
            r["playlist_name"] or r["playlist_id"],
            r["action"],
            f"{r['artist']} - {r['title']}",
            f"{r['score']:.3f}" if r["score"] is not None else "—",
            r["source"],
        )
    console.print(table)
    console.print(f"counts: {proposals.counts(conn)}")


@review_app.command("accept")
def review_accept(proposal_id: int = typer.Argument(...)) -> None:
    """Accept one proposal (records the verdict; applying it is on you / the builder)."""
    from . import proposals

    conn = db.connect()
    db.init_db(conn)
    row = proposals.decide(conn, proposal_id, "accepted")
    console.print(
        f"[green]accepted[/] #{proposal_id}: {row['artist']} - {row['title']} "
        f"→ {row['playlist_name']}"
        if row
        else f"[red]no proposal #{proposal_id}[/]"
    )


@review_app.command("reject")
def review_reject(proposal_id: int = typer.Argument(...)) -> None:
    """Reject one proposal (a durable verdict — it won't be re-proposed)."""
    from . import proposals

    conn = db.connect()
    db.init_db(conn)
    row = proposals.decide(conn, proposal_id, "rejected")
    console.print(
        f"[yellow]rejected[/] #{proposal_id}: {row['artist']} - {row['title']}"
        if row
        else f"[red]no proposal #{proposal_id}[/]"
    )


@review_app.command("feedback")
def review_feedback() -> None:
    """Self-grading report: of everything acetate surfaced, what got played after?"""
    from . import proposals

    conn = db.connect()
    db.init_db(conn)
    rows = proposals.feedback(conn)
    if not rows:
        console.print("[yellow]Nothing surfaced yet — no feedback to grade.[/]")
        return
    table = Table(title="Recommendation feedback (plays after surfacing)")
    table.add_column("surface")
    table.add_column("recommended", justify="right")
    table.add_column("played after", justify="right")
    table.add_column("hit rate", justify="right")
    for r in rows:
        table.add_row(
            r["surface"], str(r["recommended"]), str(r["played_after"]), f"{r['hit_rate']:.1%}"
        )
    console.print(table)


# --- discover: outside-library recommendation consumers ---------------------


@discover_app.command("artists")
def discover_artists(
    playlist: str = typer.Argument(..., help="Playlist name fragment or id."),
    limit: int = typer.Option(20, help="Maximum artists to return."),
    novelty: str = typer.Option("outside_library", help="outside_library or outside_playlist."),
    min_seed_coverage: int = typer.Option(
        2, help="Distinct playlist tracks that must point to an artist."
    ),
) -> None:
    """Rank new artists by votes from distinct tracks in an authored playlist."""
    from . import artist_discovery

    conn = db.connect()
    db.init_db(conn)
    try:
        result = artist_discovery.discover(
            conn,
            playlist,
            limit=limit,
            novelty=novelty,
            min_seed_coverage=min_seed_coverage,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    if result.get("error"):
        console.print(f"[red]{result['error']}[/]")
        if result.get("matches"):
            console.print("Matches: " + ", ".join(result["matches"]))
        raise typer.Exit(1)
    if not result["artists"]:
        console.print(f"[yellow]{result.get('reason', 'No matching artists.')}[/]")
        return
    table = Table(title=f"Artist discovery — {result['playlist']['name']}")
    table.add_column("artist")
    table.add_column("score", justify="right")
    table.add_column("seeds", justify="right")
    table.add_column("coverage", justify="right")
    table.add_column("best entry point")
    for artist in result["artists"]:
        connection = artist["connections"][0]
        table.add_row(
            artist["artist"],
            f"{artist['score']:.3f}",
            str(artist["seed_count"]),
            f"{artist['coverage']:.1%}",
            connection["candidate_title"],
        )
    console.print(table)


@discover_app.command("for")
def discover_for(
    playlist: str = typer.Argument(..., help="Playlist name fragment or id."),
    limit: int = typer.Option(15, help="Max candidate proposals to file."),
) -> None:
    """Score embedded candidates against one playlist's region → review queue."""
    from . import discovery

    conn = db.connect()
    db.init_db(conn)
    pl = _find_playlist(conn, playlist)
    if not pl:
        console.print(f"[red]No authored playlist matches[/] {playlist!r}")
        raise typer.Exit(1)
    res = discovery.for_playlist(conn, pl["playlist_id"], pl["name"], limit=limit)
    if res.get("skipped"):
        console.print(f"[yellow]{res['skipped']}[/]")
    else:
        console.print(
            f"[green]{res['proposed']}[/] candidate proposals for "
            f"“{res['playlist']}” (scored {res['candidates_scored']})"
        )


@publish_app.command("discovery")
def publish_discovery(
    size: int = typer.Option(30, help="Playlist size."),
) -> None:
    """Build/refresh the 🔭 Discovery playlist from top reranked candidates."""
    _require_creds()
    from . import discovery

    conn = db.connect()
    db.init_db(conn)
    sp = auth.client(open_browser=False)
    res = discovery.build_discovery_playlist(conn, sp, size=size)
    if res.get("skipped"):
        console.print(
            f"[yellow]{res['name']}[/]: no playable candidates yet "
            f"({res['no_spotify_id']} lack a Spotify id — run candidates "
            "harvest/resolve passes)"
        )
    else:
        console.print(
            f"[green]{res['name']}[/]: {res['tracks']} tracks "
            f"(https://open.spotify.com/playlist/{res['playlist_id']}); "
            f"{res['no_spotify_id']} candidates skipped for missing Spotify ids"
        )


# --- filing: the liked-but-unplaylisted inbox --------------------------------


@filing_app.command("queue")
def filing_queue(
    min_plays: int = typer.Option(1, help="Only liked songs with at least this many plays."),
    limit: int = typer.Option(30, help="Rows to show."),
) -> None:
    """Liked songs in no playlist, most-played first, with best-fit targets."""
    from . import filing

    conn = db.connect()
    db.init_db(conn)
    q = filing.queue(conn, min_plays=min_plays, limit=limit)
    total = filing.unfiled_count(conn, min_plays=min_plays)
    table = Table(title=f"Filing inbox — {total} unfiled liked songs")
    table.add_column("plays", justify="right")
    table.add_column("track")
    table.add_column("best-fit playlists (score, * = strong)")
    for it in q:
        tg = ", ".join(
            f"{t['name']} ({t['score']}{'*' if t['strong'] else ''})" for t in it["targets"]
        )
        table.add_row(str(it["plays"]), f"{it['artist']} — {it['title']}", tg)
    console.print(table if q else "[green]nothing to file[/]")


@filing_app.command("history")
def filing_history(limit: int = typer.Option(30, help="Rows to show.")) -> None:
    """The filing ledger — every add acetate made to your playlists, newest first."""
    from . import filing

    conn = db.connect()
    db.init_db(conn)
    table = Table(title="Filing ledger")
    table.add_column("id", justify="right")
    table.add_column("status")
    table.add_column("track")
    table.add_column("playlist")
    table.add_column("when")
    for a in filing.history(conn, limit=limit):
        table.add_row(
            str(a["id"]),
            a["status"],
            f"{a['artist']} — {a['title']}",
            a["playlist_name"] or "",
            (a["added_at"] or "")[:16],
        )
    console.print(table)


@filing_app.command("file")
def filing_file_cmd(
    track_id: str = typer.Argument(..., help="Spotify track id to file."),
    playlist: str = typer.Argument(..., help="Target playlist name fragment or id."),
) -> None:
    """Append a liked track to a playlist (append-only) and log it to the ledger."""
    _require_creds()
    from . import filing

    conn = db.connect()
    db.init_db(conn)
    pl = _find_playlist(conn, playlist)
    if not pl:
        console.print(f"[red]No authored playlist matches[/] {playlist!r}")
        raise typer.Exit(1)
    rec = conn.execute(
        "SELECT tr.recording_key, rs.artist, rs.title FROM track_recording tr "
        "JOIN recording_stats rs ON rs.recording_key=tr.recording_key "
        "WHERE tr.track_id=?",
        (track_id,),
    ).fetchone()
    if not rec:
        console.print(f"[red]Unknown track[/] {track_id!r}")
        raise typer.Exit(1)
    sp = auth.client(open_browser=False)
    aid = filing.file_track(
        conn,
        sp,
        recording_key=rec["recording_key"],
        track_id=track_id,
        playlist_id=pl["playlist_id"],
        playlist_name=pl["name"],
        artist=rec["artist"],
        title=rec["title"],
    )
    console.print(
        f"[green]filed[/] {rec['artist']} — {rec['title']} → {pl['name']} "
        f"(undo: [bold]acetate filing undo {aid}[/])"
    )


@filing_app.command("undo")
def filing_undo_cmd(
    action_id: int = typer.Argument(..., help="filing_actions id (see `filing history`)."),
) -> None:
    """Undo one filing — remove exactly the track acetate added, nothing else."""
    _require_creds()
    from . import filing

    conn = db.connect()
    db.init_db(conn)
    sp = auth.client(open_browser=False)
    row = filing.undo(conn, sp, action_id)
    if not row:
        console.print(
            f"[yellow]nothing to undo[/] for action {action_id} (unknown or already undone)"
        )
        raise typer.Exit(1)
    console.print(
        f"[green]undone[/] — removed {row['artist']} — {row['title']} from {row['playlist_name']}"
    )


# --- playlists: profiles, personas, labels -----------------------------------


def _find_profiled_playlist(conn, query: str) -> dict | None:
    """Resolve a name fragment (or id) against playlist_stats (any snapshotted
    playlist, own/followed/deleted alike)."""
    rows = [dict(r) for r in conn.execute("SELECT * FROM playlist_stats")]
    exact = [
        r for r in rows if r["playlist_id"] == query or (r["name"] or "").lower() == query.lower()
    ]
    if exact:
        return exact[0]
    matches = [r for r in rows if query.lower() in (r["name"] or "").lower()]
    if len(matches) > 1:
        console.print(
            "[yellow]Ambiguous[/] — matches: " + ", ".join(r["name"] for r in matches[:8])
        )
        return None
    return matches[0] if matches else None


@playlists_app.command("profile")
def playlists_profile() -> None:
    """Rebuild playlist_members + playlist_stats (usage, add-cadence, personas)."""
    from . import playlist_profile

    conn = db.connect()
    db.init_db(conn)
    res = playlist_profile.rebuild(conn)
    console.print(
        f"[green]profiled[/] {res['playlists']} playlists ({res['members']:,} member rows)"
    )
    for r in conn.execute(
        "SELECT persona, COUNT(*) n FROM playlist_stats GROUP BY persona ORDER BY n DESC"
    ):
        console.print(f"  {r['persona']}: {r['n']}")


@playlists_app.command("list")
def playlists_list(
    persona: str = typer.Option(
        None, help="Filter: rotation|archive|one-shot|dormant|acetate|deleted"
    ),
    all: bool = typer.Option(False, "--all", help="Include followed/others' playlists."),
    limit: int = typer.Option(40, help="Rows to show."),
) -> None:
    """Playlists ranked by recent member listening, with personas."""
    from . import playlist_profile

    conn = db.connect()
    db.init_db(conn)
    rows = playlist_profile.overview(conn, persona=persona, own_only=not all, limit=limit)
    table = Table(title="Playlist profiles" + (f" · {persona}" if persona else ""))
    table.add_column("playlist")
    table.add_column("persona")
    table.add_column("tracks", justify="right")
    table.add_column("plays 180d", justify="right")
    table.add_column("from-playlist", justify="right")
    table.add_column("last add")
    table.add_column("label")
    for r in rows:
        table.add_row(
            r["name"] or r["playlist_id"],
            r["persona"],
            str(r["n_tracks"]),
            str(r["member_plays_180d"]),
            str(r["plays_from"]),
            (r["last_added_at"] or "")[:10],
            r["user_label"] or "",
        )
    console.print(
        table if rows else "[yellow]No profiles yet — run `acetate playlists profile`.[/]"
    )


@playlists_app.command("label")
def playlists_label(
    playlist: str = typer.Argument(..., help="Playlist name fragment or id."),
    label: str = typer.Argument(..., help="fave | rotation | archive | one-shot | dormant | mute"),
    note: str = typer.Option(None, help="Optional note (why)."),
) -> None:
    """Record your own verdict on a playlist — overrides the persona heuristic."""
    from . import playlist_profile

    if label not in playlist_profile.KNOWN_LABELS:
        console.print(
            f"[red]Unknown label[/] — use one of {', '.join(playlist_profile.KNOWN_LABELS)}"
        )
        raise typer.Exit(1)
    conn = db.connect()
    db.init_db(conn)
    pl = _find_profiled_playlist(conn, playlist)
    if not pl:
        console.print(
            f"[red]No profiled playlist matches[/] {playlist!r} "
            "(run `acetate playlists profile` first)"
        )
        raise typer.Exit(1)
    playlist_profile.set_label(conn, pl["playlist_id"], label, note)
    console.print(f"[green]{pl['name']}[/] labeled [bold]{label}[/]")


# --- genre playlists: classify + append-only publish ------------------------


@genre_app.command("classify")
def genre_classify(
    genre: str = typer.Argument(..., help="Genre name, e.g. 'melodic techno'."),
    tags: str = typer.Option(None, help="Comma-separated Last.fm tag aliases (default: the name)."),
    styles: str = typer.Option(
        None, help="Comma-separated discogs400 style keywords (default: the name)."
    ),
    limit: int = typer.Option(30, help="Rows to display."),
) -> None:
    """Preview the three-signal genre classification (no writes)."""
    from . import genre_playlists

    conn = db.connect()
    db.init_db(conn)
    rows = genre_playlists.classify(
        conn,
        genre,
        tags=[t.strip() for t in tags.split(",")] if tags else None,
        style_keywords=[s.strip() for s in styles.split(",")] if styles else None,
    )
    anchors = genre_playlists.anchor_playlists(conn, genre)
    console.print(f"anchors: {', '.join(a['name'] for a in anchors) or '[yellow]none[/]'}")
    table = Table(title=f"“{genre}” classification ({len(rows)} matches)")
    table.add_column("artist - title")
    table.add_column("conf", justify="right")
    table.add_column("region", justify="right")
    table.add_column("style", justify="right")
    table.add_column("lastfm", justify="right")
    table.add_column("tier")
    for r in rows[:limit]:
        table.add_row(
            f"{r['artist']} - {r['title']}",
            f"{r['confidence']:.2f}",
            f"{r['region']:.2f}",
            f"{r['style']:.2f}",
            f"{r['lastfm']:.2f}",
            "[green]auto[/]" if r["tier"] == "auto" else "review",
        )
    console.print(table)
    n_auto = sum(1 for r in rows if r["tier"] == "auto")
    console.print(f"{n_auto} auto / {len(rows) - n_auto} review")


@publish_app.command("genre")
def publish_genre(
    genre: str = typer.Argument(..., help="Genre name, e.g. 'melodic techno'."),
    tags: str = typer.Option(None, help="Comma-separated Last.fm tag aliases."),
    styles: str = typer.Option(None, help="Comma-separated discogs400 style keywords."),
) -> None:
    """Classify and sync the 🎧 genre playlist (append-only); borderline → review queue."""
    _require_creds()
    from . import genre_playlists

    conn = db.connect()
    db.init_db(conn)
    sp = auth.client(open_browser=False)
    res = genre_playlists.sync(
        conn,
        sp,
        genre,
        tags=[t.strip() for t in tags.split(",")] if tags else None,
        style_keywords=[s.strip() for s in styles.split(",")] if styles else None,
    )
    if res.get("skipped"):
        console.print(
            f"[yellow]{res['name']}[/]: nothing confident enough to publish yet; "
            f"{res['review_proposed']} sent to review"
        )
    else:
        console.print(
            f"[green]{res['name']}[/]: {res['tracks']} tracks "
            f"(+{res['appended']} new, {res['review_proposed']} to review) "
            f"(https://open.spotify.com/playlist/{res['playlist_id']})"
        )


if __name__ == "__main__":
    app()
