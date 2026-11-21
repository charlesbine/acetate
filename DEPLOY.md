# Self-Hosting Acetate with Docker Compose

Acetate runs as two containers built from one slim image — **`app`** (FastAPI + HTMX
dashboard on `:8765`) and **`scheduler`** (APScheduler running periodic jobs) — plus an
optional **`embed`** sidecar (`--profile embeddings`) for on-demand and background neural
audio embeddings. All containers share `./data` (SQLite database, cached Spotify token,
preview audio, and model weights) and read configuration from `.env`.

## 1. Clone and prepare `./data`

```sh
git clone https://github.com/charlesbine/acetate.git
cd acetate
mkdir -p data
```

If you already ran `acetate auth` and `acetate import history` locally on a workstation,
copy your database and cached Spotify refresh token to the server:

```sh
scp data/acetate.db            myserver:~/acetate/data/
scp data/.spotify_token_cache  myserver:~/acetate/data/
```

The cached refresh token makes authentication headless — the containers never open a
browser. If the token is ever revoked, re-run `uv run acetate auth` locally and copy
`data/.spotify_token_cache` back up.

## 2. Configure `.env`

Copy `.env.example` to `.env` and fill in plaintext credentials:

```sh
cp .env.example .env
chmod 600 .env
```

Minimal `.env` for a self-hosted server:

```dotenv
SPOTIPY_CLIENT_ID=your_spotify_client_id
SPOTIPY_CLIENT_SECRET=your_spotify_client_secret
SPOTIPY_REDIRECT_URI=http://127.0.0.1:8080/callback

# Optional integrations
LASTFM_API_KEY=
OPENROUTER_API_KEY=

# Display timezone (DB timestamps remain UTC)
ACETATE_TZ=America/New_York

# Auth mode inside Docker (defaults to authelia; set local ONLY on a private loopback/VPN)
ACETATE_AUTH_MODE=authelia
```

*(Optional)* If you store `op://` 1Password secret references in your workstation `.env`,
resolve them to plaintext before copying to the server (`op inject -i .env -o .env.server`),
because the container image does not bundle the `op` CLI. `.env*` files are gitignored.

## 3. Launch

```sh
docker compose up -d --build
docker compose logs -f scheduler   # watch periodic jobs run
```

The dashboard binds to `127.0.0.1:8765` by default. Reach it through a reverse proxy
(Traefik, Caddy, or NGINX) or an SSH tunnel (`ssh -L 8765:127.0.0.1:8765 myserver`).

> **SSH-tunnel-only deployments (no Authelia):** the Docker image and Compose file default to
> `ACETATE_AUTH_MODE=authelia` (fail-closed) and will return HTTP `401` when no reverse proxy
> injects `Remote-User`. If you access the dashboard exclusively over an SSH tunnel to
> `127.0.0.1:8765`, set `ACETATE_AUTH_MODE=local` in `.env` — and keep port `8765` bound strictly
> to loopback.

For a shared or network-reachable deployment, put **Authelia forward-auth** in front of
the dashboard and keep `ACETATE_AUTH_MODE=authelia` (the Docker/Compose default). See
[`docs/authelia-vps-setup.md`](docs/authelia-vps-setup.md) for the proxy header contract
and `viewer` / `query` / `admin` role groups. Do not expose port `8765` directly: Acetate
trusts identity headers only because the reverse proxy is the sole caller.

## 4. Optional: embedding worker sidecar (`--profile embeddings`)

Admin chat tools (`request_embeddings` / `embedding_request_status`) and background
embedding backfills can run inside an isolated heavy ML sidecar while keeping the main
`app` and `scheduler` containers slim:

```sh
docker compose --profile embeddings up -d --build
docker compose logs -f embed
```

The `embed` service shares `./data` (database, preview audio, downloaded weights, and
lock files), caps CPU usage (`cpus: "2.0"` in `docker-compose.yml`), and requires no
API credentials. On first run it downloads model weights into `data/models/`. Note that
Essentia/MTG weights (`discogs-effnet` and `genre_discogs400`) are licensed
**CC BY-NC-ND 4.0** (personal / non-commercial use only).

For one-off local processing on a workstation using the isolated PEP-723 script envs:

```sh
uv run python scripts/embed_loop.py --once --local
```

Guardrails enforced in Python and SQLite for on-demand requests:

- At most **5 tracks per chat turn** and **20 outstanding tracks** queued overall.
- Exact identity matching, deduplication, and transactional capacity checks across web processes.
- **One active request controller** and **one inference process** across requests, background
  workers, CLI embedding commands, and CLAP text queries on the same shared data volume.
- Serial preview fetching and per-stage inference timeouts (**10 minutes** per model stage).
- Up to **3 attempts per recording**; failures remain visible and require an explicit retry.
- Background cadence is tuned with `EMBED_SLEEP` (default `900` seconds), `EMBED_BATCH`,
  `EMBED_CLAP_BATCH`, and `EMBED_CLAP_MICROBATCH`.

## 5. Optional: resource caps for `scheduler`

On small multi-tenant boxes, you can cap the scheduler's CPU share in `docker-compose.yml`
or a `docker-compose.override.yml`:

```yaml
services:
  scheduler:
    cpus: "1.0"
```

Or apply a live limit without restarting:

```sh
docker update --cpus 1 acetate-scheduler-1
```

## 6. Schedule (UTC; override in `docker-compose.yml` → `scheduler` environment)

| Job | Cadence | Default | Env var |
|-----|---------|---------|---------|
| `poll` — recently-played → history | every N minutes | `15` | `ACETATE_POLL_MINUTES` |
| `backup` — liked songs + playlists (store-on-change) | daily | `04:00` | `ACETATE_BACKUP_HOUR` |
| `canon` — rebuild duplicate-ID → recording map | daily | `backup + :30` | — |
| `monthly` — 📆 Liked + 🔥 Top Played playlists | daily | `05:00` | `ACETATE_MONTHLY_HOUR` |
| `gems` — rotate the 💎 Lost Gems playlist | weekly (Sun) | `06:00` | — |
| `hydrate` — artist images & metadata (≤3,000/run) | daily | `03:00` | — |
| `backfill` — Last.fm genre tags (chunk) | hourly `:30` | `2000` | `ACETATE_BACKFILL_BATCH` (`0` disables) |

`canon` runs right after `backup` (so the recording map sees newly ingested tracks) and
before `monthly`/`gems`, which group through it. Polling every 15 minutes reads up to 50
tracks per call (200 plays/hour capacity), avoiding dropped plays in normal listening.

## 7. Updating

Pull `main` and rebuild the Compose stack on your server:

```sh
git fetch && git pull --ff-only
docker compose up -d --build
```

If you use [`just`](https://github.com/casey/just), copy an SSH alias into `~/.ssh/config`
(or set `ACETATE_DEPLOY_HOST` in a gitignored `.env.deploy`) and run `just deploy`,
`just logs`, `just ps`, or `just restart` from your workstation.

```sshconfig
# ~/.ssh/config
Host acetate
    HostName <server-ip-or-hostname>
    User <you>
```

> Want hands-off pull-based updates? Add a `systemd` timer on the server that runs
> `git fetch && git pull --ff-only && docker compose up -d --build` periodically.

## 8. Reconciling workstation and server databases

If you run heavy embedding compute on a laptop/desktop and run the always-on poller on a
server, **do not overwrite `data/acetate.db` on the server**. The two copies have
intentionally separate table ownership:

- **Server-owned (always preserved):** play events, Spotify backup snapshots, Last.fm
  genre backfill, chat threads, scheduler cursors/runs, managed-playlist state, proposal
  verdicts, recommendation events, and playlist labels.
- **Workstation-owned (promoted):** audio-cache ledgers, embeddings, Discogs styles and
  UMAP projections, Last.fm similarity/artist caches, candidate frontier tables, harvest
  state, classic audio features, and MusicBrainz MBIDs.
- **Rebuilt after merge:** canonical track/recording mappings, `recording_stats`,
  playlist profiles, and taste regions.

Preview the merge (read-only):

```sh
just db-reconcile
```

Apply it safely (checksummed bundle, additive audio rsync, pre-apply backup, atomic swap,
integrity verification):

```sh
just db-reconcile-apply
```

If the `embed` sidecar runs on the server, include `embed` in `ACETATE_SERVICES` inside
`.env.deploy` so reconciliation stops all writers before swapping the DB.

Roll back to a pre-reconcile backup if needed:

```sh
just db-reconcile-restore acetate-pre-reconcile-YYYYMMDDTHHMMSSZ.db
```
