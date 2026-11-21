# Roadmap

Acetate is a self-hosted personal Spotify archive, audio-embedding lab, and AI playlist
workbench. This document tracks what is shipped and what is planned next.

## Shipped

- **Archive & Backup**
  - Single-file SQLite store (`data/acetate.db`) with schema migrations and WAL mode.
  - Full Spotify GDPR "Extended Streaming History" import plus a 15-minute recently-played
    poller so listening history never drops plays.
  - Append-only, store-on-change snapshots of liked songs and playlists (preserving a trace
    even if a playlist is later edited or deleted).
  - Canonical recording deduplication (`track_recording` + materialized `recording_stats`)
    so remasters, single vs. album releases, and regional re-issues roll up cleanly.
- **Analytics & Dashboard**
  - FastAPI + HTMX web app: **Home**, **Stats**, **Wrapped** (year / quarter / month / week),
    **Lost Gems** (faded / never-played / under-explored / re-ups), **Hyperspace**,
    **Playlists**, **Smart & Curated Playlists**, **Playlist Health**, **Filing Inbox**,
    **Review Queue**, **Chat**, and **Pipeline**.
  - On-demand play refresh from the UI and chat, plus a timezone-aware display layer.
  - Optional **Authelia** forward-auth integration (`viewer`, `query`, and `admin` roles).
- **Neural Audio Layer & Hyperspace**
  - Resumable 30-second preview fetcher (Deezer ISRC-first, with duration-validated search fallbacks).
  - Isolated PEP-723 ML workers (`discogs-effnet`, 400-style `genre_discogs400` head,
    pinned `LAION-CLAP` text↔audio space, UMAP 3D + HDBSCAN clustering) plus an optional
    Docker Compose `embed` sidecar.
  - Interactive 3D Three.js **Hyperspace** explorer with style / era / cluster / plays /
    rarity / label coloring, hover details, click-to-preview audio, and CLAP natural-language search.
- **Taste Model, Curation & Discovery**
  - Playlist-as-region taste model (`acetate/taste.py`) powering **Playlist Health**
    (cohesion, style entropy, misfit detection, split check), **Filing Inbox** (one-click
    append-only filing with ledgered undo), **Discovery** candidates, and **Genre Playlists**.
  - Provenance-first outside-library recommendation frontier (Last.fm co-listening, Deezer
    and Spotify artist/album expansion, YouTube Music radio, optional LAION-DISCO-12M lookup).
  - Unified `/review` proposals queue with durable accept/reject verdicts.
- **Library Chat Agent**
  - OpenRouter-backed streaming agent with read-only SQL (`mode=ro` + single-SELECT guard),
    DuckDB statistical analysis, sonic/collaborative similarity, CLAP sound search,
    human-in-the-loop playlist proposals, shared Library Memory, and rolling model-aware
    context compaction.
  - Safety gate (`_assert_managed`) guaranteeing Acetate never overwrites user-authored
    playlists (except explicit user-triggered Filing Inbox adds with exact ledgered undo).

## Planned

1. **Genre playlist calibration** — refine confidence thresholds and hyphen/space style
   matching against `/review` verdicts, with optional LLM adjudication for borderline tracks.
2. **Richer playlist-taste models** — graduate from centroid/k-NN baselines to lightweight
   supervised classifiers trained on `/review` accept/reject feedback with time-based holdouts.
3. **BPM & harmonic sequencing** — add energy-arc and Camelot-key sequencing to the
   retrieve → rerank → sequence autoplaylist pipeline (using Deezer BPM + Essentia TempoCNN/key).
4. **Hyperspace visual upgrades** — era-drift animation over time, ghost-star rendering for
   outside-library discovery candidates, and lasso-select → Smart Playlist creation.
5. **MCP server** — expose Acetate's read-only SQL, similarity, sound-search, and playlist
   proposal tools over the Model Context Protocol for Claude Desktop and coding agents.
6. **Per-category Gem playlists** — publish rotating playlists for *never-played*, *re-ups*,
   and *under-explored* lenses alongside *faded* gems.
7. **Mobile UX polish** — tighten dense tables and charts across narrow viewports.
8. **Small-group multi-user support** — optional per-user OAuth allowlist and partitioned
   tables for friend-group stats and compatibility views behind Authelia SSO.

## Known Spotify Web API Constraints

Because Spotify's Developer Mode restricts new apps to a small allowlisted user cap and
removed several endpoints after November 2024:

- Acetate follows a **"bring your own Spotify app"** self-hosted model.
- `/v1/audio-features` and artist `genres[]` are unavailable to new apps — Acetate computes
  its own audio embeddings and 400-style predictions locally from previews and pairs them
  with Last.fm tags and Deezer metadata.
- Spotify-owned algorithmic/editorial playlists (such as Discover Weekly or Release Radar)
  return `404` via the Web API for standard apps and cannot be read directly unless mirrored
  to a user-owned playlist.
