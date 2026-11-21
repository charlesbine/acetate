# Acetate workflows

This catalog is generated from `acetate/workflows.py`, the shared contract for the website
capability catalog and the LLM agent's tools. Every published workflow has one stable id,
one website home, and one agent tool.

| Workflow | Access | Website | Agent tool |
|---|---|---|---|
| [Inspect a track](#track-report) | read-only | `/chat` | `track_report` |
| [Inspect a playlist](#playlist-report) | read-only | `/playlists` | `playlist_report` |
| [Score playlist fit](#playlist-fit) | read-only | `/playlists` | `playlist_fit` |
| [Query your listening library](#query-library) | read-only | `/chat` | `query_db` |
| [Compute statistics over your library](#analyze-library) | read-only | `/chat` | `analyze_db` |
| [Propose a playlist](#propose-smart-playlist) | proposal only | `/chat` | `propose_playlist` |
| [Find similar tracks](#find-similar-tracks) | read-only | `/chat` | `similar_tracks` |
| [Search by sound](#search-by-sound) | read-only; local compute | `/hyperspace` | `search_by_sound` |
| [Discover artists from a playlist](#discover-artists) | read-only | `/tools#discover-artists` | `discover_artists` |
| [Inspect system status](#system-status) | read-only | `/pipeline` | `system_status` |
| [Request missing track embeddings](#request-embeddings) | administrator; queues local audio work | `/chat` | `request_embeddings` |
| [Check requested embeddings](#embedding-request-status) | administrator; read-only | `/chat` | `embedding_request_status` |
| [Remember library organization](#library-memory) | administrator; updates the shared organization note | `/chat#library-memory` | `update_library_memory` |
| [Refresh play history now](#refresh-plays) | administrator; writes new plays | `/pipeline` | `refresh_plays` |

## Inspect a track
<a id="track-report"></a>

Get one provenance-aware dossier for a track, including explicit enrichment states.

- **When to use:** Any question centered on one named track or Spotify track URL.
- **How it works:** Resolves the recording once and joins listening, catalog, provider-ledger, preview, embedding, style, and playlist evidence without treating absent rows as provider misses.
- **Output:** Track facts, personal history, playlist membership, and typed per-surface availability across Spotify, Last.fm, Deezer, MusicBrainz, YT Music, previews, and every embedding model.
- **Access:** read-only
- **Website:** `/chat`
- **Agent tool:** `track_report`
- **Example:** “What vibe is this track, and why is some metadata missing?”

Limitations:

- Unknown legacy rows remain explicitly unknown until rechecked.
- not_scheduled, pending, blocked, source_not_covered, provider_miss, error, and available are separate states. Only an attempted authoritative lookup may produce provider_miss; row absence is not_scheduled.

## Inspect a playlist
<a id="playlist-report"></a>

Measure a playlist's identity, cohesion, outliers, gaps, and stale links.

- **When to use:** Understanding or cleaning up one authored playlist.
- **How it works:** Combines playlist usage with audio-embedding cohesion and membership evidence.
- **Output:** Identity, cohesion, misfits, candidate additions, sibling playlists, and stale links.
- **Access:** read-only
- **Website:** `/playlists`
- **Agent tool:** `playlist_report`

## Score playlist fit
<a id="playlist-fit"></a>

Score how well named tracks fit the sound of a playlist.

- **When to use:** Checking proposed additions or removals against a playlist's sonic center.
- **How it works:** Compares track embeddings with the playlist region and member distribution.
- **Output:** Per-track fit scores and member-relative percentiles.
- **Access:** read-only
- **Website:** `/playlists`
- **Agent tool:** `playlist_fit`

## Query your listening library
<a id="query-library"></a>

Answer factual questions from the listening archive with read-only SQL.

- **When to use:** Counts, history, favorites, playlist membership, and listening patterns.
- **How it works:** Runs one guarded query through a query-only SQLite connection.
- **Output:** Columns, rows, row count, and a truncation flag.
- **Access:** read-only
- **Website:** `/chat`
- **Agent tool:** `query_db`
- **Example:** “Which artists did I play most in 2025?”

Limitations:

- One statement per call.
- Results are capped and long queries time out.

## Compute statistics over your library
<a id="analyze-library"></a>

Run statistical SQL — medians, percentiles, spread, correlation — over the archive.

- **When to use:** Any question whose answer is a computed number: averages, medians, percentiles, shares, spread, trends, correlations, or gaps between events.
- **How it works:** Attaches the SQLite archive to an in-memory DuckDB as a READ_ONLY catalog, then drops external file/network access and locks the configuration so the query cannot restore it. One guarded statement runs against that.
- **Output:** Columns, rows, row count, and a truncation flag.
- **Access:** read-only
- **Website:** `/chat`
- **Agent tool:** `analyze_db`
- **Example:** “What's the median gap between plays of my top artists?”

Limitations:

- One statement per call.
- Results are capped and long queries time out.
- Reaches only the attached archive — no files, no network.

## Propose a playlist
<a id="propose-smart-playlist"></a>

Propose a fixed curated selection or an automatically refreshed Smart Playlist.

- **When to use:** The user wants a hand-picked selection or a reusable rule that refreshes daily.
- **How it works:** Previews the specification before approval. Fixed track lists are filed under Curated Playlists and published once; live rules are filed under Smart Playlists.
- **Output:** A proposal summary and the current number of matching recordings.
- **Access:** proposal only
- **Website:** `/chat`
- **Agent tool:** `propose_playlist`
- **Example:** “Make a playlist of liked techno I have not played in six months.”

Limitations:

- Never publishes without explicit approval.

## Find similar tracks
<a id="find-similar-tracks"></a>

Find sonic neighbors and collaborative co-listening matches for one seed track.

- **When to use:** The user names a track and wants music like it.
- **How it works:** Combines EffNet nearest neighbors with cached Last.fm track similarity.
- **Output:** The resolved seed plus separate sonic and collaborative result lists.
- **Access:** read-only
- **Website:** `/chat`
- **Agent tool:** `similar_tracks`
- **Example:** “Find tracks like Jon Hopkins - Emerald Rush.”

Limitations:

- Sonic results require an embedded library seed.

## Search by sound
<a id="search-by-sound"></a>

Search audio using a natural-language description of texture, mood, or style.

- **When to use:** The user describes a sound instead of naming a track.
- **How it works:** Embeds the text with CLAP and searches the compatible audio-vector space.
- **Output:** Ranked recordings whose audio best matches the description.
- **Access:** read-only; local compute
- **Website:** `/hyperspace`
- **Agent tool:** `search_by_sound`
- **Example:** “Warm, hypnotic dub techno with no vocals.”

Limitations:

- Requires the local CLAP environment.
- A query may take around 15 seconds.

## Discover artists from a playlist
<a id="discover-artists"></a>

Find unseen artists repeatedly adjacent to distinct tracks in an authored playlist.

- **When to use:** The user wants new artists matching the character of one playlist.
- **How it works:** Retrieves cached Last.fm neighbors for every playlist recording, keeps one vote per artist per seed, then blends rank-weighted votes, distinct-seed coverage, and similarity.
- **Output:** Ranked artists, coverage, representative candidate tracks, and the playlist seeds that voted.
- **Access:** read-only
- **Website:** `/tools#discover-artists`
- **Agent tool:** `discover_artists`
- **Example:** “Find artists outside my library that match Melodic techno.”

Limitations:

- External artist identity currently uses normalized display names.
- Audio reranking is deferred until more outside-library candidates are embedded.

## Inspect system status
<a id="system-status"></a>

Inspect scheduler timing, job outcomes, embedding pipelines, and backfill progress.

- **When to use:** Checking scheduler health, poll timing, failures, queue depth, or backfill completion.
- **How it works:** Reads the scheduler's persisted heartbeat and next-fire times plus bounded aggregate state from the shared SQLite database.
- **Output:** Scheduler freshness and next runs, latest and recent jobs, seven-day outcomes, embedding coverage and queues, and backfill progress.
- **Access:** read-only
- **Website:** `/pipeline`
- **Agent tool:** `system_status`
- **Example:** “Is the scheduler healthy, and what embedding or backfill work remains?”

Limitations:

- Reports stale when the scheduler snapshot has not refreshed within its expected window.
- Does not expose controls that run, retry, pause, or modify jobs.

## Request missing track embeddings
<a id="request-embeddings"></a>

Queue missing preview audio and embeddings for a few known tracks.

- **When to use:** Missing embeddings prevent analysis of specific known library tracks or candidates.
- **How it works:** Deduplicates a durable, bounded queue. The existing embedding sidecar fetches one track's preview and runs the existing models serially. Shared process locks cover background and CLI inference.
- **Output:** Request ids, current states, missing models, worker visibility, and enforced limits.
- **Access:** administrator; queues local audio work
- **Website:** `/chat`
- **Agent tool:** `request_embeddings`
- **Example:** “Embed these unembedded tracks so we can compare their sound.”

Limitations:

- At most five tracks and one enqueue call per chat turn; at most twenty outstanding tracks overall.
- One active request and one inference process at a time; each model stage has a ten-minute timeout.
- Only exact known identities; no catalog-wide search, new models, playlist publication, or map rebuild.
- A failed request needs an explicit retry; at most three attempts per track.
- Requires the embedding sidecar or the local request worker; unavailable previews remain source coverage gaps.

## Check requested embeddings
<a id="embedding-request-status"></a>

Read progress and failures for on-demand track embedding requests.

- **When to use:** Following up on a previously queued embedding request.
- **How it works:** Reads durable request states, actual vector availability, and the worker heartbeat.
- **Output:** Bounded request details, missing models, errors, attempts, and worker freshness.
- **Access:** administrator; read-only
- **Website:** `/chat`
- **Agent tool:** `embedding_request_status`
- **Example:** “Have my requested embeddings finished?”

Limitations:

- Does not wait for completion or automatically retry failed work.

## Remember library organization
<a id="library-memory"></a>

Keep one shared Markdown note of confirmed library-organization decisions.

- **When to use:** Remembering confirmed organization decisions across conversations.
- **How it works:** Replaces one bounded text value in the existing SQLite key/value store, rejecting stale edits.
- **Output:** The saved note and its new revision, or an explicit conflict or validation error.
- **Access:** administrator; updates the shared organization note
- **Website:** `/chat#library-memory`
- **Agent tool:** `update_library_memory`
- **Example:** “Remember that the Deep Progressive collection is a fixed selection, and we still need to sort ambient.”

Limitations:

- One shared note for this library, not per-person memory or a searchable memory archive.
- At most 6000 characters. Current requests and freshly queried facts take precedence.
- The note does not authorize playlist writes and does not replace conversation summaries.

## Refresh play history now
<a id="refresh-plays"></a>

Poll Spotify for recently-played tracks immediately instead of waiting for the next scheduled run.

- **When to use:** The user just finished listening and wants their history current before you answer, or they explicitly ask for a refresh.
- **How it works:** Runs the same poll job as the scheduler, in the foreground, guarded so it cannot overlap a poll already in flight. Every run is recorded in the job audit trail.
- **Output:** Whether the poll ran, how many plays were fetched and newly inserted, and the new plays with their timestamps.
- **Access:** administrator; writes new plays
- **Website:** `/pipeline`
- **Agent tool:** `refresh_plays`
- **Example:** “I just listened to a few tracks — refresh my history and tell me what came in.”

Limitations:

- Spotify only exposes the last 50 plays, and a track is reported only after it finishes — something still playing will not appear yet.
- Returns busy without polling when another poll is already running.
- Imports plays only; it does not back up the library or rebuild playlists.
