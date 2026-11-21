"""The public capability registry for docs, web surfaces, and the chat agent.

Every entry is one user-facing workflow with one website home and one LLM tool.
Execution remains ordinary typed Python; this module owns only the shared
contract and presentation metadata so those surfaces cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workflow:
    id: str
    title: str
    summary: str
    tool_name: str
    tool_description: str
    input_schema: dict
    web_path: str
    web_label: str
    access: str
    when_to_use: str
    how_it_works: str
    output: str
    limitations: tuple[str, ...] = ()
    example: str = ""
    admin_only: bool = False

    def agent_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.tool_description,
                "parameters": self.input_schema,
            },
        }


WORKFLOWS = (
    Workflow(
        id="track-report",
        title="Inspect a track",
        summary="Get one provenance-aware dossier for a track, including explicit enrichment states.",
        tool_name="track_report",
        tool_description=(
            "Your FIRST call for any question centered on one named track or Spotify track URL: "
            "identity, vibe/genre evidence, listening history, metadata, availability, or why data "
            "is missing. Returns personal facts plus typed states that distinguish not scheduled, "
            "pending, blocked, source coverage gaps, provider misses, errors, and available data. "
            "Do not rebuild this dossier with query_db."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Spotify track URL/id or Artist - Title.",
                }
            },
            "required": ["query"],
        },
        web_path="/chat",
        web_label="Ask in Chat",
        access="read-only",
        when_to_use="Any question centered on one named track or Spotify track URL.",
        how_it_works=(
            "Resolves the recording once and joins listening, catalog, provider-ledger, preview, "
            "embedding, style, and playlist evidence without treating absent rows as provider misses."
        ),
        output=(
            "Track facts, personal history, playlist membership, and typed per-surface availability "
            "across Spotify, Last.fm, Deezer, MusicBrainz, YT Music, previews, and every embedding model."
        ),
        limitations=(
            "Unknown legacy rows remain explicitly unknown until rechecked.",
            "not_scheduled, pending, blocked, source_not_covered, provider_miss, error, and "
            "available are separate states. Only an attempted authoritative lookup may produce "
            "provider_miss; row absence is not_scheduled.",
        ),
        example="What vibe is this track, and why is some metadata missing?",
    ),
    Workflow(
        id="playlist-report",
        title="Inspect a playlist",
        summary="Measure a playlist's identity, cohesion, outliers, gaps, and stale links.",
        tool_name="playlist_report",
        tool_description=(
            "Your FIRST call for any question centered on one playlist: cleanup, outliers, "
            "what is missing, health, or identity. Returns embedding-based cohesion, misfits, "
            "gap-fill candidates, related playlist names, and stale links."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "playlist": {
                    "type": "string",
                    "description": "Playlist name (fuzzy) or Spotify id.",
                }
            },
            "required": ["playlist"],
        },
        web_path="/playlists",
        web_label="Open Playlists",
        access="read-only",
        when_to_use="Understanding or cleaning up one authored playlist.",
        how_it_works="Combines playlist usage with audio-embedding cohesion and membership evidence.",
        output="Identity, cohesion, misfits, candidate additions, sibling playlists, and stale links.",
    ),
    Workflow(
        id="playlist-fit",
        title="Score playlist fit",
        summary="Score how well named tracks fit the sound of a playlist.",
        tool_name="playlist_fit",
        tool_description=(
            "Score how well specific tracks fit a playlist's sound using audio embeddings and a "
            "percentile against current members. Use for 'would X fit in Y' and to verify additions."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "playlist": {"type": "string", "description": "Playlist name or id."},
                "tracks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Up to 20 Artist - Title strings.",
                },
            },
            "required": ["playlist", "tracks"],
        },
        web_path="/playlists",
        web_label="Open Playlists",
        access="read-only",
        when_to_use="Checking proposed additions or removals against a playlist's sonic center.",
        how_it_works="Compares track embeddings with the playlist region and member distribution.",
        output="Per-track fit scores and member-relative percentiles.",
    ),
    Workflow(
        id="query-library",
        title="Query your listening library",
        summary="Answer factual questions from the listening archive with read-only SQL.",
        tool_name="query_db",
        tool_description=(
            "Run ONE read-only SQL query (SELECT/WITH) over the user's Spotify listening "
            "database and get rows back. Use it to answer questions with real numbers and "
            "to check what a playlist rule would match before proposing it."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "A single SQLite SELECT or WITH query."}
            },
            "required": ["sql"],
        },
        web_path="/chat",
        web_label="Ask in Chat",
        access="read-only",
        when_to_use="Counts, history, favorites, playlist membership, and listening patterns.",
        how_it_works="Runs one guarded query through a query-only SQLite connection.",
        output="Columns, rows, row count, and a truncation flag.",
        limitations=("One statement per call.", "Results are capped and long queries time out."),
        example="Which artists did I play most in 2025?",
    ),
    Workflow(
        id="analyze-library",
        title="Compute statistics over your library",
        summary="Run statistical SQL — medians, percentiles, spread, correlation — over the archive.",
        tool_name="analyze_db",
        tool_description=(
            "Run ONE read-only analytical SQL query (SELECT/WITH) over the listening database "
            "using DuckDB, which has the statistics SQLite lacks: median(), quantile_cont(), "
            "mode(), stddev_samp(), var_samp(), corr(), regr_slope(), histogram(), plus the same "
            "window functions. Same tables and columns as query_db — write them unqualified. "
            "USE THIS INSTEAD OF DOING ARITHMETIC YOURSELF: any average, share, percentage, "
            "median, spread, ratio, growth rate, or per-period comparison belongs in this query, "
            "not in your head. It also works as a plain calculator over values you already have: "
            "SELECT median(x) FROM (SELECT unnest([4,8,15,16]) AS x)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A single DuckDB SELECT or WITH query. Table names are "
                    "unqualified, exactly as in the schema.",
                }
            },
            "required": ["sql"],
        },
        web_path="/chat",
        web_label="Ask in Chat",
        access="read-only",
        when_to_use=(
            "Any question whose answer is a computed number: averages, medians, percentiles, "
            "shares, spread, trends, correlations, or gaps between events."
        ),
        how_it_works=(
            "Attaches the SQLite archive to an in-memory DuckDB as a READ_ONLY catalog, then "
            "drops external file/network access and locks the configuration so the query cannot "
            "restore it. One guarded statement runs against that."
        ),
        output="Columns, rows, row count, and a truncation flag.",
        limitations=(
            "One statement per call.",
            "Results are capped and long queries time out.",
            "Reaches only the attached archive — no files, no network.",
        ),
        example="What's the median gap between plays of my top artists?",
    ),
    Workflow(
        id="propose-smart-playlist",
        title="Propose a playlist",
        summary="Propose a fixed curated selection or an automatically refreshed Smart Playlist.",
        tool_name="propose_playlist",
        tool_description=(
            "Propose a playlist for the user to approve. This does NOT create anything; "
            "the user reviews and approves it in the UI. Give a short name and a spec object. "
            "Use {track_ids:[...]} for a fixed curated selection, published once without "
            "automatic updates. Use filter/sort fields for a Smart Playlist that refreshes daily."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Playlist name."},
                "spec": {
                    "type": "object",
                    "description": "Explicit {track_ids:[...]} for a fixed selection, or Smart Playlist "
                    "filter/sort fields for a live rule.",
                },
            },
            "required": ["name", "spec"],
        },
        web_path="/chat",
        web_label="Ask in Chat",
        access="proposal only",
        when_to_use="The user wants a hand-picked selection or a reusable rule that refreshes daily.",
        how_it_works=(
            "Previews the specification before approval. Fixed track lists are filed under "
            "Curated Playlists and published once; live rules are filed under Smart Playlists."
        ),
        output="A proposal summary and the current number of matching recordings.",
        limitations=("Never publishes without explicit approval.",),
        example="Make a playlist of liked techno I have not played in six months.",
    ),
    Workflow(
        id="find-similar-tracks",
        title="Find similar tracks",
        summary="Find sonic neighbors and collaborative co-listening matches for one seed track.",
        tool_name="similar_tracks",
        tool_description=(
            "Find tracks similar to a seed track in two spaces: sonic audio-embedding neighbors "
            "and Last.fm collaborative neighbors, including outside-library tracks. Use for "
            "'more like X' questions."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Artist - title, or a fragment."},
                "k": {
                    "type": "integer",
                    "description": "Neighbors per space (default 12, max 30).",
                },
            },
            "required": ["query"],
        },
        web_path="/chat",
        web_label="Ask in Chat",
        access="read-only",
        when_to_use="The user names a track and wants music like it.",
        how_it_works="Combines EffNet nearest neighbors with cached Last.fm track similarity.",
        output="The resolved seed plus separate sonic and collaborative result lists.",
        limitations=("Sonic results require an embedded library seed.",),
        example="Find tracks like Jon Hopkins - Emerald Rush.",
    ),
    Workflow(
        id="search-by-sound",
        title="Search by sound",
        summary="Search audio using a natural-language description of texture, mood, or style.",
        tool_name="search_by_sound",
        tool_description=(
            "Semantic search over the library's audio from a free-text sound description via "
            "CLAP text-to-audio embeddings. Use for vibe, texture, mood, or instrumentation rather "
            "than a named seed. Slow; call at most once per request."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "The sound in words."},
                "k": {"type": "integer", "description": "Results (default 15, max 30)."},
            },
            "required": ["description"],
        },
        web_path="/hyperspace",
        web_label="Open Hyperspace",
        access="read-only; local compute",
        when_to_use="The user describes a sound instead of naming a track.",
        how_it_works="Embeds the text with CLAP and searches the compatible audio-vector space.",
        output="Ranked recordings whose audio best matches the description.",
        limitations=("Requires the local CLAP environment.", "A query may take around 15 seconds."),
        example="Warm, hypnotic dub techno with no vocals.",
    ),
    Workflow(
        id="discover-artists",
        title="Discover artists from a playlist",
        summary="Find unseen artists repeatedly adjacent to distinct tracks in an authored playlist.",
        tool_name="discover_artists",
        tool_description=(
            "Discover new artists for a specific authored playlist. Each playlist track votes for "
            "artists in its cached Last.fm neighbors; rankings reward strong matches across many "
            "distinct playlist tracks. Returns representative track-to-track connections explaining "
            "every artist."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "playlist": {
                    "type": "string",
                    "description": "Playlist id, name, or name fragment.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Artists to return (default 20, max 100).",
                },
                "novelty": {
                    "type": "string",
                    "enum": ["outside_library", "outside_playlist"],
                    "description": "Exclude artists in the whole library or only this playlist.",
                },
                "min_seed_coverage": {
                    "type": "integer",
                    "description": "Distinct playlist tracks that must point to the artist (default 2).",
                },
            },
            "required": ["playlist"],
        },
        web_path="/tools#discover-artists",
        web_label="Run Artist Discovery",
        access="read-only",
        when_to_use="The user wants new artists matching the character of one playlist.",
        how_it_works=(
            "Retrieves cached Last.fm neighbors for every playlist recording, keeps one vote per "
            "artist per seed, then blends rank-weighted votes, distinct-seed coverage, and similarity."
        ),
        output="Ranked artists, coverage, representative candidate tracks, and the playlist seeds that voted.",
        limitations=(
            "External artist identity currently uses normalized display names.",
            "Audio reranking is deferred until more outside-library candidates are embedded.",
        ),
        example="Find artists outside my library that match Melodic techno.",
    ),
    Workflow(
        id="system-status",
        title="Inspect system status",
        summary=(
            "Inspect scheduler timing, job outcomes, embedding pipelines, and backfill progress."
        ),
        tool_name="system_status",
        tool_description=(
            "Your FIRST call for operational questions about Acetate itself: whether the scheduler "
            "is alive, the next poll time, recent or failed jobs, embedding coverage and queues, "
            "candidate embedding state, and genre, YT Music, MusicBrainz, or candidate backfills. "
            "This is a bounded read-only snapshot and cannot run, retry, or change any job."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "recent_jobs": {
                    "type": "integer",
                    "description": "Recent job runs to include (default 12, maximum 25).",
                },
            },
        },
        web_path="/pipeline",
        web_label="Open Pipeline",
        access="read-only",
        when_to_use=(
            "Checking scheduler health, poll timing, failures, queue depth, or backfill completion."
        ),
        how_it_works=(
            "Reads the scheduler's persisted heartbeat and next-fire times plus bounded aggregate "
            "state from the shared SQLite database."
        ),
        output=(
            "Scheduler freshness and next runs, latest and recent jobs, seven-day outcomes, "
            "embedding coverage and queues, and backfill progress."
        ),
        limitations=(
            "Reports stale when the scheduler snapshot has not refreshed within its expected window.",
            "Does not expose controls that run, retry, pause, or modify jobs.",
        ),
        example="Is the scheduler healthy, and what embedding or backfill work remains?",
    ),
    Workflow(
        id="request-embeddings",
        title="Request missing track embeddings",
        summary="Queue missing preview audio and embeddings for a few known tracks.",
        tool_name="request_embeddings",
        tool_description=(
            "Queue on-demand audio embeddings for 1–5 exact known tracks, including outside-library "
            "candidates. Use when an admin asks to embed tracks or missing embeddings block their "
            "requested sonic analysis. Fetches missing previews, then EffNet/Discogs styles and CLAP; "
            "existing vectors are reused. Asynchronous: queued does not mean completed. Call once "
            "per turn. No arbitrary URLs, models, commands, bulk backfills, or parallelism settings."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "tracks": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {"type": "string", "maxLength": 500},
                    "description": "Exact Artist - Title, recording keys, or known Spotify track URLs.",
                },
            },
            "required": ["tracks"],
            "additionalProperties": False,
        },
        web_path="/chat",
        web_label="Request in Chat",
        access="administrator; queues local audio work",
        admin_only=True,
        when_to_use="Missing embeddings prevent analysis of specific known library tracks or candidates.",
        how_it_works=(
            "Deduplicates a durable, bounded queue. The existing embedding sidecar fetches one track's "
            "preview and runs the existing models serially. Shared process locks cover background and CLI inference."
        ),
        output="Request ids, current states, missing models, worker visibility, and enforced limits.",
        limitations=(
            "At most five tracks and one enqueue call per chat turn; at most twenty outstanding tracks overall.",
            "One active request and one inference process at a time; each model stage has a ten-minute timeout.",
            "Only exact known identities; no catalog-wide search, new models, playlist publication, or map rebuild.",
            "A failed request needs an explicit retry; at most three attempts per track.",
            "Requires the embedding sidecar or the local request worker; unavailable previews remain source coverage gaps.",
        ),
        example="Embed these unembedded tracks so we can compare their sound.",
    ),
    Workflow(
        id="embedding-request-status",
        title="Check requested embeddings",
        summary="Read progress and failures for on-demand track embedding requests.",
        tool_name="embedding_request_status",
        tool_description=(
            "Read on-demand embedding request states and worker heartbeat. Supply returned request ids "
            "or omit to see the latest twenty. Does not enqueue or retry anything. At most two calls "
            "per turn; report queued/running/offline state instead of waiting in a polling loop."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "request_ids": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 20,
                    "items": {"type": "integer", "minimum": 1},
                },
            },
            "additionalProperties": False,
        },
        web_path="/chat",
        web_label="Check in Chat",
        access="administrator; read-only",
        admin_only=True,
        when_to_use="Following up on a previously queued embedding request.",
        how_it_works="Reads durable request states, actual vector availability, and the worker heartbeat.",
        output="Bounded request details, missing models, errors, attempts, and worker freshness.",
        limitations=("Does not wait for completion or automatically retry failed work.",),
        example="Have my requested embeddings finished?",
    ),
    Workflow(
        id="library-memory",
        title="Remember library organization",
        summary="Keep one shared Markdown note of confirmed library-organization decisions.",
        tool_name="update_library_memory",
        tool_description=(
            "Replace the shared library memory note when the user asks you to remember something "
            "or confirms a durable library-organization decision. Preserve relevant existing notes, "
            "apply corrections, and remove facts the user asks you to forget. Store concise naming "
            "conventions, playlist roles, decisions, and unfinished organization work; distinguish "
            "plans from completed actions. Do not store guesses, tentative suggestions, credentials, "
            "or permission to publish or modify playlists. Pass the current revision from the "
            "shared memory context; a stale revision is rejected."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "maxLength": 6000,
                    "description": "Complete updated Markdown note; empty text clears it.",
                },
                "revision": {
                    "type": "string",
                    "description": "Opaque current memory revision; copy exactly, or empty when absent.",
                },
            },
            "required": ["text", "revision"],
            "additionalProperties": False,
        },
        web_path="/chat#library-memory",
        web_label="Edit library memory",
        access="administrator; updates the shared organization note",
        admin_only=True,
        when_to_use="Remembering confirmed organization decisions across conversations.",
        how_it_works="Replaces one bounded text value in the existing SQLite key/value store, rejecting stale edits.",
        output="The saved note and its new revision, or an explicit conflict or validation error.",
        limitations=(
            "One shared note for this library, not per-person memory or a searchable memory archive.",
            "At most 6000 characters. Current requests and freshly queried facts take precedence.",
            "The note does not authorize playlist writes and does not replace conversation summaries.",
        ),
        example="Remember that the Deep Progressive collection is a fixed selection, and we still need to sort ambient.",
    ),
    Workflow(
        id="refresh-plays",
        title="Refresh play history now",
        summary=(
            "Poll Spotify for recently-played tracks immediately instead of waiting for "
            "the next scheduled run."
        ),
        tool_name="refresh_plays",
        tool_description=(
            "Poll Spotify's recently-played feed RIGHT NOW and import any new plays, instead "
            "of waiting up to 15 minutes for the scheduled poll. Use it when the user says "
            "they just listened to something and wants Acetate to know about it, or before "
            "answering a question about what they played in the last few minutes. Returns "
            "what actually landed, including the new plays themselves. Takes a few seconds. "
            "Call it at most once per request — a second call cannot find more."
        ),
        input_schema={"type": "object", "properties": {}},
        web_path="/pipeline",
        web_label="Open Pipeline",
        access="administrator; writes new plays",
        admin_only=True,
        when_to_use=(
            "The user just finished listening and wants their history current before you "
            "answer, or they explicitly ask for a refresh."
        ),
        how_it_works=(
            "Runs the same poll job as the scheduler, in the foreground, guarded so it cannot "
            "overlap a poll already in flight. Every run is recorded in the job audit trail."
        ),
        output=(
            "Whether the poll ran, how many plays were fetched and newly inserted, and the "
            "new plays with their timestamps."
        ),
        limitations=(
            "Spotify only exposes the last 50 plays, and a track is reported only after it "
            "finishes — something still playing will not appear yet.",
            "Returns busy without polling when another poll is already running.",
            "Imports plays only; it does not back up the library or rebuild playlists.",
        ),
        example="I just listened to a few tracks — refresh my history and tell me what came in.",
    ),
)

BY_ID = {workflow.id: workflow for workflow in WORKFLOWS}
BY_TOOL = {workflow.tool_name: workflow for workflow in WORKFLOWS}


def visible(can_admin: bool = False) -> tuple[Workflow, ...]:
    """The workflows a caller with this role may use."""
    return tuple(w for w in WORKFLOWS if can_admin or not w.admin_only)


def agent_tools(can_admin: bool = False) -> list[dict]:
    """Tool definitions for one chat turn. Admin-only tools are not merely
    refused when called — a non-admin is never told they exist."""
    return [workflow.agent_tool() for workflow in visible(can_admin)]


def prompt_index(can_admin: bool = False) -> str:
    return "\n".join(
        f"- {workflow.tool_name}: {workflow.summary}" for workflow in visible(can_admin)
    )


def render_markdown() -> str:
    lines = [
        "# Acetate workflows",
        "",
        "This catalog is generated from `acetate/workflows.py`, the shared contract for the website",
        "capability catalog and the LLM agent's tools. Every published workflow has one stable id,",
        "one website home, and one agent tool.",
        "",
        "| Workflow | Access | Website | Agent tool |",
        "|---|---|---|---|",
    ]
    for workflow in WORKFLOWS:
        lines.append(
            f"| [{workflow.title}](#{workflow.id}) | {workflow.access} | "
            f"`{workflow.web_path}` | `{workflow.tool_name}` |"
        )
    for workflow in WORKFLOWS:
        lines.extend(
            [
                "",
                f"## {workflow.title}",
                f'<a id="{workflow.id}"></a>',
                "",
                workflow.summary,
                "",
                f"- **When to use:** {workflow.when_to_use}",
                f"- **How it works:** {workflow.how_it_works}",
                f"- **Output:** {workflow.output}",
                f"- **Access:** {workflow.access}",
                f"- **Website:** `{workflow.web_path}`",
                f"- **Agent tool:** `{workflow.tool_name}`",
            ]
        )
        if workflow.example:
            lines.append(f"- **Example:** “{workflow.example}”")
        if workflow.limitations:
            lines.extend(["", "Limitations:", ""])
            lines.extend(f"- {limitation}" for limitation in workflow.limitations)
    return "\n".join(lines) + "\n"


def write_markdown(path: Path | None = None) -> Path:
    target = path or Path(__file__).parent.parent / "docs" / "workflows.md"
    target.write_text(render_markdown())
    return target
