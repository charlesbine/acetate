-- acetate SQLite schema
-- Design: append-only *snapshot* tables preserve a deletion trace (backups),
-- normalized *dimension* tables (tracks/artists/albums) support analysis,
-- play_events accumulates listening history, track_genres caches classification.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- Dimension tables: current best-known state, upserted over time.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS artists (
    artist_id   TEXT PRIMARY KEY,
    name        TEXT,
    genres_json TEXT,            -- JSON array from GET /artists (filled in genre phase)
    popularity  INTEGER,
    uri         TEXT,
    raw_json    TEXT,
    updated_at  TEXT             -- ISO8601 of last refresh of this row
);

CREATE TABLE IF NOT EXISTS albums (
    album_id     TEXT PRIMARY KEY,
    name         TEXT,
    release_date TEXT,
    images_json  TEXT,
    raw_json     TEXT,
    updated_at   TEXT
);

CREATE TABLE IF NOT EXISTS tracks (
    track_id     TEXT PRIMARY KEY,
    name         TEXT,
    album_id     TEXT REFERENCES albums(album_id),
    duration_ms  INTEGER,
    popularity   INTEGER,
    isrc         TEXT,
    uri          TEXT,
    raw_json     TEXT,
    updated_at   TEXT
);

-- track <-> artist many-to-many, position preserves primary-artist ordering.
CREATE TABLE IF NOT EXISTS track_artists (
    track_id  TEXT NOT NULL REFERENCES tracks(track_id),
    artist_id TEXT NOT NULL REFERENCES artists(artist_id),
    position  INTEGER NOT NULL,
    PRIMARY KEY (track_id, artist_id)
);
CREATE INDEX IF NOT EXISTS idx_track_artists_artist ON track_artists(artist_id);

-- Denormalized track labels from sources that give strings but not full objects
-- (chiefly the GDPR streaming export). Provides human-readable names + a primary
-- artist string for display and Last.fm lookups before API hydration fills `tracks`.
CREATE TABLE IF NOT EXISTS track_labels (
    track_id    TEXT PRIMARY KEY,
    track_name  TEXT,
    artist_name TEXT,
    album_name  TEXT,
    source      TEXT,
    updated_at  TEXT
);

-- Canonical "recording" per track_id, collapsing Spotify's duplicate IDs for the
-- same song (single vs album vs comp). recording_key = normalized artist+title;
-- rep_track_id = the most-played id in the group; artist/title = display values.
-- Rebuilt by canon.canonicalize() as tracks are ingested. Track-level aggregations
-- (top tracks, discovery, top-played, lost gems) group through this so duplicate
-- IDs don't fragment play counts / last-played.
CREATE TABLE IF NOT EXISTS track_recording (
    track_id      TEXT PRIMARY KEY,
    recording_key TEXT,
    rep_track_id  TEXT,
    artist        TEXT,
    title         TEXT
);
CREATE INDEX IF NOT EXISTS idx_track_recording_key ON track_recording(recording_key);

-- ---------------------------------------------------------------------------
-- Snapshot tables: append-only, one row per (backup_time, entity). These are
-- the "trace" -- a deleted playlist still exists in past snapshots.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS playlist_snapshots (
    id                  INTEGER PRIMARY KEY,
    backup_time         TEXT NOT NULL,
    playlist_id         TEXT NOT NULL,
    name                TEXT,
    description         TEXT,
    owner_id            TEXT,
    owner_name          TEXT,
    public              INTEGER,
    collaborative       INTEGER,
    spotify_snapshot_id TEXT,        -- Spotify's own snapshot_id (changes on edit)
    total_tracks        INTEGER,
    content_hash        TEXT,        -- hash of ordered track_ids; a snapshot row is stored only on change
    raw_json            TEXT,
    UNIQUE (backup_time, playlist_id)
);
CREATE INDEX IF NOT EXISTS idx_playlist_snap_pid ON playlist_snapshots(playlist_id);

-- Per-backup heartbeat: which playlists existed at each backup, with their content
-- hash. Written every backup (tiny); full membership lands in playlist_track_snapshots
-- only when the hash changes. Lets us pinpoint deletions (gone from seen) and changes.
CREATE TABLE IF NOT EXISTS playlist_seen (
    backup_time  TEXT NOT NULL,
    playlist_id  TEXT NOT NULL,
    content_hash TEXT,
    PRIMARY KEY (backup_time, playlist_id)
);
CREATE INDEX IF NOT EXISTS idx_playlist_seen_pid ON playlist_seen(playlist_id);

CREATE TABLE IF NOT EXISTS playlist_track_snapshots (
    id          INTEGER PRIMARY KEY,
    backup_time TEXT NOT NULL,
    playlist_id TEXT NOT NULL,
    position    INTEGER,
    track_id    TEXT,               -- null for removed/local/unavailable items
    added_at    TEXT,
    added_by    TEXT,
    raw_json    TEXT
);
CREATE INDEX IF NOT EXISTS idx_pts_pid_bt ON playlist_track_snapshots(playlist_id, backup_time);
CREATE INDEX IF NOT EXISTS idx_pts_track ON playlist_track_snapshots(track_id);

CREATE TABLE IF NOT EXISTS saved_track_snapshots (
    id          INTEGER PRIMARY KEY,
    backup_time TEXT NOT NULL,
    track_id    TEXT,
    added_at    TEXT,               -- when the user liked the track
    raw_json    TEXT,
    UNIQUE (backup_time, track_id)
);
CREATE INDEX IF NOT EXISTS idx_sts_track ON saved_track_snapshots(track_id);
CREATE INDEX IF NOT EXISTS idx_sts_added ON saved_track_snapshots(added_at);

-- ---------------------------------------------------------------------------
-- Play history: accumulated from recently-played polling + GDPR export.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS play_events (
    id           INTEGER PRIMARY KEY,
    played_at    TEXT NOT NULL,     -- ISO8601 UTC
    track_id     TEXT,
    ms_played    INTEGER,           -- populated from GDPR export; null for recently-played
    context_type TEXT,             -- album / playlist / artist / ...
    context_uri  TEXT,
    source       TEXT NOT NULL,     -- 'recently_played' | 'gdpr_export'
    UNIQUE (played_at, track_id, source)
);
-- Covering index: with ms_played AND played_at in the index, per-track aggregations
-- (COUNT + MIN/MAX(played_at), filtered on ms_played) stream in index order instead of
-- building a temp B-tree over 222k text keys. The single biggest dashboard-speed win.
CREATE INDEX IF NOT EXISTS idx_play_track_ms ON play_events(track_id, ms_played, played_at);
CREATE INDEX IF NOT EXISTS idx_play_at ON play_events(played_at);

-- ---------------------------------------------------------------------------
-- recording_stats: a materialized per-recording rollup (duplicate track IDs
-- collapsed). Rebuilt by the canon job + refreshed on poll, so the heavy all-time
-- aggregations (top tracks, lost gems, discovery) are instant reads instead of
-- re-scanning 222k play_events on every page load.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS recording_stats (
    recording_key TEXT PRIMARY KEY,
    rep_track_id  TEXT,
    artist        TEXT,
    title         TEXT,
    plays         INTEGER NOT NULL DEFAULT 0,
    first_at      TEXT,                  -- first real play (discovery date)
    last_at       TEXT,                  -- most recent real play
    liked         INTEGER NOT NULL DEFAULT 0,  -- in the latest saved-tracks snapshot
    homed         INTEGER NOT NULL DEFAULT 0,  -- in some non-acetate playlist
    liked_at      TEXT                   -- earliest added_at among liked members (when you liked it)
);
CREATE INDEX IF NOT EXISTS idx_recstats_plays ON recording_stats(plays DESC);
CREATE INDEX IF NOT EXISTS idx_recstats_orphan ON recording_stats(liked, homed, last_at);

-- ---------------------------------------------------------------------------
-- Saved playlist specs: live Smart Playlist rules and fixed Curated Playlists
-- (specs containing track_ids). Rules refresh daily; fixed selections publish once.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS smart_playlists (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,            -- Spotify playlist name (also the managed_playlists key)
    spec        TEXT NOT NULL,            -- JSON rule (see smart.evaluate)
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT,
    updated_at  TEXT,
    published_at TEXT                  -- successful content sync; fixed selections stop here
);

-- ---------------------------------------------------------------------------
-- Chat: persisted, threaded conversations with the library agent.
-- A thread stays "current" for 24h, then auto-archives (reversibly).
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS chat_threads (
    id          INTEGER PRIMARY KEY,
    title       TEXT,                     -- auto-set from the first user message
    created_at  TEXT,
    updated_at  TEXT,
    archived    INTEGER NOT NULL DEFAULT 0,
    summary     TEXT,                     -- rolling memory; original messages are retained
    summary_through_id INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chat_messages (
    id          INTEGER PRIMARY KEY,
    thread_id   INTEGER NOT NULL,
    role        TEXT NOT NULL,            -- 'user' | 'assistant'
    content     TEXT,
    meta        TEXT,                     -- JSON {steps, proposals, model, usage} for assistant turns
    created_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_chat_msg_thread ON chat_messages(thread_id, id);
CREATE INDEX IF NOT EXISTS idx_chat_threads_recent ON chat_threads(archived, updated_at DESC);

-- ---------------------------------------------------------------------------
-- Genre classification cache: a track may carry genres from several sources.
-- source: 'spotify_artist' | 'lastfm' | 'llm' | 'manual'. 'manual' wins.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS track_genres (
    track_id    TEXT NOT NULL,
    genre       TEXT NOT NULL,
    source      TEXT NOT NULL,
    confidence  REAL,
    assigned_at TEXT,
    PRIMARY KEY (track_id, genre, source)
);
CREATE INDEX IF NOT EXISTS idx_track_genres_genre ON track_genres(genre);

-- Playlists that acetate owns and rewrites in place (monthly, genre, ...).
-- Maps a logical (kind, key) to the Spotify playlist id so re-runs update rather
-- than duplicate. kind: 'liked_month' | 'top_played_month' | 'genre'.
CREATE TABLE IF NOT EXISTS managed_playlists (
    kind        TEXT NOT NULL,
    key         TEXT NOT NULL,      -- e.g. '2026-06' or a genre name
    playlist_id TEXT NOT NULL,
    name        TEXT,
    created_at  TEXT,
    updated_at  TEXT,
    PRIMARY KEY (kind, key)
);

-- State for "smart" playlists (rotation / append-only / dedup): every track we've
-- ever put into a managed playlist, and when we last surfaced it. Lets Lost Gems
-- rotate, genre playlists append-only, and nothing repeat until the pool cycles.
CREATE TABLE IF NOT EXISTS managed_playlist_tracks (
    kind             TEXT NOT NULL,
    key              TEXT NOT NULL,
    track_id         TEXT NOT NULL,
    first_added_at   TEXT,
    last_surfaced_at TEXT,        -- last run that put it in the playlist (rotation signal)
    surface_count    INTEGER NOT NULL DEFAULT 0,
    currently_in     INTEGER NOT NULL DEFAULT 0,  -- 1 = in the Spotify playlist right now
    PRIMARY KEY (kind, key, track_id)
);
CREATE INDEX IF NOT EXISTS idx_mpt_surface ON managed_playlist_tracks(kind, key, last_surfaced_at);

-- Audit log of job runs (poll/backup/monthly/gems/hydrate/...) — feeds the Pipeline
-- page and becomes the trail for HITL agent actions later.
CREATE TABLE IF NOT EXISTS job_run (
    id          INTEGER PRIMARY KEY,
    job         TEXT NOT NULL,
    started_at  TEXT,
    finished_at TEXT,
    status      TEXT,            -- 'ok' | 'error'
    result      TEXT,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_job_run_started ON job_run(started_at DESC);

-- Small key/value store for poller cursors and run state.
CREATE TABLE IF NOT EXISTS kv_state (
    key   TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT
);

-- ---------------------------------------------------------------------------
-- Hyperspace: audio slices + embeddings over recordings.
-- audio_cache is the fetch queue/ledger for per-recording audio clips
-- (source 'deezer_preview' etc.; a full-track NAS-archive source can slot in
-- later). embeddings holds one float32 vector per (recording, model);
-- recording_styles keeps the top discogs400 style activations queryable for
-- smart playlists / chat / the viz legend.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS audio_cache (
    recording_key TEXT PRIMARY KEY,
    status        TEXT NOT NULL DEFAULT 'pending',  -- pending | fetched | miss | error
    source        TEXT,      -- 'deezer_isrc' | 'deezer_search' | 'itunes_search' | ...
    source_id     TEXT,      -- track id at the source
    path          TEXT,      -- relative to data/, e.g. audio/previews/ab/<sha1>.mp3
    duration_s    REAL,      -- duration of the cached clip's source track
    attempts      INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    updated_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_audio_cache_status ON audio_cache(status);

CREATE TABLE IF NOT EXISTS embeddings (
    recording_key TEXT NOT NULL,
    model         TEXT NOT NULL,     -- 'discogs-effnet' (1280) | 'discogs400' (400) | ...
    dim           INTEGER NOT NULL,
    vector        BLOB NOT NULL,     -- float32 little-endian, `dim` values
    updated_at    TEXT,
    PRIMARY KEY (recording_key, model)
);

CREATE TABLE IF NOT EXISTS recording_styles (
    recording_key TEXT NOT NULL,
    rank          INTEGER NOT NULL,  -- 1 = strongest activation
    style         TEXT NOT NULL,     -- discogs400 label, e.g. 'Electronic---Downtempo'
    score         REAL NOT NULL,
    PRIMARY KEY (recording_key, rank)
);
CREATE INDEX IF NOT EXISTS idx_recording_styles_style ON recording_styles(style);

-- 3D projections of the embedding spaces for the Hyperspace page (UMAP coords +
-- HDBSCAN cluster id), one row set per space ('effnet' | 'clap'), rebuilt by the
-- reduce worker whenever embeddings change.
CREATE TABLE IF NOT EXISTS hyperspace_points (
    space TEXT NOT NULL DEFAULT 'effnet',
    recording_key TEXT NOT NULL,
    x REAL, y REAL, z REAL,
    cluster INTEGER,                 -- -1 = noise/unclustered
    updated_at TEXT,
    PRIMARY KEY (space, recording_key)
);

-- Last.fm co-listening edges, cached per recording: the collaborative graph.
-- Feeds genre label-propagation, gems ("similar to favorites, never played"),
-- and the autoplaylist retrieve stage without live API calls. Targets keep
-- artist/title strings; sim_recording_key is filled when the target is ours.
CREATE TABLE IF NOT EXISTS lastfm_similar (
    recording_key     TEXT NOT NULL,
    sim_artist        TEXT NOT NULL,
    sim_title         TEXT NOT NULL,
    sim_recording_key TEXT,          -- in-library target, else NULL
    match             REAL,          -- Last.fm 0-1 co-listening strength
    updated_at        TEXT,
    PRIMARY KEY (recording_key, sim_artist, sim_title)
);
CREATE INDEX IF NOT EXISTS idx_lastfm_similar_target ON lastfm_similar(sim_recording_key);

-- Per-surface Last.fm execution truth. The payload tables above/below contain
-- data; this ledger says whether a provider call succeeded, authoritatively
-- missed, is waiting to run, or failed. Absence is never treated as a miss.
CREATE TABLE IF NOT EXISTS lastfm_track_runs (
    recording_key TEXT NOT NULL,
    surface       TEXT NOT NULL,       -- info | similar
    status        TEXT NOT NULL,       -- pending | available | provider_miss | error | unknown_legacy
    attempts      INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    detail_json   TEXT,
    updated_at    TEXT,
    PRIMARY KEY(recording_key, surface)
);
CREATE INDEX IF NOT EXISTS idx_lastfm_track_runs_status
    ON lastfm_track_runs(surface, status, attempts);

-- Last.fm global stats payload per recording (non-tag metadata). Execution
-- truth lives in lastfm_track_runs; row absence or NULL values are not a miss.
CREATE TABLE IF NOT EXISTS lastfm_info (
    recording_key TEXT PRIMARY KEY,
    listeners     INTEGER,
    playcount     INTEGER,
    updated_at    TEXT
);

-- Genre tagging historically used a track_genres `lastfm_none` sentinel. This
-- separate ledger preserves why that sentinel exists instead of conflating a
-- provider miss, an empty successful response, and a transient request error.
CREATE TABLE IF NOT EXISTS lastfm_tag_runs (
    track_id     TEXT PRIMARY KEY,
    status       TEXT NOT NULL,        -- pending | available | provider_miss | error | unknown_legacy
    attempts     INTEGER NOT NULL DEFAULT 0,
    error        TEXT,
    detail_json  TEXT,
    updated_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_lastfm_tag_runs_status
    ON lastfm_tag_runs(status, attempts);

-- Artist-level Last.fm cache. Raw payloads are retained because this is an
-- expensive four-call surface; partial successes survive retries.
CREATE TABLE IF NOT EXISTS lastfm_artist_cache (
    artist_key       TEXT PRIMARY KEY,
    artist           TEXT NOT NULL,
    corrected_artist TEXT,
    mbid             TEXT,
    listeners        INTEGER,
    playcount        INTEGER,
    tags_json        TEXT,
    bio_summary      TEXT,
    status           TEXT NOT NULL DEFAULT 'pending', -- pending | fetched | miss | error
    attempts         INTEGER NOT NULL DEFAULT 0,
    error            TEXT,
    info_json        TEXT,
    similar_json     TEXT,
    top_tracks_json  TEXT,
    top_albums_json  TEXT,
    updated_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_lastfm_artist_cache_status
    ON lastfm_artist_cache(status, attempts);

CREATE TABLE IF NOT EXISTS lastfm_artist_similar (
    artist_key     TEXT NOT NULL REFERENCES lastfm_artist_cache(artist_key) ON DELETE CASCADE,
    sim_artist_key TEXT NOT NULL,
    sim_artist     TEXT NOT NULL,
    match          REAL,
    rank           INTEGER,
    mbid           TEXT,
    url            TEXT,
    raw_json       TEXT,
    updated_at     TEXT,
    PRIMARY KEY(artist_key, sim_artist_key)
);
CREATE INDEX IF NOT EXISTS idx_lastfm_artist_similar_target
    ON lastfm_artist_similar(sim_artist_key);

CREATE TABLE IF NOT EXISTS lastfm_artist_top_tracks (
    artist_key TEXT NOT NULL REFERENCES lastfm_artist_cache(artist_key) ON DELETE CASCADE,
    track_key  TEXT NOT NULL,
    artist     TEXT NOT NULL,
    title      TEXT NOT NULL,
    rank       INTEGER,
    listeners  INTEGER,
    playcount  INTEGER,
    mbid       TEXT,
    url        TEXT,
    raw_json   TEXT,
    updated_at TEXT,
    PRIMARY KEY(artist_key, track_key)
);

CREATE TABLE IF NOT EXISTS lastfm_artist_top_albums (
    artist_key TEXT NOT NULL REFERENCES lastfm_artist_cache(artist_key) ON DELETE CASCADE,
    album_key  TEXT NOT NULL,
    artist     TEXT NOT NULL,
    album      TEXT NOT NULL,
    rank       INTEGER,
    playcount  INTEGER,
    mbid       TEXT,
    url        TEXT,
    raw_json   TEXT,
    updated_at TEXT,
    PRIMARY KEY(artist_key, album_key)
);

-- Outside-library recommendation candidates. These rows are not part of the
-- user's library yet; they are a provenance-first frontier for "what should we
-- inspect/embed next?" candidate_key uses the same normalized artist+title
-- shape as track_recording.recording_key so library dedupe is cheap.
CREATE TABLE IF NOT EXISTS candidate_recordings (
    candidate_key TEXT PRIMARY KEY,
    artist        TEXT,
    title         TEXT,
    status        TEXT NOT NULL DEFAULT 'metadata', -- metadata | resolved | preview | embedded | duplicate | accepted | dismissed
    score         REAL NOT NULL DEFAULT 0,
    source_count  INTEGER NOT NULL DEFAULT 0,
    first_seen    TEXT,
    last_seen     TEXT,
    updated_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_candidate_score ON candidate_recordings(status, score DESC);
CREATE INDEX IF NOT EXISTS idx_candidate_seen ON candidate_recordings(last_seen DESC);

-- Many provenance edges per outside-library candidate: Last.fm co-listening,
-- YT Music radio, Deezer related artists, Spotify same-album/appears_on, etc.
-- seed_recording_key is empty-string for source types that are not tied to one
-- known library recording, so the uniqueness constraint remains deterministic.
CREATE TABLE IF NOT EXISTS candidate_sources (
    id                 INTEGER PRIMARY KEY,
    candidate_key      TEXT NOT NULL REFERENCES candidate_recordings(candidate_key) ON DELETE CASCADE,
    source             TEXT NOT NULL,
    reason             TEXT NOT NULL,
    seed_recording_key TEXT NOT NULL DEFAULT '',
    seed_artist        TEXT,
    seed_title         TEXT,
    seed_external_id   TEXT,
    source_rank        INTEGER,
    strength           REAL,
    occurrences        INTEGER NOT NULL DEFAULT 1,
    raw_json           TEXT,
    first_seen         TEXT,
    last_seen          TEXT,
    updated_at         TEXT,
    UNIQUE(candidate_key, source, reason, seed_recording_key, seed_external_id)
);
CREATE INDEX IF NOT EXISTS idx_candidate_sources_candidate ON candidate_sources(candidate_key);
CREATE INDEX IF NOT EXISTS idx_candidate_sources_source ON candidate_sources(source, reason);
CREATE INDEX IF NOT EXISTS idx_candidate_sources_seed ON candidate_sources(seed_recording_key);

-- Resolver ids for candidates as they move from metadata-only names to concrete
-- Spotify/Deezer/YT/MusicBrainz/Discogs entities.
CREATE TABLE IF NOT EXISTS candidate_external_ids (
    candidate_key TEXT NOT NULL REFERENCES candidate_recordings(candidate_key) ON DELETE CASCADE,
    provider      TEXT NOT NULL,
    external_id   TEXT NOT NULL,
    id_type       TEXT,
    confidence    REAL,
    raw_json      TEXT,
    updated_at    TEXT,
    PRIMARY KEY(candidate_key, provider, external_id)
);
CREATE INDEX IF NOT EXISTS idx_candidate_external_provider ON candidate_external_ids(provider, external_id);

-- Exact identity overlap discovered after outside-library resolution. These
-- rows prevent alternate title formatting from re-entering the recommendation
-- frontier when the resolved ISRC already belongs to a library recording.
CREATE TABLE IF NOT EXISTS candidate_duplicates (
    candidate_key        TEXT PRIMARY KEY REFERENCES candidate_recordings(candidate_key) ON DELETE CASCADE,
    library_recording_key TEXT NOT NULL,
    method               TEXT NOT NULL,
    external_id          TEXT,
    updated_at           TEXT
);
CREATE INDEX IF NOT EXISTS idx_candidate_duplicate_library
    ON candidate_duplicates(library_recording_key);

-- Provider resolution attempts, including misses and transient failures. The
-- external-id table only records successful ids, so this ledger makes resolver
-- passes resumable without conflating "not tried" with "searched and missed".
CREATE TABLE IF NOT EXISTS candidate_resolutions (
    candidate_key      TEXT NOT NULL REFERENCES candidate_recordings(candidate_key) ON DELETE CASCADE,
    provider           TEXT NOT NULL,
    status             TEXT NOT NULL DEFAULT 'pending', -- pending | matched | provider_miss | error
    external_id        TEXT,
    method             TEXT,
    confidence         REAL,
    expected_duration_s REAL,
    matched_duration_s REAL,
    has_preview        INTEGER NOT NULL DEFAULT 0,
    preview_url        TEXT,
    isrc               TEXT,
    attempts           INTEGER NOT NULL DEFAULT 0,
    error              TEXT,
    raw_json           TEXT,
    updated_at         TEXT,
    PRIMARY KEY(candidate_key, provider)
);
CREATE INDEX IF NOT EXISTS idx_candidate_resolution_status
    ON candidate_resolutions(provider, status, attempts);

-- Downloaded outside-library preview clips. Kept separate from audio_cache so
-- candidates do not appear in library-only embedding/UI queries before review.
CREATE TABLE IF NOT EXISTS candidate_audio (
    candidate_key TEXT PRIMARY KEY REFERENCES candidate_recordings(candidate_key) ON DELETE CASCADE,
    status        TEXT NOT NULL DEFAULT 'pending', -- pending | fetched | source_not_covered | error
    source        TEXT,
    source_id     TEXT,
    source_url    TEXT,
    path          TEXT, -- relative to data/, e.g. audio/candidates/ab/<sha1>.mp3
    duration_s    REAL,
    content_type  TEXT,
    size_bytes    INTEGER,
    sha256        TEXT,
    attempts      INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    updated_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_candidate_audio_status ON candidate_audio(status, attempts);

-- Per-model candidate embedding attempts. Successful vectors live in the
-- shared embeddings table so candidate/library cosine is exact; this ledger
-- tracks retries without changing library audio state.
CREATE TABLE IF NOT EXISTS candidate_embedding_runs (
    candidate_key TEXT NOT NULL REFERENCES candidate_recordings(candidate_key) ON DELETE CASCADE,
    model         TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending', -- pending | available | error
    attempts      INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    updated_at    TEXT,
    PRIMARY KEY(candidate_key, model)
);
CREATE INDEX IF NOT EXISTS idx_candidate_embedding_status
    ON candidate_embedding_runs(model, status, attempts);

-- Admin-requested, targeted audio work. Runtime state stays on its originating
-- instance during enrichment reconciliation; the vectors use the existing tables.
CREATE TABLE IF NOT EXISTS embedding_requests (
    id             INTEGER PRIMARY KEY,
    recording_key  TEXT NOT NULL UNIQUE,
    kind           TEXT NOT NULL CHECK(kind IN ('library','candidate')),
    artist         TEXT,
    title          TEXT,
    requested_by   TEXT NOT NULL,
    status         TEXT NOT NULL CHECK(status IN ('queued','running','available','error','source_not_covered')),
    attempts       INTEGER NOT NULL DEFAULT 0,
    error          TEXT,
    requested_at   TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embedding_requests_queue
    ON embedding_requests(status, requested_at, id);

-- First content rerank: provenance blended with top-k affinity to the user's
-- high-signal library in EffNet, CLAP, and Discogs-style spaces.
CREATE TABLE IF NOT EXISTS candidate_rerank (
    candidate_key       TEXT PRIMARY KEY REFERENCES candidate_recordings(candidate_key) ON DELETE CASCADE,
    provenance_score    REAL NOT NULL,
    provenance_norm     REAL NOT NULL,
    effnet_similarity   REAL NOT NULL,
    clap_similarity     REAL NOT NULL,
    style_similarity    REAL NOT NULL,
    content_score       REAL NOT NULL,
    combined_score      REAL NOT NULL,
    effnet_neighbor_key TEXT,
    clap_neighbor_key   TEXT,
    style_neighbor_key  TEXT,
    input_fingerprint   TEXT,
    updated_at          TEXT
);
CREATE INDEX IF NOT EXISTS idx_candidate_rerank_score ON candidate_rerank(combined_score DESC);

-- YT Music radio harvest ledger. Row presence means a library recording has
-- been tried as a radio seed, even if the internal API returned no usable rows.
CREATE TABLE IF NOT EXISTS ytmusic_radio_seeds (
    recording_key TEXT PRIMARY KEY,
    video_id      TEXT,
    playlist_id   TEXT,
    status        TEXT NOT NULL DEFAULT 'pending', -- pending | fetched | miss | error
    result_count  INTEGER NOT NULL DEFAULT 0,
    attempts      INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    raw_json      TEXT,
    updated_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_ytmusic_radio_status ON ytmusic_radio_seeds(status, updated_at);

-- Library recording -> YT Music song id resolver cache. LAION-DISCO-12M gives
-- us song_id/title/artists/album/duration without hitting YT Music; this table
-- lets radio harvesting prefer exact catalog ids before falling back to live
-- search.
CREATE TABLE IF NOT EXISTS ytmusic_seed_matches (
    recording_key TEXT PRIMARY KEY,
    song_id       TEXT,
    title         TEXT,
    artists_json  TEXT,
    album_name    TEXT,
    album_id      TEXT,
    duration_s    INTEGER,
    source        TEXT NOT NULL, -- laion_disco_12m | search
    confidence    REAL,
    status        TEXT NOT NULL DEFAULT 'matched', -- matched | miss
    raw_json      TEXT,
    updated_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_ytmusic_seed_match_status ON ytmusic_seed_matches(status, confidence DESC);

-- Deezer enrichment: the metadata their track/album/artist objects carry that
-- we originally discarded. bpm+gain feed playlist sequencing; album LABEL is
-- the scene/genre signal for electronic music; rank/nb_fan are popularity axes;
-- related_json is the related-artists graph Spotify killed.
CREATE TABLE IF NOT EXISTS deezer_track (
    recording_key TEXT PRIMARY KEY,
    deezer_id     INTEGER,
    bpm           REAL,              -- 0/NULL = Deezer doesn't know
    gain          REAL,
    rank          INTEGER,
    release_date  TEXT,
    artist_id     INTEGER,
    album_id      INTEGER,
    updated_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_deezer_track_artist ON deezer_track(artist_id);
CREATE INDEX IF NOT EXISTS idx_deezer_track_album ON deezer_track(album_id);
CREATE TABLE IF NOT EXISTS deezer_album (
    album_id      INTEGER PRIMARY KEY,
    title         TEXT,
    label         TEXT,
    genres        TEXT,              -- JSON array of Deezer genre names
    release_date  TEXT,
    updated_at    TEXT
);
CREATE TABLE IF NOT EXISTS deezer_artist (
    artist_id     INTEGER PRIMARY KEY,
    name          TEXT,
    nb_fan        INTEGER,
    related_json  TEXT,              -- JSON [{id, name}] top related artists
    updated_at    TEXT
);

-- Per-entity execution truth for library-side Deezer metadata hydration.
-- Payload rows answer "what data do we have?"; this ledger answers whether a
-- missing track/album/artist row has not run, is retrying, failed, or was an
-- authoritative provider miss.
CREATE TABLE IF NOT EXISTS deezer_metadata_runs (
    entity_type TEXT NOT NULL,       -- track | album | artist
    entity_id   TEXT NOT NULL,
    status      TEXT NOT NULL,       -- pending | available | provider_miss | error
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    updated_at  TEXT,
    PRIMARY KEY(entity_type, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_deezer_metadata_runs_status
    ON deezer_metadata_runs(entity_type, status, attempts);

-- Resumable Deezer discography expansion. Artist top/albums are cached as
-- independent raw surfaces; album tracklists have their own queue so one large
-- or broken album cannot force an artist retry.
CREATE TABLE IF NOT EXISTS deezer_artist_expansion (
    artist_id          INTEGER PRIMARY KEY,
    artist             TEXT NOT NULL,
    seed_recording_key TEXT,
    status             TEXT NOT NULL DEFAULT 'pending', -- pending | fetched | miss | error
    attempts           INTEGER NOT NULL DEFAULT 0,
    error              TEXT,
    top_json           TEXT,
    albums_json        TEXT,
    updated_at         TEXT
);
CREATE INDEX IF NOT EXISTS idx_deezer_artist_expansion_status
    ON deezer_artist_expansion(status, attempts);

CREATE TABLE IF NOT EXISTS deezer_artist_top_tracks (
    artist_id   INTEGER NOT NULL REFERENCES deezer_artist_expansion(artist_id) ON DELETE CASCADE,
    track_id    INTEGER NOT NULL,
    artist      TEXT NOT NULL,
    title       TEXT NOT NULL,
    rank        INTEGER NOT NULL,
    album_id    INTEGER,
    duration_s  INTEGER,
    preview_url TEXT,
    raw_json    TEXT,
    updated_at  TEXT,
    PRIMARY KEY(artist_id, track_id)
);

CREATE TABLE IF NOT EXISTS deezer_artist_albums (
    artist_id     INTEGER NOT NULL REFERENCES deezer_artist_expansion(artist_id) ON DELETE CASCADE,
    album_id      INTEGER NOT NULL,
    artist        TEXT NOT NULL,
    album         TEXT NOT NULL,
    rank          INTEGER NOT NULL,
    record_type   TEXT,
    release_date  TEXT,
    fans           INTEGER,
    tracklist_url  TEXT,
    raw_json       TEXT,
    updated_at     TEXT,
    PRIMARY KEY(artist_id, album_id)
);
CREATE INDEX IF NOT EXISTS idx_deezer_artist_albums_album
    ON deezer_artist_albums(album_id);

CREATE TABLE IF NOT EXISTS deezer_album_expansion (
    album_id    INTEGER PRIMARY KEY,
    album       TEXT,
    artist_id   INTEGER,
    artist      TEXT,
    status      TEXT NOT NULL DEFAULT 'pending', -- pending | fetched | miss | error
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    tracks_json TEXT,
    updated_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_deezer_album_expansion_status
    ON deezer_album_expansion(status, attempts);

CREATE TABLE IF NOT EXISTS deezer_album_tracks (
    album_id       INTEGER NOT NULL REFERENCES deezer_album_expansion(album_id) ON DELETE CASCADE,
    track_id       INTEGER NOT NULL,
    artist_id      INTEGER,
    artist         TEXT NOT NULL,
    title          TEXT NOT NULL,
    rank           INTEGER NOT NULL,
    disk_number    INTEGER,
    track_position INTEGER,
    duration_s     INTEGER,
    isrc           TEXT,
    preview_url    TEXT,
    raw_json       TEXT,
    updated_at     TEXT,
    PRIMARY KEY(album_id, track_id)
);
CREATE INDEX IF NOT EXISTS idx_deezer_album_tracks_track
    ON deezer_album_tracks(track_id);

-- Spotify catalog expansion. Keep external artist/album tracks isolated from
-- the library dimensions until the user accepts a recommendation. Artist album
-- groups are stored explicitly because development-mode responses no longer
-- consistently include Spotify's historical album_group field.
CREATE TABLE IF NOT EXISTS spotify_expansion_capabilities (
    surface    TEXT PRIMARY KEY,
    status     TEXT NOT NULL, -- available | unavailable
    error      TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS spotify_artist_expansion (
    artist_id          TEXT PRIMARY KEY,
    artist             TEXT NOT NULL,
    seed_recording_key TEXT,
    status             TEXT NOT NULL DEFAULT 'pending', -- pending | fetched | miss | error
    attempts           INTEGER NOT NULL DEFAULT 0,
    error              TEXT,
    top_json           TEXT,
    albums_json        TEXT,
    updated_at         TEXT
);
CREATE INDEX IF NOT EXISTS idx_spotify_artist_expansion_status
    ON spotify_artist_expansion(status, attempts);

CREATE TABLE IF NOT EXISTS spotify_artist_top_tracks (
    artist_id   TEXT NOT NULL REFERENCES spotify_artist_expansion(artist_id) ON DELETE CASCADE,
    track_id    TEXT NOT NULL,
    artist      TEXT NOT NULL,
    title       TEXT NOT NULL,
    rank        INTEGER NOT NULL,
    album_id    TEXT,
    duration_ms INTEGER,
    raw_json    TEXT,
    updated_at  TEXT,
    PRIMARY KEY(artist_id, track_id)
);

CREATE TABLE IF NOT EXISTS spotify_artist_albums (
    artist_id    TEXT NOT NULL REFERENCES spotify_artist_expansion(artist_id) ON DELETE CASCADE,
    album_id     TEXT NOT NULL,
    album_group  TEXT NOT NULL, -- album | single | appears_on | compilation
    artist       TEXT NOT NULL,
    album        TEXT NOT NULL,
    rank         INTEGER NOT NULL,
    album_type   TEXT,
    release_date TEXT,
    total_tracks INTEGER,
    raw_json     TEXT,
    updated_at   TEXT,
    PRIMARY KEY(artist_id, album_id, album_group)
);
CREATE INDEX IF NOT EXISTS idx_spotify_artist_albums_album
    ON spotify_artist_albums(album_id, album_group);

CREATE TABLE IF NOT EXISTS spotify_album_expansion (
    album_id    TEXT PRIMARY KEY,
    album       TEXT,
    status      TEXT NOT NULL DEFAULT 'pending', -- pending | fetched | miss | error
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    tracks_json TEXT,
    updated_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_spotify_album_expansion_status
    ON spotify_album_expansion(status, attempts);

CREATE TABLE IF NOT EXISTS spotify_album_tracks (
    album_id       TEXT NOT NULL REFERENCES spotify_album_expansion(album_id) ON DELETE CASCADE,
    track_id       TEXT NOT NULL,
    artist         TEXT NOT NULL,
    artists_json   TEXT,
    title          TEXT NOT NULL,
    rank           INTEGER NOT NULL,
    disc_number    INTEGER,
    track_number   INTEGER,
    duration_ms    INTEGER,
    raw_json       TEXT,
    updated_at     TEXT,
    PRIMARY KEY(album_id, track_id)
);
CREATE INDEX IF NOT EXISTS idx_spotify_album_tracks_track
    ON spotify_album_tracks(track_id);

-- Classic Spotify audio features (Echo Nest lineage). The API died Nov 2024;
-- the data outlived it via public feature archives. Imported by exact
-- track-id join from data/catalogs/ (scripts/import_features.py); one row per
-- recording (rep member preferred). Feeds sequencing (energy arcs, tempo) and
-- the bake-off's feature baseline.
CREATE TABLE IF NOT EXISTS spotify_features (
    recording_key TEXT PRIMARY KEY,
    track_id      TEXT,              -- member id the features came from
    danceability REAL, energy REAL, valence REAL, tempo REAL,
    key INTEGER, mode INTEGER, loudness REAL,
    acousticness REAL, instrumentalness REAL, liveness REAL, speechiness REAL,
    time_signature INTEGER, track_popularity INTEGER,
    updated_at TEXT
);

-- Exact-join coverage for the static classic-features archive. A missing
-- spotify_features row alone is ambiguous; this ledger records whether the
-- current library recording was checked and absent from that source snapshot.
CREATE TABLE IF NOT EXISTS spotify_feature_coverage (
    recording_key TEXT PRIMARY KEY,
    status        TEXT NOT NULL,       -- available | source_not_covered | error | unknown_legacy
    source        TEXT,
    attempts      INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    updated_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_spotify_feature_coverage_status
    ON spotify_feature_coverage(status);

-- Library embedding execution ledger. Successful vectors remain in embeddings;
-- this keeps decode/model failures separate from audio preview fetch failures.
CREATE TABLE IF NOT EXISTS embedding_runs (
    recording_key TEXT NOT NULL,
    model         TEXT NOT NULL,
    status        TEXT NOT NULL,       -- pending | available | error | unknown_legacy
    attempts      INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    updated_at    TEXT,
    PRIMARY KEY(recording_key, model)
);
CREATE INDEX IF NOT EXISTS idx_embedding_runs_status
    ON embedding_runs(model, status, attempts);

-- VPS-local audit trail for Mac-owned enrichment promotions. Track reports use
-- this high-water mark to distinguish new live recordings from provider misses.
CREATE TABLE IF NOT EXISTS enrichment_sync (
    id                INTEGER PRIMARY KEY,
    source_created_at TEXT NOT NULL,
    applied_at        TEXT NOT NULL,
    bundle_name       TEXT,
    tables_json       TEXT
);
CREATE INDEX IF NOT EXISTS idx_enrichment_sync_applied
    ON enrichment_sync(applied_at DESC);

-- ISRC -> MusicBrainz recording id, the join key for ListenBrainz similarity.
CREATE TABLE IF NOT EXISTS recording_mbid (
    recording_key TEXT PRIMARY KEY,
    isrc          TEXT,
    mbid          TEXT,
    method        TEXT,              -- 'isrc' | 'search'
    status        TEXT NOT NULL DEFAULT 'pending',  -- pending | resolved | miss | error
    attempts      INTEGER NOT NULL DEFAULT 0,
    updated_at    TEXT
);

-- ---------------------------------------------------------------------------
-- Downstream consumers: taste regions, proposals, recommendation events.
-- taste_regions caches one embedding-space region per user-authored playlist
-- (the membership engine behind curation, discovery, and genre playlists).
-- playlist_proposals is the ONE ledger every consumer writes suggestions into;
-- the /review page drains it, and each accept/reject doubles as a training label.
-- recommendation_events logs every track acetate surfaced so later plays/likes
-- of it measure the recommender for real (self-grading precision).
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS taste_regions (
    playlist_id  TEXT PRIMARY KEY,
    name         TEXT,
    owner_id     TEXT,
    n_members    INTEGER NOT NULL,   -- recordings in the latest snapshot
    n_embedded   INTEGER NOT NULL,   -- members with an effnet vector (region basis)
    dim          INTEGER NOT NULL,
    centroid     BLOB NOT NULL,      -- float32, normalized mean of member vectors
    spread       REAL,               -- mean member cosine to centroid (tightness)
    member_p10   REAL,               -- 10th-percentile member membership score (misfit floor)
    built_at     TEXT
);

CREATE TABLE IF NOT EXISTS playlist_proposals (
    id            INTEGER PRIMARY KEY,
    playlist_id   TEXT NOT NULL,     -- target playlist (authored or managed)
    playlist_name TEXT,
    action        TEXT NOT NULL,     -- 'add' | 'remove'
    kind          TEXT NOT NULL,     -- 'library' | 'candidate'
    recording_key TEXT NOT NULL,     -- library recording_key or candidate_key
    track_id      TEXT,              -- Spotify id when known (enables one-click apply)
    artist        TEXT,
    title         TEXT,
    score         REAL,
    reasons       TEXT,              -- JSON list of human-readable reasons
    source        TEXT NOT NULL,     -- consumer that proposed it: 'health' | 'discovery' | 'genre'
    status        TEXT NOT NULL DEFAULT 'proposed',  -- proposed | accepted | rejected
    created_at    TEXT,
    decided_at    TEXT,
    UNIQUE (playlist_id, recording_key, action, source)
);
CREATE INDEX IF NOT EXISTS idx_proposals_status ON playlist_proposals(status, playlist_id, score DESC);

CREATE TABLE IF NOT EXISTS recommendation_events (
    id             INTEGER PRIMARY KEY,
    recording_key  TEXT NOT NULL,    -- library recording_key or candidate_key
    track_id       TEXT,             -- Spotify id when known (joins play_events for feedback)
    surface        TEXT NOT NULL,    -- 'discovery_playlist' | 'proposal:health' | ...
    target         TEXT,             -- playlist id/name it was surfaced into
    score          REAL,
    recommended_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rec_events_key ON recommendation_events(recording_key);
CREATE INDEX IF NOT EXISTS idx_rec_events_track ON recommendation_events(track_id);

-- ---------------------------------------------------------------------------
-- Playlist profile layer: what each playlist IS to the user.
-- playlist_members flattens the latest snapshot of every playlist (the
-- snapshot join is too fiddly for ad-hoc/agent SQL); playlist_stats is one
-- row per playlist with usage, add-cadence, and a derived persona ('archive'
-- dump vs 'one-shot' event playlist vs living 'rotation'); playlist_labels
-- holds the user's own overrides ('fave', 'archive', ...), which beat the
-- heuristics. Both regenerable tables are rebuilt by playlist_profile.rebuild.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS playlist_members (
    playlist_id   TEXT NOT NULL,
    track_id      TEXT NOT NULL,
    recording_key TEXT,             -- via track_recording; NULL if unmapped
    position      INTEGER,
    added_at      TEXT,
    PRIMARY KEY (playlist_id, track_id)
);
CREATE INDEX IF NOT EXISTS idx_playlist_members_rec ON playlist_members(recording_key);

CREATE TABLE IF NOT EXISTS playlist_stats (
    playlist_id       TEXT PRIMARY KEY,
    name              TEXT,
    owner_id          TEXT,
    is_own            INTEGER NOT NULL DEFAULT 0,  -- authored by the user
    is_managed        INTEGER NOT NULL DEFAULT 0,  -- created by acetate
    is_deleted        INTEGER NOT NULL DEFAULT 0,  -- gone from the latest backup
    n_tracks          INTEGER NOT NULL DEFAULT 0,
    n_liked           INTEGER NOT NULL DEFAULT 0,
    first_added_at    TEXT,           -- oldest member add
    last_added_at     TEXT,           -- newest member add
    add_days          INTEGER,        -- distinct days tracks were added (1 = one sitting)
    member_plays      INTEGER,        -- all-time real plays of member recordings
    member_plays_180d INTEGER,        -- ...in the last 180 days
    plays_from        INTEGER,        -- plays with THIS playlist as Spotify context (poll-era only)
    last_played_from  TEXT,
    persona           TEXT,           -- acetate | archive | one-shot | rotation | dormant | deleted | other
    user_label        TEXT,           -- copy of playlist_labels.label (override wins)
    built_at          TEXT
);

CREATE TABLE IF NOT EXISTS playlist_labels (
    playlist_id TEXT PRIMARY KEY,
    label       TEXT NOT NULL,       -- 'fave' | 'archive' | 'one-shot' | 'rotation' | ...
    note        TEXT,
    updated_at  TEXT
);

-- Filing ledger: the ONLY sanctioned path by which acetate writes to a playlist it
-- did not create. Every add is append-only and recorded here so it is fully traceable
-- and reversible; undo removes exactly the track this ledger says acetate added (and
-- nothing else). No other code may add/remove tracks on a non-managed playlist.
CREATE TABLE IF NOT EXISTS filing_actions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_key   TEXT NOT NULL,
    track_id        TEXT NOT NULL,        -- the exact Spotify track id acetate added
    playlist_id     TEXT NOT NULL,        -- target (a user-authored playlist)
    playlist_name   TEXT,
    artist          TEXT,
    title           TEXT,
    score           REAL,                 -- membership score that suggested this target
    source          TEXT NOT NULL DEFAULT 'filing',
    snapshot_before TEXT,                 -- Spotify snapshot_id before the add
    snapshot_after  TEXT,                 -- ...and after (audit trail)
    status          TEXT NOT NULL DEFAULT 'added',  -- 'added' | 'undone'
    added_at        TEXT NOT NULL,
    undone_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_filing_status ON filing_actions(status, added_at);
