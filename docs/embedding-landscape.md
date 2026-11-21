# The music-embedding landscape — acetate's baseline

A field guide to every tool, model, dataset, and API relevant to acetate's embedding /
similarity / recommendation layer. **Snapshot date: 2026-07-04.** Facts marked ✅ were
empirically verified in-session on that date (or 2026-07-03); unmarked entries come from the
original build brief or papers and may have drifted. Extend from here instead of re-researching.

Legend: **[IN USE]** shipping in acetate today · **[BENCH]** evaluated/planned, not wired ·
**[DEAD]** verified unavailable — do not re-litigate.

---

## 1. Audio sources (getting something for the models to hear)

| Source | Status | Notes |
|---|---|---|
| **Deezer previews by ISRC** | **[IN USE]** ✅ | `api.deezer.com/track/isrc:{isrc}`, public, no auth. 30s / 44.1kHz / 128k MP3, ~500KB. Typical hit rates: ~89% exact-ISRC, ~95% incl. search fallbacks; ~5% miss. ~10 req/s allowed (50/5s rolling); we run 4 threads behind one 0.25s global throttle. |
| **Deezer search** | **[IN USE]** ✅ | `search?q=artist:"X" track:"Y"`; trusted only with ±5s duration match vs Spotify's `duration_ms` (or exact normalized artist+title when no duration known). |
| **iTunes search** | **[IN USE]** ✅ | `itunes.apple.com/search` — last-resort fallback for the small tail that Deezer misses. **The `lookup?isrc=` endpoint returned 0/5 on real ISRCs** — treat iTunes-by-ISRC as dead, search-only. Previews are 30–90s AAC. |
| **Spotify `preview_url`** | **[DEAD]** | Removed for post-Nov-2024 apps along with audio-features etc. Old scraped `p.scdn.co` CDN links rot independently. |

**Load-bearing insight ✅:** our embedding models ingest **16 kHz mono** — every codec-quality
argument dies at the resampler. 128k previews are transparent *to the model*. Quality only
matters for human listening.

## 2. Content embedding models

| Model | Status | Input | Output | License | Notes |
|---|---|---|---|---|---|
| **discogs-effnet** (`discogs-effnet-bs64-1.pb`) | **[IN USE]** ✅ | 16kHz mono | 1280-d / patch (~3s), mean-pooled | **CC BY-NC-ND 4.0** | Essentia/MTG. ✅ 0.76–0.8s per 30s clip on M-series CPU (Py3.14 arm64 wheel EXISTS — the only version; check `pypi.org/pypi/essentia-tensorflow/json` for wheel drift). Style quality on our library: excellent (Kygo remix→Tropical House, Pretty Lights→Downtempo). |
| **genre_discogs400 head** | **[IN USE]** ✅ | effnet embedding | 400 Discogs styles | CC BY-NC-ND 4.0 | Same forward pass — free interpretable labels. Top-8/track stored in `recording_styles`. The deep electronic taxonomy Last.fm lacks. |
| **MAEST** (`discogs-maest-*`) | [BENCH] | 16kHz mono | same discogs400 taxonomy | CC BY-NC-ND 4.0 | Transformer upgrade over effnet, same head/taxonomy — drop-in swap when we want a quality bump. https://essentia.upf.edu/models.html |
| **LAION-CLAP** | **[IN USE]** ✅ | 48kHz | 512-d joint audio↔text | more permissive (check per checkpoint) | `scripts/clap_worker.py`, checkpoint **pinned** `music_audioset_epoch_15_esc_90.14.pt` (baked into model id `clap-music`). ✅ 8.6 tracks/s batched on M-series CPU; ~8k-track library embeds cleanly with zero errors. Text retrieval verified: "melodic techno with a long hypnotic build" surfaces melodic techno cleanly; weak on beatless-vs-not distinctions. `acetate search`. laion_clap gotcha: imports torchvision without declaring it. |
| **MuQ / MuQ-MuLan** | [BENCH] | 24kHz | SSL backbone / joint text space | **CC BY-NC 4.0** | 2025, edges CLAP on zero-shot tagging. `pip install muq`. https://github.com/tencent-ailab/MuQ |
| **MERT-v1-330M** | [BENCH] | 24kHz | 1024-d | Apache-ish (check card) | Best pure-sonic SSL representation; no text alignment, no labels. Alternative similarity space to A/B against effnet. https://huggingface.co/m-a-p/MERT-v1-330M |

**Licensing boundary:** MTG (`discogs-effnet`, `genre_discogs400`, `MAEST`) and MuQ weights are
licensed non-commercially (`CC BY-NC-ND 4.0` / `CC BY-NC 4.0`) — personal and research use only.
This open-source repo ships code only; users fetch model weights directly at runtime.

## 3. Genre / tag sources

| Source | Status | Notes |
|---|---|---|
| **discogs400 styles** | **[IN USE]** ✅ | Content-derived, dense over everything embedded, `Parent---Style` format. Agent + SQL queryable. Candidate input for the 🎧 classifier. |
| **Last.fm tags** | **[IN USE]** | Crowd tags w/ weights (`track.getTopTags`→`artist.getTopTags` fallback). Sparse where taste is specific — only 9% of the hand-built melodic-techno playlist carries the literal tag. NON_GENRE denylist filters geo/mood/decade noise. |
| **Spotify artist `genres[]`** | **[DEAD]** | Returns empty for post-2024 apps (verified earlier, see TODO Notes). |
| **MusicBrainz tags/genres** | [BENCH] | Exists via MB API; unexplored. We already hit MB for MBIDs at 1 req/s. |

## 4. Collaborative similarity (the "recommendation feel")

| Source | Status | Notes |
|---|---|---|
| **Last.fm `track.getSimilar`** | **[IN USE]** ✅ | Works, returns in- and out-of-library tracks with match scores; wired into `acetate similar to` + the chat agent's `similar_tracks` tool. Artist-level `artist.getSimilar` also available. |
| **ListenBrainz Labs `similar-recordings`** | **[DEAD-ish]** ✅ 2026-07-03 | Endpoint live, param shape `?recording_mbids=<uuid>&algorithm=<enum>` (valid enums are listed in the HTML form at `labs.api.listenbrainz.org/similar-recordings`) — but **returns `[]` for every MBID tried, incl. Radiohead "Creep"**. Dataset unpopulated. Re-check occasionally; our MBID ledger (`recording_mbid`) is the ready join key. |
| **ListenBrainz main API / Troi** | [BENCH] | `api.listenbrainz.org` stats endpoints + Troi playlist toolkit — unexplored, distinct from the Labs dataset hoster. https://troi.readthedocs.io |
| **MPD track2vec** | [BENCH] | Train word2vec over the Million Playlist Dataset (playlist=sentence). Caveats: US-centric, **frozen Nov 2017** (weak for post-2017 electronic), non-commercial, AIcrowd download gone — Kaggle mirror: https://www.kaggle.com/datasets/himanshuwagh/spotify-million |
| **ytmusicapi (unofficial YT Music)** | **[OPTIONAL]** | Unauthenticated reverse-engineered client lib (optional `ytmusic` extra: `uv sync --extra ytmusic`). `get_watch_playlist(radio=True)` provides per-seed radio from YouTube Music's live recommendation engine, plus song search (duration-checkable), related artists, and moods/genres browse. Unofficial/ToS-gray, can break without notice; be civil. Pairs naturally with LAION-DISCO-12M (YTM-keyed names). No BPM, no listener counts. |
| **Deezer enrichment (2026-07-04 audit)** | **[IN USE]** ✅ | `acetate embed enrich` harvests what the preview fetch discarded: track **bpm (~2/3 coverage)** + gain + rank, album **label** (the electronic scene signal) + genres + dates, artist nb_fan + **related-artists graph** (Spotify's dead endpoint, alive here). Keyed by source_ids already in audio_cache; per-entity ledgers distinguish pending work, authoritative provider misses, errors, and available payloads. |
| **Last.fm artist layer** | [BENCH] next | `artist.getSimilar` verified strong for artist-level similarity; + `artist.getInfo` listeners = artist-level obscurity. One cache pass over library artists. `tag.getTopTracks` verified chart-polluted. |
| **AcousticBrainz** | **[DEAD]** ✅ | Archived 2022; probed 2026-07-04 with resolved MBIDs — misses. We compute features from our own audio instead. |

## 5. Beyond-library datasets (the expansion question)

**Verdict ✅ (researched 2026-07-03): no available dataset ships precomputed music *audio*
embeddings at millions-of-tracks scale.** Expansion architecture = **embed candidates on
demand** with our own model (compatible by construction): candidate names from Last.fm
similar / LAION-DISCO-12M → Deezer preview → effnet (~0.8s/track).

| Dataset | Status | Notes |
|---|---|---|
| **DISCO-10M** | **[DEAD]** ✅ | Removed from Hugging Face between ~Aug 2023 and May 2024; org page survives but shards were cleared, with no public mirror. Its CLAP checkpoint was also never documented, so precomputed vectors were not reliably joinable anyway. Was: 15.3M pairs / ~2.6M unique tracks / 28GB / CC-BY-4.0. Paper: https://arxiv.org/abs/2306.13512 |
| **gaodrew/DISCO-200K-Tensors** | live ✅ | Surviving DISCO fragment: embeddings + song-name index derived from DISCO-200K-high-quality (`all_embeddings.pt` 410MB, `inverted_index.pkl`). Unknown checkpoint caveat applies to joins. https://huggingface.co/datasets/gaodrew/DISCO-200K-Tensors |
| **LAION-DISCO-12M** | live ✅ | Optional local metadata catalog (`data/catalogs/laion-disco-12m/`, 750MB, 12.32M rows). Schema: YTM song_id / title / artist_names[] / album / views / duration. Apache 2.0. Useful artist-first: matched library artists expand to full catalog discographies as discovery candidates, and `song_id` feeds YouTube Music radio directly. |
| **Sleeping-DISCO 9M** | live ✅ | ~9M songs, Genius metadata + **lyric** (not audio) embeddings; CC-BY-NC-ND; partly gated. https://huggingface.co/datasets/sleeping-ai/Sleeping-DISCO-9M |
| **MTG-Jamendo** | live ✅ | 55k CC-licensed full tracks + tags; downloadable audio → compute embeddings with a known checkpoint. Good eval corpus, wrong distribution for mainstream-electronic taste. https://github.com/MTG/mtg-jamendo-dataset |
| **fma-mert-embeddings** | live ✅ | FMA-small (8k tracks) with precomputed MERT-v1-330M 1024-d. Toy scale. |
| **Audio Atlas** (ETH) | reference ✅ | CLAP-embedding search/viz webapp (Milvus + DeepScatter) — a plumbing reference, not a dataset. Low activity. https://github.com/ETH-DISCO/audio-atlas |
| **MetaMIDI Dataset** | [BENCH] | 436k MIDIs with **10.8M audio↔MIDI matches keyed by Spotify track ID** (matched vs 32M preview clips, 2021). A joinable symbolic layer (melody/chords) for harmonic-similarity features someday. https://github.com/jeffreyjohnens/MetaMIDIDataset |
| **Caption cluster** (JamendoMaxCaps 362k, LP-MusicCaps method, MusicCaps 5.5k, Song Describer, WikiMuTe) | [BENCH] | Text↔music description data. Uses: calibrate CLAP prompt phrasing; LP-MusicCaps-style **auto-captioning of our own library** (sidebar descriptions + caption-based search); MusicBench for A/B'ing CLAP vs MuQ-MuLan. |
| **Directory: Yuan-ManX/ai-audio-datasets** | reference ✅ | ~110-entry awesome-list surveyed; above rows are the acetate-relevant delta, the rest is MIDI/symbolic/singing/regional corpora. |
| **Classic audio features (user-supplied Parquet)** | **[OPTIONAL]** | Spotify deprecated `/v1/audio-features` for new apps in Nov 2024. If you have a local Parquet archive with `track_id` and classic feature columns (e.g. a public academic/Kaggle dataset such as `rodolfofigueroa/spotify-12m-songs` or your own historical export), import it by exact ID join with `uv run scripts/import_features.py --parquet /path/to/features.parquet` into `spotify_features`. Acetate works cleanly without it. |

## 6. Identity / join keys

- **ISRC** — already in `tracks` (from Spotify track objects; 99.8% of liked, 88% of ≥5-play).
  The key that makes Deezer preview lookup exact.
- **MusicBrainz MBID** — `recording_mbid` ledger, resolved ISRC-first then scored artist+title
  search (accept ≥90). MB API: 1 req/s, identifying User-Agent required, `fmt=json`.
  ✅ ISRC lookup works (`/ws/2/isrc/{isrc}`).
- **AcoustID/Chromaprint** — [BENCH] audio-fingerprint→MBID fallback for the unmatched tail.

## 7. Reduction / clustering / ANN

- **UMAP 3D, cosine** — [IN USE] ✅ `n_neighbors=30, min_dist=0.1, random_state=42`; ~10k×1280
  fits in seconds-to-minutes. Coords centered then scaled to ±100 (center BEFORE scaling —
  learned the hard way). Stochastic across input changes: full page reload on rebuild.
- **HDBSCAN** — [IN USE] ✅ sklearn's built-in (`min_cluster_size=25, min_samples=10`) on the 3D
  coords; typically ~15 clusters / ~35% noise at ~8k points. Tunable; could cluster in higher-dim UMAP
  space for better purity.
- **ANN** — at 10⁴ vectors, **brute-force numpy dot IS the index** (✅ instant). FAISS/LanceDB
  only become relevant if the beyond-library frontier grows to 10⁶.
- **Env constraint ✅:** essentia-tensorflow needs Py3.14 (only arm64 wheel); numba/umap need
  ≤3.13 → two PEP-723 script envs (`scripts/embed_worker.py`, `scripts/reduce_worker.py`).

## 8. Visualization

- **three.js Points + custom shader** — [IN USE] ✅ additive blending, per-point size/dim
  attributes, raycast picking. Gotcha: ShaderMaterial does NOT auto-declare custom uniforms.
- **deck.gl ScatterplotLayer** — [BENCH] alternative if point count grows 10×.
- **TF Embedding Projector / Nomic Atlas** — skipped (decision: straight to the dashboard page).

## 9. DJ features (phase-4 polish, unexplored)

- **Essentia TempoCNN + key extractors** — BPM/key for harmonic (Camelot) sequencing; works on
  30s previews. Same essentia-tensorflow env as the embed worker.
- **librosa** — pure-Python fallback for the same.
- **Demucs** — stem separation before embedding (cleaner sonic vectors); heavy, GPU-preferred.

## 10. Where to pick up next

See [`ROADMAP.md`](../ROADMAP.md) for the live public roadmap. Highlights:
retrieve→rerank→sequence playlist engine · ghost-star beyond-library candidates ·
genre classifier calibration · lasso→playlist in Hyperspace · MCP server.
