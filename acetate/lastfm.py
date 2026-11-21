"""Last.fm tag lookups for genre classification.

Read-only: needs only LASTFM_API_KEY. We use crowd tags (track-level first, then
artist-level) as the genre signal since Spotify's artist genres are sparse and
audio-features is gone. Tags come with a 0-100 weight we keep as confidence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests

API_ROOT = "https://ws.audioscrobbler.com/2.0/"


class LastfmAPIError(requests.RequestException):
    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class TagLookup:
    tags: list[tuple[str, int]]
    status: str
    detail: str


def _call(method: str, **params) -> dict:
    params.update(method=method, api_key=os.environ["LASTFM_API_KEY"], format="json", autocorrect=1)
    resp = requests.get(
        API_ROOT, params=params, timeout=15, headers={"User-Agent": "acetate/0.2 (personal)"}
    )
    resp.raise_for_status()
    payload = resp.json()
    if isinstance(payload, dict) and payload.get("error") is not None:
        raise LastfmAPIError(
            str(payload.get("message") or f"Last.fm API error {payload['error']}"),
            int(payload["error"]),
        )
    return payload


def _tags(payload: dict) -> list[tuple[str, int]]:
    tags = (payload.get("toptags") or {}).get("tag") or []
    if isinstance(tags, dict):  # single-tag responses come back un-listed
        tags = [tags]
    return [(t["name"].lower(), int(t.get("count", 0) or 0)) for t in tags if t.get("name")]


def track_top_tags(artist: str, track: str) -> list[tuple[str, int]]:
    return _tags(_call("track.gettoptags", artist=artist, track=track))


def artist_top_tags(artist: str) -> list[tuple[str, int]]:
    return _tags(_call("artist.gettoptags", artist=artist))


def best_tags_result(artist: str, track: str, min_count: int = 10) -> TagLookup:
    """Fetch track tags, falling back to artist tags, without erasing provenance.

    Provider error 6 is an authoritative entity miss. Network/rate-limit/server
    errors propagate so callers can retry instead of persisting an empty result.
    """
    track_miss = False
    try:
        tags = track_top_tags(artist, track)
    except LastfmAPIError as exc:
        if exc.code != 6:
            raise
        track_miss = True
        tags = []
    if tags:
        kept = [(name, count) for name, count in tags if count >= min_count]
        return TagLookup(kept, "available", "track_tags")

    try:
        artist_tags = artist_top_tags(artist)
    except LastfmAPIError as exc:
        if exc.code != 6:
            raise
        if track_miss:
            return TagLookup([], "provider_miss", "track_and_artist_missing")
        return TagLookup([], "available", "track_empty_artist_missing")
    kept = [(name, count) for name, count in artist_tags if count >= min_count]
    detail = "artist_tags" if kept else "provider_returned_no_qualifying_tags"
    return TagLookup(kept, "available", detail)


def best_tags(artist: str, track: str, min_count: int = 10) -> list[tuple[str, int]]:
    """Compatibility wrapper; request failures deliberately propagate."""
    return best_tags_result(artist, track, min_count=min_count).tags


def track_info(artist: str, track: str) -> dict | None:
    """Global listeners/playcount — the crowd-scale popularity signal tags lack."""
    data = _call("track.getinfo", artist=artist, track=track)
    t = data.get("track") or {}
    if not t:
        return None
    return {"listeners": int(t.get("listeners") or 0), "playcount": int(t.get("playcount") or 0)}


def track_similar(artist: str, track: str, limit: int = 25) -> list[tuple[str, str, float]]:
    """Collaborative 'people also play' similarity: (artist, title, match 0-1).
    This is the co-listening signal ListenBrainz Labs would give us if its
    similar-recordings dataset were populated (checked 2026-07: it returns
    empty for everything, Radiohead included)."""
    data = _call("track.getsimilar", artist=artist, track=track, limit=limit)
    tracks = (data.get("similartracks") or {}).get("track") or []
    if isinstance(tracks, dict):
        tracks = [tracks]
    return [
        ((t.get("artist") or {}).get("name", ""), t.get("name", ""), float(t.get("match", 0) or 0))
        for t in tracks
        if t.get("name")
    ]
