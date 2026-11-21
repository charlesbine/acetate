"""Seed a self-contained, melodic-techno-focused demo SQLite database for Acetate.

Usage:
    uv run python scripts/seed_demo.py --db /tmp/acetate-demo.db
    uv run python scripts/seed_demo.py --db /tmp/acetate-demo.db --offline

Populates 10 years (2017-2026) of realistic listening history, canonical recordings,
playlists (authored, smart, curated, and managed), Discogs electronic styles, 3D
Hyperspace point clouds, playlist taste regions, and a sample Chat conversation with
a playlist proposal card. Never reads any real user database or credentials.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import random
import sqlite3
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from acetate import canon, chatstore, db, playlist_profile, taste

# 20 headline melodic techno / electronic artists with signature tracks, albums, labels,
# BPMs, Discogs styles, and Last.fm genre tags.
HEADLINE_CATALOG: list[dict] = [
    {
        "artist": "Stephan Bodzin",
        "style": "Electronic---Melodic Techno",
        "genres": ["melodic techno", "techno", "minimal techno"],
        "label": "Herzblut",
        "weight": 1.45,
        "tracks": [
            ("Singularity", "Powers of Ten", 122, 421000, 2017),
            ("Boavista", "Boavista", 123, 398000, 2021),
            ("Strand", "Powers of Ten", 122, 456000, 2017),
            ("Zulu", "Powers of Ten", 124, 412000, 2018),
            ("Luka-Leon", "Boavista", 123, 389000, 2021),
            ("Catamaran", "Boavista", 121, 405000, 2022),
            ("River", "Boavista", 124, 376000, 2023),
            ("Isaac", "Boavista", 122, 418000, 2026),
        ],
    },
    {
        "artist": "Anyma",
        "style": "Electronic---Melodic Techno",
        "genres": ["melodic techno", "techno", "progressive house"],
        "label": "Afterlife",
        "weight": 1.40,
        "tracks": [
            ("Consciousness", "Genesys", 125, 336000, 2022),
            ("Eternity", "Genesys", 124, 348000, 2023),
            ("Syren", "Genesys", 126, 329000, 2023),
            ("Welcome To The Opera", "Genesys", 125, 314000, 2023),
            ("Pictures Of You", "Genesys II", 124, 342000, 2024),
            ("Explore Your Future", "Genesys", 125, 318000, 2024),
            ("The Sign", "Genesys", 124, 351000, 2025),
            ("Hypnotized", "The End of Genesys", 125, 322000, 2026),
        ],
    },
    {
        "artist": "Tale Of Us",
        "style": "Electronic---Melodic Techno",
        "genres": ["melodic techno", "techno", "deep techno"],
        "label": "Afterlife",
        "weight": 1.32,
        "tracks": [
            ("Nova", "Realm of Consciousness", 124, 392000, 2019),
            ("Astral", "Realm of Consciousness", 123, 415000, 2018),
            ("Monument", "Endless", 121, 408000, 2017),
            ("North Star", "North Star", 123, 384000, 2017),
            ("Silent Space", "Endless", 120, 398000, 2019),
            ("Red Sky", "Realm of Consciousness", 124, 376000, 2020),
            ("Lies", "Realm of Consciousness", 123, 369000, 2022),
        ],
    },
    {
        "artist": "Innellea",
        "style": "Electronic---Melodic Techno",
        "genres": ["melodic techno", "techno", "electronica"],
        "label": "Afterlife",
        "weight": 1.25,
        "tracks": [
            ("Vigilans", "Vigilans", 125, 378000, 2020),
            ("Forced Adaptation", "Five Phases", 126, 362000, 2022),
            ("Loss Of Hope", "Five Phases", 124, 391000, 2022),
            ("The World Returns", "The Belonging", 124, 354000, 2024),
            ("Targo", "Distorted Youth", 125, 344000, 2021),
            ("Confrontation", "The Belonging", 126, 368000, 2026),
        ],
    },
    {
        "artist": "Ben Böhmer",
        "style": "Electronic---Progressive House",
        "genres": ["progressive house", "melodic house", "deep house"],
        "label": "Anjunadeep",
        "weight": 1.28,
        "tracks": [
            ("Breathing", "Breathing", 121, 355000, 2019),
            ("Beyond Beliefs", "Begin Again", 122, 348000, 2021),
            ("Father Ocean", "Begin Again", 122, 391000, 2020),
            ("Cappadocia", "Live from Cappadocia", 121, 372000, 2020),
            ("In Memoriam", "Breathing", 120, 364000, 2019),
            ("Wall Of Strings", "Begin Again", 123, 342000, 2022),
            ("Faithless", "Bloom", 122, 336000, 2026),
        ],
    },
    {
        "artist": "Mind Against",
        "style": "Electronic---Melodic Techno",
        "genres": ["melodic techno", "deep techno", "techno"],
        "label": "Afterlife",
        "weight": 1.18,
        "tracks": [
            ("Walking Away", "Walking Away", 123, 412000, 2020),
            ("Solaris", "Avalon", 124, 428000, 2018),
            ("Dreamcast", "Dreamcast", 124, 396000, 2022),
            ("Avalon", "Avalon", 122, 434000, 2017),
            ("Gravity", "Habitat", 125, 382000, 2024),
            ("Changes", "Habitat", 124, 371000, 2026),
        ],
    },
    {
        "artist": "Adriatique",
        "style": "Electronic---Melodic Techno",
        "genres": ["melodic techno", "deep house", "techno"],
        "label": "Afterlife",
        "weight": 1.15,
        "tracks": [
            ("Miracle", "Miracle", 124, 358000, 2023),
            ("Home", "Nude", 122, 394000, 2018),
            ("Ray", "Nude", 123, 412000, 2018),
            ("Voices From The Dawn", "Nude", 121, 442000, 2019),
            ("With You", "X", 124, 364000, 2024),
            ("All I Ever Wanted", "X", 125, 352000, 2026),
        ],
    },
    {
        "artist": "Massano",
        "style": "Electronic---Melodic Techno",
        "genres": ["melodic techno", "techno", "progressive house"],
        "label": "Afterlife",
        "weight": 1.12,
        "tracks": [
            ("The Feeling", "The Feeling", 126, 364000, 2021),
            ("In My System", "In My System", 125, 349000, 2022),
            ("Shut Down", "Simulate", 126, 338000, 2023),
            ("Solitude", "Simulate", 124, 372000, 2024),
            ("Drown", "Simulate", 125, 355000, 2025),
            ("Euphoria", "Simulate II", 126, 346000, 2026),
        ],
    },
    {
        "artist": "Colyn",
        "style": "Electronic---Melodic Techno",
        "genres": ["melodic techno", "progressive house", "techno"],
        "label": "Afterlife",
        "weight": 1.08,
        "tracks": [
            ("Resolve", "Resolve", 123, 405000, 2019),
            ("Amor", "Amor", 123, 392000, 2020),
            ("Wait For You", "Wait For You", 124, 378000, 2021),
            ("Oxygen Levels Low", "Oxygen Levels Low", 124, 384000, 2022),
            ("Concepts of Love", "Concepts of Love", 123, 368000, 2024),
            ("Endless Horizon", "Endless Horizon", 124, 372000, 2026),
        ],
    },
    {
        "artist": "Fideles",
        "style": "Electronic---Melodic Techno",
        "genres": ["melodic techno", "techno", "deep techno"],
        "label": "Afterlife",
        "weight": 1.04,
        "tracks": [
            ("Night After Night", "Night After Night", 124, 362000, 2023),
            ("Resonant", "Resonant", 123, 396000, 2019),
            ("Lotus", "Lotus", 123, 412000, 2018),
            ("The Last Glow", "The Last Glow", 124, 388000, 2020),
            ("Overcome", "Overcome", 125, 374000, 2022),
            ("Awe", "Awe", 124, 366000, 2026),
        ],
    },
    {
        "artist": "Boris Brejcha",
        "style": "Electronic---Minimal",
        "genres": ["minimal techno", "techno", "melodic techno"],
        "label": "Fckng Serious",
        "weight": 1.16,
        "tracks": [
            ("Gravity", "Space Diver", 125, 458000, 2019),
            ("Purple Noise", "22", 125, 432000, 2017),
            ("Space Diver", "Space Diver", 125, 412000, 2020),
            ("Never Look Back", "Space Diver", 125, 424000, 2020),
            ("House Music", "Never Stop Dancing", 125, 408000, 2021),
            ("Spicy", "Never Stop Dancing", 125, 396000, 2022),
            ("Level One", "Level One", 126, 402000, 2026),
        ],
    },
    {
        "artist": "Mathame",
        "style": "Electronic---Melodic Techno",
        "genres": ["melodic techno", "techno", "progressive house"],
        "label": "Afterlife",
        "weight": 1.06,
        "tracks": [
            ("Nothing Around Us", "Nothing Around Us", 124, 418000, 2018),
            ("Skywalking", "Skywalking", 123, 426000, 2019),
            ("Never Give Up", "Never Give Up", 124, 384000, 2020),
            ("So What", "MEMO", 125, 356000, 2023),
            ("For Every Forever", "MEMO", 124, 392000, 2022),
            ("Humans", "MEMO", 125, 364000, 2026),
        ],
    },
    {
        "artist": "ARTBAT",
        "style": "Electronic---Melodic Techno",
        "genres": ["melodic techno", "progressive house", "techno"],
        "label": "Diynamic",
        "weight": 1.10,
        "tracks": [
            ("Horizon", "Horizon", 124, 412000, 2021),
            ("Upperground", "Upperground", 124, 404000, 2018),
            ("Flame", "Flame", 125, 388000, 2021),
            ("Age of Love", "Age of Love", 126, 416000, 2021),
            ("Best of Me", "Best of Me", 123, 392000, 2020),
            ("Breathe In", "Upperground", 124, 376000, 2026),
        ],
    },
    {
        "artist": "Monolink",
        "style": "Electronic---Progressive House",
        "genres": ["progressive house", "melodic techno", "deep house"],
        "label": "Embassy One",
        "weight": 1.08,
        "tracks": [
            ("Return to Oz", "Amniotic", 122, 478000, 2018),
            ("Sirens", "Amniotic", 121, 396000, 2018),
            ("Sinner", "Under Darkening Skies", 122, 368000, 2021),
            ("Rearrange My Mind", "Amniotic", 121, 384000, 2019),
            ("Otherside", "Under Darkening Skies", 123, 372000, 2022),
            ("Powerful Play", "Under Darkening Skies", 122, 364000, 2025),
        ],
    },
    {
        "artist": "RÜFÜS DU SOL",
        "style": "Electronic---Deep House",
        "genres": ["deep house", "progressive house", "electronica"],
        "label": "Rose Avenue",
        "weight": 1.14,
        "tracks": [
            ("Innerbloom", "Bloom", 122, 578000, 2017),
            ("Underwater", "Solace", 122, 349000, 2018),
            ("On My Knees", "Surrender", 124, 321000, 2021),
            ("Alive", "Surrender", 123, 334000, 2021),
            ("Next to Me", "Surrender", 122, 314000, 2021),
            ("Lately", "Inhale / Exhale", 124, 298000, 2024),
            ("Music Is Better", "Inhale / Exhale", 123, 312000, 2026),
        ],
    },
    {
        "artist": "Nora En Pure",
        "style": "Electronic---Deep House",
        "genres": ["deep house", "progressive house", "melodic house"],
        "label": "Enormous Tunes",
        "weight": 0.96,
        "tracks": [
            ("Come With Me", "Come With Me", 122, 332000, 2017),
            ("Lake Arrowhead", "Lake Arrowhead", 122, 346000, 2017),
            ("Tears In Your Eyes", "Tears In Your Eyes", 122, 328000, 2018),
            ("Morning Dew", "Morning Dew", 121, 354000, 2019),
            ("Diving with Whales", "Diving with Whales", 122, 341000, 2020),
            ("Wholehearted", "Purified", 123, 338000, 2025),
        ],
    },
    {
        "artist": "Emancipator",
        "style": "Electronic---Downtempo",
        "genres": ["downtempo", "trip hop", "electronica"],
        "label": "Loci Records",
        "weight": 1.02,
        "tracks": [
            (
                "Soon It Will Be Cold Enough to Build Fires",
                "Soon It Will Be Cold Enough",
                94,
                312000,
                2017,
            ),
            ("First Snow", "Soon It Will Be Cold Enough", 92, 294000, 2017),
            ("Nevergreen", "Safe in the Steep Cliffs", 96, 268000, 2018),
            ("Dusk to Dawn", "Dusk to Dawn", 95, 286000, 2019),
            ("Maps", "Baralku", 98, 274000, 2020),
            ("Labyrinth", "Mountain of Memory", 96, 302000, 2023),
        ],
    },
    {
        "artist": "Jon Hopkins",
        "style": "Electronic---Ambient",
        "genres": ["ambient", "techno", "electronica"],
        "label": "Domino",
        "weight": 1.05,
        "tracks": [
            ("Open Eye Signal", "Immunity", 124, 468000, 2017),
            ("Emerald Rush", "Singularity", 120, 336000, 2018),
            ("Singularity", "Singularity", 124, 389000, 2018),
            ("Collider", "Immunity", 122, 561000, 2019),
            ("Luminous Beings", "Singularity", 118, 492000, 2020),
            ("RITUAL (evocation)", "RITUAL", 116, 424000, 2025),
        ],
    },
    {
        "artist": "Bicep",
        "style": "Electronic---Breakbeat",
        "genres": ["breakbeat", "electronica", "deep house"],
        "label": "Ninja Tune",
        "weight": 1.12,
        "tracks": [
            ("Glue", "Bicep", 130, 269000, 2017),
            ("Apricots", "Isles", 128, 246000, 2021),
            ("Atlas", "Isles", 132, 352000, 2020),
            ("Opal", "Bicep", 127, 294000, 2018),
            ("Water", "Water", 131, 278000, 2022),
            ("Meli (II)", "Meli (II)", 129, 342000, 2023),
            ("CHROMA 001 HELIUM", "CHROMA", 132, 312000, 2026),
        ],
    },
    {
        "artist": "Four Tet",
        "style": "Electronic---Breakbeat",
        "genres": ["electronica", "ambient", "breakbeat"],
        "label": "Text Records",
        "weight": 1.00,
        "tracks": [
            ("Two Thousand and Seventeen", "New Energy", 118, 252000, 2017),
            ("Lush", "New Energy", 126, 304000, 2018),
            ("Baby", "Sixteen Oceans", 129, 264000, 2020),
            ("Loved", "Three", 122, 242000, 2024),
            ("Daydream Repeat", "Three", 124, 368000, 2024),
            ("Into Dust (Still Falling)", "Three+", 125, 296000, 2026),
        ],
    },
]

# Supporting electronic roster to give the library realistic breadth & rich Hyperspace density.
SUPPORTING_ARTISTS: list[tuple[str, str, list[str], str, int]] = [
    ("Joris Voorn", "Electronic---Melodic Techno", ["melodic techno", "techno"], "Spectrum", 124),
    (
        "Yotto",
        "Electronic---Progressive House",
        ["progressive house", "deep house"],
        "Odd One Out",
        123,
    ),
    (
        "Lane 8",
        "Electronic---Progressive House",
        ["progressive house", "melodic house"],
        "This Never Happened",
        122,
    ),
    ("Solomun", "Electronic---Deep House", ["deep house", "melodic techno"], "Diynamic", 122),
    (
        "CamelPhat",
        "Electronic---Melodic Techno",
        ["melodic techno", "deep house"],
        "When Stars Align",
        124,
    ),
    ("Kölsch", "Electronic---Melodic Techno", ["melodic techno", "techno"], "Kompakt", 124),
    ("Maceo Plex", "Electronic---Techno", ["techno", "deep techno"], "Ellum", 126),
    (
        "Kevin de Vries",
        "Electronic---Melodic Techno",
        ["melodic techno", "techno"],
        "Afterlife",
        125,
    ),
    ("Argy", "Electronic---Melodic Techno", ["melodic techno", "techno"], "Afterlife", 125),
    (
        "Chris Avantgarde",
        "Electronic---Melodic Techno",
        ["melodic techno", "techno"],
        "Afterlife",
        125,
    ),
    ("KAS:ST", "Electronic---Techno", ["techno", "melodic techno"], "Afterlife", 127),
    (
        "Cassian",
        "Electronic---Melodic Techno",
        ["melodic techno", "progressive house"],
        "Rose Avenue",
        124,
    ),
    (
        "Agents Of Time",
        "Electronic---Melodic Techno",
        ["melodic techno", "techno"],
        "Afterlife",
        124,
    ),
    ("Woo York", "Electronic---Techno", ["techno", "melodic techno"], "Afterlife", 126),
    (
        "Corren Cavini",
        "Electronic---Progressive House",
        ["progressive house", "melodic techno"],
        "Purified",
        123,
    ),
    ("Stephan Jolk", "Electronic---Melodic Techno", ["melodic techno", "techno"], "Afterlife", 124),
    ("Bonobo", "Electronic---Downtempo", ["downtempo", "electronica"], "Ninja Tune", 112),
    ("Tycho", "Electronic---Downtempo", ["downtempo", "ambient"], "Ghostly International", 116),
    ("Moderat", "Electronic---Breakbeat", ["electronica", "techno"], "Monkeytown", 122),
    ("Apparat", "Electronic---Ambient", ["ambient", "electronica"], "Mute", 115),
    ("Rival Consoles", "Electronic---Ambient", ["electronica", "ambient"], "Erased Tapes", 120),
    ("Christian Löffler", "Electronic---Deep House", ["deep house", "ambient"], "Ki Records", 119),
    ("Kiasmos", "Electronic---Minimal", ["minimal techno", "ambient"], "Erased Tapes", 121),
    ("Max Cooper", "Electronic---Ambient", ["ambient", "techno"], "Mesh", 122),
    ("Floating Points", "Electronic---Breakbeat", ["electronica", "ambient"], "Ninja Tune", 124),
    ("Caribou", "Electronic---Breakbeat", ["electronica", "deep house"], "City Slang", 123),
    ("Dixon", "Electronic---Deep House", ["deep house", "melodic techno"], "Innervisions", 122),
    ("Âme", "Electronic---Deep House", ["deep house", "melodic techno"], "Innervisions", 123),
    ("Recondite", "Electronic---Minimal", ["minimal techno", "deep techno"], "Innervisions", 123),
    (
        "Einmusik",
        "Electronic---Progressive House",
        ["progressive house", "melodic techno"],
        "Einmusika",
        123,
    ),
]

SYNTH_TITLE_WORDS_A = [
    "Nocturne",
    "Solstice",
    "Vesper",
    "Meridian",
    "Halcyon",
    "Eclipse",
    "Obsidian",
    "Aether",
    "Pulse",
    "Mirage",
    "Drift",
    "Vertex",
    "Monolith",
    "Prism",
    "Velvet",
    "Horizon",
    "Kinetic",
    "Subzero",
    "Afterglow",
    "Cascade",
]
SYNTH_TITLE_WORDS_B = [
    "I",
    "II",
    "III",
    "Transit",
    "Signal",
    "Sequence",
    "Motion",
    "Echoes",
    "Phase",
    "State",
    "Arc",
    "Form",
    "Light",
    "Shadow",
    "Wave",
    "Room",
    "Night",
    "Dawn",
    "Reprise",
    "Dub",
]

# 8 style cluster centers in 3D hyperspace (organic curved electronic manifold)
STYLE_CENTERS: dict[str, tuple[float, float, float]] = {
    "Electronic---Melodic Techno": (-28.0, 14.0, 10.0),
    "Electronic---Progressive House": (-48.0, -8.0, -14.0),
    "Electronic---Deep House": (-18.0, -34.0, -18.0),
    "Electronic---Techno": (10.0, 38.0, 24.0),
    "Electronic---Minimal": (28.0, 16.0, -16.0),
    "Electronic---Breakbeat": (18.0, -24.0, 26.0),
    "Electronic---Downtempo": (48.0, -22.0, -8.0),
    "Electronic---Ambient": (56.0, 12.0, 14.0),
}

STYLE_LIST = list(STYLE_CENTERS.keys())

YEAR_PLAY_TARGETS: dict[int, int] = {
    2017: 2150,
    2018: 2680,
    2019: 3220,
    2020: 4120,
    2021: 4580,
    2022: 5140,
    2023: 5490,
    2024: 5860,
    2025: 6180,
    2026: 7290,  # +18% YoY growth vs 2025
}

# Realistic diurnal listening weights (UTC hours 0..23: evening/late-night peak)
HOUR_WEIGHTS = [
    1.5,
    1.1,
    0.7,
    0.4,
    0.2,
    0.15,
    0.15,
    0.25,
    0.45,
    0.75,
    1.0,
    1.2,
    1.35,
    1.5,
    1.7,
    1.9,
    2.1,
    2.3,
    2.5,
    2.8,
    3.1,
    3.4,
    2.9,
    2.2,
]


def _stable_id(prefix: str, key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:18]
    return f"{prefix}{digest}"


def _svg_art_data_uri(label: str, style: str, is_artist: bool = False) -> str:
    """Generate a crisp dark-vinyl geometric SVG cover as a zero-network fallback."""
    palettes = {
        "Electronic---Melodic Techno": ("#0c1218", "#c6ff00", "#1f3b4d"),
        "Electronic---Progressive House": ("#10151e", "#4de1c1", "#243b55"),
        "Electronic---Deep House": ("#15101d", "#9f7aea", "#34224a"),
        "Electronic---Techno": ("#140d10", "#ff4d6d", "#3d1a26"),
        "Electronic---Minimal": ("#0f1419", "#60a5fa", "#1e2e42"),
        "Electronic---Breakbeat": ("#16120c", "#fb923c", "#422c16"),
        "Electronic---Downtempo": ("#0f1713", "#34d399", "#1d3b2d"),
        "Electronic---Ambient": ("#11131d", "#a78bfa", "#24294a"),
    }
    bg, acc, mid = palettes.get(style, ("#111317", "#c6ff00", "#262b33"))
    h = int(hashlib.sha1(label.encode("utf-8")).hexdigest()[:8], 16)
    cx = 64 + (h % 48) - 24
    cy = 64 + ((h >> 8) % 48) - 24
    r1 = 26 + ((h >> 16) % 28)
    r2 = r1 + 18
    initials = "".join(w[0] for w in label.split()[:2] if w).upper() or "AC"
    if is_artist:
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128">'
            f'<defs><radialGradient id="g" cx="35%" cy="30%" r="80%">'
            f'<stop offset="0%" stop-color="{mid}"/><stop offset="100%" stop-color="{bg}"/>'
            f"</radialGradient></defs>"
            f'<rect width="128" height="128" fill="url(#g)"/>'
            f'<circle cx="64" cy="64" r="46" fill="none" stroke="{acc}" stroke-opacity="0.35" stroke-width="1.5"/>'
            f'<circle cx="64" cy="64" r="34" fill="none" stroke="{acc}" stroke-opacity="0.18" stroke-width="1"/>'
            f'<text x="64" y="72" text-anchor="middle" font-family="ui-monospace,monospace" '
            f'font-size="28" font-weight="700" fill="{acc}" opacity="0.9">{initials}</text></svg>'
        )
    else:
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128">'
            f'<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
            f'<stop offset="0%" stop-color="{bg}"/><stop offset="100%" stop-color="{mid}"/>'
            f"</linearGradient></defs>"
            f'<rect width="128" height="128" fill="url(#g)"/>'
            f'<circle cx="{cx}" cy="{cy}" r="{r2}" fill="none" stroke="{acc}" stroke-opacity="0.22" stroke-width="1.5"/>'
            f'<circle cx="{cx}" cy="{cy}" r="{r1}" fill="none" stroke="{acc}" stroke-opacity="0.55" stroke-width="2"/>'
            f'<circle cx="{cx}" cy="{cy}" r="6" fill="{acc}" opacity="0.85"/>'
            f"</svg>"
        )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _fetch_deezer_json(url: str) -> dict | None:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "acetate-demo/0.2 (+https://github.com/charlesbine/acetate)"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _resolve_artwork(offline: bool) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    """Fetch public Deezer artist portraits and album covers for headline artists/tracks."""
    cache_path = Path("/tmp/acetate-demo-artwork-cache.json")
    artist_art: dict[str, str] = {}
    track_art: dict[tuple[str, str], str] = {}

    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text())
            artist_art.update(cached.get("artists", {}))
            for k, v in cached.get("tracks", {}).items():
                a, t = k.split("\t", 1)
                track_art[(a, t)] = v
        except Exception:
            pass

    if offline:
        return artist_art, track_art

    dirty = False
    for entry in HEADLINE_CATALOG:
        artist = entry["artist"]
        if artist not in artist_art:
            q = urllib.parse.quote(artist)
            data = _fetch_deezer_json(f"https://api.deezer.com/search/artist?q={q}&limit=3")
            for item in (data or {}).get("data") or []:
                pic = item.get("picture_medium") or item.get("picture_big")
                if pic and "default" not in pic:
                    artist_art[artist] = pic
                    dirty = True
                    break
            time.sleep(0.08)

        for title, _album, _bpm, _dur, _yr in entry["tracks"]:
            key = (artist, title)
            if key not in track_art:
                q = urllib.parse.quote(f'artist:"{artist}" track:"{title}"')
                data = _fetch_deezer_json(f"https://api.deezer.com/search/track?q={q}&limit=3")
                items = (data or {}).get("data") or []
                if not items:
                    q2 = urllib.parse.quote(f"{artist} {title}")
                    data = _fetch_deezer_json(f"https://api.deezer.com/search/track?q={q2}&limit=3")
                    items = (data or {}).get("data") or []
                for item in items:
                    cov = (item.get("album") or {}).get("cover_medium") or (
                        item.get("album") or {}
                    ).get("cover_big")
                    if cov:
                        track_art[key] = cov
                        dirty = True
                        break
                time.sleep(0.06)

    if dirty:
        try:
            cache_path.write_text(
                json.dumps(
                    {
                        "artists": artist_art,
                        "tracks": {f"{a}\t{t}": u for (a, t), u in track_art.items()},
                    },
                    indent=1,
                )
            )
        except Exception:
            pass

    return artist_art, track_art


def seed_demo_db(db_path: Path, *, offline: bool = False) -> dict[str, int]:
    """Create and populate a complete melodic-techno Acetate demo database."""
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(20261119)
    np_rng = np.random.default_rng(20261119)
    artist_art_map, track_art_map = _resolve_artwork(offline=offline)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    db.init_db(conn)

    now_dt = datetime(2026, 11, 19, 18, 42, 0, tzinfo=timezone.utc)
    now_iso = now_dt.isoformat()
    backup_time = "2026-11-19T04:00:00+00:00"

    all_tracks: list[dict] = []
    artists_seen: dict[str, str] = {}
    albums_seen: dict[str, str] = {}

    def register_track(
        artist: str,
        title: str,
        album: str,
        style: str,
        genres: list[str],
        label: str,
        bpm: int,
        duration_ms: int,
        release_year: int,
        weight: float,
        is_headline: bool,
    ) -> dict:
        aid = artists_seen.get(artist)
        if aid is None:
            aid = _stable_id("art_", artist)
            artists_seen[artist] = aid
            img = artist_art_map.get(artist) or _svg_art_data_uri(artist, style, is_artist=True)
            conn.execute(
                "INSERT OR IGNORE INTO artists(artist_id, name, genres_json, popularity, raw_json, updated_at) "
                "VALUES(?,?,?,?,?,?)",
                (
                    aid,
                    artist,
                    json.dumps(genres),
                    rng.randint(58, 86) if is_headline else rng.randint(38, 72),
                    json.dumps({"id": aid, "name": artist, "images": [{"url": img}]}),
                    now_iso,
                ),
            )
        al_key = f"{artist}::{album}"
        alid = albums_seen.get(al_key)
        if alid is None:
            alid = _stable_id("alb_", al_key)
            albums_seen[al_key] = alid
            conn.execute(
                "INSERT OR IGNORE INTO albums(album_id, name, release_date, updated_at) "
                "VALUES(?,?,?,?)",
                (alid, album, f"{release_year}-05-15", now_iso),
            )

        tid = _stable_id("trk_", f"{artist}::{title}")
        cover = track_art_map.get((artist, title)) or _svg_art_data_uri(
            f"{artist}-{title}", style, is_artist=False
        )
        isrc = f"DE{hashlib.sha1(tid.encode()).hexdigest()[:10].upper()}"
        raw_track = {
            "id": tid,
            "name": title,
            "duration_ms": duration_ms,
            "album": {
                "id": alid,
                "name": album,
                "release_date": f"{release_year}-05-15",
                "images": [{"url": cover}],
            },
            "artists": [{"id": aid, "name": artist}],
            "external_ids": {"isrc": isrc},
        }
        conn.execute(
            "INSERT OR IGNORE INTO tracks(track_id, name, album_id, duration_ms, popularity, isrc, uri, raw_json, updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                tid,
                title,
                alid,
                duration_ms,
                rng.randint(52, 84) if is_headline else rng.randint(28, 66),
                isrc,
                f"spotify:track:{tid}",
                json.dumps(raw_track),
                now_iso,
            ),
        )
        conn.execute(
            "INSERT OR IGNORE INTO track_artists(track_id, artist_id, position) VALUES(?,?,0)",
            (tid, aid),
        )
        conn.execute(
            "INSERT OR REPLACE INTO track_labels(track_id, track_name, artist_name, album_name, "
            "source, updated_at) VALUES(?,?,?,?,?,?)",
            (tid, title, artist, album, "spotify", now_iso),
        )
        for rank_idx, g in enumerate(genres):
            conn.execute(
                "INSERT OR IGNORE INTO track_genres(track_id, genre, source, confidence, assigned_at) "
                "VALUES(?,?,?,?,?)",
                (tid, g, "lastfm", round(max(0.35, 1.0 - rank_idx * 0.22), 2), now_iso),
            )
        rec = {
            "track_id": tid,
            "artist": artist,
            "title": title,
            "album": album,
            "style": style,
            "genres": genres,
            "label": label,
            "bpm": bpm,
            "duration_ms": duration_ms,
            "release_year": release_year,
            "weight": weight,
            "is_headline": is_headline,
        }
        all_tracks.append(rec)
        return rec

    # 1) Register headline catalog (130 signature tracks)
    for entry in HEADLINE_CATALOG:
        for idx, (title, album, bpm, dur, rel_yr) in enumerate(entry["tracks"]):
            # Top tracks per artist get stronger play weight
            t_weight = entry["weight"] * (1.85 if idx == 0 else 1.35 if idx == 1 else 0.95)
            register_track(
                artist=entry["artist"],
                title=title,
                album=album,
                style=entry["style"],
                genres=entry["genres"],
                label=entry["label"],
                bpm=bpm,
                duration_ms=dur,
                release_year=rel_yr,
                weight=t_weight,
                is_headline=True,
            )

    # 2) Register supporting catalog (~3,900 tracks across 30 supporting artists for rich Hyperspace)
    for s_idx, (artist, style, genres, label, base_bpm) in enumerate(SUPPORTING_ARTISTS):
        n_tracks = 130
        for k in range(n_tracks):
            w1 = SYNTH_TITLE_WORDS_A[(s_idx * 7 + k) % len(SYNTH_TITLE_WORDS_A)]
            w2 = SYNTH_TITLE_WORDS_B[(s_idx * 3 + k * 5) % len(SYNTH_TITLE_WORDS_B)]
            title = f"{w1} {w2}" if k < 20 else f"{w1} {w2} ({k - 19})"
            album = f"{w1} Sessions"
            rel_yr = rng.choices(
                list(range(2017, 2027)),
                weights=[6, 7, 8, 10, 11, 12, 12, 11, 11, 12],
            )[0]
            register_track(
                artist=artist,
                title=title,
                album=album,
                style=style,
                genres=genres,
                label=label,
                bpm=base_bpm + rng.randint(-3, 3),
                duration_ms=rng.randint(245000, 460000),
                release_year=rel_yr,
                weight=0.18 if k < 5 else 0.02,
                is_headline=False,
            )

    # 3) Generate 10 years of play events (2017-2026) with a crisp upward curve
    play_rows: list[tuple] = []
    hours_seq = list(range(24))
    for year, target_plays in YEAR_PLAY_TARGETS.items():
        eligible = [t for t in all_tracks if t["release_year"] <= year]
        weights = []
        for t in eligible:
            w = t["weight"]
            # Boost tracks released in that exact year so each year has fresh discoveries
            if t["release_year"] == year:
                w *= 2.4 if t["is_headline"] else 1.4
            elif year - t["release_year"] == 1:
                w *= 1.35
            weights.append(w)

        chosen = rng.choices(eligible, weights=weights, k=target_plays)
        max_day = 322 if year == 2026 else 364
        start_of_year = datetime(year, 1, 1, tzinfo=timezone.utc)
        for tr in chosen:
            day_offset = rng.randint(0, max_day)
            hour = rng.choices(hours_seq, weights=HOUR_WEIGHTS, k=1)[0]
            minute = rng.randint(0, 59)
            second = rng.randint(0, 59)
            ts = start_of_year + timedelta(
                days=day_offset, hours=hour, minutes=minute, seconds=second
            )
            if ts > now_dt - timedelta(hours=3):
                ts = now_dt - timedelta(hours=rng.randint(4, 96))
            play_rows.append(
                (
                    ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    tr["track_id"],
                    tr["duration_ms"],
                    "playlist",
                    None,
                    "gdpr" if year < 2026 else "recently_played",
                )
            )

    # Add a curated recent listening session in the last 3 hours for Home's "recently played"
    recent_session = [
        ("Stephan Bodzin", "Singularity", 12),
        ("Anyma", "Consciousness", 21),
        ("Tale Of Us", "Nova", 30),
        ("Innellea", "Vigilans", 39),
        ("Ben Böhmer", "Breathing", 48),
        ("Mind Against", "Walking Away", 58),
        ("Adriatique", "Miracle", 69),
        ("Massano", "The Feeling", 81),
        ("Bicep", "Glue", 94),
        ("Four Tet", "Loved", 108),
    ]
    by_artist_title = {(t["artist"], t["title"]): t for t in all_tracks}
    for artist, title, mins_ago in recent_session:
        tr = by_artist_title[(artist, title)]
        ts = now_dt - timedelta(minutes=mins_ago)
        play_rows.append(
            (
                ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                tr["track_id"],
                tr["duration_ms"],
                "playlist",
                None,
                "recently_played",
            )
        )

    conn.executemany(
        "INSERT OR IGNORE INTO play_events(played_at, track_id, ms_played, context_type, context_uri, source) "
        "VALUES(?,?,?,?,?,?)",
        play_rows,
    )

    # 4) Liked songs snapshot (~1,240 liked tracks, including all headline tracks)
    liked_tracks = [t for t in all_tracks if t["is_headline"]] + rng.sample(
        [t for t in all_tracks if not t["is_headline"]], k=1100
    )
    saved_rows = []
    for idx, tr in enumerate(liked_tracks):
        added = datetime(
            tr["release_year"], rng.randint(1, 10), rng.randint(1, 26), tzinfo=timezone.utc
        )
        saved_rows.append((backup_time, tr["track_id"], added.strftime("%Y-%m-%dT%H:%M:%SZ")))
    conn.executemany(
        "INSERT OR IGNORE INTO saved_track_snapshots(backup_time, track_id, added_at) VALUES(?,?,?)",
        saved_rows,
    )

    # 5) Playlists (authored + managed + smart + curated)
    def tracks_for_artists(names: set[str], limit: int = 36) -> list[str]:
        picks = [t["track_id"] for t in all_tracks if t["artist"] in names and t["is_headline"]]
        extra = [t["track_id"] for t in all_tracks if t["artist"] in names and not t["is_headline"]]
        return (picks + extra)[:limit]

    authored_defs = [
        (
            "pl_afterlife_late_night",
            "Afterlife · Late Night",
            "Hypnotic melodic techno for 2am drives and warehouse lasers.",
            {
                "Tale Of Us",
                "Anyma",
                "Innellea",
                "Mind Against",
                "Adriatique",
                "Massano",
                "Colyn",
                "Fideles",
                "Mathame",
            },
            "fave",
            48,
        ),
        (
            "pl_melodic_techno_core",
            "Melodic Techno Core",
            "The spine of the library — Bodzin, Afterlife, ARTBAT, and Innervisions.",
            {
                "Stephan Bodzin",
                "Tale Of Us",
                "Anyma",
                "ARTBAT",
                "Innellea",
                "Mind Against",
                "Kölsch",
                "Joris Voorn",
            },
            "fave",
            54,
        ),
        (
            "pl_peak_time_125",
            "Peak Time · 125 BPM",
            "High-tension synth arps and rolling basslines built for peak hours.",
            {
                "Anyma",
                "Massano",
                "Boris Brejcha",
                "ARTBAT",
                "Innellea",
                "Kevin de Vries",
                "Argy",
                "KAS:ST",
            },
            "fave",
            42,
        ),
        (
            "pl_organic_melodic_house",
            "Deep & Organic House",
            "Warm sunset progressions from Ben Böhmer, Monolink, RÜFÜS DU SOL, and Nora En Pure.",
            {
                "Ben Böhmer",
                "Monolink",
                "RÜFÜS DU SOL",
                "Nora En Pure",
                "Yotto",
                "Lane 8",
                "Corren Cavini",
            },
            "fave",
            44,
        ),
        (
            "pl_bodzin_sessions",
            "Stephan Bodzin & Friends",
            "Analog Moog leads, long builds, and relentless melodic drive.",
            {
                "Stephan Bodzin",
                "Boris Brejcha",
                "Solomun",
                "Maceo Plex",
                "Recondite",
                "Dixon",
                "Âme",
            },
            "fave",
            36,
        ),
        (
            "pl_sunrise_downtempo",
            "Sunrise Downtempo & IDM",
            "After-hours cooldown: Emancipator, Jon Hopkins, Four Tet, Bonobo, and Tycho.",
            {
                "Emancipator",
                "Jon Hopkins",
                "Four Tet",
                "Bonobo",
                "Tycho",
                "Apparat",
                "Rival Consoles",
            },
            "fave",
            38,
        ),
        (
            "pl_breaks_uk_club",
            "UK Breaks & Analog Club",
            "Bicep, Four Tet, Floating Points, and Moderat — broken beats with euphoria.",
            {"Bicep", "Four Tet", "Floating Points", "Moderat", "Caribou", "Max Cooper"},
            None,
            32,
        ),
        (
            "pl_berlin_afterhours",
            "Berlin Afterhours · 2024",
            "Dark minimal and hypnotic textures from Kiasmos, Recondite, and Mind Against.",
            {"Mind Against", "Recondite", "Kiasmos", "Christian Löffler", "KAS:ST", "Woo York"},
            None,
            34,
        ),
    ]

    for pid, name, desc, artist_set, label_val, limit in authored_defs:
        tids = tracks_for_artists(artist_set, limit=limit)
        conn.execute(
            "INSERT INTO playlist_snapshots(backup_time, playlist_id, name, description, owner_id, "
            "owner_name, public, collaborative, total_tracks) VALUES(?,?,?,?,?,?,?,?,?)",
            (backup_time, pid, name, desc, "demo", "Demo", 0, 0, len(tids)),
        )
        conn.execute(
            "INSERT OR IGNORE INTO playlist_seen(backup_time, playlist_id) VALUES(?,?)",
            (backup_time, pid),
        )
        for pos, tid in enumerate(tids):
            conn.execute(
                "INSERT INTO playlist_track_snapshots(backup_time, playlist_id, position, track_id, added_at) "
                "VALUES(?,?,?,?,?)",
                (backup_time, pid, pos, tid, "2026-08-14T22:00:00Z"),
            )
        if label_val:
            conn.execute(
                "INSERT OR REPLACE INTO playlist_labels(playlist_id, label, updated_at) VALUES(?,?,?)",
                (pid, label_val, now_iso),
            )

    # Managed playlists (monthly, gems, on_loop, discovery)
    managed_defs = [
        ("monthly_liked", "2026-11", "pl_liked_2026_11", "📆 Liked · 2026-11", 28),
        ("monthly_top", "2026-11", "pl_top_2026_11", "🔥 Top Played · 2026-11", 35),
        ("lost_gems", "faded", "pl_lost_gems", "💎 Lost Gems", 30),
        ("on_loop", "current", "pl_on_loop", "🔁 On Loop", 25),
        ("discovery", "weekly", "pl_discovery", "🔭 Discovery", 24),
    ]
    headline_ids = [t["track_id"] for t in all_tracks if t["is_headline"]]
    for kind, key, pid, name, count in managed_defs:
        sample_ids = rng.sample(headline_ids, k=min(count, len(headline_ids)))
        conn.execute(
            "INSERT INTO managed_playlists(kind, key, playlist_id, name, updated_at) VALUES(?,?,?,?,?)",
            (kind, key, pid, name, now_iso),
        )
        conn.execute(
            "INSERT INTO playlist_snapshots(backup_time, playlist_id, name, description, owner_id, "
            "owner_name, public, collaborative, total_tracks) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                backup_time,
                pid,
                name,
                f"Managed automatically by acetate ({kind}).",
                "demo",
                "Demo",
                0,
                0,
                len(sample_ids),
            ),
        )
        conn.execute(
            "INSERT OR IGNORE INTO playlist_seen(backup_time, playlist_id) VALUES(?,?)",
            (backup_time, pid),
        )
        for pos, tid in enumerate(sample_ids):
            conn.execute(
                "INSERT INTO playlist_track_snapshots(backup_time, playlist_id, position, track_id, added_at) "
                "VALUES(?,?,?,?,?)",
                (backup_time, pid, pos, tid, backup_time),
            )

    # Smart playlists (daily rules + one curated fixed set)
    smart_rule_1 = json.dumps(
        {"genre": "melodic techno", "min_plays": 10, "limit": 40, "sort": "plays"}
    )
    smart_rule_2 = json.dumps(
        {"liked": True, "homed": False, "genre": "progressive house", "limit": 30}
    )
    curated_track_ids = [
        by_artist_title[("Stephan Bodzin", "Singularity")]["track_id"],
        by_artist_title[("Ben Böhmer", "Breathing")]["track_id"],
        by_artist_title[("Monolink", "Return to Oz")]["track_id"],
        by_artist_title[("Mind Against", "Walking Away")]["track_id"],
        by_artist_title[("Colyn", "Resolve")]["track_id"],
        by_artist_title[("Tale Of Us", "Nova")]["track_id"],
        by_artist_title[("Innellea", "Vigilans")]["track_id"],
        by_artist_title[("Adriatique", "Miracle")]["track_id"],
        by_artist_title[("Anyma", "Consciousness")]["track_id"],
        by_artist_title[("Massano", "The Feeling")]["track_id"],
        by_artist_title[("Fideles", "Night After Night")]["track_id"],
        by_artist_title[("Mathame", "Nothing Around Us")]["track_id"],
    ]
    conn.execute(
        "INSERT INTO smart_playlists(name, spec, enabled, created_at, updated_at, published_at) "
        "VALUES(?,?,1,?,?,?)",
        ("Smart · Hypnotic Melodic Techno", smart_rule_1, now_iso, now_iso, now_iso),
    )
    conn.execute(
        "INSERT INTO smart_playlists(name, spec, enabled, created_at, updated_at, published_at) "
        "VALUES(?,?,1,?,?,?)",
        ("Smart · Unfiled Progressive Gems", smart_rule_2, now_iso, now_iso, now_iso),
    )
    conn.commit()

    # 6) Rebuild canonical recordings & playlist profiles using Acetate's own modules
    canon.canonicalize(conn)
    playlist_profile.rebuild(conn)

    # 7) Populate embeddings, recording_styles, deezer metadata, lastfm_info, and Hyperspace 3D points
    rec_rows = conn.execute(
        "SELECT rs.recording_key, rs.rep_track_id, rs.artist, rs.title, rs.plays FROM recording_stats rs"
    ).fetchall()
    track_by_id = {t["track_id"]: t for t in all_tracks}

    # Build orthonormal-ish 16-d basis vectors per style for embeddings (lightweight & coherent)
    dim = 16
    style_bases = {}
    for idx, st in enumerate(STYLE_LIST):
        v = np.zeros(dim, dtype=np.float32)
        v[(idx * 2) % dim] = 1.0
        v[(idx * 2 + 1) % dim] = 0.55
        style_bases[st] = v / np.linalg.norm(v)

    emb_rows = []
    style_rows = []
    hp_rows = []
    dz_album_rows = {}
    dz_track_rows = []
    lfm_rows = []

    for r in rec_rows:
        rk = r["recording_key"]
        tr = track_by_id.get(r["rep_track_id"])
        if not tr:
            continue
        st = tr["style"]
        cluster_idx = STYLE_LIST.index(st)
        base = style_bases[st]
        vec = base + 0.14 * np_rng.standard_normal(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        emb_rows.append((rk, "discogs-effnet", dim, vec.tobytes(), now_iso))
        emb_rows.append((rk, "clap-music", dim, vec.tobytes(), now_iso))

        # Top 3 Discogs styles
        secondary = STYLE_LIST[(cluster_idx + 1) % len(STYLE_LIST)]
        tertiary = STYLE_LIST[(cluster_idx + 2) % len(STYLE_LIST)]
        style_rows.append((rk, 1, st, round(rng.uniform(0.62, 0.91), 3)))
        style_rows.append((rk, 2, secondary, round(rng.uniform(0.24, 0.48), 3)))
        style_rows.append((rk, 3, tertiary, round(rng.uniform(0.11, 0.23), 3)))

        # 3D UMAP-like curved electronic manifold: dense glowing core + organic bridge filaments
        cx, cy, cz = STYLE_CENTERS[st]
        nx, ny, nz = STYLE_CENTERS[secondary]
        if rng.random() < 0.24:
            # Filament bridge linking neighboring styles for an organic UMAP manifold look
            t_bridge = float(np_rng.beta(1.6, 2.2))
            bx = cx + (nx - cx) * t_bridge
            by = cy + (ny - cy) * t_bridge
            bz = cz + (nz - cz) * t_bridge
            x = bx + np_rng.normal(0.0, 7.0)
            y = by + np_rng.normal(0.0, 6.5)
            z = bz + np_rng.normal(0.0, 6.0)
        else:
            angle = np_rng.uniform(0, 2 * math.pi)
            radius = abs(np_rng.normal(0.0, 12.5))
            twist = math.sin((cx + radius) * 0.045) * 8.0
            x = cx + radius * math.cos(angle) * 1.12
            y = cy + radius * math.sin(angle) * 0.92 + twist
            z = cz + np_rng.normal(0.0, 10.5)
        hp_rows.append(
            (
                "effnet",
                rk,
                round(float(x), 3),
                round(float(y), 3),
                round(float(z), 3),
                cluster_idx,
                now_iso,
            )
        )
        hp_rows.append(
            (
                "clap",
                rk,
                round(float(x * 0.9 + y * 0.2), 3),
                round(float(y * 0.85 - z * 0.25), 3),
                round(float(z * 0.9 + x * 0.15), 3),
                cluster_idx,
                now_iso,
            )
        )

        dz_aid = abs(hash(f"{tr['artist']}:{tr['album']}")) % 900_000_000 + 1000
        dz_tid = abs(hash(tr["track_id"])) % 900_000_000 + 1000
        dz_artist_id = abs(hash(tr["artist"])) % 900_000_000 + 1000
        dz_album_rows[dz_aid] = (
            dz_aid,
            tr["album"],
            tr["label"],
            "[]",
            f"{tr['release_year']}-05-15",
            now_iso,
        )
        dz_track_rows.append(
            (
                rk,
                dz_tid,
                float(tr["bpm"]),
                -7.2,
                420000,
                f"{tr['release_year']}-05-15",
                dz_artist_id,
                dz_aid,
                now_iso,
            )
        )
        lfm_rows.append(
            (
                rk,
                rng.randint(25000, 820000) if tr["is_headline"] else rng.randint(4000, 95000),
                rng.randint(150000, 4200000),
                now_iso,
            )
        )

    conn.executemany(
        "INSERT OR REPLACE INTO embeddings(recording_key, model, dim, vector, updated_at) VALUES(?,?,?,?,?)",
        emb_rows,
    )
    conn.executemany(
        "INSERT OR REPLACE INTO recording_styles(recording_key, rank, style, score) VALUES(?,?,?,?)",
        style_rows,
    )
    conn.executemany(
        "INSERT OR REPLACE INTO hyperspace_points(space, recording_key, x, y, z, cluster, updated_at) VALUES(?,?,?,?,?,?,?)",
        hp_rows,
    )
    conn.executemany(
        "INSERT OR REPLACE INTO deezer_album(album_id, title, label, genres, release_date, updated_at) VALUES(?,?,?,?,?,?)",
        list(dz_album_rows.values()),
    )
    conn.executemany(
        "INSERT OR REPLACE INTO deezer_track(recording_key, deezer_id, bpm, gain, rank, release_date, artist_id, album_id, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
        dz_track_rows,
    )
    conn.executemany(
        "INSERT OR REPLACE INTO lastfm_info(recording_key, listeners, playcount, updated_at) VALUES(?,?,?,?)",
        lfm_rows,
    )
    conn.commit()

    # Build taste regions from the seeded authored playlists + embeddings
    taste.build(conn)

    # 8) Recent successful job_run entries so Home's status bar looks healthy & live
    recent_jobs = [
        (
            "poll",
            (now_dt - timedelta(minutes=6)).isoformat(),
            (now_dt - timedelta(minutes=6) + timedelta(seconds=3)).isoformat(),
            "ok",
            "+3 new plays",
        ),
        (
            "backup",
            "2026-11-19T04:00:00+00:00",
            "2026-11-19T04:00:22+00:00",
            "ok",
            "13 playlists, 1,230 liked (stored)",
        ),
        (
            "canon",
            "2026-11-19T04:30:00+00:00",
            "2026-11-19T04:30:04+00:00",
            "ok",
            "1,810 canonical recordings",
        ),
        (
            "monthly",
            "2026-11-19T05:00:00+00:00",
            "2026-11-19T05:00:06+00:00",
            "ok",
            "updated 2026-11 playlists",
        ),
    ]
    conn.executemany(
        "INSERT INTO job_run(job, started_at, finished_at, status, result) VALUES(?,?,?,?,?)",
        recent_jobs,
    )

    # 9) Library memory + Chat conversation with tool traces & interactive playlist proposal
    memory_note = (
        "- Keep late-night Afterlife / melodic techno sets separate from daytime progressive house.\n"
        "- Prefer 121-126 BPM builds for curated sets; avoid repeating tracks played 5+ times this week.\n"
        "- Next: audit Sunrise Downtempo for tempo outliers."
    )
    conn.execute(
        "INSERT OR REPLACE INTO kv_state(key, value, updated_at) VALUES(?,?,?)",
        (chatstore.LIBRARY_MEMORY_KEY, memory_note, now_iso),
    )

    # Older thread for the threads drawer
    t_old = conn.execute(
        "INSERT INTO chat_threads(title, created_at, updated_at, archived) VALUES(?,?,?,1)",
        (
            "Melodic techno evolution (2017-2026)",
            "2026-11-12T20:10:00+00:00",
            "2026-11-12T20:14:00+00:00",
        ),
    ).lastrowid
    conn.execute(
        "INSERT INTO chat_messages(thread_id, role, content, created_at) VALUES(?,?,?,?)",
        (
            t_old,
            "user",
            "How has my melodic techno listening evolved since 2017?",
            "2026-11-12T20:10:00+00:00",
        ),
    )

    # Active showcase thread
    t_cur = conn.execute(
        "INSERT INTO chat_threads(title, created_at, updated_at, archived) VALUES(?,?,?,0)",
        (
            "Late-night Afterlife & melodic techno set",
            (now_dt - timedelta(minutes=14)).isoformat(),
            (now_dt - timedelta(minutes=12)).isoformat(),
        ),
    ).lastrowid

    user_q = (
        "Build me a 12-track late-night Afterlife & melodic techno set anchored around "
        "Stephan Bodzin's Singularity — pull from my library, skip anything overplayed "
        "this week, and sequence it from hypnotic buildup to peak."
    )
    conn.execute(
        "INSERT INTO chat_messages(thread_id, role, content, created_at) VALUES(?,?,?,?)",
        (t_cur, "user", user_q, (now_dt - timedelta(minutes=14)).isoformat()),
    )

    assistant_md = (
        "Pulled the nearest sonic neighbors to **Stephan Bodzin — Singularity** in `discogs-effnet` "
        "space, cross-checked your Afterlife & Melodic Techno Core taste regions, and filtered out "
        "anything you overplayed this week. The 12-track arc builds cleanly from **121 → 126 BPM** "
        "(*Singularity* & *Breathing* warmup → *Resolve* & *Vigilans* tension → *Consciousness* & "
        "*The Feeling* peak)."
    )
    proposal_preview = [
        {"title": track_by_id[tid]["title"], "artist": track_by_id[tid]["artist"]}
        for tid in curated_track_ids[:4]
    ]
    assistant_meta = {
        "model": "anthropic/claude-sonnet-4.5",
        "usage": {
            "cost": 0.0042,
            "input_tokens": 2840,
            "output_tokens": 612,
            "cached_tokens": 1920,
        },
        "steps": [
            {
                "tool": "similar_tracks",
                "query": "Stephan Bodzin - Singularity",
                "ok": True,
                "duration_ms": 38,
                "row_count": 25,
            },
            {
                "tool": "query_db",
                "sql": (
                    "SELECT rs.artist, rs.title, dt.bpm, rs.plays\n"
                    "FROM recording_stats rs\n"
                    "JOIN deezer_track dt USING(recording_key)\n"
                    "JOIN recording_styles st USING(recording_key)\n"
                    "WHERE st.rank = 1 AND st.style LIKE '%Techno%'\n"
                    "  AND dt.bpm BETWEEN 121 AND 126\n"
                    "ORDER BY dt.bpm ASC, rs.plays DESC LIMIT 20"
                ),
                "ok": True,
                "duration_ms": 11,
                "row_count": 20,
            },
            {
                "tool": "propose_playlist",
                "ok": True,
                "duration_ms": 6,
            },
        ],
        "proposals": [
            {
                "name": "After Hours · Melodic Techno Set",
                "summary": "12-track late-night arc (121–126 BPM) anchored on Stephan Bodzin — Singularity",
                "matches": len(curated_track_ids),
                "preview": proposal_preview,
                "spec": {"track_ids": curated_track_ids},
            }
        ],
    }
    conn.execute(
        "INSERT INTO chat_messages(thread_id, role, content, meta, created_at) VALUES(?,?,?,?,?)",
        (
            t_cur,
            "assistant",
            assistant_md,
            json.dumps(assistant_meta),
            (now_dt - timedelta(minutes=12)).isoformat(),
        ),
    )

    conn.commit()
    summary = {
        "plays": conn.execute("SELECT COUNT(*) FROM play_events").fetchone()[0],
        "recordings": conn.execute("SELECT COUNT(*) FROM recording_stats").fetchone()[0],
        "artists": conn.execute("SELECT COUNT(*) FROM artists").fetchone()[0],
        "liked": conn.execute("SELECT COUNT(*) FROM saved_track_snapshots").fetchone()[0],
        "hyperspace_points": conn.execute(
            "SELECT COUNT(*) FROM hyperspace_points WHERE space='effnet'"
        ).fetchone()[0],
        "playlists": conn.execute(
            "SELECT COUNT(DISTINCT playlist_id) FROM playlist_snapshots"
        ).fetchone()[0],
    }
    conn.close()
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed a melodic-techno demo Acetate SQLite DB.")
    ap.add_argument("--db", type=Path, required=True, help="Output SQLite DB path.")
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Skip public Deezer artwork lookups and use generative SVG covers only.",
    )
    args = ap.parse_args()
    summary = seed_demo_db(args.db, offline=args.offline)
    print("Seeded demo DB:", args.db, json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
