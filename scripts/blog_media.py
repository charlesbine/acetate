# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["numpy", "librosa", "matplotlib", "soundfile", "numba>=0.60", "umap-learn>=0.5"]
# ///
"""Generate the /how blog's model-demo media from REAL library data.

Repro:
    uv run scripts/blog_media.py --track "corren cavini - listen to the silence"

Writes to assets/blog/:
  spectrogram-pair.png   linear-frequency STFT vs log-mel of the track's cached
                         preview, rendered at 16 kHz mono — exactly what the
                         embedding model receives
  effnet_demo.json       the track's real discogs400 top activations (from
                         recording_styles) for the in-page bar chart
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUT = REPO / "assets" / "blog"

BG = "#0a0a0c"
INK = "#888d99"
# after-hours lime ramp: silence fades to the page background, energy glows
CMAP = LinearSegmentedColormap.from_list(
    "acetate", ["#0a0a0c", "#232a10", "#4d5a0c", "#7e9712", "#b2d916", "#e2f77e"]
)

matplotlib.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": INK,
        "axes.edgecolor": "#23252e",
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "font.family": "monospace",
        "font.size": 8,
    }
)


def find_track(conn: sqlite3.Connection, query: str) -> sqlite3.Row:
    artist, _, title = (p.strip() for p in query.partition(" - "))
    row = conn.execute(
        """
        SELECT rs.recording_key, rs.artist, rs.title, rs.plays, a.path
        FROM recording_stats rs JOIN audio_cache a USING(recording_key)
        WHERE a.status='fetched' AND rs.artist LIKE ? AND rs.title LIKE ?
        ORDER BY rs.plays DESC LIMIT 1
    """,
        (f"%{artist}%", f"%{title or artist}%"),
    ).fetchone()
    if not row:
        sys.exit(f"no fetched track matching {query!r}")
    return row


def render_zoom_pair(path: Path, out: Path, start_s: float = 8.0, dur_s: float = 4.0) -> None:
    """A few-second zoom, linear vs mel side by side — kick columns at full legibility."""
    y, sr = librosa.load(path, sr=16000, mono=True, offset=start_s, duration=dur_s)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.4), dpi=160)
    stft_db = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=64)), ref=np.max)
    librosa.display.specshow(
        stft_db,
        sr=sr,
        hop_length=64,
        x_axis="time",
        y_axis="hz",
        ax=axes[0],
        cmap=CMAP,
        vmin=-70,
        vmax=0,
    )
    axes[0].set_title(f"linear · {dur_s:.0f}s zoom @ {start_s:.0f}s", color="#f3f3ef", pad=8)
    mel_db = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=64, n_mels=128), ref=np.max
    )
    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=64,
        x_axis="time",
        y_axis="mel",
        ax=axes[1],
        cmap=CMAP,
        vmin=-70,
        vmax=0,
    )
    axes[1].set_title(f"log-mel · same {dur_s:.0f}s", color="#f3f3ef", pad=8)
    for ax in axes:
        ax.set_xlabel("time (s)")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(pad=1.2)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def render_full_track(path: Path, out: Path, marks: list[tuple[float, str]]) -> None:
    """Whole-track log-mel — the structural arc no 30s snippet (and no ~3s model
    patch) can see. `marks` = [(seconds, label), ...] annotation lines."""
    y, sr = librosa.load(path, sr=16000, mono=True)
    fig, ax = plt.subplots(figsize=(13.5, 3.2), dpi=160)
    mel_db = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128),
        ref=np.max,
    )
    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        cmap=CMAP,
        vmin=-70,
        vmax=0,
    )
    for t, label in marks:
        ax.axvline(t, color="#ff5d8f", lw=0.9, ls=(0, (4, 3)), alpha=0.85)
        ax.text(t + 2, 7600, label, color="#ff5d8f", fontsize=7.5, va="top")
    ax.set_title("log-mel, full track", color="#f3f3ef", pad=8)
    ax.set_xlabel("time (s)")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=1.2)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


PAL = [
    "#b2d916",
    "#ff70a0",
    "#38d6f5",
    "#ffb224",
    "#b39dff",
    "#2fe6a8",
    "#ff8a4d",
    "#8fa9ff",
    "#5b5f6b",
]  # site categorical palette; slot 9 = Other


def render_clap_map(conn: sqlite3.Connection, textvec_path: Path, out: Path) -> None:
    """The CLAP joint space, seen: UMAP of all audio vectors (colored by style
    group) with TEXT prompts projected into the same 2D map as landmarks."""
    rows = conn.execute(
        "SELECT e.recording_key, e.vector FROM embeddings e "
        "JOIN recording_stats rs USING(recording_key) "
        "WHERE e.model='clap-music'"
    ).fetchall()
    keys = [r["recording_key"] for r in rows]
    mat = np.vstack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9

    # style-group index: top-8 sub-styles by frequency + Other (matches the API)
    style = {
        k: (v or "").split("---")[-1]
        for k, v in conn.execute("SELECT recording_key, style FROM recording_styles WHERE rank=1")
    }
    from collections import Counter

    top = [s for s, _ in Counter(style.get(k) for k in keys if style.get(k)).most_common(8)]
    gidx = {s: i for i, s in enumerate(top)}
    groups = np.array([gidx.get(style.get(k), 8) for k in keys])

    tv = json.loads(textvec_path.read_text())
    tmat = np.asarray(tv["vectors"], dtype=np.float32)
    tmat /= np.linalg.norm(tmat, axis=1, keepdims=True) + 1e-9

    import umap

    reducer = umap.UMAP(
        n_components=2, metric="cosine", n_neighbors=30, min_dist=0.15, random_state=42
    )
    xy = reducer.fit_transform(mat)
    txy = reducer.transform(tmat)

    fig, ax = plt.subplots(figsize=(11.5, 7), dpi=160)
    for g in range(9):
        m = groups == g
        ax.scatter(
            xy[m, 0],
            xy[m, 1],
            s=2.2,
            c=PAL[g],
            alpha=0.45,
            lw=0,
            label=(top[g] if g < 8 else "Other"),
        )
    for (x, y), label in zip(txy, tv["prompts"]):
        ax.scatter(
            [x], [y], marker="X", s=130, c="#ff5d8f", edgecolors="#0a0a0c", linewidths=1.2, zorder=5
        )
        ax.annotate(
            f"“{label}”",
            (x, y),
            xytext=(8, 7),
            textcoords="offset points",
            color="#ff5d8f",
            fontsize=8.5,
            zorder=6,
            path_effects=None,
            fontweight="bold",
        )
    ax.legend(loc="upper right", frameon=False, fontsize=7, markerscale=3, labelcolor="#f3f3ef")
    ax.set_title(
        "the CLAP joint space — every song, plus sentences, in one map", color="#f3f3ef", pad=10
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=1.0)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def render_spectrogram_pair(path: Path, out: Path) -> None:
    # 16 kHz mono = the embedding model's actual input; every plot claim on the
    # blog stays honest to what the network sees.
    y, sr = librosa.load(path, sr=16000, mono=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.4), dpi=160)

    stft_db = librosa.amplitude_to_db(
        np.abs(librosa.stft(y, n_fft=1024, hop_length=256)), ref=np.max
    )
    librosa.display.specshow(
        stft_db,
        sr=sr,
        hop_length=256,
        x_axis="time",
        y_axis="hz",
        ax=axes[0],
        cmap=CMAP,
        vmin=-70,
        vmax=0,
    )
    axes[0].set_title("linear-frequency spectrogram (STFT)", color="#f3f3ef", pad=8)

    mel_db = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128),
        ref=np.max,
    )
    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=256,
        x_axis="time",
        y_axis="mel",
        ax=axes[1],
        cmap=CMAP,
        vmin=-70,
        vmax=0,
    )
    axes[1].set_title("log-mel spectrogram (what the model sees)", color="#f3f3ef", pad=8)

    for ax in axes:
        ax.set_xlabel("time (s)")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(pad=1.2)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="corren cavini - listen to the silence")
    ap.add_argument("--zoom-start", type=float, default=8.0)
    ap.add_argument(
        "--full",
        type=Path,
        default=None,
        help="Path to a full-length audio file for the structure spectrogram.",
    )
    ap.add_argument(
        "--marks",
        default="",
        help="Full-track annotations: 'sec:label,sec:label' (e.g. '0:intro,64:drop').",
    )
    ap.add_argument(
        "--clap-map",
        type=Path,
        default=None,
        metavar="TEXTVEC_JSON",
        help="Render the CLAP joint-space map using text vectors from this file "
        "(produce it with: clap_worker.py textvec --prompt ... --out f.json). "
        "Renders ONLY the map, then exits.",
    )
    args = ap.parse_args()

    conn = sqlite3.connect(DATA / "acetate.db")
    conn.row_factory = sqlite3.Row
    OUT.mkdir(parents=True, exist_ok=True)
    if args.clap_map:
        render_clap_map(conn, args.clap_map, OUT / "clap-space.png")
        return 0
    rec = find_track(conn, args.track)

    render_spectrogram_pair(DATA / rec["path"], OUT / "spectrogram-pair.png")
    render_zoom_pair(DATA / rec["path"], OUT / "spectrogram-zoom.png", start_s=args.zoom_start)
    if args.full:
        marks = [
            (float(m.split(":")[0]), m.split(":")[1]) for m in args.marks.split(",") if ":" in m
        ]
        render_full_track(args.full, OUT / "spectrogram-full.png", marks)

    styles = [
        {"style": r["style"].split("---")[-1], "score": round(r["score"], 3)}
        for r in conn.execute(
            "SELECT style, score FROM recording_styles WHERE recording_key=? ORDER BY rank LIMIT 8",
            (rec["recording_key"],),
        )
    ]
    demo = {"artist": rec["artist"], "title": rec["title"], "plays": rec["plays"], "styles": styles}
    (OUT / "effnet_demo.json").write_text(json.dumps(demo, indent=1))
    print(
        f"wrote effnet_demo.json ({rec['artist']} — {rec['title']}, "
        f"top style {styles[0]['style']} {styles[0]['score']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
