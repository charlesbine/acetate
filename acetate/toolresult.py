"""Fitting tool results into the model's context without corrupting them.

Every tool result is JSON the model has to parse. Slicing the serialized string
(``json.dumps(payload)[:6000]``) respects the byte budget but hands back a
truncated, unparseable object — and says nothing about what was dropped, so the
model quietly treats a partial sample as the whole answer.

So truncate in ROW space, never in byte space: drop whole rows until the payload
fits, then say so in-band and tell the model how to get the rest. The result is
always valid JSON, and "there is more" is a fact the model can act on rather than
a silent corruption it has to guess at.
"""

from __future__ import annotations

import json
from typing import Any

# A single pathological cell (a giant text blob) can blow the budget on its own,
# so cap cells before deciding how many rows fit.
MAX_CELL = 400

RETRY_HINT = (
    "Re-run with the work pushed into SQL — an aggregate "
    "(count/median/quantile_cont), a tighter WHERE, fewer columns, or an "
    "explicit LIMIT. Do not extrapolate a total from these rows."
)


def _clip_cell(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_CELL:
        return value[:MAX_CELL] + f"…[+{len(value) - MAX_CELL} chars]"
    return value


def _clip_rows(rows: list) -> list:
    return [
        [_clip_cell(cell) for cell in row] if isinstance(row, list) else _clip_cell(row)
        for row in rows
    ]


def pack(payload: Any, budget: int) -> str:
    """Serialize `payload` to at most `budget` characters of *valid* JSON.

    A tabular payload ({"rows": [...], ...}) loses whole rows from the end and
    gains `truncated`/`note`. Anything else falls back to a labelled envelope —
    still parseable, still explicit about being partial.
    """
    text = json.dumps(payload)
    if len(text) <= budget:
        return text

    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = _clip_rows(payload["rows"])
        # Largest prefix of rows that still fits, found by bisection so a wide
        # result doesn't cost a serialization per row.
        lo, hi = 0, len(rows)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(json.dumps(_tabular(payload, rows[:mid], len(rows)))) <= budget:
                lo = mid
            else:
                hi = mid - 1
        return json.dumps(_tabular(payload, rows[:lo], len(rows)))

    return json.dumps(
        {
            "truncated": True,
            "note": f"Result too large for one tool call ({len(text)} chars). {RETRY_HINT}",
            "partial": text[: max(0, budget - len(RETRY_HINT) - 200)],
        }
    )


def _tabular(payload: dict, rows: list, available: int) -> dict:
    kept = len(rows)
    out = {**payload, "rows": rows, "row_count": kept, "truncated": True}
    out["note"] = f"Showing {kept} of {available}+ rows — the rest did not fit. {RETRY_HINT}"
    return out
