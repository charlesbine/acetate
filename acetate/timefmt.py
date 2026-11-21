"""One display-timezone contract, shared by the website and the chat agent.

Every timestamp in the database is UTC (see `db.utcnow`). People think in local
wall-clock time, so both surfaces convert at their own edge: the Jinja
`localtime` filter renders HTML, and `localize` rewrites tool results before the
LLM ever sees them. Converting one step above the model means it never has to
juggle offsets — and the two surfaces cannot disagree about when "today" ended.

Storage stays UTC. This module is display-only, in both directions it is used.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_TZ = "America/New_York"

# A complete ISO-8601 instant, and nothing else: date-only keys ("2026-07"),
# durations, and free text must pass through untouched. Anchored on both ends so
# a timestamp embedded in a sentence is left alone rather than half-rewritten.
_ISO_INSTANT = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$"
)


def tz_name() -> str:
    return (os.environ.get("ACETATE_TZ") or "").strip() or DEFAULT_TZ


def display_tz() -> ZoneInfo:
    """The configured display timezone, or America/New_York if it is unusable.

    A typo in ACETATE_TZ should not take down a page or a chat turn.
    """
    try:
        return ZoneInfo(tz_name())
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo(DEFAULT_TZ)


def parse_utc(value: str | None) -> datetime | None:
    """Parse a stored timestamp as UTC. Naive values are assumed UTC, as stored."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def to_local(value: str | None, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Render a stored UTC timestamp in the display timezone, or '' if unparseable."""
    parsed = parse_utc(value)
    if parsed is None:
        return value if isinstance(value, str) else ""
    return parsed.astimezone(display_tz()).strftime(fmt)


def local_iso(value: str | None) -> str | None:
    """A stored timestamp as a local ISO-8601 instant, offset included.

    The offset is what makes this safe to hand to a model: the value is still an
    unambiguous instant, it just reads in the user's own clock.
    """
    parsed = parse_utc(value)
    if parsed is None:
        return None
    return parsed.astimezone(display_tz()).isoformat(timespec="seconds")


def now_local() -> datetime:
    return datetime.now(timezone.utc).astimezone(display_tz())


def localize(value: Any) -> Any:
    """Recursively rewrite complete ISO-8601 instants into local time.

    Anything that is not a full timestamp — ids, titles, month keys, cron
    strings — is returned untouched.
    """
    if isinstance(value, str):
        if _ISO_INSTANT.match(value):
            return local_iso(value) or value
        return value
    if isinstance(value, dict):
        return {key: localize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [localize(item) for item in value]
    return value


def localize_json_text(text: str) -> str:
    """Localize a JSON tool result in place, leaving non-JSON payloads alone.

    Tool handlers return JSON text (or an "ERROR: ..." string); both go through
    here on the way to the model, and only the former is rewritten.
    """
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
    return json.dumps(localize(parsed), default=str)
