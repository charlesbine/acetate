"""Persistence + threading for the chat agent.

A thread holds an ordered list of messages. The newest non-archived thread stays
"current" for THREAD_TTL_HOURS; after that it auto-archives and a fresh blank thread
opens. Archiving is reversible — opening any thread un-archives it.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from . import db, smart

THREAD_TTL_HOURS = 24
LIBRARY_MEMORY_KEY = "library_organization_memory"
LIBRARY_MEMORY_LIMIT = 6000


def library_memory(conn: sqlite3.Connection) -> dict:
    """One shared Markdown note, independent of conversation summaries."""
    row = conn.execute(
        "SELECT value, updated_at FROM kv_state WHERE key=?", (LIBRARY_MEMORY_KEY,)
    ).fetchone()
    return (
        {"text": row["value"] or "", "revision": row["updated_at"] or ""}
        if row
        else {"text": "", "revision": ""}
    )


def save_library_memory(conn: sqlite3.Connection, text: str, revision: str) -> dict:
    if not isinstance(text, str) or len(text) > LIBRARY_MEMORY_LIMIT:
        raise ValueError(
            f"Memory must be Markdown text of at most {LIBRARY_MEMORY_LIMIT} characters"
        )
    if not isinstance(revision, str):
        raise ValueError("Memory revision must be text")
    now = db.utcnow()
    with conn:
        if revision:
            saved = conn.execute(
                "UPDATE kv_state SET value=?, updated_at=? WHERE key=? AND updated_at=?",
                (text, now, LIBRARY_MEMORY_KEY, revision),
            )
        else:
            saved = conn.execute(
                "INSERT INTO kv_state(key,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO NOTHING",
                (LIBRARY_MEMORY_KEY, text, now),
            )
        if saved.rowcount != 1:
            raise LookupError(
                "Memory changed elsewhere. Reload the note before saving your changes."
            )
    return {"text": text, "revision": now}


def _age_hours(iso: str | None) -> float:
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    except Exception:
        return 1e9


def get_thread(conn: sqlite3.Connection, tid: int) -> dict | None:
    r = conn.execute("SELECT * FROM chat_threads WHERE id=?", (tid,)).fetchone()
    return dict(r) if r else None


def create_thread(conn: sqlite3.Connection) -> dict:
    now = db.utcnow()
    cur = conn.execute(
        "INSERT INTO chat_threads(title, created_at, updated_at, archived) VALUES(NULL,?,?,0)",
        (now, now),
    )
    conn.commit()
    return get_thread(conn, cur.lastrowid)


def get_active_thread(conn: sqlite3.Connection) -> dict:
    """Continue the newest non-archived thread if it's younger than the TTL; otherwise
    archive it and open a fresh blank one."""
    r = conn.execute(
        "SELECT * FROM chat_threads WHERE archived=0 ORDER BY updated_at DESC LIMIT 1"
    ).fetchone()
    if r:
        if _age_hours(r["updated_at"]) <= THREAD_TTL_HOURS:
            return dict(r)
        conn.execute("UPDATE chat_threads SET archived=1 WHERE id=?", (r["id"],))
        conn.commit()
    return create_thread(conn)


def open_thread(conn: sqlite3.Connection, tid: int) -> dict | None:
    """Open a specific thread, un-archiving it so it becomes current (reversible archive)."""
    if not get_thread(conn, tid):
        return None
    conn.execute("UPDATE chat_threads SET archived=0, updated_at=? WHERE id=?", (db.utcnow(), tid))
    conn.commit()
    return get_thread(conn, tid)


def list_threads(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM chat_threads ORDER BY archived ASC, updated_at DESC LIMIT ?", (limit,)
        )
    ]


def add_message(
    conn: sqlite3.Connection, tid: int, role: str, content: str, meta: dict | None = None
) -> int:
    cur = conn.execute(
        "INSERT INTO chat_messages(thread_id, role, content, meta, created_at) VALUES(?,?,?,?,?)",
        (tid, role, content, json.dumps(meta) if meta else None, db.utcnow()),
    )
    if role == "user":
        t = get_thread(conn, tid)
        if t and not t.get("title") and content:
            conn.execute("UPDATE chat_threads SET title=? WHERE id=?", (content[:48], tid))
    conn.execute("UPDATE chat_threads SET updated_at=? WHERE id=?", (db.utcnow(), tid))
    conn.commit()
    return cur.lastrowid


def messages(conn: sqlite3.Connection, tid: int) -> list[dict]:
    out = []
    for r in conn.execute(
        "SELECT id, role, content, meta FROM chat_messages WHERE thread_id=? ORDER BY id", (tid,)
    ):
        m = {"id": r["id"], "role": r["role"], "content": r["content"]}
        if r["meta"]:
            try:
                m["meta"] = json.loads(r["meta"])
            except Exception:
                pass
        out.append(m)
    return out


def history_for_agent(
    conn: sqlite3.Connection, tid: int, *, through_id: int | None = None
) -> list[dict]:
    """All text after the summary checkpoint, with IDs for advancing that checkpoint.

    The agent strips IDs at the upstream boundary and budgets by tokens rather
    than silently discarding history after a fixed number of exchanges.
    """
    if through_id is None:
        thread = get_thread(conn, tid)
        through_id = thread["summary_through_id"] if thread else 0
    return [
        dict(r)
        for r in conn.execute(
            "SELECT id, role, content FROM chat_messages WHERE thread_id=? AND id > ? "
            "AND content IS NOT NULL AND content != '' ORDER BY id",
            (tid, through_id),
        )
    ]


def save_summary(
    conn: sqlite3.Connection, tid: int, summary: str, through_id: int, expected_through_id: int
) -> bool:
    """Save a checkpoint without overwriting one advanced by a concurrent request."""
    if not summary.strip() or through_id < expected_through_id:
        return False
    cur = conn.execute(
        "UPDATE chat_threads SET summary=?, summary_through_id=? "
        "WHERE id=? AND summary_through_id=? AND EXISTS "
        "(SELECT 1 FROM chat_messages WHERE thread_id=? AND id=?)",
        (summary, through_id, tid, expected_through_id, tid, through_id),
    )
    conn.commit()
    return bool(cur.rowcount)


def saved_playlist_id(conn: sqlite3.Connection, proposal: dict) -> int | None:
    """Resolve an approval, including rules saved before cards tracked their state."""
    sid = proposal.get("saved_playlist_id")
    if sid is not None:
        row = smart.get(conn, sid)
        # SQLite can reuse a deleted rule's integer ID for an unrelated rule.
        return (
            sid if row and row["created_at"] == proposal.get("saved_playlist_created_at") else None
        )
    for row in conn.execute(
        "SELECT id, spec FROM smart_playlists WHERE name=? ORDER BY id",
        (proposal.get("name", "").strip(),),
    ):
        if json.loads(row["spec"]) == proposal.get("spec"):
            return row["id"]
    return None


def playlist_drafts(conn: sqlite3.Connection) -> list[dict]:
    """Unapproved playlist proposals across conversations, newest distinct selection first."""
    rows = conn.execute(
        "SELECT m.id, m.thread_id, m.created_at, t.title thread_title, "
        "json_extract(m.meta, '$.proposals') proposals FROM chat_messages m "
        "JOIN chat_threads t ON t.id=m.thread_id "
        "WHERE m.role='assistant' AND json_type(CASE WHEN json_valid(m.meta) "
        "THEN m.meta ELSE '{}' END, '$.proposals')='array' ORDER BY m.id DESC"
    )
    out, seen = [], set()
    for row in rows:
        for index, proposal in enumerate(json.loads(row["proposals"])):
            if not isinstance(proposal, dict) or not isinstance(proposal.get("spec"), dict):
                continue
            name = proposal.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            # Removing an accepted playlist must not resurrect its original draft.
            if proposal.get("saved_playlist_id") is not None or saved_playlist_id(conn, proposal):
                continue
            key = (name.strip(), json.dumps(proposal["spec"], sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    **proposal,
                    "matches": proposal.get("matches"),
                    "message_id": row["id"],
                    "index": index,
                    "thread_id": row["thread_id"],
                    "thread_title": row["thread_title"],
                    "created_at": row["created_at"],
                }
            )
    return out


def approve_playlist_proposal(conn: sqlite3.Connection, message_id: int, index: int) -> int:
    """Save the rule and its chat reference atomically; repeat approvals reuse it."""
    with conn:
        # Serialize concurrent clicks across requests and web workers.
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT meta FROM chat_messages WHERE id=? AND role='assistant'", (message_id,)
        ).fetchone()
        meta = json.loads(row["meta"]) if row and row["meta"] else {}
        proposals = meta.get("proposals", [])
        if not 0 <= index < len(proposals):
            raise LookupError("Playlist proposal not found")
        proposal = proposals[index]
        name, spec = proposal.get("name", "").strip(), proposal.get("spec")
        if not name or not isinstance(spec, dict):
            raise ValueError("This proposal needs a playlist name and a rule object")
        sid = saved_playlist_id(conn, proposal)
        if sid is None:
            try:
                smart.evaluate(conn, spec)
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"Invalid playlist specification: {exc}") from exc
            sid = smart.create(conn, name, spec, commit=False)
        proposal["saved_playlist_id"] = sid
        proposal["saved_playlist_created_at"] = smart.get(conn, sid)["created_at"]
        conn.execute("UPDATE chat_messages SET meta=? WHERE id=?", (json.dumps(meta), message_id))
    return sid
