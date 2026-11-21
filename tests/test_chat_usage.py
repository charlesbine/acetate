from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from acetate import agent


class _FakeConnection:
    def close(self):
        pass


class _FakeCompletions:
    def __init__(self, streams):
        self.streams = list(streams)

    def create(self, **kwargs):
        return iter(self.streams.pop(0))


def _chunk(*, content=None, tool_calls=None, usage=None):
    choices = (
        []
        if content is None and tool_calls is None
        else [
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ]
    )
    return SimpleNamespace(choices=choices, usage=usage)


def test_usage_aggregates_every_upstream_round():
    totals = agent._usage_totals()
    agent._merge_usage(
        totals,
        {
            "prompt_tokens": 1000,
            "completion_tokens": 40,
            "prompt_tokens_details": {"cached_tokens": 700},
            "cost": 0.0012,
        },
    )
    agent._merge_usage(
        totals,
        {
            "prompt_tokens": 1200,
            "completion_tokens": 60,
            "prompt_tokens_details": {"cached_tokens": 900},
            "cost": 0.0015,
        },
    )

    usage = agent._final_usage(totals)

    assert usage == {
        "input_tokens": 2200,
        "output_tokens": 100,
        "cached_tokens": 1600,
        "cost": pytest.approx(0.0027),
    }


def test_usage_omits_cost_when_any_round_is_unpriced():
    totals = agent._usage_totals()
    agent._merge_usage(totals, {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.001})
    agent._merge_usage(totals, {"prompt_tokens": 20, "completion_tokens": 4})

    assert agent._final_usage(totals) == {
        "input_tokens": 30,
        "output_tokens": 6,
        "cached_tokens": 0,
    }


def test_stream_aggregates_usage_from_terminal_chunks_across_tool_rounds(monkeypatch):
    tool_call = SimpleNamespace(
        index=0,
        id="call-1",
        function=SimpleNamespace(name="query_db", arguments='{"sql":"SELECT 1"}'),
    )
    streams = [
        [
            _chunk(tool_calls=[tool_call]),
            _chunk(
                usage={
                    "prompt_tokens": 1000,
                    "completion_tokens": 20,
                    "prompt_tokens_details": {"cached_tokens": 700},
                    "cost": 0.001,
                }
            ),
        ],
        [
            _chunk(content="Done."),
            _chunk(
                usage={
                    "prompt_tokens": 1200,
                    "completion_tokens": 30,
                    "prompt_tokens_details": {"cached_tokens": 900},
                    "cost": 0.0015,
                }
            ),
        ],
    ]
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeCompletions(streams)),
    )
    monkeypatch.setattr(agent, "_client", lambda: fake_client)
    monkeypatch.setattr(agent.query, "readonly_connect", _FakeConnection)
    monkeypatch.setattr(agent, "system_prompt", lambda conn, can_admin=False: "system")
    monkeypatch.setattr(agent, "exec_tool", lambda conn, name, args, seen, **kw: ("[]", None))

    events = list(
        agent.chat_stream(
            [{"role": "user", "content": "test"}],
            model="deepseek/deepseek-v4-flash",
        )
    )

    done = next(event for event in events if event["t"] == "done")
    assert done["v"] == {
        "model": "deepseek/deepseek-v4-flash",
        "usage": {
            "input_tokens": 2200,
            "output_tokens": 50,
            "cached_tokens": 1600,
            "cost": pytest.approx(0.0025),
        },
    }


def test_chat_persists_and_renders_effective_usage(conn, monkeypatch):
    path = conn.execute("PRAGMA database_list").fetchone()["file"]
    monkeypatch.setenv("ACETATE_DB", path)
    monkeypatch.setenv("ACETATE_AUTH_MODE", "local")
    from acetate import auth

    monkeypatch.setattr(auth, "load_credentials", lambda: None)
    from acetate import web

    usage = {
        "input_tokens": 1234,
        "output_tokens": 56,
        "cached_tokens": 1000,
        "cost": 0.0015,
    }

    def fake_stream(messages, model=None, can_admin=False, requested_by="administrator", **kwargs):
        yield {"t": "token", "v": "Done."}
        yield {"t": "done", "v": {"model": model, "usage": usage}}

    monkeypatch.setattr(web.agent, "chat_stream", fake_stream)
    client = TestClient(web.app)

    response = client.post(
        "/chat/stream",
        data={
            "q": "Summarize this playlist",
            "model": "minimax/minimax-m2.7:nitro",
        },
    )

    assert response.status_code == 200
    row = conn.execute(
        "SELECT meta FROM chat_messages WHERE role='assistant' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert json.loads(row["meta"])["usage"] == usage

    page = client.get("/chat")
    assert page.status_code == 200
    assert (
        "via minimax/minimax-m2.7:nitro · $0.0015 effective · 1,234 in · 56 out · 1,000 cached"
    ) in page.text
