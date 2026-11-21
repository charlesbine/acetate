"""The transient thinking stream: reasoning deltas reach the browser, and
nothing about them leaks into the conversation record.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from acetate import agent


class _Delta(SimpleNamespace):
    """A stream delta with only the fields a given provider actually sends."""

    def __init__(self, **fields) -> None:
        super().__init__(**{"content": None, "tool_calls": None, **fields})


def _chunk(delta) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None)


class _FakeCompletions:
    def __init__(self, streams) -> None:
        self._streams = list(streams)
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return iter(self._streams.pop(0))


class _FakeConnection:
    def execute(self, *args, **kwargs):
        return SimpleNamespace(fetchone=lambda: None, fetchall=lambda: [])

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
#  Reading reasoning off a chunk
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "delta, expected",
    [
        (_Delta(reasoning="the user asked "), "the user asked "),  # OpenRouter
        (_Delta(reasoning_content="via a gateway"), "via a gateway"),  # LiteLLM
        (
            _Delta(reasoning_details=[{"type": "reasoning.text", "text": "structured"}]),
            "structured",
        ),  # OpenRouter, structured
        (
            _Delta(reasoning_details=[{"type": "reasoning.summary", "summary": "a summary"}]),
            "a summary",
        ),
        (_Delta(content="just an answer"), ""),  # no reasoning at all
        (_Delta(reasoning=None, reasoning_content=""), ""),  # present but empty
        (_Delta(reasoning_details="not a list"), ""),  # malformed, not a crash
    ],
)
def test_reasoning_delta_reads_every_provider_shape(delta, expected) -> None:
    assert agent._reasoning_delta(delta) == expected


def test_reasoning_delta_joins_multi_part_details() -> None:
    delta = _Delta(
        reasoning_details=[
            {"type": "reasoning.text", "text": "first "},
            {"type": "reasoning.text", "text": "second"},
            {"type": "reasoning.encrypted", "data": "opaque"},  # nothing readable — skipped
        ]
    )

    assert agent._reasoning_delta(delta) == "first second"


# --------------------------------------------------------------------------- #
#  The stream
# --------------------------------------------------------------------------- #


def _run_stream(monkeypatch, streams) -> tuple[list[dict], _FakeCompletions]:
    completions = _FakeCompletions(streams)
    monkeypatch.setattr(
        agent, "_client", lambda: SimpleNamespace(chat=SimpleNamespace(completions=completions))
    )
    monkeypatch.setattr(agent.query, "readonly_connect", _FakeConnection)
    monkeypatch.setattr(agent, "system_prompt", lambda conn, can_admin=False: "system")
    events = list(agent.chat_stream([{"role": "user", "content": "hi"}], model="z-ai/glm-5.2"))
    return events, completions


def test_reasoning_streams_as_think_events_before_the_answer(monkeypatch) -> None:
    events, _ = _run_stream(
        monkeypatch,
        [
            [
                _chunk(_Delta(reasoning="let me check ")),
                _chunk(_Delta(reasoning="the play counts")),
                _chunk(_Delta(content="You played 12 tracks.")),
            ]
        ],
    )

    kinds = [event["t"] for event in events]
    assert kinds == ["think", "think", "token", "done"]
    assert "".join(e["v"] for e in events if e["t"] == "think") == "let me check the play counts"
    assert "".join(e["v"] for e in events if e["t"] == "token") == "You played 12 tracks."


def test_thinking_is_never_sent_back_upstream_or_returned_as_answer(monkeypatch) -> None:
    """Reasoning is for watching. It must not enter the conversation the model sees."""
    events, completions = _run_stream(
        monkeypatch,
        [
            [
                _chunk(
                    _Delta(
                        reasoning="I should call a tool",
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_1",
                                function=SimpleNamespace(
                                    name="query_db", arguments='{"sql": "SELECT 1"}'
                                ),
                            )
                        ],
                    )
                )
            ],
            [_chunk(_Delta(reasoning="now to phrase it")), _chunk(_Delta(content="Done."))],
        ],
    )

    assert [e["t"] for e in events][:2] == ["think", "step"]
    second_request = completions.requests[1]["messages"]
    assistant = next(m for m in second_request if m.get("role") == "assistant")
    assert "I should call a tool" not in str(second_request)  # not replayed to the model
    assert assistant.get("content") is None  # and not smuggled in as content


def test_a_model_that_does_not_think_streams_exactly_as_before(monkeypatch) -> None:
    events, _ = _run_stream(monkeypatch, [[_chunk(_Delta(content="Straight answer."))]])

    assert [event["t"] for event in events] == ["token", "done"]


def test_usage_is_requested_on_every_stream(monkeypatch) -> None:
    """Without stream_options the spec sends no usage, and the cost/token line
    silently disappears — which is exactly how it was broken before."""
    _, completions = _run_stream(monkeypatch, [[_chunk(_Delta(content="hi"))]])

    assert completions.requests[0]["stream_options"] == {"include_usage": True}


def test_no_reasoning_parameter_is_sent_upstream(monkeypatch) -> None:
    """Reasoning comes back by default; forcing `enabled` would switch it ON for
    hybrid models and change what the user pays."""
    _, completions = _run_stream(monkeypatch, [[_chunk(_Delta(content="hi"))]])

    request = completions.requests[0]
    assert "reasoning" not in request
    assert "include_reasoning" not in request
    assert "reasoning" not in str(request.get("extra_body") or {})


# --------------------------------------------------------------------------- #
#  The page
# --------------------------------------------------------------------------- #


def test_chat_page_ships_the_thinking_toggle_and_panel(conn, monkeypatch) -> None:
    path = conn.execute("PRAGMA database_list").fetchone()["file"]
    monkeypatch.setenv("ACETATE_DB", path)
    monkeypatch.setenv("ACETATE_AUTH_MODE", "local")
    from acetate import auth, web

    monkeypatch.setattr(auth, "load_credentials", lambda: None)

    page = TestClient(web.app).get("/chat").text

    assert "id=thinkopt" in page  # the show/hide control
    assert "acetate.think" in page  # remembered across reloads
    assert "ev.t==='think'" in page  # the stream branch that fills the panel
    assert "thought for " in page  # collapses to a duration when done
