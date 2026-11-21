from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import requests
from fastapi.testclient import TestClient

from acetate import agent, chat_context, chatstore, db
from acetate.chat_context import Budget, ChatContext, estimate_tokens


def _response(
    content="User prefers instrumental techno; playlist P1 remains a proposal.",
    finish_reason="stop",
    usage=None,
):
    message = SimpleNamespace(content=content, tool_calls=None)
    message.model_dump = lambda **kwargs: {"role": "assistant", "content": content}
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)], usage=usage
    )


class Completions:
    def __init__(self, responses=None, responder=None):
        self.responses = iter(responses) if responses is not None else None
        self.responder = responder
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.responder:
            return self.responder(kwargs)
        if self.responses is None:
            return _response()
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def _client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def _history(turns=8, size=1000):
    return [
        {
            "id": i + 1,
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"message {i}: " + "music " * (size // 6),
        }
        for i in range(turns * 2)
    ]


QUESTION = {"role": "user", "content": "What did we decide about playlist P1?"}
REQUEST = {"model": "gateway/example", "extra_body": {"provider": {"only": ["example"]}}}
SMALL = Budget(12_000, 1500)


def test_catalog_lookup_is_cached_and_unauthenticated(monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"data": [{"id": "test/model", "context_length": 80_000}]},
        )

    monkeypatch.setattr(chat_context, "_catalog_expires", 0)
    monkeypatch.setattr(chat_context.requests, "get", get)
    assert chat_context.model_budget("test/model").window == 80_000
    assert chat_context.model_budget("test/model").window == 80_000
    assert calls == [(chat_context.MODELS_URL, {"timeout": 5})]


def test_catalog_failure_and_unknown_model_use_conservative_fallback(monkeypatch):
    calls = []

    def get(*args, **kwargs):
        calls.append(args)
        raise requests.Timeout("offline")

    monkeypatch.setattr(chat_context, "_catalog_expires", 0)
    monkeypatch.setattr(chat_context.requests, "get", get)
    assert chat_context.model_budget("unknown/model").window == 32_768
    assert chat_context.model_budget("unknown/model").window == 32_768
    assert len(calls) == 1


def test_model_limits_nitro_provider_and_gateway_cap(monkeypatch):
    monkeypatch.setattr(
        chat_context,
        "_catalog",
        {
            "test/model": {
                "context_length": 128_000,
                "top_provider": {"context_length": 64_000, "max_completion_tokens": 2000},
            },
        },
    )
    assert chat_context.model_budget("test/model:nitro") == Budget(64_000, 2000)
    monkeypatch.setenv("ACETATE_CHAT_CONTEXT_TOKENS", "16000")
    assert chat_context.model_budget("test/model:nitro") == Budget(16_000, 2000)
    monkeypatch.setenv("ACETATE_CHAT_CONTEXT_TOKENS", "256000")
    assert chat_context.model_budget("test/model").window == 64_000
    assert chat_context.model_budget("custom/gateway").window == 256_000


@pytest.mark.parametrize("value", ["bad", "0", "4000"])
def test_invalid_context_override_is_explicit(monkeypatch, value):
    monkeypatch.setenv("ACETATE_CHAT_CONTEXT_TOKENS", value)
    with pytest.raises(ValueError, match="ACETATE_CHAT_CONTEXT_TOKENS"):
        chat_context.model_budget("test/model")


def test_short_turns_are_not_limited_to_twelve_exchanges():
    history = _history(turns=30, size=12)
    completions = Completions([])
    context = ChatContext("system", history + [QUESTION], [], SMALL)
    prepared = context.prepare(_client(completions), REQUEST, lambda usage: None)
    assert len(prepared) == len(history) + 2
    assert not completions.requests
    assert all("id" not in message for message in prepared)
    assert prepared[1]["content"] == history[0]["content"]


@pytest.mark.parametrize("repetitions", [55_000, 100_000])
def test_dense_history_and_summary_chunks_fit_provider_window(repetitions):
    # GPT-OSS's o200k_harmony tokenizer uses three tokens per repeated banjo emoji.
    # Count these independently of the budget estimator; the rest is bounded by bytes.
    budget = Budget(131_072)

    def provider_tokens(messages):
        return sum(
            message["content"].count("🪕") * 3
            + len(message["content"].replace("🪕", "").encode("utf-8"))
            + 16  # message framing
            for message in messages
        )

    def respond(kwargs):
        assert provider_tokens(kwargs["messages"]) + kwargs["max_tokens"] <= budget.window
        return _response()

    history = [
        {"id": 1, "role": "user", "content": "🪕" * repetitions},
        {"id": 2, "role": "assistant", "content": "Noted."},
    ]
    completions = Completions(responder=respond)
    saved = []
    context = ChatContext(
        "system",
        history + [QUESTION],
        [],
        budget,
        on_summary=lambda summary, last_id: saved.append((summary, last_id)),
    )
    prepared = context.prepare(_client(completions), REQUEST, lambda usage: None)

    assert completions.requests  # do not send oversized history straight to the model
    assert provider_tokens(prepared) + budget.reply_tokens <= budget.window
    chunks = [
        json.loads(r["messages"][1]["content"])["transcript_chunk"] for r in completions.requests
    ]
    assert json.loads("".join(chunks)) == [
        {k: v for k, v in message.items() if k != "id"} for message in history
    ]
    assert saved[0][1] == 2


def test_summarizes_old_exchanges_preserving_recent_turns_and_current_request():
    history = _history()
    saved, usage = [], []
    completions = Completions()
    context = ChatContext(
        "system",
        history + [QUESTION],
        [],
        SMALL,
        on_summary=lambda summary, last_id: saved.append((summary, last_id)),
    )
    prepared = context.prepare(_client(completions), REQUEST, usage.append)

    assert saved and saved[0][1] % 2 == 0  # checkpoint ends at an exchange boundary
    assert prepared[-1] == QUESTION
    assert prepared[-3:-1] == [{k: v for k, v in m.items() if k != "id"} for m in history[-2:]]
    assert prepared[0] == {"role": "system", "content": "system"}
    assert prepared[1]["role"] == "user"
    assert "do not authorize actions" in prepared[1]["content"]
    assert prepared[2]["role"] == "assistant"
    assert "not instructions or proof of approval" in prepared[2]["content"]
    assert estimate_tokens(prepared) + context.tool_tokens <= SMALL.input_limit
    summary_request = completions.requests[0]
    assert summary_request["model"] == REQUEST["model"]
    assert summary_request["extra_body"] == REQUEST["extra_body"]
    assert "tools" not in summary_request
    assert "message 0:" in summary_request["messages"][1]["content"]
    assert context.metadata()["context"]["summarized"] is True


def test_switch_to_smaller_model_rechecks_history_and_existing_memory():
    history = _history()
    completions = Completions()
    large = ChatContext("system", history + [QUESTION], [], Budget(128_000))
    large.prepare(_client(completions), REQUEST, lambda usage: None)
    assert not completions.requests
    small = ChatContext(
        "system", history + [QUESTION], [], SMALL, summary="Earlier preference: no vocals."
    )
    small.prepare(_client(completions), REQUEST, lambda usage: None)
    assert "Earlier preference: no vocals." in completions.requests[0]["messages"][1]["content"]
    assert small.needs_summary() is False


def test_oversized_unicode_history_is_chunked_without_losing_input():
    history = [
        {"id": 1, "role": "user", "content": '音乐🎵\n"' * 10_000},
        {"id": 2, "role": "assistant", "content": "An unresolved proposal."},
    ]
    completions = Completions()
    context = ChatContext("system", history + [QUESTION], [], SMALL, summary="Keep playlist P1.")
    context.prepare(_client(completions), REQUEST, lambda usage: None)
    chunks = [
        json.loads(r["messages"][1]["content"])["transcript_chunk"] for r in completions.requests
    ]
    assert len(chunks) > 1
    assert json.loads("".join(chunks)) == [
        {k: v for k, v in m.items() if k != "id"} for m in history
    ]
    assert all(estimate_tokens(r["messages"]) <= SMALL.input_limit for r in completions.requests)
    assert all(r["max_tokens"] == SMALL.reply_tokens for r in completions.requests)
    assert all(
        json.loads(r["messages"][1]["content"])["previous_summary"] for r in completions.requests
    )


@pytest.mark.parametrize(
    "response",
    [_response(""), _response("partial", "length"), RuntimeError("provider unavailable")],
)
def test_summary_failure_preserves_checkpoint_and_history(response):
    history, saved = _history(), []
    context = ChatContext(
        "system",
        history + [QUESTION],
        [],
        SMALL,
        summary="old memory",
        on_summary=lambda *args: saved.append(args),
    )
    with pytest.raises((ValueError, RuntimeError)):
        context.prepare(_client(Completions([response])), REQUEST, lambda usage: None)
    assert context.summary == "old memory"
    assert context.history == history
    assert not saved


def test_oversized_current_request_fails_before_any_provider_call():
    completions = Completions([])
    context = ChatContext(
        "system", _history() + [{"role": "user", "content": "x" * 40_000}], [], SMALL
    )
    with pytest.raises(ValueError, match="Shorten the request"):
        context.prepare(_client(completions), REQUEST, lambda usage: None)
    assert not completions.requests


def test_large_existing_summary_is_recompressed_after_switch_without_new_history():
    saved = []
    original_summary = "old preference " * 1600
    context = ChatContext(
        "system",
        [QUESTION],
        [],
        SMALL,
        summary=original_summary,
        on_summary=lambda summary, last_id: saved.append((summary, last_id)),
    )
    completions = Completions()
    prepared = context.prepare(_client(completions), REQUEST, lambda usage: None)
    chunks = [
        json.loads(r["messages"][1]["content"])["transcript_chunk"] for r in completions.requests
    ]
    assert len(chunks) > 1
    assert json.loads("".join(chunks)) == [
        {"role": "assistant", "content": chat_context.SUMMARY_LABEL + original_summary}
    ]
    assert saved and saved[0][1] is None  # keep the existing durable message boundary
    assert prepared[-1] == QUESTION
    assert context.needs_summary() is False


def test_missing_summary_usage_does_not_report_partial_cost_as_total():
    totals = agent._usage_totals()
    context = ChatContext("system", _history() + [QUESTION], [], SMALL)
    context.prepare(
        _client(Completions()), REQUEST, lambda usage: agent._merge_usage(totals, usage)
    )
    agent._merge_usage(totals, {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.001})
    assert "cost" not in agent._final_usage(totals)


def test_tool_schema_and_current_tool_rounds_count_toward_budget_without_breaking_pairs():
    history = _history(turns=4, size=400)
    tools = [{"description": "tool description " * 150}]
    context = ChatContext("system", history + [QUESTION], tools, SMALL)
    client = _client(Completions())
    context.prepare(client, REQUEST, lambda usage: None)
    assert not context.summary
    calls = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "query_db", "arguments": "{}"},
            }
        ],
    }
    result = {"role": "tool", "tool_call_id": "call-1", "content": "result " * 600}
    context.current.extend([calls, result])
    prepared = context.prepare(client, REQUEST, lambda usage: None)
    assert context.summary
    assert prepared[-3:] == [QUESTION, calls, result]
    assert estimate_tokens(prepared) + estimate_tokens(tools) <= SMALL.input_limit


def test_checkpoint_is_thread_scoped_monotonic_and_keeps_original_messages(conn):
    tid = chatstore.create_thread(conn)["id"]
    other = chatstore.create_thread(conn)["id"]
    for message in _history(turns=3, size=12):
        chatstore.add_message(conn, tid, message["role"], message["content"])
    original = chatstore.messages(conn, tid)
    history = chatstore.history_for_agent(conn, tid)
    boundary = history[3]["id"]
    assert not chatstore.save_summary(conn, other, "wrong thread", boundary, 0)
    assert chatstore.save_summary(conn, tid, "memory", boundary, 0)
    assert chatstore.history_for_agent(conn, tid) == history[4:]
    assert chatstore.history_for_agent(conn, tid, through_id=0) == history
    assert not chatstore.save_summary(conn, tid, "stale", history[-1]["id"], 0)
    assert not chatstore.save_summary(conn, tid, "backwards", history[1]["id"], boundary)
    assert chatstore.messages(conn, tid) == original
    assert chatstore.get_thread(conn, tid)["summary"] == "memory"


def test_existing_chat_schema_migrates_idempotently(tmp_path):
    conn = db.connect(tmp_path / "old.db")
    conn.execute(
        "CREATE TABLE chat_threads (id INTEGER PRIMARY KEY, title TEXT, created_at TEXT, "
        "updated_at TEXT, archived INTEGER NOT NULL DEFAULT 0)"
    )
    conn.execute("INSERT INTO chat_threads(id,title) VALUES(1,'old conversation')")
    db.init_db(conn)
    db.init_db(conn)
    assert chatstore.get_thread(conn, 1)["summary_through_id"] == 0
    assert chatstore.get_thread(conn, 1)["title"] == "old conversation"


def _stream(content="Done.", usage=None, finish_reason="stop", tool_calls=None):
    return iter(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=content, tool_calls=tool_calls),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None, tool_calls=None),
                        finish_reason=finish_reason,
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(choices=[], usage=usage),
        ]
    )


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "output", ["reasoning_only", "partial_answer", "partial_tool_call", "complete_tool_call"]
)
def test_output_exhaustion_is_an_error_before_executing_tools(monkeypatch, streaming, output):
    content = "A partial answer" if output == "partial_answer" else None
    tool_calls = (
        [
            SimpleNamespace(
                index=0,
                id="call-1",
                function=SimpleNamespace(
                    name="query_db",
                    arguments='{"sql":"SELECT 1"}'
                    if output == "complete_tool_call"
                    else '{"sql":"SELECT',
                ),
            )
        ]
        if output.endswith("tool_call")
        else None
    )
    if streaming:
        response = list(_stream(content, finish_reason="length", tool_calls=tool_calls))
        if output == "reasoning_only":
            response[0].choices[0].delta.reasoning = "Still thinking."
    else:
        response = _response(content, "length")
        response.choices[0].message.tool_calls = tool_calls
    completions = Completions([response])
    closed, executed = [], []
    monkeypatch.setattr(agent, "_client", lambda: _client(completions))
    monkeypatch.setattr(
        agent.query, "readonly_connect", lambda: SimpleNamespace(close=lambda: closed.append(True))
    )
    monkeypatch.setattr(agent, "system_prompt", lambda *args, **kwargs: "system")
    monkeypatch.setattr(agent, "tools_for", lambda can_admin: [])
    monkeypatch.setattr(agent, "exec_tool", lambda *args, **kwargs: executed.append(args))
    if streaming:
        events = list(agent.chat_stream([QUESTION]))
        assert events[-1]["t"] == "error"
        assert "output limit" in events[-1]["v"]
        assert not any(event["t"] == "done" for event in events)
    else:
        with pytest.raises(ValueError, match="output limit"):
            agent.chat([QUESTION])
    assert not executed
    assert closed == [True]
    assert len(completions.requests) == 1  # no automatic paid retry


def test_sync_and_stream_summary_usage_includes_all_calls(monkeypatch):
    for streaming in (False, True):
        usage = {"prompt_tokens": 200, "completion_tokens": 30, "cost": 0.001}

        def respond(kwargs):
            if kwargs.get("stream"):
                return _stream(usage=usage)
            return _response(usage=usage)

        completions = Completions(responder=respond)
        monkeypatch.setattr(agent, "_client", lambda: _client(completions))
        monkeypatch.setattr(
            agent.query, "readonly_connect", lambda: SimpleNamespace(close=lambda: None)
        )
        monkeypatch.setattr(agent, "system_prompt", lambda *args, **kwargs: "system")
        monkeypatch.setattr(agent.chat_context, "model_budget", lambda model: SMALL)
        monkeypatch.setattr(agent, "tools_for", lambda can_admin: [])
        if streaming:
            events = list(agent.chat_stream(_history() + [QUESTION]))
            assert events[0] == {"t": "status", "v": "Summarizing earlier messages…"}
            result = events[-1]["v"]
        else:
            result = agent.chat(_history() + [QUESTION])
        calls = len(completions.requests)
        assert calls >= 3  # chunked summaries plus the final answer
        assert result["usage"] == {
            "input_tokens": 200 * calls,
            "output_tokens": 30 * calls,
            "cached_tokens": 0,
            "cost": pytest.approx(0.001 * calls),
        }
        assert result["context"]["summarized"]
        assert completions.requests[-1]["max_tokens"] == SMALL.reply_tokens


def test_web_saves_reuses_summary_and_renders_it_after_reload(conn, monkeypatch):
    monkeypatch.setenv("ACETATE_DB", conn.execute("PRAGMA database_list").fetchone()["file"])
    monkeypatch.setenv("ACETATE_AUTH_MODE", "local")
    from acetate import auth

    monkeypatch.setattr(auth, "load_credentials", lambda: None)
    from acetate import web

    memory = chatstore.save_library_memory(conn, "- Keep playlist P1 instrumental.", "")
    tid = chatstore.create_thread(conn)["id"]
    for message in _history():
        chatstore.add_message(conn, tid, message["role"], message["content"])
    completions = Completions(
        responder=lambda kwargs: _stream() if kwargs.get("stream") else _response()
    )
    monkeypatch.setattr(agent, "_client", lambda: _client(completions))
    monkeypatch.setattr(agent, "system_prompt", lambda *args, **kwargs: "system")
    monkeypatch.setattr(agent.chat_context, "model_budget", lambda model: SMALL)
    monkeypatch.setattr(agent, "tools_for", lambda can_admin: [])
    client = TestClient(web.app)
    response = client.post("/chat/stream", data={"q": QUESTION["content"], "thread": tid})
    assert '"t": "status"' in response.text
    assert '"t": "error"' not in response.text
    checkpoint = chatstore.get_thread(conn, tid)
    assert checkpoint["summary_through_id"] > 0
    assert len(chatstore.messages(conn, tid)) == 18
    assert "earlier messages summarized" in client.get(f"/chat?thread={tid}").text
    assert chatstore.library_memory(conn) == memory
    first_call_count = len(completions.requests)
    response = client.post("/chat/stream", data={"q": "Continue", "thread": tid})
    assert '"t": "status"' not in response.text
    assert len(completions.requests) == first_call_count + 1
    assert checkpoint["summary"] in completions.requests[-1]["messages"][2]["content"]
    assert "message 0:" not in str(completions.requests[-1]["messages"])


def test_failed_summary_does_not_save_a_partial_exchange(conn, monkeypatch):
    monkeypatch.setenv("ACETATE_DB", conn.execute("PRAGMA database_list").fetchone()["file"])
    monkeypatch.setenv("ACETATE_AUTH_MODE", "local")
    from acetate import auth

    monkeypatch.setattr(auth, "load_credentials", lambda: None)
    from acetate import web

    tid = chatstore.create_thread(conn)["id"]
    for message in _history():
        chatstore.add_message(conn, tid, message["role"], message["content"])
    original = chatstore.messages(conn, tid)
    monkeypatch.setattr(agent, "_client", lambda: _client(Completions([_response("", "length")])))
    monkeypatch.setattr(agent, "system_prompt", lambda *args, **kwargs: "system")
    monkeypatch.setattr(agent.chat_context, "model_budget", lambda model: SMALL)
    monkeypatch.setattr(agent, "tools_for", lambda can_admin: [])
    response = TestClient(web.app).post("/chat/stream", data={"q": "continue", "thread": tid})
    assert '"t": "error"' in response.text
    assert chatstore.messages(conn, tid) == original
    assert chatstore.get_thread(conn, tid)["summary_through_id"] == 0


@pytest.mark.parametrize("content", [None, "An unfinished answer"])
def test_exhausted_answer_does_not_persist_a_partial_exchange(conn, monkeypatch, content):
    monkeypatch.setenv("ACETATE_DB", conn.execute("PRAGMA database_list").fetchone()["file"])
    monkeypatch.setenv("ACETATE_AUTH_MODE", "local")
    from acetate import auth, web

    monkeypatch.setattr(auth, "load_credentials", lambda: None)
    tid = chatstore.create_thread(conn)["id"]
    chatstore.add_message(conn, tid, "user", "Remember playlist P1.")
    chatstore.add_message(conn, tid, "assistant", "Noted.")
    original = chatstore.messages(conn, tid)
    completions = Completions([_stream(content, finish_reason="length")])
    monkeypatch.setattr(agent, "_client", lambda: _client(completions))
    monkeypatch.setattr(agent, "system_prompt", lambda *args, **kwargs: "system")
    monkeypatch.setattr(agent, "tools_for", lambda can_admin: [])
    response = TestClient(web.app).post("/chat/stream", data={"q": "continue", "thread": tid})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1]["t"] == "error"
    assert "output limit" in events[-1]["v"]
    assert not any(event["t"] == "done" for event in events)
    assert chatstore.messages(conn, tid) == original
