from fastapi.testclient import TestClient

from acetate import agent


def test_model_catalog_groups_flatten_to_unique_allowlist():
    grouped = [model for group in agent.MODEL_GROUPS for model in group["models"]]
    selections = [agent.model_id(model) for model in agent.MODELS]

    assert agent.MODELS == grouped
    assert len(agent.MODELS) == 32
    assert len(set(selections)) == len(agent.MODELS)
    assert len({model["label"] for model in agent.MODELS}) == len(agent.MODELS)
    assert agent.ALLOWED == set(selections)


def test_model_catalog_keeps_stable_default_and_covers_major_providers(monkeypatch):
    monkeypatch.delenv("ACETATE_LLM_MODEL", raising=False)

    assert agent.default_model() == "z-ai/glm-5.2"
    assert {slug.split("/", 1)[0] for slug in agent.ALLOWED} >= {
        "anthropic",
        "deepseek",
        "google",
        "minimax",
        "moonshotai",
        "openai",
        "x-ai",
        "z-ai",
    }
    monkeypatch.setenv("ACETATE_LLM_MODEL", "vendor/unlisted-model")
    assert agent.default_model() == "z-ai/glm-5.2"


def test_web_model_selection_only_accepts_allowlisted_slugs(monkeypatch):
    monkeypatch.delenv("ACETATE_LLM_MODEL", raising=False)

    assert agent.resolve_model("anthropic/claude-sonnet-5") == "anthropic/claude-sonnet-5"
    assert agent.resolve_model("vendor/unlisted-model") == "z-ai/glm-5.2"


def test_provider_pinned_selection_uses_canonical_model_and_exact_provider(monkeypatch):
    assert agent.model_request_kwargs("openai/gpt-oss-120b@cerebras") == {
        "model": "openai/gpt-oss-120b",
        "extra_body": {"provider": {"only": ["cerebras"]}},
    }
    assert agent.model_request_kwargs("openai/gpt-oss-120b:nitro") == {
        "model": "openai/gpt-oss-120b:nitro",
    }
    monkeypatch.setattr(agent, "MODEL_PREFIX", "openrouter/")
    assert agent.model_request_kwargs("z-ai/glm-4.7@cerebras") == {
        "model": "openrouter/z-ai/glm-4.7",
        "extra_body": {"provider": {"only": ["cerebras"]}},
    }


def test_chat_page_renders_grouped_model_picker(conn, monkeypatch):
    path = conn.execute("PRAGMA database_list").fetchone()["file"]
    monkeypatch.setenv("ACETATE_DB", path)
    monkeypatch.setenv("ACETATE_AUTH_MODE", "local")
    from acetate import auth

    monkeypatch.setattr(auth, "load_credentials", lambda: None)
    from acetate import web

    page = TestClient(web.app).get("/chat")

    assert page.status_code == 200
    assert '<optgroup label="Recommended">' in page.text
    assert '<optgroup label="High-speed routes">' in page.text
    assert '<optgroup label="Flagship &amp; frontier">' in page.text
    assert '<optgroup label="Value &amp; experimental">' in page.text
    assert page.text.count("<option") == len(agent.MODELS)
    assert 'value="z-ai/glm-5.2" selected' in page.text
    assert 'value="openai/gpt-oss-120b@cerebras"' in page.text
