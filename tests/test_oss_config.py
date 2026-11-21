from __future__ import annotations

from acetate import agent, mbid


def test_mbid_headers_default_and_override(monkeypatch) -> None:
    monkeypatch.delenv("ACETATE_USER_AGENT", raising=False)
    assert mbid._headers()["User-Agent"] == mbid.DEFAULT_USER_AGENT
    assert "@" not in mbid._headers()["User-Agent"]

    monkeypatch.setenv("ACETATE_USER_AGENT", "custom-acetate/1.0 (+https://example.org)")
    assert mbid._headers()["User-Agent"] == "custom-acetate/1.0 (+https://example.org)"


def test_agent_client_uses_public_url_env_override(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    monkeypatch.delenv("ACETATE_PUBLIC_URL", raising=False)
    default_client = agent._client()
    assert (
        default_client.default_headers["HTTP-Referer"] == "https://github.com/charlesbine/acetate"
    )

    monkeypatch.setenv("ACETATE_PUBLIC_URL", "https://music.example.com")
    custom_client = agent._client()
    assert custom_client.default_headers["HTTP-Referer"] == "https://music.example.com"
