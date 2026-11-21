from types import SimpleNamespace

from acetate import auth


def test_load_credentials_prefers_env_local(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=from-env\n")
    (tmp_path / ".env.local").write_text("OPENROUTER_API_KEY=from-local\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    auth.load_credentials()

    assert auth.os.environ["OPENROUTER_API_KEY"] == "from-local"


def test_op_reference_resolves_when_cli_is_available(monkeypatch) -> None:
    ref = "op://Personal/OpenRouter/acetate"
    monkeypatch.setenv("OPENROUTER_API_KEY", ref)
    monkeypatch.setattr(auth.shutil, "which", lambda command: "/usr/local/bin/op")
    monkeypatch.setattr(
        auth.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="resolved-key\n", stderr=""),
    )
    auth._OP_CACHE.clear()

    auth._resolve_op_refs()

    assert auth.os.environ["OPENROUTER_API_KEY"] == "resolved-key"


def test_op_reference_is_unset_when_cli_is_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "op://Personal/OpenRouter/acetate")
    monkeypatch.setattr(auth.shutil, "which", lambda command: None)
    auth._OP_CACHE.clear()

    auth._resolve_op_refs()

    assert "OPENROUTER_API_KEY" not in auth.os.environ
