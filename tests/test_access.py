from __future__ import annotations

import pytest
from fastapi import HTTPException

from acetate import access


def test_local_mode_is_an_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACETATE_AUTH_MODE", "local")
    monkeypatch.setenv("ACETATE_LOCAL_USER", "alice-local")

    identity = access.identity_from_headers({})

    assert identity.username == "alice-local"
    assert identity.source == "local"
    assert identity.can_query
    assert identity.can_admin


@pytest.mark.parametrize(
    ("groups", "role"),
    [
        ("friends", access.Role.VIEWER),
        ("friends,acetate-query", access.Role.QUERY),
        ("acetate-admin", access.Role.ADMIN),
    ],
)
def test_authelia_groups_map_to_roles(
    monkeypatch: pytest.MonkeyPatch, groups: str, role: access.Role
) -> None:
    monkeypatch.setenv("ACETATE_AUTH_MODE", "authelia")

    identity = access.identity_from_headers(
        {
            "Remote-User": "friend",
            "Remote-Groups": groups,
            "Remote-Email": "friend@example.com",
            "Remote-Name": "Friendly Person",
        }
    )

    assert identity.role is role
    assert identity.name == "Friendly Person"
    assert identity.email == "friend@example.com"


def test_configured_admin_username_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACETATE_AUTH_MODE", "authelia")
    monkeypatch.setenv("ACETATE_ADMIN_USERS", "Alice")

    identity = access.identity_from_headers({"remote-user": "ALICE"})

    assert identity.role is access.Role.ADMIN


def test_authelia_requires_remote_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACETATE_AUTH_MODE", "authelia")

    with pytest.raises(HTTPException) as exc:
        access.identity_from_headers({"Remote-Groups": "acetate-admin"})

    assert exc.value.status_code == 401


def test_unknown_auth_mode_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACETATE_AUTH_MODE", "typo")

    with pytest.raises(RuntimeError, match="ACETATE_AUTH_MODE"):
        access.identity_from_headers({})
