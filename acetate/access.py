"""Request identity and role adapter for the web dashboard.

The application has one authorization vocabulary regardless of where identity
comes from. Local development supplies an explicit admin identity; production
trusts headers injected by the reverse proxy after Authelia forward-auth.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping

from fastapi import HTTPException, Request


class Role(IntEnum):
    VIEWER = 10
    QUERY = 20
    ADMIN = 30


@dataclass(frozen=True)
class Identity:
    username: str
    name: str
    email: str | None
    groups: frozenset[str]
    role: Role
    source: str

    @property
    def can_query(self) -> bool:
        return self.role >= Role.QUERY

    @property
    def can_admin(self) -> bool:
        return self.role >= Role.ADMIN


identity_context: ContextVar[Identity | None] = ContextVar("acetate_identity", default=None)


def _items(value: str | None) -> frozenset[str]:
    return frozenset(item.strip().lower() for item in (value or "").split(",") if item.strip())


def _configured_role(username: str, groups: frozenset[str]) -> Role:
    username = username.lower()
    if username in _items(os.environ.get("ACETATE_ADMIN_USERS")):
        return Role.ADMIN
    if groups & _items(os.environ.get("ACETATE_ADMIN_GROUPS", "acetate-admin")):
        return Role.ADMIN
    if groups & _items(os.environ.get("ACETATE_QUERY_GROUPS", "acetate-query")):
        return Role.QUERY
    return Role.VIEWER


def identity_from_headers(headers: Mapping[str, str]) -> Identity:
    """Resolve an identity using the configured provider.

    Header names are case-insensitive in ASGI, but accepting a Mapping keeps
    this adapter independent and easy to test.
    """
    mode = os.environ.get("ACETATE_AUTH_MODE", "local").strip().lower()
    if mode == "local":
        username = os.environ.get("ACETATE_LOCAL_USER", "local-dev").strip() or "local-dev"
        return Identity(username, username, None, frozenset({"acetate-admin"}), Role.ADMIN, "local")
    if mode != "authelia":
        raise RuntimeError("ACETATE_AUTH_MODE must be 'local' or 'authelia'")

    normalized = {key.lower(): value for key, value in headers.items()}
    username = normalized.get("remote-user", "").strip()
    if not username:
        raise HTTPException(status_code=401, detail="Missing trusted Authelia identity")
    groups = _items(normalized.get("remote-groups"))
    return Identity(
        username=username,
        name=normalized.get("remote-name", "").strip() or username,
        email=normalized.get("remote-email", "").strip() or None,
        groups=groups,
        role=_configured_role(username, groups),
        source="authelia",
    )


def current_identity(request: Request) -> Identity:
    identity = getattr(request.state, "identity", None)
    if identity is None:
        identity = identity_from_headers(request.headers)
        request.state.identity = identity
    return identity


def require_query(request: Request) -> Identity:
    identity = current_identity(request)
    if not identity.can_query:
        raise HTTPException(status_code=403, detail="Query access required")
    return identity


def require_admin(request: Request) -> Identity:
    identity = current_identity(request)
    if not identity.can_admin:
        raise HTTPException(status_code=403, detail="Administrator access required")
    return identity
