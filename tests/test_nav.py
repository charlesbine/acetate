"""The nav bar: every page reachable, exactly once, in a bar that actually fits."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from acetate import web

# Rough width of the links row at desktop size: Space Mono at .72rem with .12em
# tracking is ~8.3px per character, plus 1.4rem of horizontal padding per item.
# The row shares a 1080px container with the brand (~110px) and identity (~120px),
# so ~700px is the point where things start colliding.
_PX_PER_CHAR = 8.3
_PX_PER_ITEM_PADDING = 22.0
_LINKS_BUDGET_PX = 700


def _all_entries() -> list[tuple]:
    entries = list(web.NAV_PRIMARY)
    for _, group in web.NAV_GROUPS:
        entries.extend(group)
    return entries


def _top_level_labels() -> list[str]:
    return [label for _, _, label in web.NAV_PRIMARY] + [
        label + " ▾" for label, _ in web.NAV_GROUPS
    ]


def test_top_level_nav_fits_the_desktop_bar() -> None:
    labels = _top_level_labels()
    width = sum(len(label) * _PX_PER_CHAR + _PX_PER_ITEM_PADDING for label in labels)

    assert width < _LINKS_BUDGET_PX, (
        f"{len(labels)} top-level items ≈ {width:.0f}px, over the {_LINKS_BUDGET_PX}px "
        f"budget — move something into a group: {labels}"
    )


def test_every_page_appears_exactly_once() -> None:
    hrefs = [href for _, href, _ in _all_entries()]
    keys = [key for key, _, _ in _all_entries() if key is not None]

    assert len(hrefs) == len(set(hrefs))
    assert len(keys) == len(set(keys))


def test_nav_covers_every_page_the_app_renders() -> None:
    """A new page with an `active=` key must be reachable from the bar."""
    rendered = set(re.findall(r'active="([a-z]+)"', web.__doc__ or "")) or set(
        re.findall(r'active="([a-z]+)"', _web_source())
    )
    navigable = {key for key, _, _ in _all_entries() if key is not None}

    assert rendered, "could not read active keys from the web module"
    assert rendered <= navigable, f"pages missing from the nav: {sorted(rendered - navigable)}"
    assert navigable <= rendered, (
        f"nav points at pages that don't exist: {sorted(navigable - rendered)}"
    )


def _web_source() -> str:
    from pathlib import Path

    return Path(web.__file__).read_text()


def test_active_page_lights_up_its_group() -> None:
    nav = web.nav_for("smart")
    playlists, system = nav["groups"]

    assert playlists["label"] == "Playlists"
    assert playlists["on"] is True  # the group shows where you are
    assert [it["on"] for it in playlists["items"]].count(True) == 1
    assert next(it for it in playlists["items"] if it["on"])["key"] == "smart"
    assert system["on"] is False
    assert all(not it["on"] for it in nav["primary"])


def test_a_primary_page_lights_up_alone() -> None:
    nav = web.nav_for("hyperspace")

    assert [it["label"] for it in nav["primary"] if it["on"]] == ["Hyperspace"]
    assert all(not group["on"] for group in nav["groups"])


def test_external_links_never_claim_to_be_the_current_page() -> None:
    externals = [it for _, group in web.NAV_GROUPS for it in group if it[0] is None]

    assert externals, "expected at least the offsite build-journey link"
    for _, href, _ in externals:
        for active in (None, "home", "pipeline"):
            item = next(
                i for g in web.nav_for(active)["groups"] for i in g["items"] if i["href"] == href
            )
            assert item["on"] is False
            assert item["external"] is True


@pytest.fixture
def client(conn, monkeypatch) -> TestClient:
    path = conn.execute("PRAGMA database_list").fetchone()["file"]
    monkeypatch.setenv("ACETATE_DB", path)
    monkeypatch.setenv("ACETATE_AUTH_MODE", "local")
    from acetate import auth

    monkeypatch.setattr(auth, "load_credentials", lambda: None)
    return TestClient(web.app)


def test_rendered_bar_has_grouped_menus_and_every_link(client) -> None:
    page = client.get("/").text
    bar = page[page.index("<nav>") : page.index("</nav>")]

    assert bar.count('<details class="navgroup') == len(web.NAV_GROUPS)
    for _, href, label in _all_entries():
        assert f'href="{href}"' in bar, f"{label} missing from the bar"
    assert "target=_blank" in bar  # the offsite link opens away


def test_the_open_page_is_marked_in_the_rendered_bar(client) -> None:
    # /stats is skipped here: it divides by a max play count and cannot render
    # against an empty database — unrelated to the nav.
    primary = client.get("/hyperspace").text
    assert 'href="/hyperspace" class="on"' in primary
    assert 'class="navgroup on"' not in primary

    grouped = client.get("/gems").text
    assert 'class="navgroup on"' in grouped  # Playlists group, holding Lost Gems
    assert 'href="/gems" class="on"' in grouped
