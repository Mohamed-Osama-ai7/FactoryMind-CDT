"""Repository-level proof that incident idempotency and reads are tenant- and asset-scoped."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from factorymind.auth.principal import Principal, Role
from factorymind.db.repository import DuplicateIncidentError, IncidentRepo
from factorymind.db.session import Database
from tests.auth_helpers import ASSET

START = datetime(2020, 1, 9, 6, 0)


def principal(tenant: str, assets: frozenset[str] = frozenset({ASSET})) -> Principal:
    return Principal(
        actor_id="eve", tenant_id=tenant, roles=frozenset({Role.ENGINEER}), asset_ids=assets
    )


def new_incident(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "asset_id": ASSET,
        "detector": "unit-detector",
        "model_version": "m1",
        "episode_start": START,
        "episode_end": START + timedelta(hours=4),
        "raised_at": START + timedelta(minutes=20),
        "peak_score": 1.5,
        "evidence": [],
    }
    values.update(overrides)
    return values


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(f"sqlite:///{(tmp_path / 'repo.sqlite3').as_posix()}")


def test_the_same_episode_in_two_tenants_is_two_incidents_and_each_finds_only_its_own(
    db: Database,
) -> None:
    ids: dict[str, str] = {}
    for tenant in ("acme", "globex"):
        with db.session() as session:
            ids[tenant] = IncidentRepo(session, principal(tenant)).create(**new_incident()).id
    assert ids["acme"] != ids["globex"]  # the unique key includes the tenant

    for tenant, other in (("acme", "globex"), ("globex", "acme")):
        with db.session() as session:
            repo = IncidentRepo(session, principal(tenant))
            found = repo.find_by_episode(ASSET, "m1", START)
            assert found is not None and found.id == ids[tenant]
            assert repo.get(ids[other]) is None  # the other tenant's incident does not exist here
            items, total = repo.list_page(status=None, limit=25, offset=0)
            assert total == 1 and [i.id for i in items] == [ids[tenant]]

    with db.session() as session:  # a third tenant sees neither
        repo = IncidentRepo(session, principal("initech"))
        assert repo.find_by_episode(ASSET, "m1", START) is None
        assert repo.list_page(status=None, limit=25, offset=0)[1] == 0


def test_a_duplicate_within_one_tenant_is_rejected_and_leaves_one_incident(db: Database) -> None:
    with db.session() as session:
        IncidentRepo(session, principal("acme")).create(**new_incident())
    with pytest.raises(DuplicateIncidentError), db.session() as session:
        IncidentRepo(session, principal("acme")).create(**new_incident())
    with db.session() as session:
        assert (
            IncidentRepo(session, principal("acme")).list_page(status=None, limit=25, offset=0)[1]
            == 1
        )


def test_a_new_model_version_is_a_new_incident_for_the_same_episode(db: Database) -> None:
    with db.session() as session:
        repo = IncidentRepo(session, principal("acme"))
        first = repo.create(**new_incident(model_version="m1")).id
        second = repo.create(**new_incident(model_version="m2")).id
    assert first != second


def test_a_principal_without_the_asset_cannot_create_or_find_it(db: Database) -> None:
    with db.session() as session:
        IncidentRepo(session, principal("acme")).create(**new_incident())
    blind = principal("acme", assets=frozenset())
    with db.session() as session:
        repo = IncidentRepo(session, blind)
        assert repo.find_by_episode(ASSET, "m1", START) is None
        assert repo.list_page(status=None, limit=25, offset=0)[1] == 0
        with pytest.raises(PermissionError):
            repo.create(**new_incident(episode_start=START + timedelta(days=1)))
