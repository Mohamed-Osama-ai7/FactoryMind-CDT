"""Principals, roles and permissions. Authorization is deny-by-default.

A principal is built only from a *verified* token. Unknown role strings are ignored, so a token
with no recognised role has no permission at all. Tenant and asset scope come from the token (or,
for the tenant, an explicitly configured single-tenant default) and are never taken from a request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from factorymind.config import IDENTIFIER, Settings


class Role(StrEnum):
    VIEWER = "viewer"
    ENGINEER = "engineer"
    APPROVER = "approver"
    AUDITOR = "auditor"
    ADMIN = "admin"


class Permission(StrEnum):
    ANOMALY_READ = "anomaly:read"
    INCIDENT_READ = "incident:read"
    INCIDENT_CREATE = "incident:create"
    OBSERVATION_CREATE = "observation:create"
    DECISION_CREATE = "decision:create"
    AUDIT_READ = "audit:read"


_VIEW = {Permission.ANOMALY_READ, Permission.INCIDENT_READ}

ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.VIEWER: frozenset(_VIEW),
    Role.ENGINEER: frozenset(_VIEW | {Permission.INCIDENT_CREATE, Permission.OBSERVATION_CREATE}),
    Role.APPROVER: frozenset(_VIEW | {Permission.DECISION_CREATE, Permission.OBSERVATION_CREATE}),
    Role.AUDITOR: frozenset({Permission.INCIDENT_READ, Permission.AUDIT_READ}),
    # separation of duties: administrators manage and audit but do not approve
    Role.ADMIN: frozenset(set(Permission) - {Permission.DECISION_CREATE}),
}

_ACTOR = re.compile(r"^[A-Za-z0-9._:@|-]{1,128}$")
MAX_ASSET_GRANTS = 100


class IdentityError(Exception):
    """The token is valid but cannot be turned into a usable principal."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Principal:
    actor_id: str
    tenant_id: str
    roles: frozenset[Role]
    asset_ids: frozenset[str]  # explicit grants; empty means no asset access

    @property
    def permissions(self) -> frozenset[Permission]:
        granted: set[Permission] = set()
        for role in self.roles:
            granted |= ROLE_PERMISSIONS[role]
        return frozenset(granted)

    def can(self, permission: Permission) -> bool:
        return permission in self.permissions

    def can_access_asset(self, asset_id: str) -> bool:
        return asset_id in self.asset_ids

    def data_source_denial(self, settings: Settings) -> str | None:
        """Why this principal may not use the served data source (None when it may)."""
        if settings.data_source_owner is None:
            return "data_source_unowned"
        if self.tenant_id != settings.data_source_owner:
            return "tenant_mismatch"
        if not self.can_access_asset(settings.data_source_asset_id):
            return "asset_not_granted"
        return None


def _dig(claims: dict[str, Any], path: str) -> Any:
    node: Any = claims
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def principal_from_claims(claims: dict[str, Any], settings: Settings) -> Principal:
    actor = claims.get("sub")
    if not isinstance(actor, str) or not _ACTOR.match(actor):
        raise IdentityError("bad_subject")

    tenant = _dig(claims, settings.auth_tenant_claim)
    if tenant is None and settings.single_tenant_mode:
        tenant = settings.default_tenant_id
    if not isinstance(tenant, str) or not IDENTIFIER.match(tenant):
        raise IdentityError("no_tenant")

    raw_roles = _dig(claims, settings.auth_roles_claim)
    items = [raw_roles] if isinstance(raw_roles, str) else raw_roles
    roles = frozenset(
        Role(r)
        for r in (items if isinstance(items, list) else [])
        if isinstance(r, str) and r in Role._value2member_map_
    )

    raw_assets = _dig(claims, settings.auth_assets_claim)
    grants = raw_assets if isinstance(raw_assets, list) else []
    if len(grants) > MAX_ASSET_GRANTS:
        raise IdentityError("too_many_assets")
    if "*" in grants:
        assets = frozenset(settings.asset_ids)
    else:
        assets = frozenset(a for a in grants if isinstance(a, str) and a in settings.asset_ids)
    return Principal(actor_id=actor, tenant_id=tenant, roles=roles, asset_ids=assets)
