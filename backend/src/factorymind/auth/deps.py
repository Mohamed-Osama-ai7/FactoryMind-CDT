"""FastAPI dependencies: authenticate, rate-limit and authorize (deny by default).

Order for every protected route: authentication-failure throttle → token verification → principal →
per-principal rate limit → permission check. Each failure is audited with a reason code only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from factorymind.api.security import request_id_of
from factorymind.auth.principal import IdentityError, Permission, Principal, principal_from_claims
from factorymind.auth.tokens import TokenError, TokenVerifier
from factorymind.config import Settings
from factorymind.db.audit import PRE_AUTH_TENANT, append_audit
from factorymind.db.session import Database

log = logging.getLogger("factorymind.security")


def _audit(
    request: Request,
    *,
    action: str,
    outcome: str,
    tenant_id: str = PRE_AUTH_TENANT,
    actor_id: str = "-",
    roles: frozenset[object] = frozenset(),
    object_id: str = "-",
    reason: str = "",
) -> None:
    db: Database = request.app.state.db
    try:
        with db.session() as session:
            append_audit(
                session,
                tenant_id=tenant_id,
                actor_id=actor_id,
                roles=roles,
                action=action,
                object_type="route",
                object_id=object_id,
                outcome=outcome,
                request_id=request_id_of(request),
                details={"reason": reason, "method": request.method},
            )
    except Exception:  # the request outcome must not depend on audit availability
        log.exception("audit write failed for %s", action)


def _unauthorized(reason_visible: bool) -> HTTPException:
    challenge = 'Bearer error="invalid_token"' if reason_visible else "Bearer"
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": challenge},
    )


def _client_key(request: Request) -> str:
    return f"authfail:{request.client.host if request.client else 'unknown'}"


def get_principal(request: Request) -> Principal:
    settings: Settings = request.app.state.settings
    limiter = request.app.state.limiter
    verifier: TokenVerifier | None = request.app.state.verifier
    key = _client_key(request)

    blocked = limiter.blocked(key, settings.rate_limit_auth_failures_per_min)
    if not blocked.allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed authentication attempts",
            headers={"Retry-After": str(blocked.retry_after)},
        )

    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    presented = scheme.lower() == "bearer" and bool(token.strip())

    def fail(reason: str) -> HTTPException:
        limiter.record(key)
        _audit(request, action="authn.failure", outcome="failure", reason=reason)
        return _unauthorized(presented)

    if verifier is None:
        raise fail("auth_not_configured")
    if not presented:
        raise fail("no_token")
    try:
        claims = verifier.verify(token.strip())
    except TokenError as exc:
        raise fail(exc.reason) from None
    try:
        return principal_from_claims(claims, settings)
    except IdentityError as exc:
        _audit(request, action="authz.denied", outcome="denied", reason=exc.reason)
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Forbidden") from None


def require(
    permission: Permission, *, write: bool = False, cost: int = 1
) -> Callable[..., Principal]:
    """Dependency factory: authenticated + rate-limited + holds ``permission``."""

    def dependency(
        request: Request, principal: Annotated[Principal, Depends(get_principal)]
    ) -> Principal:
        settings: Settings = request.app.state.settings
        limit = settings.rate_limit_write_per_min if write else settings.rate_limit_read_per_min
        kind = "w" if write else "r"
        decision = request.app.state.limiter.allow(
            f"p:{principal.tenant_id}:{principal.actor_id}:{kind}", limit, cost
        )
        if not decision.allowed:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(decision.retry_after)},
            )
        if not principal.can(permission):
            _audit(
                request,
                action="authz.denied",
                outcome="denied",
                tenant_id=principal.tenant_id,
                actor_id=principal.actor_id,
                roles=principal.roles,
                object_id=request.url.path,
                reason=f"missing:{permission.value}",
            )
            raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Forbidden")
        return principal

    return dependency
