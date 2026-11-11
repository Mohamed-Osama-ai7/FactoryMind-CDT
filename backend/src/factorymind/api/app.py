"""FastAPI application factory.

Safety boundary: FactoryMind is advisory and read-only toward industrial systems. The API must
never expose routes that write to PLCs, change setpoints or actuate equipment; this is enforced
by ``tests/test_safety_boundary.py``. Every route except ``/api/v1/health`` requires a valid
access token and an explicit permission (deny by default); ``tests/test_auth_routes.py`` checks it.
"""

import logging
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel

from factorymind import __version__
from factorymind.api.anomaly import router as anomaly_router
from factorymind.api.assets import router as assets_router
from factorymind.api.incidents import router as incidents_router
from factorymind.api.security import SecurityMiddleware, SlidingWindowLimiter
from factorymind.auth.tokens import TokenVerifier
from factorymind.config import Settings, get_settings
from factorymind.db.audit import AuditUnavailableError
from factorymind.db.session import Database

log = logging.getLogger("factorymind")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    environment: str
    operating_mode: Literal["advisory-read-only"]
    auth: Literal["configured", "not_configured"]


def _operation_id(route: APIRoute) -> str:
    # Stable, readable operation IDs (e.g. "health") give clean generated client names.
    return route.name


def _build_verifier(settings: Settings) -> TokenVerifier | None:
    if not settings.auth_configured:
        log.warning("authentication is not configured: every protected route will answer 401")
        return None
    try:
        return TokenVerifier(settings)
    except Exception:  # unreadable JWKS file, malformed keys, ...: fail closed
        log.exception("token verifier could not be initialised: protected routes answer 401")
        return None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    production = settings.environment == "production"
    app = FastAPI(
        title="FactoryMind CDT API",
        version=__version__,
        description="Evidence-grounded, uncertainty-aware decision support. Advisory only.",
        generate_unique_id_function=_operation_id,
        # interactive documentation and the schema are not served in production
        docs_url=None if production else "/docs",
        redoc_url=None if production else "/redoc",
        openapi_url=None if production else "/openapi.json",
    )
    app.state.settings = settings
    app.state.verifier = _build_verifier(settings)
    app.state.db = Database(settings.sqlalchemy_url)
    app.state.limiter = SlidingWindowLimiter()
    app.add_middleware(SecurityMiddleware, max_body_bytes=settings.max_body_bytes)

    @app.exception_handler(AuditUnavailableError)
    async def audit_unavailable(_: Request, exc: AuditUnavailableError) -> JSONResponse:
        # a state change whose audit event cannot be recorded is rolled back, never applied
        app.state.db.note_audit_failure()
        log.critical("AUDIT FAILURE: state change refused (audit log unavailable)", exc_info=exc)
        return JSONResponse(
            {"detail": "Audit log temporarily unavailable; the request was not applied"},
            status_code=503,
            headers={"Retry-After": "1"},
        )

    @app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            environment=settings.environment,
            operating_mode="advisory-read-only",
            auth="configured" if app.state.verifier is not None else "not_configured",
        )

    app.include_router(anomaly_router)
    app.include_router(assets_router)
    app.include_router(incidents_router)
    return app


app = create_app()
