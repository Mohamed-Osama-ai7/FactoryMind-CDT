"""FastAPI application factory.

Safety boundary: FactoryMind is advisory and read-only toward industrial systems. The API must
never expose routes that write to PLCs, change setpoints or actuate equipment; this is enforced
by ``tests/test_safety_boundary.py``.
"""

from typing import Literal

from fastapi import FastAPI
from fastapi.routing import APIRoute
from pydantic import BaseModel

from factorymind import __version__
from factorymind.config import get_settings


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    environment: str
    operating_mode: Literal["advisory-read-only"]


def _operation_id(route: APIRoute) -> str:
    # Stable, readable operation IDs (e.g. "health") give clean generated client names.
    return route.name


def create_app() -> FastAPI:
    app = FastAPI(
        title="FactoryMind CDT API",
        version=__version__,
        description="Evidence-grounded, uncertainty-aware decision support. Advisory only.",
        generate_unique_id_function=_operation_id,
    )

    @app.get("/api/v1/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            environment=get_settings().environment,
            operating_mode="advisory-read-only",
        )

    return app


app = create_app()
