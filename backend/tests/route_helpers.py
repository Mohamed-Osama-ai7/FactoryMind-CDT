"""Enumerate every HTTP route of an application.

FastAPI 0.16x wraps routers added with ``include_router`` in ``_IncludedRouter`` objects, so
``app.routes`` no longer lists the individual ``APIRoute`` objects. A guard built on it passes
vacuously. The OpenAPI document lists every operation regardless of how routers are nested.
"""

from __future__ import annotations

from fastapi import FastAPI

HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}


def all_operations(app: FastAPI) -> list[tuple[str, str]]:
    """(METHOD, path template) for every operation, e.g. ("POST", "/api/v1/incidents")."""
    paths = app.openapi()["paths"]
    return [
        (method.upper(), path)
        for path, item in paths.items()
        for method in item
        if method in HTTP_METHODS
    ]
