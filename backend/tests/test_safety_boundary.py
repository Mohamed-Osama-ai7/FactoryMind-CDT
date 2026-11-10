"""Guards the advisory/read-only safety boundary of the API.

Any new state-changing route must be added to ``ALLOWED_MUTATING_PATHS`` deliberately, and no
route may ever target industrial control (PLC writes, setpoints, actuation).
"""

import re

from fastapi.routing import APIRoute

from factorymind.api.app import create_app

# Internal workflow writes (incidents, reviews, audit) will be listed here as they are added.
ALLOWED_MUTATING_PATHS: frozenset[str] = frozenset(
    {
        "/api/v1/anomaly/score",  # stateless scoring of submitted samples; persists nothing
    }
)

FORBIDDEN_PATH_PATTERN = re.compile(
    r"(plc|setpoint|actuat|write[-_]?tag|command|control)", re.IGNORECASE
)


def _routes() -> list[APIRoute]:
    return [r for r in create_app().routes if isinstance(r, APIRoute)]


def test_no_route_targets_industrial_control() -> None:
    offending = [r.path for r in _routes() if FORBIDDEN_PATH_PATTERN.search(r.path)]
    assert offending == []


def test_mutating_routes_are_explicitly_allowlisted() -> None:
    mutating = {r.path for r in _routes() if r.methods & {"POST", "PUT", "PATCH", "DELETE"}}
    assert mutating <= ALLOWED_MUTATING_PATHS
