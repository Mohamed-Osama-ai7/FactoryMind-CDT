"""Guards the advisory/read-only safety boundary of the API.

Any new state-changing route must be added to ``ALLOWED_MUTATING_OPERATIONS`` deliberately, and no
route may ever target industrial control (PLC writes, setpoints, actuation). A decision recorded
through the API is a human review record in FactoryMind's own database; it triggers nothing else.
"""

import re

from factorymind.api.app import create_app
from tests.route_helpers import all_operations

ALLOWED_MUTATING_OPERATIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("POST", "/api/v1/anomaly/score"),  # stateless scoring of submitted samples
        ("POST", "/api/v1/incidents"),  # records an incident re-detected by the server
        ("POST", "/api/v1/incidents/{incident_id}/decisions"),  # human review record
        ("POST", "/api/v1/incidents/{incident_id}/observations"),  # human field observation
    }
)

FORBIDDEN_PATH_PATTERN = re.compile(
    r"(plc|setpoint|actuat|write[-_]?tag|command|control|opc|modbus)", re.IGNORECASE
)
MUTATING = {"POST", "PUT", "PATCH", "DELETE"}


def test_the_guard_sees_the_applications_routes() -> None:
    operations = all_operations(create_app())
    assert len(operations) >= 11  # fails loudly if route discovery ever stops working
    assert ("GET", "/api/v1/health") in operations
    assert ("POST", "/api/v1/incidents") in operations


def test_no_route_targets_industrial_control() -> None:
    offending = [p for _, p in all_operations(create_app()) if FORBIDDEN_PATH_PATTERN.search(p)]
    assert offending == []


def test_mutating_routes_are_exactly_the_allowlist() -> None:
    mutating = {(m, p) for m, p in all_operations(create_app()) if m in MUTATING}
    assert mutating == ALLOWED_MUTATING_OPERATIONS  # no new write route, no stale allowlist entry
