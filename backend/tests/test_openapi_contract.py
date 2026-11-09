"""The committed OpenAPI file is the contract the React client is generated from."""

from pathlib import Path

import pytest

from factorymind.api.openapi_export import render

CONTRACT = Path(__file__).resolve().parents[2] / "frontend" / "openapi.json"


@pytest.mark.skipif(not CONTRACT.exists(), reason="frontend contract file not present")
def test_committed_openapi_matches_application() -> None:
    assert CONTRACT.read_text(encoding="utf-8") == render(), (
        "frontend/openapi.json is stale: run "
        "`uv run python -m factorymind.api.openapi_export ../frontend/openapi.json` "
        "and `npm run generate:api` in frontend/."
    )
