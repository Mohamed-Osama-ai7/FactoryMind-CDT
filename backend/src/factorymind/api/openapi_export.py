"""Export the API's OpenAPI schema; the frontend's typed client is generated from this file.

Usage: ``uv run python -m factorymind.api.openapi_export ../frontend/openapi.json``
"""

import json
import sys
from pathlib import Path
from typing import Any

from factorymind.api.app import create_app


def openapi_document() -> dict[str, Any]:
    return create_app().openapi()


def render() -> str:
    return json.dumps(openapi_document(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m factorymind.api.openapi_export <output.json>", file=sys.stderr)
        return 2
    Path(argv[0]).write_text(render(), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
