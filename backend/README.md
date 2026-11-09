# factorymind (backend)

Python package for the FactoryMind CDT backend: data pipeline, models, evidence/RCA,
uncertainty, simulation and the FastAPI service.

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy
uv run uvicorn factorymind.api.app:app --reload
```

Large data, model caches and experiment runs live outside the repository; see
`FACTORYMIND_DATA_DIR` in `src/factorymind/config.py`.
