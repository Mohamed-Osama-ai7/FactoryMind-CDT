from pathlib import Path

import pytest
from pydantic import ValidationError

from factorymind.config import Settings


def test_data_dir_is_configurable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FACTORYMIND_DATA_DIR", str(tmp_path))
    settings = Settings()
    assert settings.raw_dir == tmp_path / "raw"
    assert settings.processed_dir == tmp_path / "processed"


def test_invalid_environment_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACTORYMIND_ENVIRONMENT", "staging-ish")
    with pytest.raises(ValidationError):
        Settings()
