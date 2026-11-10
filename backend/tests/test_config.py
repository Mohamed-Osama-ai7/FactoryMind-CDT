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


@pytest.mark.parametrize("algorithms", [["HS256"], ["none"], ["RS256", "HS512"], []])
def test_symmetric_and_unknown_jwt_algorithms_are_refused(algorithms: list[str]) -> None:
    with pytest.raises(ValidationError):
        Settings(auth_algorithms=algorithms)


@pytest.mark.parametrize(
    ("url", "environment", "valid"),
    [
        ("https://idp.example/jwks", "production", True),
        ("http://idp.example/jwks", "development", False),
        ("http://localhost:8080/jwks", "development", True),
        ("http://localhost:8080/jwks", "production", False),
        ("file:///etc/passwd", "development", False),
    ],
)
def test_jwks_url_must_be_https_except_local_development(
    url: str, environment: str, valid: bool
) -> None:
    if valid:
        assert Settings(auth_jwks_url=url, environment=environment).auth_jwks_url == url
    else:
        with pytest.raises(ValidationError):
            Settings(auth_jwks_url=url, environment=environment)


def test_identifiers_and_limits_are_validated() -> None:
    for kwargs in (
        {"asset_ids": ["../x"]},
        {"asset_ids": []},
        {"default_tenant_id": "Bad Tenant"},
        {"auth_leeway_seconds": 3600},
        {"max_body_bytes": 10},
    ):
        with pytest.raises(ValidationError):
            Settings(**kwargs)


def test_auth_is_configured_only_when_issuer_audience_and_keys_are_all_set(tmp_path: Path) -> None:
    assert not Settings(auth_issuer=None, auth_audience=None).auth_configured
    partial = Settings(auth_issuer="https://idp", auth_audience="api", auth_jwks_url=None)
    assert not partial.auth_configured
    full = Settings(auth_issuer="https://idp", auth_audience="api", auth_jwks_file=tmp_path / "k")
    assert full.auth_configured


def test_database_defaults_to_a_file_under_the_data_directory(tmp_path: Path) -> None:
    url = Settings(data_dir=tmp_path, database_url=None).sqlalchemy_url
    assert url.startswith("sqlite:///") and "factorymind.sqlite3" in url
