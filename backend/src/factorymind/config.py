"""Runtime configuration loaded from environment variables (prefix ``FACTORYMIND_``)."""

import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ALLOWED_ALGORITHMS = frozenset({"RS256", "RS384", "RS512", "ES256", "ES384", "PS256"})
IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


class Settings(BaseSettings):
    """Validated application settings.

    Large artefacts (raw/processed data, model caches, experiment runs) are kept outside the
    Git repository under ``data_dir``.
    """

    model_config = SettingsConfigDict(env_prefix="FACTORYMIND_", env_file=".env", extra="ignore")

    environment: str = Field(default="development", pattern="^(development|test|production)$")
    data_dir: Path = Field(default=Path("D:/FactoryMind/data"))
    artifacts_dir: Path = Field(default=Path("D:/FactoryMind/artifacts"))
    anomaly_model_name: str = Field(default="metropt3-anomaly", pattern=r"^[a-z0-9][a-z0-9_.-]*$")

    # --- authentication (OIDC resource server) ---
    auth_issuer: str | None = None
    auth_audience: str | None = None
    auth_jwks_url: str | None = None  # https only (http://localhost allowed in development)
    auth_jwks_file: Path | None = None  # static JWKS for air-gapped sites and tests
    auth_algorithms: list[str] = ["RS256", "ES256"]
    auth_leeway_seconds: int = Field(default=30, ge=0, le=120)
    auth_roles_claim: str = "roles"  # dotted path allowed, e.g. realm_access.roles
    auth_tenant_claim: str = "tenant_id"
    auth_assets_claim: str = "assets"  # list of asset ids, or ["*"] for every asset of the tenant
    # Single-tenant mode (explicit opt-in): a token without a tenant claim belongs to
    # ``default_tenant_id``. Otherwise a missing tenant claim is rejected.
    single_tenant_mode: bool = False
    default_tenant_id: str | None = None
    # The tenant that owns the telemetry/model this deployment serves (defaults to the single
    # tenant in single-tenant mode). With neither set, anomaly and incident data are denied.
    data_source_tenant_id: str | None = None

    # --- authorization, assets, persistence ---
    asset_ids: list[str] = ["metropt3-apu-1"]
    # The one asset whose telemetry the configured model/dataset actually describes.
    data_source_asset_id: str = "metropt3-apu-1"
    require_separation_of_duties: bool = True
    database_url: str | None = None  # default: SQLite file under data_dir

    # --- abuse controls ---
    rate_limit_read_per_min: int = Field(default=120, ge=1)
    rate_limit_write_per_min: int = Field(default=30, ge=1)
    rate_limit_auth_failures_per_min: int = Field(default=20, ge=1)
    max_body_bytes: int = Field(default=1_048_576, ge=1024)

    @field_validator("auth_algorithms")
    @classmethod
    def _algorithms(cls, value: list[str]) -> list[str]:
        if not value or any(a not in _ALLOWED_ALGORITHMS for a in value):
            raise ValueError(f"auth_algorithms must be a subset of {sorted(_ALLOWED_ALGORITHMS)}")
        return value

    @field_validator("asset_ids")
    @classmethod
    def _assets(cls, value: list[str]) -> list[str]:
        if not value or any(not IDENTIFIER.match(a) for a in value):
            raise ValueError("asset ids must match [a-z0-9][a-z0-9_.-]{0,63}")
        return value

    @field_validator("default_tenant_id", "data_source_tenant_id")
    @classmethod
    def _tenant(cls, value: str | None) -> str | None:
        if value is not None and not IDENTIFIER.match(value):
            raise ValueError("tenant ids must match [a-z0-9][a-z0-9_.-]{0,63}")
        return value

    @model_validator(mode="after")
    def _tenancy_and_data_source(self) -> "Settings":
        if self.single_tenant_mode and self.default_tenant_id is None:
            raise ValueError("single_tenant_mode requires default_tenant_id")
        if not IDENTIFIER.match(self.data_source_asset_id):
            raise ValueError("data_source_asset_id must match [a-z0-9][a-z0-9_.-]{0,63}")
        if self.data_source_asset_id not in self.asset_ids:
            raise ValueError("data_source_asset_id must be one of asset_ids")
        return self

    @property
    def data_source_owner(self) -> str | None:
        """Tenant that owns the served data; None means nobody (deny by default)."""
        if self.data_source_tenant_id is not None:
            return self.data_source_tenant_id
        return self.default_tenant_id if self.single_tenant_mode else None

    @model_validator(mode="after")
    def _jwks_url_scheme(self) -> "Settings":
        if self.auth_jwks_url:
            parsed = urlparse(self.auth_jwks_url)
            local = parsed.hostname in {"localhost", "127.0.0.1"}
            if parsed.scheme != "https" and not (
                parsed.scheme == "http" and local and self.environment != "production"
            ):
                raise ValueError("auth_jwks_url must be https")
        return self

    @property
    def auth_configured(self) -> bool:
        return bool(
            self.auth_issuer and self.auth_audience and (self.auth_jwks_url or self.auth_jwks_file)
        )

    @property
    def sqlalchemy_url(self) -> str:
        return (
            self.database_url or f"sqlite:///{(self.data_dir / 'factorymind.sqlite3').as_posix()}"
        )

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


@lru_cache
def get_settings() -> Settings:
    return Settings()
