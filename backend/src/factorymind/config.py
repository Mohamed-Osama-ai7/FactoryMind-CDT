"""Runtime configuration loaded from environment variables (prefix ``FACTORYMIND_``)."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
