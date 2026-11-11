"""Industrial Asset Digital Passport registry."""

from factorymind.assets.passport import (
    REGISTERED_ASSET_IDS,
    get_asset_passport,
    list_asset_passports,
    read_active_model_version,
)

__all__ = [
    "REGISTERED_ASSET_IDS",
    "get_asset_passport",
    "list_asset_passports",
    "read_active_model_version",
]
