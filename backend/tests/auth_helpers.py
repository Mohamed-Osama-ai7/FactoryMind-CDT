"""Test identity provider: an RSA key pair, a static JWKS, token minting and forged tokens."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from factorymind.anomaly.service import AnomalyService
from factorymind.api.anomaly import get_anomaly_service
from factorymind.api.app import create_app
from factorymind.config import Settings

ISSUER = "https://idp.test/realms/factory"
AUDIENCE = "factorymind-api"
KID = "test-key-1"
ASSET = "metropt3-apu-1"


def b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


class AuthKit:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.private_pem = self._key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        self.public_pem = self._key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(self._key.public_key()))
        jwk.update({"kid": KID, "use": "sig", "alg": "RS256"})
        self.jwks_path = root / "jwks.json"
        self.jwks_path.write_text(json.dumps({"keys": [jwk]}), encoding="utf-8")

    def settings(self, **overrides: Any) -> Settings:
        values: dict[str, Any] = {
            "environment": "test",
            "auth_issuer": ISSUER,
            "auth_audience": AUDIENCE,
            "auth_jwks_file": self.jwks_path,
            "data_source_tenant_id": "acme",
            "data_dir": self.root / "data",
            "artifacts_dir": self.root / "artifacts",
            "database_url": f"sqlite:///{(self.root / 'app.sqlite3').as_posix()}",
        }
        values.update(overrides)
        return Settings(**values)

    def claims(
        self,
        sub: str = "user-1",
        roles: Iterable[str] = ("viewer",),
        tenant: str | None = "acme",
        assets: Iterable[str] | None = (ASSET,),
        **extra: Any,
    ) -> dict[str, Any]:
        now = int(time.time())
        claims: dict[str, Any] = {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": sub,
            "iat": now,
            "exp": now + 600,
            "roles": list(roles),
        }
        if tenant is not None:
            claims["tenant_id"] = tenant
        if assets is not None:
            claims["assets"] = list(assets)
        claims.update(extra)
        return claims

    def sign(self, claims: dict[str, Any], *, kid: str = KID, alg: str = "RS256") -> str:
        return jwt.encode(claims, self.private_pem, algorithm=alg, headers={"kid": kid})

    def token(self, **kwargs: Any) -> str:
        return self.sign(self.claims(**kwargs))

    def headers(self, **kwargs: Any) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token(**kwargs)}"}

    # -- forged tokens (built by hand; PyJWT refuses to create some of these) --

    def forge_none(self, claims: dict[str, Any]) -> str:
        header = b64(json.dumps({"alg": "none", "typ": "JWT", "kid": KID}).encode())
        return f"{header}.{b64(json.dumps(claims).encode())}."

    def forge_hs256_with_public_key(self, claims: dict[str, Any]) -> str:
        header = b64(json.dumps({"alg": "HS256", "typ": "JWT", "kid": KID}).encode())
        body = b64(json.dumps(claims).encode())
        sig = hmac.new(self.public_pem, f"{header}.{body}".encode(), hashlib.sha256).digest()
        return f"{header}.{body}.{b64(sig)}"


def make_client(
    kit: AuthKit, service: AnomalyService | None = None, **overrides: Any
) -> TestClient:
    app = create_app(kit.settings(**overrides))
    if service is not None:
        app.dependency_overrides[get_anomaly_service] = lambda: service
    return TestClient(app)
