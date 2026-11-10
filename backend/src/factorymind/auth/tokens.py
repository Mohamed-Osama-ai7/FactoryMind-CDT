"""Access-token verification against an OIDC provider's signing keys.

Rules: only asymmetric algorithms from an allowlist; the algorithm comes from configuration and not
from the token (so ``alg=none`` and HS256-with-public-key confusion cannot succeed); issuer,
audience, expiry and subject are required; keys are looked up by ``kid`` from a static JWKS file or
a cached JWKS URL. Failures carry a short reason code, never token contents.
"""

from __future__ import annotations

from typing import Any, Protocol

import jwt
from jwt import PyJWKClient, PyJWKSet

from factorymind.config import Settings

MAX_TOKEN_CHARS = 8192


class TokenError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class KeyProvider(Protocol):
    def key_for(self, token: str) -> Any: ...


class StaticJwks:
    def __init__(self, jwks_json: str) -> None:
        self._set = PyJWKSet.from_json(jwks_json)

    def key_for(self, token: str) -> Any:
        kid = jwt.get_unverified_header(token).get("kid")
        if not isinstance(kid, str):
            raise TokenError("no_kid")
        for key in self._set.keys:
            if key.key_id == kid:
                return key.key
        raise TokenError("unknown_key")


class RemoteJwks:
    def __init__(self, url: str) -> None:
        self._client = PyJWKClient(
            url, cache_keys=True, max_cached_keys=16, lifespan=300, timeout=5
        )

    def key_for(self, token: str) -> Any:
        try:
            return self._client.get_signing_key_from_jwt(token).key
        except jwt.PyJWKClientError as exc:
            raise TokenError("unknown_key") from exc


class TokenVerifier:
    def __init__(self, settings: Settings, keys: KeyProvider | None = None) -> None:
        if not settings.auth_configured or settings.auth_issuer is None:
            raise ValueError("authentication is not configured")
        self._issuer = settings.auth_issuer
        self._audience = settings.auth_audience
        self._algorithms = list(settings.auth_algorithms)
        self._leeway = settings.auth_leeway_seconds
        if keys is not None:
            self._keys = keys
        elif settings.auth_jwks_file is not None:
            self._keys = StaticJwks(settings.auth_jwks_file.read_text(encoding="utf-8"))
        elif settings.auth_jwks_url is not None:
            self._keys = RemoteJwks(settings.auth_jwks_url)
        else:  # pragma: no cover - guarded by auth_configured
            raise ValueError("no key source configured")

    def verify(self, token: str) -> dict[str, Any]:
        if not token or len(token) > MAX_TOKEN_CHARS:
            raise TokenError("malformed")
        try:
            header = jwt.get_unverified_header(token)
            if header.get("alg") not in self._algorithms:
                raise TokenError("bad_algorithm")
            key = self._keys.key_for(token)
            claims: dict[str, Any] = jwt.decode(
                token,
                key=key,
                algorithms=self._algorithms,
                audience=self._audience,
                issuer=self._issuer,
                leeway=self._leeway,
                options={"require": ["exp", "sub", "iss", "aud"]},
            )
        except TokenError:
            raise
        except jwt.ExpiredSignatureError as exc:
            raise TokenError("expired") from exc
        except jwt.ImmatureSignatureError as exc:
            raise TokenError("not_yet_valid") from exc
        except jwt.InvalidIssuerError as exc:
            raise TokenError("bad_issuer") from exc
        except jwt.InvalidAudienceError as exc:
            raise TokenError("bad_audience") from exc
        except jwt.InvalidSignatureError as exc:
            raise TokenError("bad_signature") from exc
        except jwt.MissingRequiredClaimError as exc:
            raise TokenError("missing_claim") from exc
        except jwt.InvalidAlgorithmError as exc:
            raise TokenError("bad_algorithm") from exc
        except jwt.DecodeError as exc:
            raise TokenError("malformed") from exc
        except jwt.InvalidTokenError as exc:
            raise TokenError("invalid") from exc
        return claims
