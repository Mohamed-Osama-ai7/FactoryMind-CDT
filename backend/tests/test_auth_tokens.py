import time

import pytest

from factorymind.auth.principal import (
    IdentityError,
    Permission,
    Role,
    principal_from_claims,
)
from factorymind.auth.tokens import MAX_TOKEN_CHARS, TokenError, TokenVerifier
from tests.auth_helpers import ASSET, AUDIENCE, ISSUER, AuthKit


@pytest.fixture
def verifier(kit: AuthKit) -> TokenVerifier:
    return TokenVerifier(kit.settings())


def reason(verifier: TokenVerifier, token: str) -> str:
    with pytest.raises(TokenError) as info:
        verifier.verify(token)
    return info.value.reason


def test_valid_token_is_accepted(kit: AuthKit, verifier: TokenVerifier) -> None:
    claims = verifier.verify(kit.token(sub="alice"))
    assert claims["sub"] == "alice" and claims["iss"] == ISSUER


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"exp": int(time.time()) - 3600}, "expired"),
        ({"nbf": int(time.time()) + 3600}, "not_yet_valid"),
        ({"iss": "https://evil.example"}, "bad_issuer"),
        ({"aud": "someone-else"}, "bad_audience"),
    ],
)
def test_claim_checks(kit: AuthKit, verifier: TokenVerifier, override: dict, expected: str) -> None:
    assert reason(verifier, kit.sign(kit.claims(**override))) == expected


@pytest.mark.parametrize("missing", ["sub", "exp", "iss", "aud"])
def test_required_claims_must_be_present(
    kit: AuthKit, verifier: TokenVerifier, missing: str
) -> None:
    claims = kit.claims()
    del claims[missing]
    assert reason(verifier, kit.sign(claims)) == "missing_claim"


def test_alg_none_is_rejected(kit: AuthKit, verifier: TokenVerifier) -> None:
    assert reason(verifier, kit.forge_none(kit.claims())) == "bad_algorithm"


def test_hs256_signed_with_the_public_key_is_rejected(
    kit: AuthKit, verifier: TokenVerifier
) -> None:
    # classic algorithm-confusion attack: the public key used as an HMAC secret
    assert reason(verifier, kit.forge_hs256_with_public_key(kit.claims())) == "bad_algorithm"


def test_unknown_key_id_is_rejected(kit: AuthKit, verifier: TokenVerifier) -> None:
    assert reason(verifier, kit.sign(kit.claims(), kid="rotated-away")) == "unknown_key"


def test_tampered_signature_and_payload_are_rejected(kit: AuthKit, verifier: TokenVerifier) -> None:
    token = kit.token()
    head, body, sig = token.split(".")
    flipped = sig[:-4] + ("AAAA" if not sig.endswith("AAAA") else "BBBB")
    assert reason(verifier, f"{head}.{body}.{flipped}") == "bad_signature"
    other_body = kit.token(sub="mallory").split(".")[1]
    assert reason(verifier, f"{head}.{other_body}.{sig}") == "bad_signature"


@pytest.mark.parametrize("token", ["", "not-a-jwt", "a.b.c", "x" * (MAX_TOKEN_CHARS + 1)])
def test_malformed_and_oversized_tokens_are_rejected(verifier: TokenVerifier, token: str) -> None:
    assert reason(verifier, token) == "malformed"


def test_error_reason_never_contains_the_token(kit: AuthKit, verifier: TokenVerifier) -> None:
    token = kit.sign(kit.claims(exp=int(time.time()) - 3600))
    with pytest.raises(TokenError) as info:
        verifier.verify(token)
    assert token not in str(info.value) and token not in repr(info.value)


def test_verifier_requires_configuration(kit: AuthKit) -> None:
    with pytest.raises(ValueError, match="not configured"):
        TokenVerifier(kit.settings(auth_issuer=None))


# ---------------------------------------------------------------- claims -> principal


def principal(kit: AuthKit, claims: dict, **settings: object):
    return principal_from_claims(claims, kit.settings(**settings))


def test_roles_permissions_and_asset_grants(kit: AuthKit) -> None:
    p = principal(kit, kit.claims(roles=["engineer", "nonsense"]))
    assert p.roles == {Role.ENGINEER} and p.tenant_id == "acme" and p.actor_id == "user-1"
    assert p.can(Permission.INCIDENT_CREATE) and not p.can(Permission.DECISION_CREATE)
    assert p.can_access_asset(ASSET) and not p.can_access_asset("other-asset")


def test_unrecognised_roles_grant_nothing(kit: AuthKit) -> None:
    p = principal(kit, kit.claims(roles=["superuser", "root"]))
    assert p.roles == frozenset() and p.permissions == frozenset()


def test_admin_cannot_decide_separation_of_duties(kit: AuthKit) -> None:
    p = principal(kit, kit.claims(roles=["admin"]))
    assert p.can(Permission.AUDIT_READ) and not p.can(Permission.DECISION_CREATE)


def test_dotted_roles_claim_and_single_string_role(kit: AuthKit) -> None:
    p = principal(
        kit,
        kit.claims(roles=[], realm_access={"roles": ["approver"]}),
        auth_roles_claim="realm_access.roles",
    )
    assert p.roles == {Role.APPROVER}
    single = kit.claims()
    single["roles"] = "viewer"  # some providers emit a bare string
    assert principal(kit, single).roles == {Role.VIEWER}


def test_assets_are_explicit_and_filtered_to_the_registry(kit: AuthKit) -> None:
    assert principal(kit, kit.claims(assets=None)).asset_ids == frozenset()  # deny by default
    assert principal(kit, kit.claims(assets=["unknown-asset"])).asset_ids == frozenset()
    assert principal(kit, kit.claims(assets=["*"])).asset_ids == {ASSET}


def test_tenant_comes_from_the_token_or_an_explicit_single_tenant_default(kit: AuthKit) -> None:
    with pytest.raises(IdentityError) as info:
        principal(kit, kit.claims(tenant=None))
    assert info.value.reason == "no_tenant"
    assert principal(kit, kit.claims(tenant=None), default_tenant_id="solo").tenant_id == "solo"
    with pytest.raises(IdentityError):
        principal(kit, kit.claims(tenant="Bad Tenant!"))


@pytest.mark.parametrize("sub", ["", "a b", "x" * 200, "line\nbreak", 5])
def test_subject_must_be_a_safe_identifier(kit: AuthKit, sub: object) -> None:
    claims = kit.claims()
    claims["sub"] = sub
    with pytest.raises(IdentityError) as info:
        principal(kit, claims)
    assert info.value.reason == "bad_subject"


def test_audience_constant_matches_settings(kit: AuthKit) -> None:
    assert kit.settings().auth_audience == AUDIENCE
