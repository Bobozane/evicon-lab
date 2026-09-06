from __future__ import annotations

import socket
from pathlib import Path

import pytest

from evicon.conformity_identification import sha256_file
from evicon.conformity_source_relation_comprehension_provider_subset_timeout_v3_approval import (
    DEFAULT_APPROVAL,
    DEFAULT_CONFIG,
    DEFAULT_MODULE,
    PARENT_APPROVAL,
    PARENT_ATTEMPT_CLAIM,
    PARENT_CONFIG,
    PARENT_MODULE,
    PROVIDER_SCHEMA_MODULE,
    SourceRelationProviderSubsetTimeoutApprovalError,
    compatibility_approval_sha256,
    compatibility_config_sha256,
    load_compatibility_approval,
    load_compatibility_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _approval_text() -> str:
    return (ROOT / DEFAULT_APPROVAL).read_text(encoding="utf-8")


def _accepted_copy(tmp_path: Path) -> Path:
    raw = _approval_text()
    raw = raw.replace('accepted_by = "researcher_user"', 'accepted_by = "test-reviewer"')
    path = tmp_path / "accepted-timeout-amendment.toml"
    path.write_text(raw, encoding="utf-8")
    return path


def test_default_review_is_accepted_and_cannot_open_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_network(*args: object, **kwargs: object) -> object:
        raise AssertionError("approval loading must remain offline")

    monkeypatch.setattr(socket, "create_connection", forbidden_network)
    config = load_compatibility_config()
    approval = load_compatibility_approval()

    assert approval.acceptance_status == "accepted"
    assert approval.accepted_by == "researcher_user"
    assert approval.accepted_on == "2026-08-29"
    assert approval.persistent_network_authorization_granted is False
    assert approval.explicit_command_authorization_required is True
    assert config.network_enabled_by_default is False
    assert config.explicit_allow_network_required is True


def test_v3_changes_only_timeout_and_preserves_one_shot_parameters() -> None:
    config = load_compatibility_config()

    assert config.timeout_only_amendment is True
    assert config.parent_timeout_seconds == 5.0
    assert config.timeout_seconds == 15.0
    assert config.one_shot_http_request_cap == 1
    assert config.max_retries == 0
    assert config.max_tokens == 128
    assert config.temperature == 0.0
    assert config.seed == 20261401
    assert config.reasoning_effort == "none"
    assert config.provider_schema_changed is False
    assert config.canonical_parser_changed is False
    assert config.prompt_semantics_changed is False
    assert config.public_material_changed is False
    assert config.generation_parameters_other_than_timeout_changed is False


def test_parent_timeout_attempt_and_all_hash_bindings_are_immutable() -> None:
    config = load_compatibility_config()
    approval = load_compatibility_approval()

    assert config.parent_attempt_status == "provider_error"
    assert config.parent_failure_category == "timeout"
    assert config.parent_attempt_count == 1
    assert config.parent_attempt_claim_empty is True
    assert config.parent_receipt_created is False
    assert config.parent_compatibility_config_sha256 == sha256_file(PARENT_CONFIG)
    assert config.parent_compatibility_approval_sha256 == sha256_file(PARENT_APPROVAL)
    assert config.parent_compatibility_module_sha256 == sha256_file(PARENT_MODULE)
    assert config.parent_attempt_claim_sha256 == sha256_file(PARENT_ATTEMPT_CLAIM)
    assert config.provider_schema_module_sha256 == sha256_file(PROVIDER_SCHEMA_MODULE)
    assert config.compatibility_module_sha256 == sha256_file(DEFAULT_MODULE)
    assert approval.compatibility_config_sha256 == compatibility_config_sha256()
    assert compatibility_approval_sha256()


def test_accepted_copy_loads_only_after_complete_review(tmp_path: Path) -> None:
    accepted = load_compatibility_approval(_accepted_copy(tmp_path))
    assert accepted.acceptance_status == "accepted"
    assert accepted.accepted_by == "test-reviewer"
    assert accepted.accepted_on == "2026-08-29"


@pytest.mark.parametrize(
    "field",
    [
        "confirm_timeout_only_change",
        "confirm_one_shot_only",
        "confirm_parent_timeout_and_no_receipt",
        "confirm_not_effect_or_causal_evidence",
    ],
)
def test_accepted_copy_requires_every_confirmation(
    tmp_path: Path,
    field: str,
) -> None:
    path = _accepted_copy(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            f"{field} = true",
            f"{field} = false",
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        SourceRelationProviderSubsetTimeoutApprovalError,
        match="source_relation_provider_subset_timeout_approval_invalid",
    ):
        load_compatibility_approval(path)


def test_config_parent_hash_tampering_is_rejected(tmp_path: Path) -> None:
    source = (ROOT / DEFAULT_CONFIG).read_text(encoding="utf-8")
    path = tmp_path / "tampered-config.toml"
    path.write_text(
        source.replace(
            f'parent_compatibility_config_sha256 = "{sha256_file(PARENT_CONFIG)}"',
            'parent_compatibility_config_sha256 = "'
            + ("0" * 64)
            + '"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        SourceRelationProviderSubsetTimeoutApprovalError,
        match="source_relation_provider_subset_timeout_binding_mismatch",
    ):
        load_compatibility_config(path)


def test_approval_config_hash_tampering_is_rejected(tmp_path: Path) -> None:
    path = _accepted_copy(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            f'compatibility_config_sha256 = "{compatibility_config_sha256()}"',
            'compatibility_config_sha256 = "' + ("0" * 64) + '"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        SourceRelationProviderSubsetTimeoutApprovalError,
        match="source_relation_provider_subset_timeout_approval_binding_mismatch",
    ):
        load_compatibility_approval(path)
