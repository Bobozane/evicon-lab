from __future__ import annotations

from pathlib import Path

import pytest

from evicon.conformity_identification import sha256_file
from evicon.conformity_source_relation_comprehension_provider_subset_v2_approval import (
    DEFAULT_APPROVAL,
    DEFAULT_CONFIG,
    DEFAULT_MODULE,
    SourceRelationProviderSubsetApprovalError,
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
    raw = raw.replace('acceptance_status = "pending"', 'acceptance_status = "accepted"')
    raw = raw.replace('accepted_by = ""', 'accepted_by = "test-reviewer"')
    raw = raw.replace('accepted_by = "researcher_user"', 'accepted_by = "test-reviewer"')
    raw = raw.replace('accepted_on = ""', 'accepted_on = "2026-08-29"')
    path = tmp_path / "accepted.toml"
    path.write_text(raw, encoding="utf-8")
    return path


def test_default_amendment_is_accepted_and_parent_failure_is_bound() -> None:
    config = load_compatibility_config()
    approval = load_compatibility_approval()
    assert approval.acceptance_status == "accepted"
    assert approval.accepted_by == "researcher_user"
    assert approval.accepted_on == "2026-08-29"
    assert config.parent_failure_category == "response_format_unsupported"
    assert config.parent_receipt_created is False
    assert config.network_enabled_by_default is False
    assert config.explicit_allow_network_required is True
    assert config.one_shot_http_request_cap == 1


def test_hash_bindings_are_current() -> None:
    config = load_compatibility_config()
    approval = load_compatibility_approval()
    assert config.compatibility_module_path == DEFAULT_MODULE
    assert config.compatibility_module_sha256 == sha256_file(DEFAULT_MODULE)
    assert approval.compatibility_config_sha256 == compatibility_config_sha256()
    assert approval.calibration_config_sha256 == config.calibration_config_sha256
    assert approval.provider_schema_sha256 == config.provider_schema_sha256
    assert approval.compatibility_module_sha256 == config.compatibility_module_sha256
    assert compatibility_approval_sha256()


def test_accepted_copy_requires_all_confirmations(tmp_path: Path) -> None:
    path = _accepted_copy(tmp_path)
    accepted = load_compatibility_approval(path)
    assert accepted.acceptance_status == "accepted"
    assert accepted.accepted_by == "test-reviewer"

    tampered = path.read_text(encoding="utf-8").replace(
        "confirm_provider_subset_only = true",
        "confirm_provider_subset_only = false",
    )
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(
        SourceRelationProviderSubsetApprovalError,
        match="source_relation_provider_subset_approval_invalid",
    ):
        load_compatibility_approval(path)


def test_config_tampering_is_rejected(tmp_path: Path) -> None:
    source = (ROOT / DEFAULT_CONFIG).read_text(encoding="utf-8")
    path = tmp_path / "tampered.toml"
    path.write_text(
        source.replace(
            'provider_schema_sha256 = "',
            'provider_schema_sha256 = "0000000000000000000000000000000000000000000000000000000000000000" # ',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        SourceRelationProviderSubsetApprovalError,
        match="source_relation_provider_subset_config_invalid",
    ):
        load_compatibility_config(path)
