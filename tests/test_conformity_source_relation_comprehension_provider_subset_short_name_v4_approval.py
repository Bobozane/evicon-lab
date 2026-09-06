from __future__ import annotations

from pathlib import Path

import pytest

from evicon.conformity_identification import sha256_file
from evicon.conformity_source_relation_comprehension_provider_subset_short_name_v4_approval import (
    DEFAULT_APPROVAL,
    DEFAULT_CONFIG,
    DEFAULT_MODULE,
    MAX_PROVIDER_SCHEMA_NAME_LENGTH,
    SHORT_PROVIDER_SCHEMA_NAME,
    SourceRelationProviderSubsetShortNameApprovalError,
    load_compatibility_approval,
    load_compatibility_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_accepted_approval(tmp_path: Path) -> Path:
    raw = (ROOT / DEFAULT_APPROVAL).read_text(encoding="utf-8")
    raw = raw.replace('accepted_by = "researcher_user"', 'accepted_by = "test-reviewer"')
    path = tmp_path / "accepted.toml"
    path.write_text(raw, encoding="utf-8")
    return path


def _write_pending_approval(tmp_path: Path) -> Path:
    raw = (ROOT / DEFAULT_APPROVAL).read_text(encoding="utf-8")
    raw = raw.replace('acceptance_status = "accepted"', 'acceptance_status = "pending"')
    raw = raw.replace('accepted_by = "researcher_user"', 'accepted_by = ""')
    raw = raw.replace('accepted_on = "2026-08-30"', 'accepted_on = ""')
    path = tmp_path / "pending.toml"
    path.write_text(raw, encoding="utf-8")
    return path


def test_scope_binds_runtime_provider_schema_and_consumed_parents(
    tmp_path: Path,
) -> None:
    config = load_compatibility_config()
    approval = load_compatibility_approval(_write_pending_approval(tmp_path))
    assert approval.acceptance_status == "pending"
    assert config.compatibility_module_sha256 == sha256_file(DEFAULT_MODULE)
    assert config.provider_schema_name == SHORT_PROVIDER_SCHEMA_NAME
    assert config.provider_schema_name_length == len(SHORT_PROVIDER_SCHEMA_NAME) == 42
    assert config.provider_schema_name_length <= MAX_PROVIDER_SCHEMA_NAME_LENGTH
    assert config.v3_provider_schema_name_length == 71
    assert config.v3_payload_delta == ("response_format.json_schema.name",)
    assert config.provider_schema_body_changed is False
    assert config.canonical_parser_changed is False
    assert config.prompt_semantics_changed is False
    assert config.public_material_changed is False
    assert config.generation_parameters_changed is False
    assert config.strict_json_schema_retained is True


def test_accepted_copy_requires_complete_review_and_correct_bindings(
    tmp_path: Path,
) -> None:
    accepted = _write_accepted_approval(tmp_path)
    approval = load_compatibility_approval(accepted)
    assert approval.acceptance_status == "accepted"
    assert approval.compatibility_config_sha256 == sha256_file(DEFAULT_CONFIG)
    assert approval.compatibility_module_sha256 == sha256_file(DEFAULT_MODULE)

    missing_identity = tmp_path / "missing-identity.toml"
    raw = accepted.read_text(encoding="utf-8").replace(
        'accepted_by = "test-reviewer"', 'accepted_by = ""'
    )
    missing_identity.write_text(raw, encoding="utf-8")
    with pytest.raises(SourceRelationProviderSubsetShortNameApprovalError):
        load_compatibility_approval(missing_identity)


@pytest.mark.parametrize(
    "replacement",
    [
        'provider_schema_name = "different_name"',
        "provider_schema_name_length = 64",
        "provider_schema_name_max_length = 41",
        'v3_payload_delta = ["response_format.json_schema.schema"]',
    ],
)
def test_invalid_short_name_scope_is_rejected(
    replacement: str, tmp_path: Path
) -> None:
    raw = (ROOT / DEFAULT_CONFIG).read_text(encoding="utf-8")
    if replacement.startswith("provider_schema_name ="):
        raw = raw.replace(
            'provider_schema_name = "source_relation_comprehension_v1_subset_v4"',
            replacement,
        )
    elif replacement.startswith("provider_schema_name_length"):
        raw = raw.replace("provider_schema_name_length = 42", replacement)
    elif replacement.startswith("provider_schema_name_max_length"):
        raw = raw.replace("provider_schema_name_max_length = 64", replacement)
    else:
        raw = raw.replace(
            'v3_payload_delta = ["response_format.json_schema.name"]', replacement
        )
    path = tmp_path / "invalid-config.toml"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(SourceRelationProviderSubsetShortNameApprovalError):
        load_compatibility_config(path)
