from __future__ import annotations

import socket
from pathlib import Path

import pytest

from evicon.conformity_identification import sha256_file
from evicon.conformity_source_relation_comprehension_compatibility_approval_v1 import (
    DEFAULT_COMPATIBILITY_APPROVAL,
    DEFAULT_COMPATIBILITY_CONFIG,
    DEFAULT_COMPATIBILITY_MODULE,
    SourceRelationCompatibilityApprovalError,
    compatibility_approval_sha256,
    compatibility_config_sha256,
    load_compatibility_approval,
    load_compatibility_config,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _approval_text() -> str:
    return (_REPOSITORY_ROOT / DEFAULT_COMPATIBILITY_APPROVAL).read_text(
        encoding="utf-8"
    )


def _accepted_approval(tmp_path: Path) -> Path:
    raw = _approval_text()
    raw = raw.replace('acceptance_status = "pending"', 'acceptance_status = "accepted"')
    raw = raw.replace('accepted_by = ""', 'accepted_by = "test-reviewer"')
    raw = raw.replace('accepted_by = "researcher_user"', 'accepted_by = "test-reviewer"')
    raw = raw.replace('accepted_on = ""', 'accepted_on = "2026-08-29"')
    path = tmp_path / "accepted-compatibility-approval.toml"
    path.write_text(raw, encoding="utf-8")
    return path


def test_default_accepted_approval_is_static_and_does_not_open_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_network(*args: object, **kwargs: object) -> object:
        raise AssertionError("approval loading must not open a network connection")

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
    assert config.one_shot_http_request_cap == 1


def test_default_config_and_approval_are_hash_bound() -> None:
    config = load_compatibility_config()
    approval = load_compatibility_approval()

    assert config.compatibility_module_path == DEFAULT_COMPATIBILITY_MODULE
    assert config.compatibility_module_sha256 == sha256_file(
        DEFAULT_COMPATIBILITY_MODULE
    )
    assert approval.compatibility_config_sha256 == compatibility_config_sha256()
    assert approval.calibration_config_sha256 == config.calibration_config_sha256
    assert approval.calibration_protocol_sha256 == config.calibration_protocol_sha256
    assert approval.response_schema_sha256 == config.response_schema_sha256
    assert approval.compatibility_module_sha256 == config.compatibility_module_sha256
    assert compatibility_approval_sha256()


def test_accepted_copy_requires_identity_and_loads_when_complete(
    tmp_path: Path,
) -> None:
    path = _accepted_approval(tmp_path)

    approval = load_compatibility_approval(path)

    assert approval.acceptance_status == "accepted"
    assert approval.accepted_by == "test-reviewer"
    assert approval.accepted_on == "2026-08-29"


def test_config_hash_tampering_is_rejected(tmp_path: Path) -> None:
    config = load_compatibility_config()
    source = _REPOSITORY_ROOT / DEFAULT_COMPATIBILITY_CONFIG
    path = tmp_path / "tampered-compatibility-config.toml"
    path.write_text(
        source.read_text(encoding="utf-8").replace(
            'calibration_config_sha256 = "'
            + config.calibration_config_sha256
            + '"',
            "calibration_config_sha256 = "
            '"0000000000000000000000000000000000000000000000000000000000000000"',
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        SourceRelationCompatibilityApprovalError,
        match="source_relation_compatibility_config_binding_mismatch",
    ):
        load_compatibility_config(path)


def test_approval_hash_tampering_is_rejected(tmp_path: Path) -> None:
    path = _accepted_approval(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'compatibility_config_sha256 = "' + compatibility_config_sha256() + '"',
            "compatibility_config_sha256 = "
            '"0000000000000000000000000000000000000000000000000000000000000000"',
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        SourceRelationCompatibilityApprovalError,
        match="source_relation_compatibility_approval_binding_mismatch",
    ):
        load_compatibility_approval(path)


@pytest.mark.parametrize(
    "replacement",
    [
        'accepted_by = ""',
        'accepted_on = "not-a-date"',
        "confirm_one_shot_only = false",
    ],
)
def test_accepted_approval_rejects_incomplete_review(
    tmp_path: Path,
    replacement: str,
) -> None:
    path = _accepted_approval(tmp_path)
    if replacement.startswith("accepted_by"):
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                'accepted_by = "test-reviewer"', replacement
            ),
            encoding="utf-8",
        )
    elif replacement.startswith("accepted_on"):
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                'accepted_on = "2026-08-29"', replacement
            ),
            encoding="utf-8",
        )
    else:
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "confirm_one_shot_only = true", replacement
            ),
            encoding="utf-8",
        )

    with pytest.raises(
        SourceRelationCompatibilityApprovalError,
        match="source_relation_compatibility_approval_invalid",
    ):
        load_compatibility_approval(path)
