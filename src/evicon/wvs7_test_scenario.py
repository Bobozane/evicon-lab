from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .scenario_loader import ScenarioLoader, ScenarioLoadError
from .models import ScenarioSpec
from .models._validation import normalized_text, identifier_list

_ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_SCENARIO_ID = "wvs7-english-baseline-scenario"
TEST_SCENARIO_ID = "wvs7-english-protocol-blind-v2-test-scenario-001"


class TestScenarioAuthoringError(ValueError):
    """Stable offline authoring error."""


class TestScenarioAuthoring(BaseModel):
    __test__ = False
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1)
    scenario_path: str = Field(min_length=1)
    author_id: str = Field(min_length=1)
    authoring_version: str = Field(min_length=1)
    language: Literal["en"]
    source_kind: Literal["author_original"]
    license_note: str = Field(min_length=1)
    material_status: Literal["pending_manual_review", "accepted"]
    scenario_sha256: str = Field(min_length=64, max_length=64)
    development_scenario_id: str = Field(min_length=1)

    @field_validator(
        "scenario_id", "scenario_path", "author_id", "authoring_version",
        "license_note", "development_scenario_id",
    )
    @classmethod
    def text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("scenario_sha256")
    @classmethod
    def sha256(cls, value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
            raise ValueError("scenario_sha256 must be a SHA-256 digest")
        return value.lower()


class TestScenarioReview(BaseModel):
    __test__ = False
    model_config = ConfigDict(extra="forbid", frozen=True)

    review_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    scenario_sha256: str = Field(min_length=64, max_length=64)
    status: Literal["pending_manual_review", "accepted", "rejected"]
    reviewer_id: str = Field(min_length=1)
    reviewed_on: str = Field(min_length=1)
    approve_distinct_from_development: bool
    approve_public_synthetic_content: bool
    approve_protocol_blind_use: bool
    notes: str = Field(min_length=1)

    @field_validator("review_id", "scenario_id", "reviewer_id", "reviewed_on", "notes")
    @classmethod
    def text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("scenario_sha256")
    @classmethod
    def sha256(cls, value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
            raise ValueError("review scenario_sha256 must be a SHA-256 digest")
        return value.lower()


class TestScenarioGateReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["blocked", "ready"]
    scenario_id: str
    scenario_sha256: str
    review_status: str
    blocking_reasons: list[str] = Field(default_factory=list)


def load_authoring(path: str | Path) -> TestScenarioAuthoring:
    try:
        payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        return TestScenarioAuthoring.model_validate(payload)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        raise TestScenarioAuthoringError("test_scenario_authoring_invalid") from exc


def load_review(path: str | Path) -> TestScenarioReview:
    try:
        payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        return TestScenarioReview.model_validate(payload)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValidationError, ValueError) as exc:
        raise TestScenarioAuthoringError("test_scenario_review_invalid") from exc


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _ROOT / candidate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distinct_from_development(test: ScenarioSpec) -> bool:
    development = ScenarioLoader.load(_ROOT / "configs/scenarios/wvs7/english-baseline-scenario.toml")
    test_evidence = {(card.evidence_id, tuple(card.supports), tuple(card.contradicts)) for card in test.evidence_cards}
    dev_evidence = {(card.evidence_id, tuple(card.supports), tuple(card.contradicts)) for card in development.evidence_cards}
    return (
        test.scenario_id != development.scenario_id
        and test.title != development.title
        and test.initial_context != development.initial_context
        and test.description != development.description
        and test_evidence != dev_evidence
        and test.metadata.get("decision_options") != development.metadata.get("decision_options")
    )


def validate_test_scenario(
    authoring: TestScenarioAuthoring,
    review: TestScenarioReview,
    *,
    require_review: bool = True,
) -> TestScenarioGateReport:
    reasons: list[str] = []
    scenario_path = _resolve(authoring.scenario_path)
    try:
        actual_hash = _sha256(scenario_path)
        scenario = ScenarioLoader.load(scenario_path)
    except (OSError, ScenarioLoadError, ValueError):
        raise TestScenarioAuthoringError("test_scenario_file_invalid") from None

    if authoring.scenario_id != TEST_SCENARIO_ID or authoring.development_scenario_id != DEVELOPMENT_SCENARIO_ID:
        reasons.append("scenario_id_or_development_binding_invalid")
    if authoring.scenario_id != scenario.scenario_id or review.scenario_id != authoring.scenario_id:
        reasons.append("scenario_id_mismatch")
    if actual_hash != authoring.scenario_sha256 or actual_hash != review.scenario_sha256:
        reasons.append("test_scenario_hash_mismatch")
    metadata = scenario.metadata
    required_metadata = {
        "language": "en",
        "source_kind": "author_original",
        "material_status": "pending_manual_review",
        "synthetic": True,
        "not_wvs": True,
    }
    if any(metadata.get(key) != value for key, value in required_metadata.items()):
        reasons.append("scenario_metadata_safety_contract_invalid")
    if not _distinct_from_development(scenario):
        reasons.append("test_scenario_reuses_development_structure")
    if authoring.material_status != "pending_manual_review":
        reasons.append("authoring_status_must_start_pending")
    if review.status == "rejected":
        reasons.append("test_scenario_review_rejected")
    review_date_valid = False
    if review.status == "accepted":
        try:
            from datetime import date
            date.fromisoformat(review.reviewed_on)
            review_date_valid = True
        except ValueError:
            review_date_valid = False
    if require_review and (
        review.status != "accepted"
        or not review.approve_distinct_from_development
        or not review.approve_public_synthetic_content
        or not review.approve_protocol_blind_use
        or review.reviewer_id.startswith("TODO_")
        or not review_date_valid
    ):
        reasons.append("test_scenario_manual_review_required")
    reasons = sorted(set(reasons))
    return TestScenarioGateReport(
        status="ready" if not reasons else "blocked",
        scenario_id=authoring.scenario_id,
        scenario_sha256=actual_hash,
        review_status=review.status,
        blocking_reasons=reasons,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the English v2 confirmatory test scenario offline.")
    parser.add_argument("--authoring", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args(argv)
    try:
        authoring = load_authoring(args.authoring)
        review = load_review(args.review)
        report = validate_test_scenario(authoring, review, require_review=not args.allow_pending)
    except TestScenarioAuthoringError as exc:
        print(json.dumps({"status": "blocked", "blocking_reasons": [str(exc)]}, sort_keys=True))
        return 1
    print(json.dumps(report.model_dump(mode="json"), sort_keys=True))
    return 0 if report.status == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEVELOPMENT_SCENARIO_ID",
    "TEST_SCENARIO_ID",
    "TestScenarioAuthoring",
    "TestScenarioAuthoringError",
    "TestScenarioGateReport",
    "TestScenarioReview",
    "load_authoring",
    "load_review",
    "validate_test_scenario",
]
