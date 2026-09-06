from __future__ import annotations

from pathlib import Path

import pytest

from evicon.conformity_identification import IdentificationError
from evicon.conformity_source_behavior_analysis_lock import load_analysis_lock
from evicon.conformity_source_behavior_descriptive_analysis import (
    aggregate_safe_case_audits,
    safe_summary_json,
)
from evicon.conformity_source_behavior_qualification import (
    AdoptionDecision,
    SafeBehaviorCaseAudit,
    SharingDecision,
)
from evicon.conformity_source_manipulation import SourceProjection


def _audit(
    scenario: str,
    projection: SourceProjection,
    adoption: AdoptionDecision,
    sharing: SharingDecision,
) -> SafeBehaviorCaseAudit:
    return SafeBehaviorCaseAudit(
        case_id=f"fake-{scenario}-{projection.value}",
        scenario_id=scenario,
        projection=projection,
        parser_status="valid",
        adoption_decision=adoption,
        sharing_decision=sharing,
        used_content_count=0,
        visible_root_count={
            SourceProjection.SOURCE_FREE: 0,
            SourceProjection.SAME_ROOT: 1,
            SourceProjection.INDEPENDENT_ROOTS: 2,
        }[projection],
    )


def _fake_audits() -> tuple[SafeBehaviorCaseAudit, ...]:
    # adoption: same-source = [+1, 0, 0, -1]; independent = [0, +1, 0, 0]
    # sharing: same-source = [+1, +1, 0, -1]; independent = [0, 0, 0, 0]
    values = {
        "s1": (
            (AdoptionDecision.WITHHOLD, SharingDecision.DO_NOT_SHARE),
            (AdoptionDecision.ADOPT_VISIBLE_CLAIM, SharingDecision.SHARE_VISIBLE_CONTENT),
            (AdoptionDecision.ADOPT_VISIBLE_CLAIM, SharingDecision.SHARE_VISIBLE_CONTENT),
        ),
        "s2": (
            (AdoptionDecision.WITHHOLD, SharingDecision.DO_NOT_SHARE),
            (AdoptionDecision.WITHHOLD, SharingDecision.SHARE_VISIBLE_CONTENT),
            (AdoptionDecision.ADOPT_VISIBLE_CLAIM, SharingDecision.SHARE_VISIBLE_CONTENT),
        ),
        "s3": (
            (AdoptionDecision.ADOPT_VISIBLE_CLAIM, SharingDecision.DO_NOT_SHARE),
            (AdoptionDecision.ADOPT_VISIBLE_CLAIM, SharingDecision.DO_NOT_SHARE),
            (AdoptionDecision.ADOPT_VISIBLE_CLAIM, SharingDecision.DO_NOT_SHARE),
        ),
        "s4": (
            (AdoptionDecision.ADOPT_VISIBLE_CLAIM, SharingDecision.SHARE_VISIBLE_CONTENT),
            (AdoptionDecision.WITHHOLD, SharingDecision.DO_NOT_SHARE),
            (AdoptionDecision.WITHHOLD, SharingDecision.DO_NOT_SHARE),
        ),
    }
    return tuple(
        _audit(scenario, projection, adoption, sharing)
        for scenario, rows in values.items()
        for projection, (adoption, sharing) in zip(SourceProjection, rows, strict=True)
    )


def test_fake_audits_produce_only_locked_aggregate_summaries() -> None:
    result = aggregate_safe_case_audits(_fake_audits(), plan=load_analysis_lock())
    observed = [item.model_dump() for item in result.paired_summaries]
    assert observed == [
        {
            "outcome": "adoption_decision",
            "contrast_id": "same_root_minus_source_free",
            "eligible_scenario_count": 4,
            "positive_difference_count": 1,
            "zero_difference_count": 2,
            "negative_difference_count": 1,
            "mean_paired_difference": 0.0,
        },
        {
            "outcome": "adoption_decision",
            "contrast_id": "independent_roots_minus_same_root",
            "eligible_scenario_count": 4,
            "positive_difference_count": 1,
            "zero_difference_count": 3,
            "negative_difference_count": 0,
            "mean_paired_difference": 0.25,
        },
        {
            "outcome": "sharing_decision",
            "contrast_id": "same_root_minus_source_free",
            "eligible_scenario_count": 4,
            "positive_difference_count": 2,
            "zero_difference_count": 1,
            "negative_difference_count": 1,
            "mean_paired_difference": 0.25,
        },
        {
            "outcome": "sharing_decision",
            "contrast_id": "independent_roots_minus_same_root",
            "eligible_scenario_count": 4,
            "positive_difference_count": 0,
            "zero_difference_count": 4,
            "negative_difference_count": 0,
            "mean_paired_difference": 0.0,
        },
    ]
    assert result.inferential_tests_run is False
    assert result.case_level_values_included is False
    assert result.no_causal_conclusion is True


def test_aggregation_is_pure_and_does_not_read_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = load_analysis_lock()

    def forbidden(*args, **kwargs):
        raise AssertionError("the aggregate core must not read any file")

    monkeypatch.setattr(Path, "read_text", forbidden)
    result = aggregate_safe_case_audits(_fake_audits(), plan=plan)
    assert result.scenario_count == 4


def test_aggregation_fails_closed_on_missing_case() -> None:
    with pytest.raises(IdentificationError) as error:
        aggregate_safe_case_audits(_fake_audits()[:-1], plan=load_analysis_lock())
    assert error.value.code == "source_behavior_analysis_case_count_mismatch"


def test_aggregation_fails_closed_on_duplicate_case_id() -> None:
    audits = list(_fake_audits())
    audits[-1] = audits[-1].model_copy(update={"case_id": audits[0].case_id})
    with pytest.raises(IdentificationError) as error:
        aggregate_safe_case_audits(audits, plan=load_analysis_lock())
    assert error.value.code == "source_behavior_analysis_duplicate_case_id"


def test_aggregation_fails_closed_on_duplicate_projection_coordinate() -> None:
    audits = list(_fake_audits())
    audits[-1] = audits[-1].model_copy(
        update={
            "case_id": "fake-s3-independent-roots-extra",
            "scenario_id": "s3",
        }
    )
    with pytest.raises(IdentificationError) as error:
        aggregate_safe_case_audits(audits, plan=load_analysis_lock())
    assert error.value.code == "source_behavior_analysis_duplicate_scenario_projection"


def test_serialized_summary_has_no_case_coordinates_or_raw_decisions() -> None:
    payload = safe_summary_json(
        aggregate_safe_case_audits(_fake_audits(), plan=load_analysis_lock())
    )
    for forbidden in (
        "case_id",
        "scenario_id",
        "fake-s",
        "adopt_visible_claim",
        "withhold",
        "share_visible_content",
        "do_not_share",
        "system_prompt",
        "raw_response",
        "api_key",
        '"evaluator_private_truth":',
    ):
        assert forbidden not in payload
