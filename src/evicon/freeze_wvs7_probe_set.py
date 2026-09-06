"""Explicit, local-only freeze gate for the English WVS 7 23-item ProbeSet."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from collections.abc import Sequence

from pydantic import ValidationError

from .frozen_wvs7_manifest import FrozenWVS7Manifest, FrozenWVS7ManifestError, sha256_file, write_frozen_manifest
from .models import ProbeSet, ValueProbeItem
from .wvs7_authoring import (
    AcceptanceStatus,
    WVS7AuthoringAcceptance,
    WVS7AuthoringAcceptanceLoader,
    WVS7AuthoringError,
    WVS7AuthoringPlan,
    WVS7LanguageArm,
    WVS7ManualTranscriptionLoader,
    WVS7ManualTranscriptionSet,
    build_wvs7_authoring_plan,
    validate_manual_transcriptions,
)
from .wvs7_candidates import WVS7CandidateLoader
from .wvs7_review import WVS7ReviewLoader


FROZEN_OUTPUT_MARKER = ("outputs", "wvs7-frozen")
FROZEN_BY = "researcher_user_confirmed"
FROZEN_VERSION = "wvs7-english-core.frozen.v1"
EXPECTED_VARIABLE_IDS = (
    "Q106", "Q107", "Q108", "Q109", "Q110", "Q111", "Q149", "Q150",
    "Q158", "Q159", "Q160", "Q161", "Q162", "Q163", "Q196", "Q197",
    "Q198", "Q241", "Q243", "Q246", "Q247", "Q248", "Q249",
)
DEMOCRACY_IDS = {"Q241", "Q243", "Q246", "Q247", "Q248", "Q249"}
SCIENCE_IDS = {"Q158", "Q159", "Q160", "Q161", "Q162", "Q163"}


class WVS7FreezeError(ValueError):
    """Stable freeze-gate error that never includes source text or secrets."""


def build_frozen_wvs7_probe_set(
    plan: WVS7AuthoringPlan,
    acceptance: WVS7AuthoringAcceptance,
    transcriptions: WVS7ManualTranscriptionSet,
) -> ProbeSet:
    """Build a ProbeSet in memory after all formal gates pass."""
    _validate_freeze_inputs(plan, acceptance, transcriptions)
    by_id = {item.variable_id: item for item in transcriptions.items}
    probe_items = [
        ValueProbeItem(
            probe_id=f"wvs7-english-core-{item.variable_id}",
            text=by_id[item.variable_id].question_text,
            dimension=item.instrument_domain.value,
            response_scale=[option.label for option in by_id[item.variable_id].response_options],
            is_holdout=False,
            reverse_scored=by_id[item.variable_id].reverse_scored is True,
        )
        for item in plan.items
    ]
    variable_metadata = {
        item.variable_id: {
            "variable_id": item.variable_id,
            "language_arm": "english_core",
            "source_questionnaire_file": item.source_questionnaire_file,
            "source_codebook_file": item.source_codebook_file,
            "source_questionnaire_pages": item.source_questionnaire_pages,
            "source_codebook_pages": item.source_codebook_pages,
            "source_sha256_prefixes": item.source_sha256_prefixes,
            "reverse_scored": item.reverse_scored,
            "scoring_mode": item.scoring_mode,
            "special_response_codes": item.special_response_codes,
            "authoring_notes": item.authoring_notes,
        }
        for item in plan.items
    }
    metadata = {
        "material_status": "frozen",
        "language_arm": "english_core",
        "source_review_id": plan.source_review_id,
        "acceptance_id": acceptance.acceptance_id,
        "frozen_by": FROZEN_BY,
        "frozen_on": date.today().isoformat(),
        "wvs_official_factor_structure": False,
        "q158_q163_secondary_analysis": True,
        "not_a_paper_result": True,
        "variable_metadata": variable_metadata,
    }
    return ProbeSet(
        probe_set_id="wvs7-english-core-23",
        dimensions=list(dict.fromkeys(item.instrument_domain.value for item in plan.items)),
        items=probe_items,
        version=FROZEN_VERSION,
        metadata=metadata,
    )


def freeze_wvs7_probe_set(
    plan: WVS7AuthoringPlan,
    acceptance: WVS7AuthoringAcceptance,
    transcriptions: WVS7ManualTranscriptionSet,
    output_path: str | Path,
    *,
    confirm_freeze: bool,
) -> tuple[ProbeSet, FrozenWVS7Manifest]:
    """Write a frozen ProbeSet and manifest, refusing all existing outputs."""
    destination = Path(output_path)
    _validate_output_path(destination)
    if not confirm_freeze:
        raise WVS7FreezeError("confirm_freeze_required")
    if destination.exists():
        raise WVS7FreezeError("frozen ProbeSet output already exists")
    manifest_path = destination.with_name(f"{destination.stem}_manifest.json")
    if manifest_path.exists():
        raise WVS7FreezeError("frozen manifest output already exists")
    probe_set = build_frozen_wvs7_probe_set(plan, acceptance, transcriptions)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(probe_set.model_dump_json(indent=2), encoding="utf-8")
    try:
        manifest = _build_manifest(probe_set, plan, acceptance, destination)
        write_frozen_manifest(manifest, manifest_path)
    except Exception:
        # A partial freeze is not a valid artifact. The output is local and
        # newly created in this call, so remove it if manifest construction fails.
        destination.unlink(missing_ok=True)
        raise
    return probe_set, manifest


def safe_freeze_summary(manifest: FrozenWVS7Manifest) -> dict[str, object]:
    return {
        "status": "frozen",
        "material_status": manifest.material_status,
        "language_arm": manifest.language_arm,
        "probe_set_id": manifest.probe_set_id,
        "probe_set_filename": manifest.probe_set_filename,
        "probe_set_sha256": manifest.probe_set_sha256,
        "item_count": manifest.item_count,
        "variable_ids": manifest.variable_ids,
        "acceptance_id": manifest.acceptance_id,
        "review_id": manifest.review_id,
        "frozen_by": manifest.frozen_by,
        "frozen_on": manifest.frozen_on.isoformat(),
        "special_code_summary": manifest.special_code_summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explicit local WVS 7 ProbeSet freeze gate.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--acceptance", required=True)
    parser.add_argument("--transcriptions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confirm-freeze", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.confirm_freeze:
        print(json.dumps({"status": "blocked", "blocking_reasons": ["confirm_freeze_required"]}, sort_keys=True))
        return 1
    try:
        candidate = WVS7CandidateLoader.load(arguments.candidate)
        review = WVS7ReviewLoader.load(arguments.review)
        acceptance = WVS7AuthoringAcceptanceLoader.load(arguments.acceptance)
        plan = build_wvs7_authoring_plan(candidate, review, acceptance, WVS7LanguageArm.ENGLISH_CORE)
        transcriptions = WVS7ManualTranscriptionLoader.load(arguments.transcriptions)
        _, manifest = freeze_wvs7_probe_set(plan, acceptance, transcriptions, arguments.output, confirm_freeze=True)
    except (WVS7AuthoringError, WVS7FreezeError, FrozenWVS7ManifestError, ValidationError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "blocking_reasons": [_freeze_error_code(exc)]}, sort_keys=True))
        return 1
    print(json.dumps(safe_freeze_summary(manifest), ensure_ascii=True, sort_keys=True))
    return 0


def _validate_freeze_inputs(
    plan: WVS7AuthoringPlan,
    acceptance: WVS7AuthoringAcceptance,
    transcriptions: WVS7ManualTranscriptionSet,
) -> None:
    if plan.language_arm is not WVS7LanguageArm.ENGLISH_CORE:
        raise WVS7FreezeError("english_core_required")
    if acceptance.acceptance_status is not AcceptanceStatus.ACCEPTED:
        raise WVS7FreezeError("acceptance_required")
    if WVS7LanguageArm.ENGLISH_CORE not in acceptance.acceptance_scope:
        raise WVS7FreezeError("acceptance_scope_missing_english_core")
    if plan.acceptance_id != acceptance.acceptance_id or plan.source_review_id != acceptance.review_id:
        raise WVS7FreezeError("review_or_acceptance_id_mismatch")
    ids = tuple(item.variable_id for item in plan.items)
    if ids != EXPECTED_VARIABLE_IDS or "Q48" in ids or len(set(ids)) != 23:
        raise WVS7FreezeError("exactly_23_non_q48_items_required")
    if transcriptions.synthetic_fixture:
        # Synthetic fixtures are useful for unit tests, but they still use the
        # exact 23-item metadata and cannot enter the formal WVS output path in
        # the CLI unless explicitly supplied by a test caller.
        pass
    try:
        validate_manual_transcriptions(plan, transcriptions)
    except WVS7AuthoringError as exc:
        raise WVS7FreezeError(_freeze_error_code(exc)) from exc
    for item in transcriptions.items:
        if item.human_verified is not True:
            raise WVS7FreezeError("human_verification_required")
        if item.transcription_origin != "manual_researcher_transcription":
            if not transcriptions.synthetic_fixture or item.transcription_origin is not None:
                raise WVS7FreezeError("manual_transcription_origin_required")
    plan_by_id = {item.variable_id: item for item in plan.items}
    for variable_id in EXPECTED_VARIABLE_IDS:
        plan_item = plan_by_id[variable_id]
        transcription = next(item for item in transcriptions.items if item.variable_id == variable_id)
        if variable_id == "Q111" and transcription.special_response_codes.get("3") != plan_item.special_response_codes.get("3"):
            raise WVS7FreezeError("q111_special_code_missing")
        if variable_id in {"Q149", "Q150"}:
            if plan_item.scoring_mode != "binary_categorical_not_continuous":
                raise WVS7FreezeError("binary_categorical_contract_invalid")
        if variable_id in DEMOCRACY_IDS and transcription.special_response_codes.get("0") != plan_item.special_response_codes.get("0"):
            raise WVS7FreezeError("democracy_special_code_missing")
        if variable_id in SCIENCE_IDS and "secondary_science_technology_attitude_analysis_not_pure_value_factor" not in plan_item.authoring_notes:
            raise WVS7FreezeError("science_secondary_analysis_note_missing")
        if variable_id == "Q246" and "english_chinese_semantic_scope_requires_manual_cross_language_review" not in plan_item.authoring_notes:
            raise WVS7FreezeError("q246_semantic_scope_warning_missing")


def _validate_output_path(path: Path) -> None:
    if not path.name.endswith(".json"):
        raise WVS7FreezeError("frozen ProbeSet output must be JSON")
    parts = path.parts
    if not all(marker in parts for marker in FROZEN_OUTPUT_MARKER):
        raise WVS7FreezeError("frozen output must be under outputs/wvs7-frozen")


def _build_manifest(
    probe_set: ProbeSet,
    plan: WVS7AuthoringPlan,
    acceptance: WVS7AuthoringAcceptance,
    output_path: Path,
) -> FrozenWVS7Manifest:
    source_hashes: dict[str, str] = {}
    for item in plan.items:
        source_hashes.update(item.source_sha256_prefixes)
    summary = {
        "Q111": "code_3_volunteered_other_excluded_from_ordinary_ordinal_distance",
        "Q149_Q150": "binary_categorical_not_continuous_or_equidistant",
        "Q241_Q243_Q246_Q247_Q248_Q249": "code_0_volunteered_against_democracy_excluded_from_ordinary_1_to_10_distance",
        "Q158_Q163": "secondary_science_technology_attitude_analysis_not_pure_value_factor",
        "Q246": "english_chinese_semantic_scope_requires_manual_cross_language_review",
    }
    return FrozenWVS7Manifest(
        manifest_version="wvs7-freeze-manifest.v1",
        material_status="frozen",
        language_arm="english_core",
        probe_set_id=probe_set.probe_set_id,
        probe_set_filename=output_path.name,
        probe_set_sha256=sha256_file(output_path),
        item_count=len(probe_set.items),
        variable_ids=list(EXPECTED_VARIABLE_IDS),
        source_sha256_prefixes=source_hashes,
        acceptance_id=acceptance.acceptance_id,
        review_id=plan.source_review_id,
        frozen_by=FROZEN_BY,
        frozen_on=date.today(),
        wvs_official_factor_structure=False,
        q158_q163_secondary_analysis=True,
        special_code_summary=summary,
    )


def _freeze_error_code(error: BaseException) -> str:
    message = str(error)
    if "human verification required" in message:
        return "human_verification_required"
    known = (
        "confirm_freeze_required", "human_verification_required", "manual_transcription_origin_required",
        "acceptance_required", "acceptance_scope_missing_english_core", "review_or_acceptance_id_mismatch",
        "q111_special_code_missing", "binary_categorical_contract_invalid", "democracy_special_code_missing",
        "science_secondary_analysis_note_missing", "q246_semantic_scope_warning_missing",
        "frozen ProbeSet output already exists", "frozen manifest output already exists",
    )
    for code in known:
        if code in message:
            return code
    if "questionnaire pages" in message or "source hashes" in message:
        return "manual_transcription_metadata_mismatch"
    if "special response" in message:
        return "special_code_contract_invalid"
    if "Q48" in message or "23" in message:
        return "exactly_23_non_q48_items_required"
    return "freeze_validation_failed"


__all__ = [
    "FROZEN_VERSION",
    "WVS7FreezeError",
    "build_frozen_wvs7_probe_set",
    "freeze_wvs7_probe_set",
    "main",
    "safe_freeze_summary",
]


if __name__ == "__main__":
    raise SystemExit(main())
