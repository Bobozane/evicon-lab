"""Safe CLI for the WVS Wave 7 pre-freeze authoring gate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .wvs7_authoring import (
    AuthoringGateStatus,
    WVS7AuthoringError,
    WVS7LanguageArm,
    load_and_evaluate_wvs7_authoring_gate,
    safe_gate_summary,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a local WVS Wave 7 authoring gate without PDF access or network calls.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--acceptance", required=True)
    parser.add_argument("--language", required=True, choices=("english", "chinese"))
    parser.add_argument("--transcriptions", help="Explicit ignored local transcription TOML input.")
    parser.add_argument("--confirm-freeze", action="store_true", help="Never writes a formal WVS ProbeSet in this CLI.")
    parser.add_argument("--strict", action="store_true", help="Return nonzero when the requested arm is blocked.")
    arguments = parser.parse_args(argv)
    language = WVS7LanguageArm.ENGLISH_CORE if arguments.language == "english" else WVS7LanguageArm.CHINESE_APPLIED
    try:
        _, report = load_and_evaluate_wvs7_authoring_gate(
            arguments.candidate, arguments.review, arguments.acceptance, language, arguments.transcriptions
        )
    except WVS7AuthoringError as error:
        error_code = _error_code(error)
        status = "blocked" if arguments.transcriptions and error_code in {"manual_transcription_invalid", "human_verification_required"} else "invalid"
        print(json.dumps({"status": status, "blocking_reasons": [error_code]}, sort_keys=True))
        return 1
    payload = safe_gate_summary(report)
    if arguments.confirm_freeze:
        payload["blocking_reasons"] = [*payload["blocking_reasons"], "formal_wvs_freeze_not_enabled_in_cli"]
        payload["status"] = "blocked"
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 1 if arguments.strict and report.status is AuthoringGateStatus.BLOCKED else 0


def _error_code(error: WVS7AuthoringError) -> str:
    message = str(error)
    if "human verification required" in message:
        return "human_verification_required"
    if "acceptance" in message:
        return "acceptance_contract_invalid"
    if "transcription" in message:
        return "manual_transcription_invalid"
    if "review" in message:
        return "preliminary_review_invalid"
    return "authoring_validation_failed"


if __name__ == "__main__":
    raise SystemExit(main())
