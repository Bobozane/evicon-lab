"""Offline authoring/validation entry point for the v2 test-scenario gate."""

from __future__ import annotations

import argparse
import json

from .wvs7_confirmatory_study import (
    ConfirmatoryStudyError,
    load_confirmatory_manifest,
    validate_confirmatory_manifest,
)
from .wvs7_test_scenario import (
    TestScenarioAuthoringError,
    load_authoring,
    load_review,
    validate_test_scenario,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate confirmatory test-scenario authoring and its future manifest without providers."
    )
    parser.add_argument("--authoring", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--manifest")
    args = parser.parse_args(argv)
    try:
        scenario = validate_test_scenario(
            load_authoring(args.authoring),
            load_review(args.review),
            require_review=True,
        )
        payload: dict[str, object] = {
            "status": scenario.status,
            "scenario_id": scenario.scenario_id,
            "scenario_sha256": scenario.scenario_sha256,
            "review_status": scenario.review_status,
            "blocking_reasons": list(scenario.blocking_reasons),
        }
        if args.manifest:
            study = validate_confirmatory_manifest(load_confirmatory_manifest(args.manifest))
            payload["confirmatory_status"] = study.status
            payload["blocking_reasons"] = sorted(
                set(payload["blocking_reasons"]) | set(study.blocking_reasons)
            )
            payload["status"] = "ready" if scenario.status == "ready" and study.status == "ready_for_confirmatory" else "blocked"
    except (TestScenarioAuthoringError, ConfirmatoryStudyError, OSError, ValueError):
        print(json.dumps({"status": "blocked", "blocking_reasons": ["confirmatory_authoring_validation_failed"]}, sort_keys=True))
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
