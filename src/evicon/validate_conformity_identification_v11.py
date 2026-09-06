"""Final network-free design preflight for Conformity Identification Study v1.1."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .conformity_identification import safe_json, sha256_file
from .conformity_identification_analysis_plan import DEFAULT_ANALYSIS_PLAN, load_analysis_plan
from .conformity_identification_v11 import DEFAULT_V11_CONFIG, load_v11_config, validate_branch_plan
from .conformity_identification_v11_smoke import run_fake_smoke
from .conformity_source_manipulation import DEFAULT_MANIPULATION_CONFIG, load_manipulation_config
from .conformity_source_manipulation_smoke import run_fake_smoke as run_manipulation_smoke


def preflight(path: str = DEFAULT_V11_CONFIG) -> dict[str, object]:
    try:
        config, _ = load_v11_config(path)
        validate_branch_plan(config)
        analysis = load_analysis_plan(DEFAULT_ANALYSIS_PLAN)
        manipulation = load_manipulation_config(DEFAULT_MANIPULATION_CONFIG)
        smoke = run_fake_smoke(path)
        manipulation_smoke = run_manipulation_smoke(DEFAULT_MANIPULATION_CONFIG)
        reasons = [
            "exact_hash_human_approval_required",
            "real_source_manipulation_check_required",
            "protocol_stability_probe_required",
            "provider_compatibility_check_required",
        ]
        return {
            "status": "offline_design_ready",
            "ready_for_network": False,
            "blocking_reasons": reasons,
            "study_id": config.study_id,
            "config_sha256": sha256_file(path),
            "protocol_sha256": config.protocol_sha256,
            "branch_contract_sha256": config.branch_contract_sha256,
            "analysis_plan_sha256": sha256_file(DEFAULT_ANALYSIS_PLAN),
            "manipulation_config_sha256": sha256_file(DEFAULT_MANIPULATION_CONFIG),
            "matched_group_count": config.matched_group_count,
            "branch_run_count": config.branch_run_count,
            "logical_request_cap": config.logical_request_cap,
            "completion_reservation_cap": config.completion_reservation_cap,
            "primary_contrast_count": len(analysis.primary_contrasts),
            "fake_logical_request_count": smoke["logical_request_count"],
            "fake_unique_fingerprint_count": smoke["unique_request_fingerprint_count"],
            "manipulation_fake_passed_count": manipulation_smoke["passed_case_count"],
            "output_root_exists": Path(config.output_root).exists(),
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
            "private_truth_exposed": False,
            "effect_estimated": False,
            "not_paper_result": True,
            "no_causal_conclusion": True,
        }
    except Exception as exc:
        return {
            "status": "blocked",
            "ready_for_network": False,
            "blocking_reasons": [getattr(exc, "code", "v11_preflight_failed")],
            "network": "disabled",
            "provider_constructed": False,
            "api_key_read": False,
            "results_written": False,
            "private_truth_exposed": False,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Conformity Identification Study v1.1")
    parser.add_argument("--config", default=DEFAULT_V11_CONFIG)
    args = parser.parse_args(argv)
    result = preflight(args.config)
    print(safe_json(result))
    return 0 if result.get("status") == "offline_design_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
