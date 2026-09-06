"""Offline 48-run FakeProvider integration smoke for H-C."""
from __future__ import annotations
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from .cascade_protocol import CascadeScenarioLoader
from .cascade_intervention_calibration import CascadeInterventionCalibrationRunner
from .provenance_cascade_preregistration import CascadeCondition
from .cascade_real_agent_runner import CascadeRealAgentRunner
from .llm_contract import LLMResponse
from .provenance_cascade_preflight import load_hb_config

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "configs" / "provenance_cascade" / "pilot" / "provenance_cascade_pilot_hb.v1.toml"

class _FakeProvider:
    def __init__(self, mode: str = "valid") -> None:
        self.mode = mode
        self.calls = 0
    def complete(self, request: object) -> LLMResponse:
        self.calls += 1
        payload = json.loads(getattr(request, "user_prompt"))
        contents = payload.get("visible_contents", [])
        evidence = payload.get("visible_evidence", [])
        # The fixture chooses only IDs already in this public request.
        body = {"stance": "uncertain", "content_ids_used": [contents[0]["content_id"]] if contents else [], "evidence_ids_used": [evidence[0]["evidence_id"]] if evidence else [], "share_content_id": contents[0]["content_id"] if contents else None}
        if self.mode == "malformed":
            return LLMResponse(request_id=request.request_id, model_name=request.model_name, content="{", finish_reason="stop", prompt_tokens=2, completion_tokens=1, total_tokens=3, latency_ms=1.0)
        return LLMResponse(request_id=request.request_id, model_name=request.model_name, content=json.dumps(body, sort_keys=True), finish_reason="stop", prompt_tokens=8, completion_tokens=8, total_tokens=16, latency_ms=1.0)

def run_smoke() -> dict[str, object]:
    config = load_hb_config(_CONFIG)
    runner = CascadeRealAgentRunner()
    summaries: list[dict[str, object]] = []
    provider_calls = 0
    applied = 0
    proposals = 0
    with TemporaryDirectory(prefix="evicon-hc-") as tmp:
        root = Path(tmp)
        for spec in config.runs:
            ref = next(item for item in config.scenario_materials if item.scenario_id == spec.scenario_id)
            scenario = CascadeScenarioLoader.load(_CONFIG.parent / ref.config_path)
            provider = _FakeProvider()
            record = runner.run_scenario(
                scenario, spec.seed, spec.condition, provider=provider, run_id=spec.run_id,
                ledger_path=root / spec.run_id / "request_ledger.jsonl",
                model_name="fake-cascade-agent", request_cap=spec.expected_provider_requests,
                completion_reservation_cap=spec.completion_reservation,
            )
            provider_calls += provider.calls
            applied += record.directive_applied_count
            proposals += record.proposal_count
            summaries.append({"replay": record.replay.status.value, "logical_requests": record.logical_request_count, "ledger_sha256": __import__("hashlib").sha256((root / spec.run_id / "request_ledger.jsonl").read_bytes()).hexdigest()})
        calibration = CascadeInterventionCalibrationRunner.from_file()
        calibration_scenario = next(
            item.scenario for item in calibration.scenarios
            if item.scenario.scenario_type.value == "false_majority"
        )
        calibration_record = CascadeRealAgentRunner(policy_config=calibration.policy).run_scenario(
            calibration_scenario, 20260901, CascadeCondition.PROVENANCE_AWARE_CONTROLLER,
            provider=_FakeProvider(), run_id="hc-trigger-calibration",
            ledger_path=root / "hc-trigger-calibration" / "request_ledger.jsonl",
            model_name="fake-cascade-agent", request_cap=18,
            completion_reservation_cap=4608, policy_config=calibration.policy,
        )
    ledger_digest = __import__("hashlib").sha256(
        "".join(sorted(str(item["ledger_sha256"]) for item in summaries)).encode()
    ).hexdigest()
    return {"status": "completed", "run_count": len(summaries), "matched_group_count": len(config.matched_groups), "provider_call_count": provider_calls, "logical_request_count": sum(int(x["logical_requests"]) for x in summaries), "retry_count": 0, "recovery_count": 0, "proposal_count": proposals, "directive_applied_count": applied, "calibration_directive_applied_count": calibration_record.directive_applied_count, "calibration_replay_status": calibration_record.replay.status.value, "replay_status": "passed" if all(x["replay"] == "passed" for x in summaries) else "blocked", "ledger_sha256": ledger_digest, "network": "disabled", "private_truth_exposed": False, "not_paper_result": True, "temporary_results_only": True}

def main() -> None:
    print(json.dumps(run_smoke(), sort_keys=True))

if __name__ == "__main__":
    main()
