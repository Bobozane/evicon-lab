"""Offline 48-run FakeProvider smoke for H-D.2 strict protocol."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from tempfile import TemporaryDirectory
from .cascade_agent_protocol_v2 import CascadeAgentProtocolV2Runtime, load_hd2_config
from .cascade_protocol import CascadeScenarioLoader
from .provenance_cascade_amendment import load_hd_config
from .cascade_real_agent_runner import CascadeRealAgentRunner, CascadeRealAgentRunError
from .llm_contract import LLMRequest, LLMResponse

class V2FakeProvider:
    def __init__(self) -> None: self.calls = 0
    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        payload = json.loads(request.user_prompt)
        contents = [x.get("content_id") for x in payload.get("visible_contents", []) if isinstance(x, dict) and isinstance(x.get("content_id"), str)]
        evidence = [x.get("evidence_id") for x in payload.get("visible_evidence", []) if isinstance(x, dict) and isinstance(x.get("evidence_id"), str)]
        response = {"stance":"uncertain", "content_ids_used":contents[:1], "evidence_ids_used":evidence[:1], "share_content_id":contents[0] if contents else None}
        return LLMResponse(request_id=request.request_id, model_name=request.model_name, content=json.dumps(response, sort_keys=True, separators=(",", ":")), finish_reason="stop", prompt_tokens=10, completion_tokens=8, total_tokens=18, latency_ms=1.0, provider_metadata={"provider_name":"hd2_fake"})

def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()

def run_smoke(config_path: str | Path) -> dict[str, object]:
    config, _ = load_hd2_config(config_path)
    old, _ = load_hd_config((Path(config_path).resolve().parent / config.parent_config_path).resolve())
    materials = {item.scenario_id: item for item in old.scenario_materials}
    runner = CascadeRealAgentRunner()
    records = []
    provider_calls = 0
    with TemporaryDirectory(prefix="evicon-hd2-fake-") as tmp:
        root = Path(tmp)
        for spec in config.runs:
            material = materials[spec.scenario_id]
            scenario_path = (Path(config_path).resolve().parent / config.parent_config_path).resolve().parent / material.config_path
            scenario = CascadeScenarioLoader.load(scenario_path)
            provider = V2FakeProvider()
            record = runner.run_scenario(scenario, spec.seed, spec.condition, provider=provider, run_id=spec.run_id, ledger_path=root / spec.run_id / "request_ledger.jsonl", model_name="hd2-fake-v2", agent_temperature=0.2, agent_max_tokens=256, request_cap=spec.expected_provider_requests, completion_reservation_cap=spec.completion_reservation, runtime=CascadeAgentProtocolV2Runtime())
            provider_calls += provider.calls
            records.append(record)
    replay_passed = all(record.replay is not None and record.replay.status.value == "passed" for record in records)
    ledger_hash = _sha([(r.run_id, r.exposure_ledger_sha256, r.application_ledger_sha256, r.outcome_ledger_sha256) for r in records])
    return {"status":"passed" if len(records)==48 and replay_passed and provider_calls==864 else "failed", "study_id":config.study_id, "protocol_version":config.protocol_version, "template_version":config.template_version, "run_count":len(records), "matched_group_count":len({r.matched_group_id for r in config.runs}), "provider_call_count":provider_calls, "logical_request_count":sum(r.logical_request_count for r in records), "request_cap":config.request_cap, "completion_reservation_cap":config.completion_reservation_cap, "replay_passed_count":sum(r.replay is not None and r.replay.status.value == "passed" for r in records), "directive_applied_count":sum(r.directive_applied_count for r in records), "ledger_hash":ledger_hash, "network":"disabled", "private_truth_exposed":False, "old_pilot_touched":False, "results_written":False, "safety":{"development_only":True,"not_paper_result":True,"no_causal_conclusion":True}}

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config", default="configs/provenance_cascade/pilot/provenance_cascade_pilot_hd2.v1.toml"); args=parser.parse_args(); print(json.dumps(run_smoke(args.config), ensure_ascii=True, sort_keys=True))
if __name__ == "__main__": main()
