"""Offline H-D.2 protocol and provider compatibility preflight."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from .cascade_agent_protocol_v2 import (HD2ConfigError, V2_PROTOCOL_VERSION, V2_SCHEMA_NAME, V2_TEMPLATE_VERSION, load_hd2_config)
from .openai_provider import ProviderConfig, ResponseFormatMode

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def preflight(path: str | Path) -> dict[str, object]:
    config_path = Path(path).resolve()
    try:
        config, _ = load_hd2_config(config_path)
    except HD2ConfigError as exc:
        return {"status":"blocked", "error_code":exc.code, "network":"disabled", "provider_constructed":False, "transport_called":False}
    try:
        ProviderConfig(
            model_name="offline-preflight-model", allow_network=False, max_retries=1,
            timeout_seconds=15.0, temperature=0.2, max_tokens=256, seed=0,
            response_format=ResponseFormatMode(config.response_format),
            response_schema_name=config.response_schema_name,
        )
    except Exception:
        return {"status":"blocked", "error_code":"provider_config_invalid", "network":"disabled", "provider_constructed":False, "transport_called":False}
    return {
        "status":"ready_offline", "study_id":config.study_id, "config_version":config.config_version,
        "protocol_version":V2_PROTOCOL_VERSION, "template_version":V2_TEMPLATE_VERSION,
        "scenario_count":len(config.scenario_ids), "condition_count":len(config.conditions),
        "seed_count":len(config.seeds), "run_count":len(config.runs),
        "matched_group_count":len({r.matched_group_id for r in config.runs}),
        "logical_request_count":config.request_cap, "completion_reservation_cap":config.completion_reservation_cap,
        "response_format":config.response_format, "response_schema_name":V2_SCHEMA_NAME,
        "network":"disabled", "provider_constructed":False, "transport_called":False,
        "api_key_read":False, "old_pilot_touched":False,
        "config_sha256":_sha(config_path),
        "safety":{"development_only":True,"not_paper_result":True,"no_causal_conclusion":True,"private_truth_exposed":False},
    }

def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args=parser.parse_args()
    print(json.dumps(preflight(args.config), ensure_ascii=True, sort_keys=True))

if __name__ == "__main__":
    main()
