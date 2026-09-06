from __future__ import annotations

import json
from pathlib import Path

from evicon.cascade_agent_protocol_hg231 import HG231_RESPONSE_JSON_SCHEMA, render_hg231_turn
from evicon.cascade_agent_protocol_hg232 import HG232_RESPONSE_JSON_SCHEMA, render_hg232_turn
from evicon.cascade_agent_protocol_hg232_audit import AuditedHG232Runtime, load_response_audit
from evicon.llm_contract import LLMResponse
from evicon.provenance_cascade_hg23 import HG23FakeProvider
from evicon.provenance_cascade_hg231_compatibility import build_minimal_hg231_context
from evicon.provenance_cascade_hg232 import (
    preflight, run_stability_probe,
)


class TruncatedProvider:
    def complete(self, request):
        return LLMResponse(
            request_id=request.request_id,model_name=request.model_name,
            content='{"epistemic_stance":"uncertain"',finish_reason="length",
            prompt_tokens=10,completion_tokens=2048,total_tokens=2058,latency_ms=2.0,
        )


def test_stability_probe_keeps_prompt_schema_and_parser_contract() -> None:
    receipt=run_stability_probe()
    assert receipt.status=="passed" and receipt.prompt_bytes_identical
    assert receipt.schema_fields_identical and receipt.parser_outcomes_identical
    context=build_minimal_hg231_context()
    old,new=render_hg231_turn(context),render_hg232_turn(context)
    assert old.system_prompt==new.system_prompt and old.user_prompt==new.user_prompt
    assert HG231_RESPONSE_JSON_SCHEMA==HG232_RESPONSE_JSON_SCHEMA
    assert old.request_id!=new.request_id


def test_response_audit_persists_safe_success_metadata(tmp_path:Path)->None:
    path=tmp_path/"response_audit.jsonl"; binding="a"*64
    runtime=AuditedHG232Runtime(path,binding)
    result=runtime.execute(build_minimal_hg231_context(),HG23FakeProvider())
    assert result.status.value=="completed"
    entry=load_response_audit(path,binding)[0]
    assert entry.finish_reason=="stop" and entry.http_status_class=="2xx"
    assert entry.completion_tokens==45 and entry.completion_token_limit_reached is False
    assert entry.parser_valid is True


def test_response_audit_records_truncation_without_content(tmp_path:Path)->None:
    path=tmp_path/"response_audit.jsonl"; binding="b"*64
    result=AuditedHG232Runtime(path,binding).execute(build_minimal_hg231_context(),TruncatedProvider())
    assert result.status.value=="parser_invalid"
    entry=load_response_audit(path,binding)[0]
    assert entry.finish_reason=="length" and entry.completion_tokens==2048
    assert entry.completion_token_limit_reached is True and entry.parser_valid is False
    raw=path.read_text().lower()
    for forbidden in ("system_prompt","user_prompt","raw_response","api_key","authorization","ground_truth_label"):
        assert forbidden not in raw


def test_preflight_confirms_approval_and_compatibility_are_registered() -> None:
    report=preflight()
    assert report["blocking_reasons"] == []
    assert report["ready_for_calibration"] is False
    assert report["network"]=="disabled" and report["provider_constructed"] is False
