"""Append-only content-free response audit for H-G.2.3.2."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from .cascade_agent_protocol_hg2 import HG2RuntimeResult
from .cascade_agent_protocol_hg232 import HG232Runtime
from .cascade_agent_runtime import CascadeAgentRuntimeStatus
from .cascade_real_agent_runner import CascadeRealAgentRunError
from .request_ledger import request_fingerprint_facts

AUDIT_VERSION="provenance_cascade_hg232_response_audit.v1"


class HG232ResponseAuditEntry(BaseModel):
    model_config=ConfigDict(extra="forbid",frozen=True)
    audit_version:str=AUDIT_VERSION
    binding_sha256:str=Field(min_length=64,max_length=64)
    fingerprint:str=Field(min_length=64,max_length=64)
    agent_id:str; round_id:int=Field(ge=0)
    status:str; finish_reason:str|None=None; http_status_class:str
    prompt_tokens:int|None=Field(default=None,ge=0)
    completion_tokens:int|None=Field(default=None,ge=0)
    total_tokens:int|None=Field(default=None,ge=0)
    completion_token_limit_reached:bool|None=None
    latency_ms:float|None=Field(default=None,ge=0)
    parser_valid:bool|None=None; stable_error_code:str|None=None
    response_format_mode:str="json_schema"
    response_schema_name:str="cascade_agent_epistemic_adoption_sharing_v1_provider_subset_2048"
    prompt_stored:bool=False; response_stored:bool=False; secrets_stored:bool=False
    provider_metadata_stored:bool=False; private_truth_exposed:bool=False


def load_response_audit(path:Path,binding:str)->tuple[HG232ResponseAuditEntry,...]:
    if not path.exists(): return ()
    try: entries=tuple(HG232ResponseAuditEntry.model_validate_json(x) for x in path.read_text().splitlines() if x.strip())
    except Exception as exc: raise CascadeRealAgentRunError("response_audit_invalid") from exc
    if any(x.binding_sha256!=binding for x in entries): raise CascadeRealAgentRunError("response_audit_binding_mismatch")
    if len({x.fingerprint for x in entries})!=len(entries): raise CascadeRealAgentRunError("response_audit_duplicate")
    return entries


class AuditedHG232Runtime(HG232Runtime):
    def __init__(self,path:Path,binding_sha256:str,max_tokens:int=2048)->None:
        super().__init__(); self.audit_path=path; self.binding_sha256=binding_sha256; self.max_tokens=max_tokens

    def execute(self,context,provider,*,request_metadata:Mapping[str,Any]|None=None)->HG2RuntimeResult:
        request=self.render_request(context)
        if request_metadata:
            allowed={k:v for k,v in request_metadata.items() if k in {"protocol","condition","phase","run_id","seed","matched_group_id"}}
            request=request.model_copy(update={"metadata":{**request.metadata,**allowed}})
        fingerprint=str(request_fingerprint_facts(request)["fingerprint"])
        result=super().execute(context,provider,request_metadata=request_metadata)
        existing={x.fingerprint:x for x in load_response_audit(self.audit_path,self.binding_sha256)}
        if fingerprint in existing: return result
        audit=result.audit
        received=audit.finish_reason is not None or audit.prompt_tokens is not None or audit.completion_tokens is not None
        category="2xx" if received else ({"authentication_failed":"4xx","rate_limited":"4xx","http_client_error":"4xx","http_server_error":"5xx"}.get(audit.error_code,"unavailable"))
        at_limit=None if audit.completion_tokens is None else audit.completion_tokens==self.max_tokens
        entry=HG232ResponseAuditEntry(
            binding_sha256=self.binding_sha256,fingerprint=fingerprint,agent_id=context.public_context.agent_id,
            round_id=context.public_context.round_id,status=result.status.value,finish_reason=audit.finish_reason,
            http_status_class=category,prompt_tokens=audit.prompt_tokens,completion_tokens=audit.completion_tokens,
            total_tokens=audit.total_tokens,completion_token_limit_reached=at_limit,latency_ms=audit.latency_ms,
            parser_valid=audit.parser_valid,stable_error_code=audit.error_code,
        )
        self.audit_path.parent.mkdir(parents=True,exist_ok=True)
        with self.audit_path.open("a",encoding="utf-8") as h: h.write(entry.model_dump_json()+"\n")
        return result
