"""Prompt-repaired factorized source-behavior protocol.

This is a versioned prompt revision of the frozen factorized response
contract.  The public scenarios, eight conditions, response fields, and
parser rules remain unchanged; the system instruction now states the
cross-field invariants explicitly so a provider cannot be mistaken for a
usable experimental protocol merely because it returns JSON.

The v1 protocol and its failed pilot remain importable and are never rewritten.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final

from .conformity_source_behavior_offline_simulation_v1 import (
    DEFAULT_CONFIG,
    RESPONSE_SCHEMA,
    RESPONSE_SCHEMA_SHA256,
    SCHEMA_NAME,
    FactorCase,
    FactorCondition,
    FactorizationResponse,
    SafeFactorObservation,
    build_cases,
    parse_factor_response,
    safe_factor_observation,
    _public_payload,
)
from .llm_contract import LLMRequest


PROTOCOL_VERSION: Final[str] = "conformity_source_behavior_factorized_protocol.v2"
TEMPLATE_VERSION: Final[str] = "conformity_source_behavior_factorized_turn.v2"

SYSTEM_PROMPT: Final[str] = (
    "Return exactly one JSON object with these six fields and no others: "
    "ordinal_judgment, public_action, share_decision, confidence, "
    "content_ids_used, and share_content_id. Use only these exact values: "
    "ordinal_judgment is one of strongly_reject, lean_reject, uncertain, "
    "lean_adopt, strongly_adopt; public_action is one of adopt, reject, "
    "withhold; share_decision is one of share, share_with_caveat, "
    "do_not_share; confidence is an integer from 1 through 7. Use the "
    "following consistency rules: "
    "strongly_reject or lean_reject requires public_action=reject; uncertain "
    "requires public_action=withhold; lean_adopt or strongly_adopt requires "
    "public_action=adopt. content_ids_used must be a non-empty list containing "
    "only IDs from visible_content; when the initial material informed the "
    "judgment, include content-initial, including when reflection_only=true "
    "and there is no new material. Do not use a placeholder such as none or "
    "unknown for a content ID. "
    "If share_decision=do_not_share, share_content_id must be null. Otherwise "
    "share_content_id must be one of content_ids_used and visible_content. "
    "Use only the visible public material. Equal public root IDs indicate "
    "reports traced to one shown root; distinct public root IDs indicate "
    "distinct shown roots; null means the relation is not shown. A root relation "
    "does not establish truth or source quality. An evidence ID identifies a "
    "directly visible evidence card. Do not invent IDs, sources, evidence, truth "
    "labels, private labels, or hidden information."
)


def render_factor_request(case: FactorCase) -> LLMRequest:
    """Render the v2 public request with explicit output invariants."""

    payload = dict(_public_payload(case))
    payload["template_version"] = TEMPLATE_VERSION
    user_prompt = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(
        f"{case.case_id}|{case.seed}|{user_prompt}".encode("utf-8")
    ).hexdigest()[:24]
    return LLMRequest(
        request_id=f"source-factor-v2-{digest}",
        model_name=case.model_name,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=case.temperature,
        max_tokens=case.max_tokens,
        seed=case.seed,
        metadata={"protocol_version": PROTOCOL_VERSION, "template_version": TEMPLATE_VERSION},
    )


def protocol_self_check() -> dict[str, object]:
    """Return safe static facts used by offline tests and preflight tooling."""

    case = build_cases(DEFAULT_CONFIG)[0]
    request = render_factor_request(case)
    prompt = request.system_prompt + request.user_prompt
    forbidden = (
        case.case_id,
        case.group_id,
        case.scenario_id,
        case.condition.value,
        str(case.seed),
        "latent_focal_root_ids",
        "evaluator_private",
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "template_version": TEMPLATE_VERSION,
        "schema_name": SCHEMA_NAME,
        "schema_sha256": RESPONSE_SCHEMA_SHA256,
        "prompt_contains_internal_coordinate": any(item in prompt for item in forbidden),
        "explicit_action_mapping": "strongly_reject or lean_reject requires public_action=reject" in request.system_prompt,
        "explicit_content_rule": "content_ids_used must be a non-empty list" in request.system_prompt,
    }


__all__ = [
    "DEFAULT_CONFIG",
    "FactorCase",
    "FactorCondition",
    "FactorizationResponse",
    "PROTOCOL_VERSION",
    "RESPONSE_SCHEMA",
    "RESPONSE_SCHEMA_SHA256",
    "SCHEMA_NAME",
    "SafeFactorObservation",
    "SYSTEM_PROMPT",
    "TEMPLATE_VERSION",
    "build_cases",
    "parse_factor_response",
    "protocol_self_check",
    "render_factor_request",
    "safe_factor_observation",
]
