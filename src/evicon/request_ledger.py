"""Append-only, content-free request accounting for explicitly enabled pilots."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .llm_contract import LLMProvider, LLMProviderError, LLMRequest, LLMResponse, ProviderErrorCode


class RequestLedgerStatus(str, Enum):
    """Lifecycle state for one logical provider attempt."""

    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class RequestLedgerEntry(BaseModel):
    """Safe request receipt. It deliberately has neither prompt nor response fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(min_length=1)
    fingerprint: str = Field(min_length=64, max_length=64)
    request_key: str = Field(min_length=64, max_length=64)
    model_name: str = Field(min_length=1)
    template_version: str = Field(min_length=1)
    public_content_sha256: str = Field(min_length=64, max_length=64)
    condition: str = Field(min_length=1)
    phase: str = Field(min_length=1)
    agent_id: str | None = None
    round_id: int | None = Field(default=None, ge=0)
    probe_id: str | None = None
    status: RequestLedgerStatus
    attempt_count: int = Field(ge=0)
    reserved_tokens: int = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    error_code: str | None = None
    recovery_kind: str | None = None
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("fingerprint", "request_key", "public_content_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        normalized = value.lower().strip()
        if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
            raise ValueError("ledger hash fields must be SHA-256 digests")
        return normalized


class RequestLedgerSummary(BaseModel):
    """Safe aggregate suitable for batch summaries and CLI output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_cap: int = Field(gt=0)
    completion_reservation_cap: int = Field(gt=0)
    provider_call_count: int = Field(ge=0)
    unique_logical_request_count: int = Field(default=0, ge=0)
    transport_attempt_count: int = Field(default=0, ge=0)
    parser_recovery_transport_attempt_count: int = Field(default=0, ge=0)
    completion_reserved_token_count: int = Field(ge=0)
    actual_prompt_token_count: int | None = Field(default=None, ge=0)
    actual_completion_token_count: int | None = Field(default=None, ge=0)
    actual_total_token_count: int | None = Field(default=None, ge=0)
    completed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    unknown_usage_count: int = Field(ge=0)


class RequestLedgerError(LLMProviderError):
    """Stable cap and resume errors without prompt, response, or secret data."""


class RequestLedger:
    """Read and append safe JSONL receipts; existing receipt lines are immutable."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def entries(self) -> list[RequestLedgerEntry]:
        if not self.path.exists():
            return []
        entries: list[RequestLedgerEntry] = []
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        raise RequestLedgerError(ProviderErrorCode.INVALID_RESPONSE, "request ledger contains a blank receipt")
                    entries.append(RequestLedgerEntry.model_validate_json(line))
        except (OSError, ValidationError, ValueError) as exc:
            raise RequestLedgerError(ProviderErrorCode.INVALID_RESPONSE, "request ledger is invalid") from exc
        return entries

    def append(self, entry: RequestLedgerEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(entry.model_dump_json())
            handle.write("\n")

    def summary(self, *, request_cap: int, completion_reservation_cap: int) -> RequestLedgerSummary:
        entries = self.entries()
        started = [entry for entry in entries if entry.status is RequestLedgerStatus.STARTED]
        # Caps apply to unique, pre-registered logical request fingerprints.
        # Retrying that exact fingerprint on explicit resume remains visible as
        # another attempt, but cannot silently expand the 384-request plan.
        unique_started = {entry.fingerprint: entry for entry in started}
        terminal_by_attempt = _terminal_entries(entries)
        terminal = list(terminal_by_attempt.values())
        completed = [entry for entry in terminal if entry.status is RequestLedgerStatus.COMPLETED]
        totals_known = all(entry.total_tokens is not None for entry in completed)
        prompts_known = all(entry.prompt_tokens is not None for entry in completed)
        completions_known = all(entry.completion_tokens is not None for entry in completed)
        return RequestLedgerSummary(
            request_cap=request_cap,
            completion_reservation_cap=completion_reservation_cap,
            provider_call_count=len(unique_started),
            unique_logical_request_count=len(unique_started),
            transport_attempt_count=len(started),
            parser_recovery_transport_attempt_count=sum(
                entry.status is RequestLedgerStatus.STARTED and entry.recovery_kind is not None
                for entry in entries
            ),
            completion_reserved_token_count=sum(entry.reserved_tokens for entry in unique_started.values()),
            actual_prompt_token_count=(
                sum(entry.prompt_tokens for entry in completed if entry.prompt_tokens is not None)
                if prompts_known else None
            ),
            actual_completion_token_count=(
                sum(entry.completion_tokens for entry in completed if entry.completion_tokens is not None)
                if completions_known else None
            ),
            actual_total_token_count=(
                sum(entry.total_tokens for entry in completed if entry.total_tokens is not None)
                if totals_known else None
            ),
            completed_count=sum(entry.status is RequestLedgerStatus.COMPLETED for entry in terminal),
            failed_count=sum(entry.status is RequestLedgerStatus.FAILED for entry in terminal),
            unknown_usage_count=sum(entry.status is RequestLedgerStatus.COMPLETED and entry.total_tokens is None for entry in terminal),
        )


class LedgeredProvider:
    """Wrap an injected provider with caps and persisted safe request receipts."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        ledger: RequestLedger,
        request_cap: int,
        completion_reservation_cap: int,
        resume: bool = False,
        recovery_fingerprints: set[str] | None = None,
        recovery_kind: str | None = None,
        recovery_phases: set[str] | None = None,
        max_recovery_attempts: int = 1,
        min_interval_seconds: float = 0.0,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if request_cap <= 0 or completion_reservation_cap <= 0:
            raise ValueError("request_cap and completion_reservation_cap must be positive")
        if min_interval_seconds < 0:
            raise ValueError("min_interval_seconds must be non-negative")
        if max_recovery_attempts < 0:
            raise ValueError("max_recovery_attempts must be non-negative")
        self._provider = provider
        self._ledger = ledger
        self._request_cap = request_cap
        self._completion_reservation_cap = completion_reservation_cap
        self._resume = resume
        self._recovery_fingerprints = frozenset(recovery_fingerprints or set())
        self._recovery_kind = recovery_kind
        self._recovery_phases = frozenset(recovery_phases or {"probe"})
        self._max_recovery_attempts = max_recovery_attempts
        self._recovery_consumed = False
        self._min_interval_seconds = min_interval_seconds
        self._sleeper = sleeper

    def complete(self, request: LLMRequest) -> LLMResponse:
        facts = request_fingerprint_facts(request)
        entries = self._ledger.entries()
        _reject_fingerprint_conflict(entries, facts)
        terminal = _latest_terminal(entries, facts["fingerprint"])
        recovery_mode = False
        if self._recovery_fingerprints and not self._recovery_consumed and terminal is None:
            if facts["phase"] not in self._recovery_phases or facts["fingerprint"] not in self._recovery_fingerprints:
                raise RequestLedgerError(
                    ProviderErrorCode.INVALID_REQUEST,
                    "parser_recovery_fingerprint_changed",
                )
        if terminal is not None and terminal.status is RequestLedgerStatus.COMPLETED:
            if (
                not self._resume
                or facts["fingerprint"] not in self._recovery_fingerprints
                or facts["phase"] not in self._recovery_phases
            ):
                raise RequestLedgerError(ProviderErrorCode.INVALID_REQUEST, "completed_request_fingerprint_exists")
            recovery_attempts = sum(
                entry.status is RequestLedgerStatus.STARTED
                and entry.fingerprint == facts["fingerprint"]
                and entry.recovery_kind == self._recovery_kind
                for entry in entries
            )
            if recovery_attempts >= self._max_recovery_attempts:
                raise RequestLedgerError(ProviderErrorCode.INVALID_REQUEST, "parser_recovery_attempt_limit_reached")
            recovery_mode = True
            self._recovery_consumed = True
        if terminal is not None and terminal.status is RequestLedgerStatus.FAILED and not self._resume:
            raise RequestLedgerError(ProviderErrorCode.INVALID_REQUEST, "failed_request_requires_resume")
        if terminal is None and any(
            entry.fingerprint == facts["fingerprint"] and entry.status is RequestLedgerStatus.STARTED
            for entry in entries
        ) and not self._resume:
            raise RequestLedgerError(ProviderErrorCode.INVALID_REQUEST, "incomplete_request_requires_resume")

        summary = self._ledger.summary(
            request_cap=self._request_cap,
            completion_reservation_cap=self._completion_reservation_cap,
        )
        existing_logical_request = any(
            entry.status is RequestLedgerStatus.STARTED
            and entry.fingerprint == facts["fingerprint"]
            for entry in entries
        )
        if (
            not recovery_mode
            and not existing_logical_request
            and summary.provider_call_count >= self._request_cap
        ):
            raise RequestLedgerError(ProviderErrorCode.INVALID_REQUEST, "request_cap_reached")
        if (
            not recovery_mode
            and not existing_logical_request
            and summary.completion_reserved_token_count + request.max_tokens
            > self._completion_reservation_cap
        ):
            raise RequestLedgerError(
                ProviderErrorCode.INVALID_REQUEST,
                "completion_reservation_cap_reached",
            )
        if self._min_interval_seconds and self._sleeper is not None and summary.provider_call_count:
            self._sleeper(self._min_interval_seconds)

        attempt_count = 1 + sum(entry.status is RequestLedgerStatus.STARTED and entry.fingerprint == facts["fingerprint"] for entry in entries)
        started = _entry_from_facts(
            facts, request, status=RequestLedgerStatus.STARTED, attempt_count=attempt_count,
            reserved_tokens=request.max_tokens,
            recovery_kind=self._recovery_kind if recovery_mode else None,
        )
        self._ledger.append(started)
        try:
            response = self._provider.complete(request)
        except LLMProviderError as error:
            self._ledger.append(_entry_from_facts(
                facts, request, status=RequestLedgerStatus.FAILED, attempt_count=attempt_count,
                reserved_tokens=request.max_tokens, error_code=error.code.value,
                recovery_kind=self._recovery_kind if recovery_mode else None,
            ))
            raise
        except Exception:
            self._ledger.append(_entry_from_facts(
                facts, request, status=RequestLedgerStatus.FAILED, attempt_count=attempt_count,
                reserved_tokens=request.max_tokens, error_code=ProviderErrorCode.PROVIDER_FAILURE.value,
                recovery_kind=self._recovery_kind if recovery_mode else None,
            ))
            raise LLMProviderError(ProviderErrorCode.PROVIDER_FAILURE, "provider request failed") from None
        self._ledger.append(_entry_from_facts(
            facts, request, status=RequestLedgerStatus.COMPLETED, attempt_count=attempt_count,
            reserved_tokens=request.max_tokens, prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens, total_tokens=response.total_tokens,
            latency_ms=response.latency_ms,
            recovery_kind=self._recovery_kind if recovery_mode else None,
        ))
        return response


def request_fingerprint_facts(request: LLMRequest) -> dict[str, str | int | None]:
    """Derive stable, content-free receipt facts from an in-memory rendered request."""
    template = str(request.metadata.get("template_version") or "unknown")
    agent_id = _optional_text(request.metadata.get("agent_id"))
    round_id = _optional_int(request.metadata.get("round_id"))
    condition = _optional_text(request.metadata.get("protocol")) or "unknown"
    phase = "agent_turn" if template in {"agent_turn.v1", "cascade_agent_turn.v1", "cascade_agent_turn.hg1_public_content.v1", "cascade_agent_turn.hg11_public_content_1024.v1", "cascade_agent_turn.hg1_1_public_content.v1", "cascade_agent_turn.hg2_epistemic_behavior.v1"} else "probe"
    probe_id: str | None = None
    if template in {"contextual_value_probe.v1", "contextual_value_probe.v2"}:
        try:
            payload = json.loads(request.user_prompt)
            context = payload["measurement_context"]
            probe = payload["probe_item"]
            if isinstance(context, dict):
                condition = _optional_text(context.get("protocol")) or condition
            if isinstance(probe, dict):
                probe_id = _optional_text(probe.get("probe_id"))
        except (TypeError, ValueError, json.JSONDecodeError):
            # The Runtime parser remains the final authority; the ledger only
            # needs a stable opaque fingerprint for malformed local requests.
            pass
    public_hash = _sha256({"system": request.system_prompt, "user": request.user_prompt})
    coordinates = {
        "condition": condition,
        "phase": phase,
        "agent_id": agent_id,
        "round_id": round_id,
        "probe_id": probe_id,
    }
    fingerprint = _sha256({
        "model": request.model_name,
        "template_version": template,
        "public_content_sha256": public_hash,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "seed": request.seed,
        **coordinates,
    })
    return {
        "fingerprint": fingerprint,
        "request_key": _sha256(coordinates),
        "public_content_sha256": public_hash,
        "template_version": template,
        **coordinates,
    }


def _entry_from_facts(
    facts: dict[str, str | int | None], request: LLMRequest, *, status: RequestLedgerStatus,
    attempt_count: int, reserved_tokens: int, prompt_tokens: int | None = None,
    completion_tokens: int | None = None, total_tokens: int | None = None,
    latency_ms: float | None = None, error_code: str | None = None,
    recovery_kind: str | None = None,
) -> RequestLedgerEntry:
    return RequestLedgerEntry(
        request_id=request.request_id,
        fingerprint=str(facts["fingerprint"]), request_key=str(facts["request_key"]),
        model_name=request.model_name, template_version=str(facts["template_version"]),
        public_content_sha256=str(facts["public_content_sha256"]), condition=str(facts["condition"]),
        phase=str(facts["phase"]), agent_id=_optional_text(facts.get("agent_id")),
        round_id=_optional_int(facts.get("round_id")), probe_id=_optional_text(facts.get("probe_id")),
        status=status, attempt_count=attempt_count, reserved_tokens=reserved_tokens,
        recovery_kind=recovery_kind,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens,
        latency_ms=latency_ms, error_code=error_code,
    )


def _terminal_entries(entries: list[RequestLedgerEntry]) -> dict[tuple[str, int], RequestLedgerEntry]:
    return {
        (entry.fingerprint, entry.attempt_count): entry
        for entry in entries if entry.status in {RequestLedgerStatus.COMPLETED, RequestLedgerStatus.FAILED}
    }


def _latest_terminal(entries: list[RequestLedgerEntry], fingerprint: str) -> RequestLedgerEntry | None:
    terminal = [entry for entry in entries if entry.fingerprint == fingerprint and entry.status in {RequestLedgerStatus.COMPLETED, RequestLedgerStatus.FAILED}]
    return max(terminal, key=lambda entry: entry.attempt_count) if terminal else None


def _reject_fingerprint_conflict(entries: list[RequestLedgerEntry], facts: dict[str, str | int | None]) -> None:
    for entry in entries:
        if entry.request_key == facts["request_key"] and entry.fingerprint != facts["fingerprint"]:
            raise RequestLedgerError(ProviderErrorCode.INVALID_REQUEST, "request_fingerprint_mismatch")


def _sha256(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _optional_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


__all__ = [
    "LedgeredProvider",
    "RequestLedger",
    "RequestLedgerEntry",
    "RequestLedgerError",
    "RequestLedgerStatus",
    "RequestLedgerSummary",
    "request_fingerprint_facts",
]
