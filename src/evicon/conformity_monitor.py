"""Pure, deterministic monitoring of observable conformity-risk signals."""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models._validation import Metadata, normalized_text


class RiskLevel(str, Enum):
    """Ordered operational labels, not experimental or causal conclusions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    INVALID = "invalid"


class MonitorReason(str, Enum):
    """Stable explanation labels emitted by ``ConformityMonitor``."""

    DIVERSITY_DECREASE = "diversity_decrease"
    DIVERSITY_COLLAPSE = "diversity_collapse"
    COVERAGE_LOSS = "coverage_loss"
    MINORITY_REDUCTION = "minority_reduction"
    MINORITY_LOSS = "minority_loss"
    HARM_RISK = "harm_risk"
    TASK_QUALITY_CONCERN = "task_quality_concern"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    EVIDENCE_SUPPORTED_UPDATE = "evidence_supported_update"
    NO_OBSERVABLE_PEERS = "no_observable_peers"


class MonitorInput(BaseModel):
    """One online-observable round summary, intentionally excluding private probes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    current_diversity: float = Field(ge=0.0)
    previous_diversity: float = Field(ge=0.0)
    current_coverage: float = Field(ge=0.0, le=1.0)
    previous_coverage: float = Field(ge=0.0, le=1.0)
    minority_loss: float = Field(ge=0.0, le=1.0)
    evidence_gain: float = Field(ge=0.0, le=1.0)
    evidence_quality: float = Field(ge=0.0, le=1.0)
    harm_risk: float = Field(ge=0.0, le=1.0)
    task_quality: float = Field(ge=0.0, le=1.0)
    remaining_budget: float = Field(ge=0.0)
    observable_peer_count: int = Field(ge=0)
    metadata: Metadata = Field(default_factory=dict)

    @field_validator("run_id", "scenario_id")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator(
        "current_diversity",
        "previous_diversity",
        "current_coverage",
        "previous_coverage",
        "minority_loss",
        "evidence_gain",
        "evidence_quality",
        "harm_risk",
        "task_quality",
        "remaining_budget",
        mode="before",
    )
    @classmethod
    def validate_finite_number(cls, value: object, info: object) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{getattr(info, 'field_name', 'value')} must be a number")
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{getattr(info, 'field_name', 'value')} must be a number") from exc
        if not math.isfinite(normalized):
            raise ValueError(f"{getattr(info, 'field_name', 'value')} must be finite")
        return normalized

    @field_validator("round_id", "observable_peer_count", mode="before")
    @classmethod
    def validate_nonnegative_integer(cls, value: object, info: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{getattr(info, 'field_name', 'value')} must be an integer")
        return value


class MonitorConfig(BaseModel):
    """Calibrated monitoring parameters; no formal defaults are supplied here."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    collapse_weight: float = Field(ge=0.0)
    minority_weight: float = Field(ge=0.0)
    evidence_weight: float = Field(ge=0.0)
    evidence_quality_weight: float = Field(ge=0.0)
    harm_weight: float = Field(ge=0.0)
    collapse_trigger: float = Field(ge=0.0, le=1.0)
    minority_trigger: float = Field(ge=0.0, le=1.0)
    evidence_sufficient_threshold: float = Field(ge=0.0, le=1.0)
    high_risk_threshold: float = Field(ge=0.0, le=1.0)
    medium_risk_threshold: float = Field(ge=0.0, le=1.0)
    version: str = Field(min_length=1)
    task_quality_sufficient_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator(
        "collapse_weight",
        "minority_weight",
        "evidence_weight",
        "evidence_quality_weight",
        "harm_weight",
        "collapse_trigger",
        "minority_trigger",
        "evidence_sufficient_threshold",
        "high_risk_threshold",
        "medium_risk_threshold",
        "task_quality_sufficient_threshold",
        mode="before",
    )
    @classmethod
    def validate_finite_number(cls, value: object, info: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"{getattr(info, 'field_name', 'value')} must be a number")
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{getattr(info, 'field_name', 'value')} must be a number") from exc
        if not math.isfinite(normalized):
            raise ValueError(f"{getattr(info, 'field_name', 'value')} must be finite")
        return normalized

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return normalized_text(value, "version")

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "MonitorConfig":
        if self.medium_risk_threshold > self.high_risk_threshold:
            raise ValueError("medium_risk_threshold must be less than or equal to high_risk_threshold")
        return self


class MonitorSignals(BaseModel):
    """Input-derived components included in every result for auditing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    collapse_amount: float = Field(ge=0.0)
    coverage_loss: float = Field(ge=0.0, le=1.0)
    minority_loss: float = Field(ge=0.0, le=1.0)
    evidence_gain: float = Field(ge=0.0, le=1.0)
    evidence_quality: float = Field(ge=0.0, le=1.0)
    evidence_support: float = Field(ge=0.0, le=1.0)
    harm_risk: float = Field(ge=0.0, le=1.0)
    task_quality: float = Field(ge=0.0, le=1.0)


class MonitorResult(BaseModel):
    """One deterministic eligibility assessment; it never selects an action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    round_id: int = Field(ge=0)
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    should_intervene: bool
    evidence_supported_update: bool
    signals: MonitorSignals
    reasons: list[MonitorReason] = Field(default_factory=list)
    config_version: str = Field(min_length=1)
    valid: bool

    @field_validator("run_id", "scenario_id", "config_version")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))


class ConformityMonitor:
    """A side-effect-free risk scorer for online-observable summaries only."""

    @staticmethod
    def evaluate(monitor_input: MonitorInput, config: MonitorConfig) -> MonitorResult:
        """Return a stable risk assessment without reading state, files, or providers."""
        collapse_amount = max(0.0, monitor_input.previous_diversity - monitor_input.current_diversity)
        normalized_collapse = min(1.0, collapse_amount)
        coverage_loss = max(0.0, monitor_input.previous_coverage - monitor_input.current_coverage)
        evidence_support = monitor_input.evidence_gain * monitor_input.evidence_quality
        raw_risk = (
            config.collapse_weight * normalized_collapse
            + config.minority_weight * monitor_input.minority_loss
            + config.harm_weight * monitor_input.harm_risk
            - config.evidence_weight * evidence_support
            - config.evidence_quality_weight * monitor_input.evidence_quality
        )
        risk_score = min(1.0, max(0.0, raw_risk))
        quality_sufficient = (
            config.task_quality_sufficient_threshold is None
            or monitor_input.task_quality >= config.task_quality_sufficient_threshold
        )
        evidence_supported_update = (
            monitor_input.evidence_gain > 0.0
            and evidence_support >= config.evidence_sufficient_threshold
            and quality_sufficient
        )
        risk_level = _risk_level(risk_score, config)
        reasons = _reasons(
            monitor_input,
            config,
            collapse_amount=collapse_amount,
            normalized_collapse=normalized_collapse,
            coverage_loss=coverage_loss,
            evidence_supported_update=evidence_supported_update,
        )
        return MonitorResult(
            run_id=monitor_input.run_id,
            scenario_id=monitor_input.scenario_id,
            round_id=monitor_input.round_id,
            risk_score=risk_score,
            risk_level=risk_level,
            should_intervene=(
                risk_score >= config.high_risk_threshold and not evidence_supported_update
            ),
            evidence_supported_update=evidence_supported_update,
            signals=MonitorSignals(
                collapse_amount=collapse_amount,
                coverage_loss=coverage_loss,
                minority_loss=monitor_input.minority_loss,
                evidence_gain=monitor_input.evidence_gain,
                evidence_quality=monitor_input.evidence_quality,
                evidence_support=evidence_support,
                harm_risk=monitor_input.harm_risk,
                task_quality=monitor_input.task_quality,
            ),
            reasons=reasons,
            config_version=config.version,
            valid=True,
        )


def _risk_level(risk_score: float, config: MonitorConfig) -> RiskLevel:
    if risk_score >= config.high_risk_threshold:
        return RiskLevel.HIGH
    if risk_score >= config.medium_risk_threshold:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _reasons(
    monitor_input: MonitorInput,
    config: MonitorConfig,
    *,
    collapse_amount: float,
    normalized_collapse: float,
    coverage_loss: float,
    evidence_supported_update: bool,
) -> list[MonitorReason]:
    reasons: list[MonitorReason] = []
    if collapse_amount > 0.0:
        reasons.append(
            MonitorReason.DIVERSITY_COLLAPSE
            if normalized_collapse >= config.collapse_trigger
            else MonitorReason.DIVERSITY_DECREASE
        )
    if coverage_loss > 0.0:
        reasons.append(MonitorReason.COVERAGE_LOSS)
    if monitor_input.minority_loss > 0.0:
        reasons.append(
            MonitorReason.MINORITY_LOSS
            if monitor_input.minority_loss >= config.minority_trigger
            else MonitorReason.MINORITY_REDUCTION
        )
    if monitor_input.harm_risk > 0.0:
        reasons.append(MonitorReason.HARM_RISK)
    if (
        config.task_quality_sufficient_threshold is not None
        and monitor_input.task_quality < config.task_quality_sufficient_threshold
    ):
        reasons.append(MonitorReason.TASK_QUALITY_CONCERN)
    if evidence_supported_update:
        reasons.append(MonitorReason.EVIDENCE_SUPPORTED_UPDATE)
    elif collapse_amount > 0.0 or coverage_loss > 0.0 or monitor_input.minority_loss > 0.0:
        reasons.append(MonitorReason.INSUFFICIENT_EVIDENCE)
    if monitor_input.observable_peer_count == 0:
        reasons.append(MonitorReason.NO_OBSERVABLE_PEERS)
    return reasons
