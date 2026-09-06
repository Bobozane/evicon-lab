"""Run configuration and persisted result-record contracts."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ._validation import normalized_text
from .agents import ValueProfile
from .dialogue import DialogueTurn
from .evidence import EvidenceExposure
from .scenarios import ScenarioSpec
from .protocols import InterventionDecision, ProtocolCondition


class RunStatus(str, Enum):
    """Lifecycle values for a recorded run; no runner behavior is implemented."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunConfig(BaseModel):
    """Fully serializable input configuration for a future experiment runner."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    scenario_file: str | None = None
    model_name: str = Field(min_length=1)
    protocol: ProtocolCondition
    agent_count: int = Field(gt=0)
    max_rounds: int = Field(gt=0)
    seed: int
    intervention_budget: float = Field(ge=0.0)
    output_dir: str = Field(min_length=1)

    @field_validator("run_id", "scenario_id", "model_name", "output_dir")
    @classmethod
    def validate_required_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("scenario_file")
    @classmethod
    def validate_optional_scenario_file(cls, value: str | None) -> str | None:
        return None if value is None else normalized_text(value, "scenario_file")


class RunRecord(BaseModel):
    """Serializable output envelope; it contains records but executes nothing."""

    model_config = ConfigDict(extra="forbid")

    config: RunConfig
    scenario: ScenarioSpec | None = None
    turns: list[DialogueTurn] = Field(default_factory=list)
    value_profiles: list[ValueProfile] = Field(default_factory=list)
    evidence_exposures: list[EvidenceExposure] = Field(default_factory=list)
    intervention_decisions: list[InterventionDecision] = Field(default_factory=list)
    status: RunStatus = RunStatus.PENDING
    error_message: str | None = None

    @field_validator("error_message")
    @classmethod
    def validate_optional_error_message(cls, value: str | None) -> str | None:
        return None if value is None else normalized_text(value, "error_message")
