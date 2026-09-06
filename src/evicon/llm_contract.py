"""Offline, serializable contracts for mediator-capable LLM providers."""

from __future__ import annotations

import math
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from .models._validation import Metadata, normalized_text


class ProviderErrorCode(str, Enum):
    """Stable provider failures that contain no credential or prompt text."""

    INVALID_REQUEST = "invalid_request"
    NETWORK_DISABLED = "network_disabled"
    MISSING_API_KEY = "missing_api_key"
    MISSING_BASE_URL = "missing_base_url"
    MISSING_MODEL = "missing_model"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    AUTHENTICATION_FAILED = "authentication_failed"
    HTTP_CLIENT_ERROR = "http_client_error"
    HTTP_SERVER_ERROR = "http_server_error"
    CONNECTION_FAILURE = "connection_failure"
    DNS_OR_URL_FAILURE = "dns_or_url_failure"
    RESPONSE_FORMAT_UNSUPPORTED = "response_format_unsupported"
    MALFORMED_RESPONSE = "malformed_response"
    INVALID_RESPONSE = "invalid_response"
    USAGE_MISMATCH = "usage_mismatch"
    PROVIDER_FAILURE = "provider_failure"


class LLMProviderError(RuntimeError):
    """Structured exception boundary for provider implementations."""

    def __init__(self, code: ProviderErrorCode, message: str, *, status_code: int | None = None) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code if isinstance(status_code, int) and not isinstance(status_code, bool) else None
        super().__init__(f"{code.value}: {message}")

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "message": self.message}


class LLMRequest(BaseModel):
    """A fully rendered request. Secrets are never a field of this contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    system_prompt: str
    user_prompt: str
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0)
    seed: int
    metadata: Metadata = Field(default_factory=dict)

    @field_validator("request_id", "model_name")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("system_prompt", "user_prompt")
    @classmethod
    def validate_prompt_text(cls, value: str, info: object) -> str:
        _reject_sensitive_text(value, getattr(info, "field_name", "prompt"))
        return value

    @field_validator("temperature", mode="before")
    @classmethod
    def validate_temperature(cls, value: object) -> float:
        if isinstance(value, bool):
            raise ValueError("temperature must be a finite number")
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("temperature must be a finite number") from exc
        if not math.isfinite(normalized):
            raise ValueError("temperature must be a finite number")
        return normalized

    @field_validator("max_tokens", "seed", mode="before")
    @classmethod
    def validate_integer(cls, value: object, info: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{getattr(info, 'field_name', 'value')} must be an integer")
        return value

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: Metadata) -> Metadata:
        _reject_nonpublic_metadata(value, "LLMRequest.metadata")
        return value


class LLMResponse(BaseModel):
    """Serializable provider output with explicit known or unknown usage fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    content: str
    finish_reason: str = Field(min_length=1)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0.0)
    provider_metadata: Metadata = Field(default_factory=dict)

    @field_validator("request_id", "model_name", "finish_reason")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        return normalized_text(value, getattr(info, "field_name", "text"))

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        _reject_sensitive_text(value, "content")
        return value

    @field_validator("prompt_tokens", "completion_tokens", "total_tokens", mode="before")
    @classmethod
    def validate_optional_integer(cls, value: object, info: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{getattr(info, 'field_name', 'value')} must be an integer or None")
        return value

    @field_validator("latency_ms", mode="before")
    @classmethod
    def validate_latency(cls, value: object) -> float:
        if isinstance(value, bool):
            raise ValueError("latency_ms must be a finite number")
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("latency_ms must be a finite number") from exc
        if not math.isfinite(normalized):
            raise ValueError("latency_ms must be a finite number")
        return normalized

    @field_validator("provider_metadata")
    @classmethod
    def validate_metadata(cls, value: Metadata) -> Metadata:
        _reject_nonpublic_metadata(value, "LLMResponse.provider_metadata")
        return value

    @model_validator(mode="after")
    def validate_usage(self) -> "LLMResponse":
        usage = (self.prompt_tokens, self.completion_tokens, self.total_tokens)
        if all(value is None for value in usage):
            return self
        if any(value is None for value in usage):
            raise ValueError("usage fields must be all integers or all None")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens must equal prompt_tokens plus completion_tokens")
        return self


class LLMProvider(Protocol):
    """Provider boundary for rendered offline mediator requests."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return a serializable response or raise ``LLMProviderError``."""


def _reject_sensitive_text(value: str, field_name: str) -> None:
    lowered = value.lower()
    if any(token in lowered for token in ("api_key", "apikey", "authorization:", "bearer ", "sk-")):
        raise ValueError(f"{field_name} must not contain credential material")


def _reject_nonpublic_metadata(value: Metadata, field_name: str) -> None:
    """Reject credential, private-evaluation, and prompt-copy material recursively."""
    forbidden = (
        "api_key",
        "apikey",
        "authorization",
        "secret",
        "password",
        "hidden_probe",
        "hidden probe",
        "hidden_profile",
        "hidden profile",
        "valueprofile",
        "evaluationreport",
        "evaluation_report",
        "system_prompt",
        "user_prompt",
    )

    def visit(item: JsonValue, path: str) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                if any(token in key.lower() for token in forbidden):
                    raise ValueError(f"{field_name} contains prohibited field at {path}.{key}")
                visit(nested, f"{path}.{key}")
        elif isinstance(item, list):
            for index, nested in enumerate(item):
                visit(nested, f"{path}[{index}]")
        elif isinstance(item, str) and any(token in item.lower() for token in forbidden[:11]):
            raise ValueError(f"{field_name} contains prohibited private or credential material at {path}")

    visit(value, field_name)


__all__ = [
    "LLMProvider",
    "LLMProviderError",
    "LLMRequest",
    "LLMResponse",
    "ProviderErrorCode",
]
