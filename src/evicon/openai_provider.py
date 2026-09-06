"""Protected OpenAI-compatible provider with an opt-in network boundary."""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator, model_validator

from .llm_contract import LLMProviderError, LLMRequest, LLMResponse, ProviderErrorCode
from .models._validation import normalized_text


class ReasoningEffort(str, Enum):
    """GPT-5.6 reasoning effort values accepted by the provider contract."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class ResponseFormatMode(str, Enum):
    """Opt-in response-format hints supported by OpenAI-compatible gateways."""

    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


class ProviderConfig(BaseModel):
    """Serializable provider settings. API keys intentionally are not a field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: str | None = None
    model_name: str | None = None
    timeout_seconds: float = Field(default=15.0, gt=0.0)
    max_retries: int = Field(default=1, ge=0)
    allow_network: bool = False
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=64, gt=0)
    seed: int = 0
    reasoning_effort: ReasoningEffort | None = None
    response_format: ResponseFormatMode | None = None
    response_schema_name: str | None = None
    response_schema: dict[str, JsonValue] | None = None

    @field_validator("base_url", "model_name")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("response_schema_name")
    @classmethod
    def normalize_schema_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_response_format(self) -> "ProviderConfig":
        if (self.response_schema_name is not None or self.response_schema is not None) and self.response_format is not ResponseFormatMode.JSON_SCHEMA:
            raise ValueError("response schema settings require response_format=json_schema")
        return self

    @field_validator("timeout_seconds", "temperature", mode="before")
    @classmethod
    def validate_float(cls, value: object, info: object) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{getattr(info, 'field_name', 'value')} must be a finite number")
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{getattr(info, 'field_name', 'value')} must be a finite number") from exc
        if normalized != normalized or normalized in (float("inf"), float("-inf")):
            raise ValueError(f"{getattr(info, 'field_name', 'value')} must be finite")
        return normalized

    @field_validator("max_retries", "max_tokens", "seed", mode="before")
    @classmethod
    def validate_int(cls, value: object, info: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{getattr(info, 'field_name', 'value')} must be an integer")
        return value

    @classmethod
    def from_env(cls, *, allow_network: bool = False, environment: Mapping[str, str] | None = None) -> "ProviderConfig":
        """Read non-secret settings from the documented environment variables."""
        source = os.environ if environment is None else environment
        return cls(
            base_url=source.get("EVICON_LLM_BASE_URL") or None,
            model_name=source.get("EVICON_LLM_MODEL") or None,
            allow_network=allow_network,
        )


@dataclass(frozen=True)
class TransportResponse:
    """Minimal transport result usable by injected offline tests."""

    status_code: int
    body: str


class OpenAICompatibleTransport(Protocol):
    """Injectable HTTPS boundary. Tests should use a fake implementation."""

    def post(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> TransportResponse:
        """Submit one JSON request or raise a transport exception."""


class UrllibTransport:
    """Standard-library transport used only after ``allow_network=True``."""

    def post(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> TransportResponse:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
            headers=dict(headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                return TransportResponse(
                    status_code=int(response.status),
                    body=response.read().decode("utf-8"),
                )
        except urllib.error.HTTPError as error:
            return TransportResponse(status_code=error.code, body=error.read().decode("utf-8", errors="replace"))
        except (TimeoutError, socket.timeout) as error:
            raise TimeoutError("provider transport timed out") from error
        except urllib.error.URLError as error:
            if isinstance(error.reason, socket.gaierror):
                raise socket.gaierror("provider URL resolution failed") from None
            raise ConnectionError("provider transport failed") from None


class OpenAICompatibleProvider:
    """OpenAI chat-completions provider guarded by an explicit network opt-in."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        transport: OpenAICompatibleTransport | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self._transport = transport or UrllibTransport()
        source = os.environ if environment is None else environment
        self._api_key = source.get("EVICON_LLM_API_KEY") or None

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Perform an explicitly enabled chat-completions request with finite retries."""
        self._validate_prerequisites()
        started_at = time.perf_counter()
        retries = 0
        while True:
            try:
                response = self._transport.post(
                    _completion_url(self.config.base_url or ""),
                    {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self._api_key}",
                    },
                    self._payload(request),
                    self.config.timeout_seconds,
                )
            except (TimeoutError, socket.timeout):
                if retries < self.config.max_retries:
                    retries += 1
                    continue
                raise LLMProviderError(ProviderErrorCode.TIMEOUT, "provider request timed out") from None
            except (socket.gaierror, urllib.error.URLError, ValueError):
                if retries < self.config.max_retries:
                    retries += 1
                    continue
                raise LLMProviderError(ProviderErrorCode.DNS_OR_URL_FAILURE, "provider URL resolution failed") from None
            except ConnectionError:
                if retries < self.config.max_retries:
                    retries += 1
                    continue
                raise LLMProviderError(ProviderErrorCode.CONNECTION_FAILURE, "provider connection failed") from None
            except Exception:
                raise LLMProviderError(ProviderErrorCode.PROVIDER_FAILURE, "provider transport failed") from None

            if 200 <= response.status_code < 300:
                return self._parse_response(
                    response,
                    request=request,
                    retries=retries,
                    latency_ms=(time.perf_counter() - started_at) * 1000.0,
                )
            if response.status_code == 429:
                if retries < self.config.max_retries:
                    retries += 1
                    continue
                raise LLMProviderError(ProviderErrorCode.RATE_LIMITED, "provider rate limit exhausted", status_code=response.status_code)
            if 500 <= response.status_code < 600:
                if retries < self.config.max_retries:
                    retries += 1
                    continue
                raise LLMProviderError(ProviderErrorCode.HTTP_SERVER_ERROR, "provider server error", status_code=response.status_code)
            if response.status_code == 401:
                raise LLMProviderError(ProviderErrorCode.AUTHENTICATION_FAILED, "provider authentication failed", status_code=response.status_code)
            if 400 <= response.status_code < 500:
                if response.status_code == 400 and self.config.response_format is not None and _response_format_unsupported(response.body):
                    raise LLMProviderError(ProviderErrorCode.RESPONSE_FORMAT_UNSUPPORTED, "provider response format is unsupported", status_code=response.status_code)
                raise LLMProviderError(ProviderErrorCode.HTTP_CLIENT_ERROR, "provider client error", status_code=response.status_code)
            raise LLMProviderError(ProviderErrorCode.PROVIDER_FAILURE, "provider returned an unsupported status", status_code=response.status_code)

    def _validate_prerequisites(self) -> None:
        if not self.config.allow_network:
            raise LLMProviderError(ProviderErrorCode.NETWORK_DISABLED, "network access is disabled")
        if not self.config.base_url:
            raise LLMProviderError(ProviderErrorCode.MISSING_BASE_URL, "provider base URL is not configured")
        if not self.config.model_name:
            raise LLMProviderError(ProviderErrorCode.MISSING_MODEL, "provider model is not configured")
        if not self._api_key:
            raise LLMProviderError(ProviderErrorCode.MISSING_API_KEY, "provider API key is not configured")

    def _payload(self, request: LLMRequest) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "seed": self.config.seed,
        }
        if self.config.reasoning_effort is not None:
            payload["reasoning_effort"] = self.config.reasoning_effort.value
        if self.config.response_format is ResponseFormatMode.JSON_OBJECT:
            payload["response_format"] = {"type": ResponseFormatMode.JSON_OBJECT.value}
        elif self.config.response_format is ResponseFormatMode.JSON_SCHEMA:
            payload["response_format"] = {
                "type": ResponseFormatMode.JSON_SCHEMA.value,
                "json_schema": {
                    "name": self.config.response_schema_name or "evicon_response",
                    "strict": True,
                    "schema": self.config.response_schema or {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "stance": {"type": "string", "enum": ["endorses", "rejects", "uncertain", "no_position"]},
                            "content_ids_used": {"type": "array", "items": {"type": "string"}},
                            "evidence_ids_used": {"type": "array", "items": {"type": "string"}},
                            "share_content_id": {"type": ["string", "null"]},
                        },
                        "required": ["stance", "content_ids_used", "evidence_ids_used", "share_content_id"],
                    },
                },
            }
        return payload

    def _parse_response(
        self,
        response: TransportResponse,
        *,
        request: LLMRequest,
        retries: int,
        latency_ms: float,
    ) -> LLMResponse:
        try:
            decoded = json.loads(response.body)
        except (TypeError, json.JSONDecodeError):
            raise LLMProviderError(ProviderErrorCode.MALFORMED_RESPONSE, "provider returned malformed JSON", status_code=response.status_code) from None
        if not isinstance(decoded, dict):
            raise LLMProviderError(ProviderErrorCode.INVALID_RESPONSE, "provider response must be an object", status_code=response.status_code)
        choices = decoded.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LLMProviderError(ProviderErrorCode.INVALID_RESPONSE, "provider response has no choice", status_code=response.status_code)
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise LLMProviderError(ProviderErrorCode.INVALID_RESPONSE, "provider response has no message content", status_code=response.status_code)
        finish_reason = choice.get("finish_reason")
        if not isinstance(finish_reason, str) or not finish_reason.strip():
            finish_reason = "unknown"
        prompt_tokens, completion_tokens, total_tokens = _usage(decoded.get("usage"), status_code=response.status_code)
        model_name = decoded.get("model")
        if not isinstance(model_name, str) or not model_name.strip():
            model_name = self.config.model_name or request.model_name
        try:
            return LLMResponse(
                request_id=request.request_id,
                model_name=model_name,
                content=message["content"],
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                provider_metadata={
                    "provider_name": "openai_compatible",
                    "status_code": response.status_code,
                    "retry_count": retries,
                    "request_id": request.request_id,
                    "model_name": model_name,
                },
            )
        except ValidationError:
            raise LLMProviderError(ProviderErrorCode.INVALID_RESPONSE, "provider response violates the output contract", status_code=response.status_code) from None


def _response_format_unsupported(body: object) -> bool:
    """Return true only for a narrow, non-sensitive 400 format diagnostic.

    The body is inspected transiently and never included in an exception,
    ledger, metadata object, or CLI result. Requiring both a format marker and
    an explicit unsupported/invalid capability marker avoids classifying an
    arbitrary client error as a compatibility failure.
    """
    if not isinstance(body, str):
        return False
    lowered = body.lower()
    format_marker = any(marker in lowered for marker in (
        "response_format", "response format", "json_schema", "json schema",
    ))
    unsupported_marker = any(marker in lowered for marker in (
        "unsupported", "not support", "unrecognized", "unknown", "not allowed",
    ))
    return format_marker and unsupported_marker


def _completion_url(base_url: str) -> str:
    normalized = normalized_text(base_url, "base_url").rstrip("/")
    return normalized if normalized.endswith("/chat/completions") else f"{normalized}/chat/completions"


def _usage(value: object, *, status_code: int | None = None) -> tuple[int | None, int | None, int | None]:
    if value is None:
        return None, None, None
    if not isinstance(value, dict):
        raise LLMProviderError(ProviderErrorCode.INVALID_RESPONSE, "provider usage must be an object", status_code=status_code)
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    if any(field not in value for field in fields):
        return None, None, None
    parsed: list[int] = []
    for field in fields:
        token_count = value[field]
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
            raise LLMProviderError(ProviderErrorCode.INVALID_RESPONSE, "provider usage values must be non-negative integers", status_code=status_code)
        parsed.append(token_count)
    if parsed[2] != parsed[0] + parsed[1]:
        raise LLMProviderError(ProviderErrorCode.USAGE_MISMATCH, "provider usage totals are inconsistent", status_code=status_code)
    return parsed[0], parsed[1], parsed[2]


__all__ = [
    "OpenAICompatibleProvider",
    "OpenAICompatibleTransport",
    "ProviderConfig",
    "ReasoningEffort",
    "TransportResponse",
    "UrllibTransport",
]
