"""Provider configuration is serializable and deliberately key-free."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evicon.openai_provider import ProviderConfig


def test_provider_config_round_trips_without_an_api_key() -> None:
    config = ProviderConfig(
        base_url="https://example.invalid/v1",
        model_name="test-model",
        timeout_seconds=2.0,
        max_retries=3,
        allow_network=False,
        temperature=0.1,
        max_tokens=32,
        seed=4,
    )

    assert ProviderConfig.model_validate_json(config.model_dump_json()) == config
    assert "api_key" not in config.model_dump_json().lower()
    with pytest.raises(ValidationError, match="api_key"):
        ProviderConfig.model_validate({**config.model_dump(), "api_key": "forbidden"})


@pytest.mark.parametrize(
    "field,value",
    [("timeout_seconds", -1.0), ("max_retries", -1), ("max_tokens", 0)],
)
def test_provider_config_rejects_invalid_limits(field: str, value: object) -> None:
    values = {
        "base_url": "https://example.invalid/v1",
        "model_name": "test-model",
        field: value,
    }

    with pytest.raises(ValidationError):
        ProviderConfig.model_validate(values)
