# OpenAI-Compatible Provider

`OpenAICompatibleProvider` implements the Stage 14A `LLMProvider` contract
for OpenAI-compatible `/chat/completions` endpoints. It is not connected to a
protocol runner or the adaptive control loop.

The non-secret settings are represented by serializable `ProviderConfig`:
`base_url`, `model_name`, `timeout_seconds`, `max_retries`, `allow_network`,
`temperature`, `max_tokens`, `seed`, and optional `reasoning_effort`. The API
key is deliberately not a configuration field. It is read only from
`EVICON_LLM_API_KEY` at provider construction time. `ProviderConfig.from_env()`
reads the two non-secret environment variables `EVICON_LLM_BASE_URL` and
`EVICON_LLM_MODEL`.

`reasoning_effort` is optional and accepts only the GPT-5.6 values `none`,
`low`, `medium`, `high`, `xhigh`, and `max`. When it is `None`, the provider
omits the field from the OpenAI-compatible request payload entirely, preserving
compatibility with relays that do not support it. It is not configured from an
environment variable, and it contains no credential material.

Network access is disabled by default. Calling `complete()` with
`allow_network=False` fails with `network_disabled` before resolving provider
settings or invoking the transport. With networking enabled, missing base URL,
model, or key fail with `missing_base_url`, `missing_model`, or
`missing_api_key`.

The provider retries only timeouts, HTTP 429, and 5xx responses, up to
`max_retries`. It does not retry 4xx responses. Every transport, HTTP, or
response-shape failure becomes `LLMProviderError` with a stable error code.
The provider does not retain response bodies for failed requests.

Usage is preserved only when the compatible response supplies consistent
`prompt_tokens`, `completion_tokens`, and `total_tokens`. Absent or partial
usage becomes three explicit `None` values; inconsistent totals are rejected
with `usage_mismatch`. No token values are estimated by this provider.
