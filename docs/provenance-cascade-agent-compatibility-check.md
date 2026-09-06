# H-D.3 Provider Compatibility Check

H-D.3 is a one-shot transport compatibility check for the versioned
`provenance_cascade_agent_protocol.v2` contract. It is independent of the H-D
Pilot and does not read pilot scenarios, WVS material, private truth, result
directories, request ledgers, or historical fingerprints.

The request uses the `cascade_agent_turn.v2.strict_json` contract and an
opt-in OpenAI-compatible `response_format` of `json_schema`, named
`cascade_agent_response_v2`. The schema allows only `stance`,
`content_ids_used`, `evidence_ids_used`, and nullable `share_content_id`.
Provider configuration is fixed for this check at `max_retries=0`, a short
five-second timeout, temperature `0.0`, and `max_tokens=64`. Therefore the
explicit network path can issue at most one HTTP request.

Default invocation is offline:

```bash
uv run python -m evicon.cascade_agent_protocol_v2_compatibility
```

It returns `network_disabled`, constructs no Provider, reads no API key, and
writes no files. A real compatibility request would require a separate,
explicit authorization and:

```bash
uv run python -m evicon.cascade_agent_protocol_v2_compatibility --allow-network
```

Before that command, `EVICON_LLM_BASE_URL`, `EVICON_LLM_MODEL`, and
`EVICON_LLM_API_KEY` must be configured locally. The key is used only in the
HTTP Authorization header and is never printed or retained. The command must
remain a compatibility check only: it does not run an Agent, Pilot, ledger,
resume path, or experiment.

Output is limited to status, safe provider error code, model, finish reason,
parser validity, token usage, latency, response format/schema, attempt count,
network status, and safety flags.

## H-D.3.2 generation-parameter alignment

The compatibility request reads the H-D.2 v1 seed list and its
hash-registered parent provider settings: `temperature=0.2`, `max_tokens=256`,
and the first H-D.2 seed (`20260911`) as the deterministic representative of
the per-run seed schedule. The v1 sidecar does not redeclare the provider
settings, so the loader follows its explicit `parent_config_path` and verifies
the registered parent SHA-256 instead of inventing defaults. H-D.2 does not declare an optional `reasoning_effort`, so
that field remains absent from the payload. The check still enforces
`max_retries=0`, a five-second timeout, and at most one request. These settings
are copied into the one-shot compatibility request only; neither H-D.2
configuration is changed.

The earlier 64-token check returned `finish_reason="length"` and an invalid
strict-JSON parse because the response was truncated at that artificial
compatibility budget. That observation is a truncation diagnostic, not
evidence that the JSON Schema is incompatible. A 256-token response that
finishes with `stop` is the aligned compatibility case; a response that still
finishes with `length` remains an invalid response and is reported without
repair or retry.

H-D.3.1 adds only stable transport categories: `timeout`,
`connection_failure`, `dns_or_url_failure`, `authentication_failed`,
`http_client_error`, `http_server_error`, `rate_limited`,
`malformed_provider_response`, `invalid_provider_response`,
`response_format_unsupported`, and `provider_failure`. HTTP output is reduced
to a class such as `2xx`, `4xx`, or `5xx`; response bodies are inspected only
transiently to recognize a narrowly supported `response_format` capability
error and are never retained. The safe result also reports
`transport_category`, `http_status_class`, `response_format_requested`, and
`attempt_count` (with the existing request-attempt field retained). This does
not fall back to `json_object`, change the protocol, or add retries.
 It never includes the request, response,
headers, provider metadata, or prompt. A successful check means the endpoint
accepted the request format and returned a parser-compatible response; it does
not establish model quality or any research result. A failure should first be
handled as a provider/schema compatibility issue, without changing the H-D.2
protocol or retrying outside the fixed one-shot boundary.
