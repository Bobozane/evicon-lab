# LLM Provider Contract

Stage 14A defines an offline provider boundary in `evicon.llm_contract`.
`LLMRequest` contains a request ID, model name, rendered system and user
instructions, generation parameters, seed, and safe metadata. `LLMResponse`
contains the matching request ID, content, finish reason, token accounting,
latency, and safe provider metadata. Both models reject unknown fields and
serialize through Pydantic JSON.

`LLMProvider.complete(request)` is the only provider operation. Provider
implementations must return `LLMResponse` or raise `LLMProviderError` with a
stable error code. Credentials are deliberately absent from these contracts;
metadata rejects API keys, authorization material, secrets, copied prompts,
hidden probes, hidden profiles, and offline evaluation content.

`FakeLLMProviderAdapter` is a deterministic local adapter for this contract.
It synthesizes JSON from request ID, seed, fixed action, and target IDs, and
reports deterministic token and latency values without sleeping or connecting
to a network. It supports explicit invalid-JSON and invalid-evidence modes for
parser tests. It is not a real LLM provider and produces no experimental
result.

This stage does not log rendered prompts to `events.jsonl`, does not read API
keys, and does not register a real SDK or network provider. A future real
provider must implement this boundary separately and keep credential handling
outside serializable request, response, event, and result objects.
