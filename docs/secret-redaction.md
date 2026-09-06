# Secret Redaction

`EVICON_LLM_API_KEY` is read from the environment and held only by the
provider instance. It is used only to form the outbound HTTP `Authorization`
header. It is not a `ProviderConfig` field and is never placed in
`LLMRequest`, `LLMResponse`, provider metadata, event logs, result files, or
structured provider errors.

The request and response metadata validators reject credential-like fields,
including API-key, authorization, password, and secret names. They also reject
hidden probe/profile and offline evaluation material, as well as copied system
or user prompt fields. The provider metadata is limited to safe operational
facts: provider name, status code, retry count, local request ID, and model
name.

Rendered prompts remain transient request fields; Stage 14B does not append them to `events.jsonl` or persist them to results.
The default smoke does not emit prompts. Exception messages use fixed summaries rather than transport or response-body text.
