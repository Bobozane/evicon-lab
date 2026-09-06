# H-G.1 Provider Compatibility Gate

The H-G.1 identifiability pilot uses a separate, one-shot Provider compatibility check before any pilot authorization. The check renders only a synthetic public library-schedule context. It does not open H-G.1 scenario files, evaluator-private truth fixtures, Pilot output directories, or request ledgers.

The check is offline by default. Its opt-in path permits exactly one HTTP request with `response_format=json_schema`, `cascade_agent_response_v2_1`, `max_tokens=512`, `temperature=0.2`, `seed=20261001`, `max_retries=0`, and a five-second timeout. A successful check is evidence only that this response format can be parsed for the fixed request contract; it is not a Pilot result or a claim about intervention effects.

The safe receipt stores only completion status, parser status, finish reason, HTTP status class, model identifier, response-format and schema identifiers, one-shot attempt count, token counts, latency, and bindings to the H-G.1 config, protocol/template, and amendment hashes. It never stores prompts, model output, request IDs, headers, API keys, provider metadata, evaluator truth, or scenario text.

Receipt registration requires a separately accepted H-G.1 approval record. The approval receives the receipt SHA-256 only after the completed one-shot result has passed receipt validation. Until both the receipt and acceptance exist, H-G.1 final preflight remains blocked and no Pilot Provider may be constructed.
