# H-G.2.3.1 Provider-Subset Amendment

The accepted H-G.2.3 one-shot compatibility request was rejected before model
generation with the safe category `response_format_unsupported`. No successful
response, prompt, or compatibility receipt was stored.

H-G.2.3.1 changes only the Provider-facing JSON Schema. It removes
`uniqueItems` from the two ID arrays. The strict local H-G.2.3 parser continues
to reject duplicate content or evidence IDs, unknown fields, unavailable IDs,
and inconsistent sharing fields.

The H-G.2.3 system prompt and user prompt bytes are unchanged. Epistemic
stance, observable adoption, sharing semantics, scenario material, exposure
schedule, controller policy, evaluator boundary, metric names, metric
operationalization, model parameters, request cap, and token reservation are
unchanged. The transport protocol, schema name, request IDs, fingerprints, and
future output root are versioned independently.

This is a compatibility amendment, not a behavioral change or an effectiveness
claim. It remains development-only and requires exact-hash approval followed by
one new one-shot compatibility request before any real calibration.

