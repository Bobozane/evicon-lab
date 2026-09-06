# H-D.2 Strict Agent Protocol

H-D.2 is an isolated, development-only revision of the provenance-cascade Agent output contract. The historical `provenance-cascade-pilot-hd-v2` directory, batch record, recovery amendment, and request ledger remain append-only and are never resumed by this protocol.

The new template is `cascade_agent_turn.v2.strict_json` and the protocol is `provenance_cascade_agent_protocol.v2`. A response is exactly one JSON object with exactly these fields: `stance`, `content_ids_used`, `evidence_ids_used`, and nullable `share_content_id`. IDs must be visible in the current public view; sharing must be a member of the used content IDs. No claim, target, action, provenance root, truth label, or controller field can be returned by an Agent.

Parsing is strict and never extracts Markdown, drops fields, repairs JSON, or retries parsing. Diagnostics are stable categories only: malformed JSON, top-level type, missing field, extra field, field type, invalid stance, unavailable content/evidence ID, or share-field error. Duplicate keys and private/controller key attempts are reported only as `extra_field`; no separate value-bearing diagnostic exists. Raw response, field values, prompts, credentials, provider metadata, and evaluator truth are not retained in diagnostics or audit summaries.

The OpenAI-compatible Provider has an opt-in `response_format`. With no setting, payload behavior is unchanged. H-D.2 uses `json_schema` with a strict four-field schema; `json_object` is also supported. This is a request-format compatibility hint, not an experimental manipulation or result claim. A real compatibility check would require separate authorization for one explicitly bounded network request with configured base URL, model, and key; none is performed by default.

The H-D.2 configuration keeps four scenarios, four conditions, three seeds, 48 runs, 864 logical requests, and completion reservation 221184. FakeProvider smoke uses a temporary directory and is not a paper result.
