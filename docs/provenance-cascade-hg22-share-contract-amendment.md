# H-G.2.2 explicit share-contract amendment

H-G.2.1 execution v2 stopped after the second successful Provider transport response because the local parser returned the stable category `share_field`. The saved audit contains no response body, prompt, field value, key, or Provider metadata.

The old parser checks exact field presence before share semantics, so a missing `share_content_id` is ruled out. It also checks the behavioral-decision field type and enum before share semantics. However, the old `share_field` category intentionally combined two cases: an invalid `share_content_id` type and a cross-field inconsistency. The existing safe artifacts cannot distinguish those two cases without retaining prohibited response content. Because the request used a compatibility-tested strict JSON Schema that constrains `share_content_id` to string or null, the evidence is most consistent with a cross-field inconsistency. This is a technical diagnosis, not an experimental finding.

H-G.2.2 preserves the five-field output, epistemic stances, behavioral decisions, public scenarios, roles, controller rules, evaluator boundary, seed, and calibration scale. It changes only protocol identity and the explicit share contract:

- `share` and `share_with_caveat` require one non-null, visible `share_content_id` that is also listed in `content_ids_used`;
- `do_not_share`, `support_reversible_action`, and `defer_action` require `share_content_id = null`;
- `content_ids_used` may still record visible content considered by a non-sharing decision;
- the Provider JSON Schema validates only object shape, enums, and field types;
- the local validator rejects duplicate or unavailable IDs and cross-field inconsistencies;
- stable errors distinguish `share_field_type` from `share_behavior_inconsistent`.

There is no lenient parsing, Markdown extraction, field deletion, response repair, semantic retry, or parser-invalid recovery. H-G.2.1 v1 and v2 failure directories remain immutable and must not be resumed or combined with H-G.2.2.

H-G.2.2 uses a new protocol/template/schema name, run namespace, request fingerprints, approval, compatibility receipt, and output root. It remains a one-seed development eligibility calibration with 16 runs, four matched groups, 288 logical requests, and a 294,912 completion reservation. It is not a Pilot effectiveness result or paper result.
