# Hidden Probe Isolation

Value probes measure value state for later offline analysis. They are not mediator prompts and are never used to tell an online controller how to intervene.

`ValueProbeResponse` and `ProbeResult` remain outside `DialogueState`. They are excluded from `DialogueTurn`, exposure planning, evidence exposure, peer history, and the ordinary `events.jsonl` stream. A protocol run cannot discover a holdout answer through its visible context.

Probe results are stored only in a separate file:

```text
results/<run_id>/probes/probe_results.jsonl
```

This file contains private raw responses and normalized profiles for future offline metrics. `read_probe_results()` is an offline API. `ProtocolRunner`, `ExposurePlan`, and future mediator code do not import or call it automatically.

Ordinary protocol events contain no probe answer, value vector, holdout item text, `raw_response`, or `normalized_score`. A later reporting or metrics stage must make any aggregate disclosure explicitly and must not feed private probe results back into the online negotiation loop.
