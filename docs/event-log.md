# Event Log Contract

## Files

A successful run creates a new, previously absent directory:

```text
<output_dir>/<run_id>/
  events.jsonl
  run_record.json
```

The runner refuses to use an existing run directory. This prevents an
append-only event log from being confused with a second execution of the same
run ID.

## JSONL event shape

Each `events.jsonl` line is one valid JSON object with deterministic fields:

```json
{
  "event_id": "run-id:000001",
  "event_type": "run_started",
  "run_id": "run-id",
  "round_id": 0,
  "sequence": 1,
  "payload": {}
}
```

`sequence` replaces wall-clock timestamps, so a fixed configuration and seed
produce identical core event content. The event vocabulary is:

```text
run_started
round_started
exposure_created
llm_request
llm_response
turn_completed
round_completed
run_completed
run_failed
```

Exposure and request events retain visible history IDs, peer-turn IDs, and
evidence IDs. Response and turn events retain the local response and the same
visibility references. This is sufficient to replay the visibility inputs of a
run without reading hidden probe answers.

`evicon.replay_validate` performs read-only structural validation against the
ScenarioSpec snapshot in `run_record.json`; it does not call a model provider.

## Failure behavior

If a local provider or contract check raises after the output directory exists,
the runner preserves all prior event lines, appends `run_failed`, and writes a
failed `run_record.json`. Failure payloads contain only the exception type, not
the original exception text, credentials, or environment values.

The logger never receives hold-out probe responses, API keys, provider request
headers, or environment variables.
