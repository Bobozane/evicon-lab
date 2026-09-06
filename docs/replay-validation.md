# Replay Validation

The replay validator reads an existing `events.jsonl` and `run_record.json`. It never calls `FakeLLM`, contacts a provider, or modifies either input.

```bash
uv run python -m evicon.replay_validate \
  --events results/scenario-smoke-run/events.jsonl \
  --record results/scenario-smoke-run/run_record.json
```

It checks that event sequences are continuous from 1, `run_started` is first, a terminal event is last, started rounds are completed, and every LLM request has one later response. It also checks unique completed turn IDs and compares the event log with the persisted `RunRecord`.

Visibility validation uses the scenario snapshot stored in `run_record.json`:

- Peer turn references must be from a strictly earlier round.
- Evidence must have been introduced by the turn's round.
- Evidence must be visible to the speaking agent.
- Holdout/value-probe material is prohibited in the event log.

Completed runs must cover all configured rounds and agree with the record's run ID and turn count. A malformed log produces a nonzero CLI exit with a specific validation error.

The validator establishes structural consistency only. It does not judge claim truth, compute diversity or conformity metrics, execute value probes, run a monitor, or select mediation actions.
