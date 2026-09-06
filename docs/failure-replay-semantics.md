# Failure Replay Semantics

Replay validation distinguishes completed and failed runs without rerunning a provider or changing the log.

For a completed run, `run_started` is first, every configured round has matching `round_started` and `round_completed`, and `run_completed` is the final event. No round may remain open.

For a failed run, `run_started` is first and `run_failed` is final. One final round may be open because a provider can fail after that round begins. Every already-written LLM request still has a structured response: successful requests have a normal response, and the failing request receives a response with only `status="failed"` and an exception type. No exception text, prompt, or credential is logged.

`run_failed` must be the final event. A missing terminal event, an event after `run_failed`, non-continuous sequence numbers, unmatched requests and responses, premature evidence, or invisible evidence is invalid.

The CLI prints `validation=passed` for completed runs and `validation=passed_with_failure` for valid failed runs.
