# Conformity Monitor

`ConformityMonitor` is a deterministic, pure risk-monitoring component. It
accepts a single `MonitorInput` summary and an explicit `MonitorConfig`, then
returns a `MonitorResult`. It does not create files, call a model or network,
read a prompt, modify `DialogueState`, select an intervention action, or invoke
`OfflineEvaluator`.

## Contracts

`MonitorInput` contains only online-observable round summaries:

- run and scenario IDs, round ID, remaining budget, and observable peer count;
- current and previous diversity values, both non-negative;
- current and previous coverage, minority loss, evidence gain and quality,
  harm risk, and task quality, all in `[0, 1]`;
- metadata restricted to JSON values.

The contract rejects unknown fields. In particular, it has no field for a
hidden probe response or hidden `ValueProfile`. It has no future-round input.

`MonitorConfig` explicitly supplies non-negative weights, operational
thresholds in `[0, 1]`, and a version. It has no formal default configuration.
`task_quality_sufficient_threshold` is optional; when supplied, it is the
configured quality floor required before an evidence-supported update can be
recognized. Smoke parameters live only in `evicon.monitor_smoke` and are not
formal experimental values.

`MonitorResult` includes a clamped risk score, `low` / `medium` / `high` / `invalid`
level, eligibility flag, evidence-supported flag, auditable signal values,
stable reason enums, config version, and a validity flag. `invalid` is reserved
for future persisted-result handling; malformed inputs are rejected by Pydantic
with a clear validation error rather than silently converted to a result.

## Formula

The monitor computes:

```text
collapse_amount    = max(0, previous_diversity - current_diversity)
normalized_collapse = min(1, collapse_amount)
coverage_loss      = max(0, previous_coverage - current_coverage)
evidence_support   = evidence_gain * evidence_quality

raw_risk = collapse_weight * normalized_collapse
         + minority_weight * minority_loss
         + harm_weight * harm_risk
         - evidence_weight * evidence_support
         - evidence_quality_weight * evidence_quality

risk_score = clamp(raw_risk, 0, 1)
```

Coverage loss is always returned in `signals` and recorded in reasons, but it
does not receive a separate risk weight in this initial contract. This prevents
an uncalibrated additional coefficient from being introduced as if it were an
empirical result. Task quality is also always recorded, but it is not treated
as proof of value homogeneity. Because the input has no prior task-quality
value, the monitor cannot infer a task-quality decline.

An update is `evidence_supported_update=true` only when there is nonzero new
evidence, `evidence_support` reaches the configured sufficiency threshold, and
the optional configured task-quality floor is met. In that case,
`should_intervene=false` by default. Otherwise, `should_intervene=true` only
when `risk_score >= high_risk_threshold`. This flag means only that a later
policy could be eligible to act; it does not choose or generate an action.

Reason labels include diversity decrease/collapse, coverage loss, minority
reduction/loss, harm risk, task-quality concern, insufficient evidence,
evidence-supported update, and no observable peers. They are stable enums, not
generated prose.

## Online And Offline Boundary

The monitor may consume only structured online-observable state supplied by its
caller. Hidden probes remain isolated for offline evaluation. In particular:

- hidden probe responses and hidden profiles cannot enter `MonitorInput`;
- `social_influence_loss` remains a matched, offline `OfflineEvaluator` metric;
- the monitor does not treat that offline contrast as an online observation;
- the monitor is not connected to a Mediator or `ProtocolRunner`;
- `should_intervene` is not an intervention action or prompt.

The monitor cannot prove a strict causal mechanism, cannot diagnose a human
psychological state, and does not establish that a risk score represents social
conformity in people. It reports potential conformity risk in configured agent
behavior only. Weights and thresholds must be calibrated on a validation split
before any substantive experimental use; smoke values are not calibration.

## Smoke Command

Run the three local demonstration cases with:

```bash
uv run python -m evicon.monitor_smoke
```

The command prints one JSON object each for `stable`,
`social-collapse-without-evidence`, and `evidence-supported-update`. Each object
contains only its case label, risk score, risk level, eligibility flag, and
structured reason strings. The output is a branch check, not an experiment.

## Not Implemented

This stage does not implement a Mediator, action selection, real providers,
online protocol changes, persistence, calibration procedures, statistical
analysis, or causal claims.
