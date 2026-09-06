# Intervention Policy

`InterventionPolicy` is a deterministic, side-effect-free eligibility and
targeting layer. It consumes a round-level `MonitorResult` and an explicitly
provided set of online-observable `TargetCandidate` values, then returns a
structured policy-specific `InterventionDecision`. It does not execute an
action, call an LLM, create a prompt, write an event, or modify
`DialogueState` or `ProtocolRunner`.

## Contracts

`PolicyInput` contains the run, scenario, and round identifiers; the matching
`MonitorResult`; candidates; remaining intervention budget; cooldown state; a
round-ordered intervention history; and metadata. It forbids unknown fields,
so hidden probe answers, hidden value profiles, prompt text, and future-round
data cannot enter through its top-level contract.

Each `TargetCandidate` has a unique agent identifier, marginal coverage gain,
minority-representation score, safety risk, estimated cost, eligibility flag,
and stable reason codes. All candidate scores are in `[0, 1]`. `PolicyConfig`
holds all selection thresholds, action costs, cooldown settings, and bounded
restructure settings explicitly. The smoke configuration is a local example,
not a calibrated experimental parameter set.

`InterventionDecision` records `action`, `target_agent_ids`, `risk_score`,
`estimated_cost`, stable `reason_codes`, `policy_version`, `cooldown_until`,
`round_id`, and `valid`. This policy decision is intentionally separate from
the legacy run-record decision model, preserving the online runner's existing
contract until a later integration phase.

## Target Selection

The pure `select_targets` function filters candidates in this order:

1. Candidate is eligible.
2. Marginal gain meets `min_marginal_gain`.
3. Safety risk does not exceed `max_target_safety_risk`.

Surviving candidates are ranked deterministically by marginal coverage gain
descending, minority-representation score descending, safety risk ascending,
estimated cost ascending, then `agent_id` ascending. Empty selections retain a
stable structured reason such as `empty_candidates`, `no_eligible_candidates`,
or `all_above_safety_limit`.

## Policy Boundaries

The policy returns `no_op` when the monitor result is invalid or not eligible
for intervention, budget is exhausted, or cooldown is active. It does not
override an evidence-supported update. For an eligible high-risk result, it
uses a conservative evidence request when evidence is insufficient. With a
safe selected minority representative, minority loss takes precedence over the
generic evidence fallback and produces `solicit_dissent`. Harm signals avoid
unsafe targets and can request evidence or produce a structured minority
report. Action costs are checked against the remaining budget before a
non-`no_op` decision is returned.

`restructure` is available only when the explicitly configured number of
consecutive non-no-op failures is reached, risk meets the configured threshold,
and no previous restructure appears in the supplied history. That one-time
guard prevents unbounded escalation. `should_intervene` still only means
eligibility; this policy does not apply an action.

The policy is not a factual verifier, causal estimator, value judge, prompt
generator, or mediator. It uses neither hidden probes nor offline matched
metrics such as social influence loss. Thresholds and costs must be calibrated
and reported by a later experiment design; none here establishes an empirical
or causal conclusion.

## Local Smoke

Run the local-only deterministic examples with:

```bash
uv run python -m evicon.policy_smoke
```

It prints one JSON object for `stable`, `social-collapse-without-evidence`, and
`repeated-failure-restructure`. Each object contains only decision fields:
action, targets, risk score, estimated cost, reason codes, and cooldown.
