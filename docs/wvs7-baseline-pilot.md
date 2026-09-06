# WVS7 English Baseline Pilot

This is development-only orchestration for a locally frozen English core WVS
ProbeSet. It plans one explicit matched group for each seed across
`independent`, `social_only`, `evidence_only`, and `evidence_social`. It is
not a WVS benchmark replication, a four-condition research result, or a claim
about human values.

## Immutable Inputs

Before any mode is accepted, the pilot validates the local ProbeSet SHA-256
against both the study configuration and frozen manifest. It also requires the
English-only, 23-item, non-`Q48` contract and preserves the special-response
metadata for Q111, Q149/Q150, and Q241/Q243/Q246-Q249. Q158-Q163 remain a
secondary science/technology-attitude analysis, and the Q246 cross-language
semantic-scope warning remains visible in metadata. EviCon operational domains
are not asserted to be WVS official factors.

The scenario is original English development material. It is not a WVS question
and does not stand in for a real social case. Every condition shares the same
scenario, ordered agent IDs, model settings, rounds, seed, ProbeSet, and
pre/post item selection. Run IDs and matched group IDs are explicit in the
configuration; pairing is never inferred from filenames or result directories.

## Modes

`wvs7_baseline_dry_run` only reads and validates the plan. It makes no model
calls and writes no result artifacts. The configured smoke plan has four
conditions, one seed, two agents, two rounds, 16 public Agent turns, 184 pre
probe calls, 184 post probe calls, at most 384 provider requests, and a
conservative 51,200-token upper bound.

`fake-smoke` uses the existing deterministic `FakeLLM` through a local provider
adapter. It writes the existing public trajectory and separate private pre/post
probe artifacts, then validates every completed public run with the existing
replay validator. Same-round snapshots are created before any Agent response,
so a later Agent cannot see an earlier Agent response from the same round.

`real-pilot` remains intentionally disabled in this stage. It requires both the
explicit mode and `--allow-network` before reaching its guarded branch, which
still returns `real_pilot_not_enabled`. This module makes no implicit provider,
environment lookup, or network call.

## Failure and Recovery

Each condition records `planned`, `running`, `completed`, or `failed` in the
safe batch record. A failure stops remaining conditions in that matched group
and leaves them planned; incomplete groups are never presented as evaluable
matches. Existing run and batch directories are never overwritten. An
interrupted batch with completed runs and remaining planned runs can resume with
the same manifest and batch ID. A failed run directory is not silently reused.

## Offline Evaluation

After a completed FakeLLM group, the existing offline evaluator is invoked
without changing metric definitions. It requests pairwise diversity, structural
diversity, value-dimension coverage, matched social-influence loss, and audit
metrics. The frozen 23-item set has no separate holdout subset, so holdout
profile drift is skipped rather than fabricated.

FakeLLM output is engineering smoke data only. A matched social-influence-loss
estimate is not a strict causal effect, WVS responses are not claims about real
human values, and no result is fed to an online controller, Mediator, Monitor,
Policy, or Executor.
