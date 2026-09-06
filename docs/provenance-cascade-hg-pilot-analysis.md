# H-G Provenance-Cascade Pilot Evaluator Analysis

This stage performs an evaluator-only, network-free analysis of the completed H-G public-content-identifiable Pilot. The Pilot receipt, final batch record, request ledgers, checkpoints, run records, H-D.2.1 artifacts, H-F report, WVS assets, and calibration artifacts remain read-only.

## Integrity chain

The analysis plan binds by SHA-256:

- the H-G config, Agent protocol/template, and controller policy;
- the public-identifiability amendment and its receipt;
- researcher approval and Provider compatibility receipt;
- the H-G replay technical amendment, historical validator, and H-G validator;
- the final Pilot receipt and final batch record;
- all 48 run records and their public exposure/application/outcome sidecars;
- the 24A preregistration and four evaluator-private truth fixtures.

The technical replay amendment is treated as a historical recovery authorization. It binds the pre-resume failed batch state, while the analysis manifest separately binds the completed final batch. The analysis does not rewrite either record.

All public hashes, run coordinates, matched groups, request ledgers, and three replay layers are validated before evaluator-private truth is loaded. Private truth is then joined only by stable scenario and claim IDs for offline scoring. It is never copied into the manifest or report.

## Frozen metrics

The nine operations remain those registered in 24A and H-F: false-cascade adoption, supported-correction retention, beneficial receptivity, harmful conformity, intervention false-positive rate, provenance diversity, replay audit pass rate, token cost, and latency. Every run-level metric retains scenario, seed, condition, numerator, denominator, and `not_applicable` status.

Condition summaries, scenario-by-condition summaries, and matched differences are descriptive. The report records whether condition means differ numerically, but this is not a significance test or a registered Go/No-Go threshold.

## Current Pilot interpretation boundary

The completed H-G Pilot passes data-integrity validation. It does not show substantive condition separation in the registered behavioral outcomes:

- false-cascade adoption is zero in every condition among the limited eligible units;
- supported-correction retention is identical across conditions;
- beneficial receptivity and harmful conformity have no eligible units and remain `not_applicable`;
- applied interventions have zero observed false positives, but only nine applied events exist;
- numerical differences occur only in token and latency summaries.

These observations do not establish effectiveness or causality. Because the preregistration contains no numeric Go/No-Go thresholds, the machine-readable decision remains `thresholds_not_pre_registered`. The scientifically conservative next step is a versioned re-pilot amendment that repairs behavioral eligibility and condition separation before any confirmatory design. Existing H-G results must not be merged with a revised Pilot.

## Commands

Validate without writing:

```bash
uv run python -m evicon.validate_provenance_cascade_hg_analysis \
  --config configs/provenance_cascade/pilot/provenance_cascade_pilot_hg_analysis.v1.toml
```

Write the isolated analysis artifacts:

```bash
uv run python -m evicon.provenance_cascade_hg_analysis \
  --config configs/provenance_cascade/pilot/provenance_cascade_pilot_hg_analysis.v1.toml
```

Neither command constructs a Provider, reads an API key, or accesses the network.
