# Provenance Cascade Pilot Preregistration

This record freezes an offline engineering pilot for auditing unsupported
information cascades in a synthetic six-agent network. It is a preregistered
configuration, not a result and not a claim about real public opinion.

The pilot has four scenario types (`false_majority`,
`true_minority_correction`, `independent_true_consensus`, and
`unresolved_disagreement`) and four explicitly matched conditions:
`no_intervention`, `generic_dissent`, `source_blind_controller`, and
`provenance_aware_controller`. Every future cell uses six agents, three rounds,
one six-node ring topology, and one of three preregistered seeds. This stage
creates no runs or result directories.

Evaluator truth labels and human source-independence labels are private
annotations. They are used only by offline evaluators and never enter prompts,
controller inputs, event metadata, or public exposure. A controller may use
public verification state, public evidence, source-root relations, and actual
exposure only. Majority/minority counts alone are never intervention triggers.

The two uncertain scenarios default to `abstain`. Pilot success is defined as
reducing false cascade adoption without materially harming retention of a
supported correction. The fixed primary metrics are false-cascade adoption,
supported-correction retention, beneficial receptivity, harmful conformity,
intervention false-positive rate, provenance diversity, replay audit pass rate,
token cost, and latency. No thresholds or outcomes are inferred here.

All material is author-original, English, synthetic, and development-only. The
offline validator reads only the TOML file and emits a safe summary. A later
stage must provide synthetic evidence/scenario contracts, evaluator-only label
storage, exposure/audit fixtures, and pre-registered metric implementations
before any runner can be considered.
