# Source-Behavior v2 Descriptive Analysis Lock

The 12-case source-behavior qualification finished before this analysis lock
was created. This document and its TOML are therefore a post-collection,
outcome-blind analysis lock, not a preregistration. The lock binds the completed
qualification receipt and safe audit by SHA-256 before any adoption or sharing
distribution is computed.

The lock is intentionally narrow. It treats `scenario_id`, not a case row, as
the pairing unit. Adoption is the primary descriptive outcome and sharing is
secondary. Both are binary public decisions. The two fixed contrasts are:

- `same_root - source_free`;
- `independent_roots - same_root`.

For each outcome and contrast, a future approved report may contain only the
eligible scenario count, positive/zero/negative paired-difference counts, and
the mean paired difference. It may not add p-values, confidence intervals,
significance thresholds, go/no-go criteria, case-level values, private truth,
or historical result joins.

## Mandatory limitation

The runner presented projections in the fixed order `source_free`, `same_root`,
then `independent_roots` within every scenario. Projection and request order are
therefore perfectly confounded. With one model, one seed, and four scenarios,
the resulting contrasts are descriptive qualification observations only. They
cannot identify a source-structure effect, support a causal claim, or justify
population or model generalization.

## Offline gate

The frozen lock and its independent approval sidecar are:

```text
configs/provenance_cascade/identification/conformity_source_behavior_analysis_lock.v2.toml
configs/provenance_cascade/identification/conformity_source_behavior_analysis_approval.v2.toml
```

The researcher accepted the analysis lock on 2026-08-28. This command performs
only an offline hash and safe-receipt preflight:

```bash
.venv/bin/python -m evicon.conformity_source_behavior_analysis_lock
```

It does not parse case-level outcomes, compute decision distributions, write an
analysis result, read private truth, construct a Provider, read an API key, or
use the network. Acceptance of the sidecar does not authorize analysis
execution; that remains a separate explicit step after the analysis
implementation and its Fake-data tests are reviewed.

## Implemented Fake-data core

`evicon.conformity_source_behavior_descriptive_analysis` now implements the
locked aggregation over caller-supplied, in-memory `SafeBehaviorCaseAudit`
objects. Its Fake-data tests cover the two contrasts, aggregate-only output,
missing cases, duplicate case IDs, duplicate scenario/projection coordinates,
and the absence of file reads. The module deliberately has no paths, audit
loader, CLI, report writer, Provider integration, or network behavior.

Consequently, it cannot read the completed qualification audit by default or
create an analysis result. A future dedicated execution gate must first bind
this implementation to the accepted analysis lock, require a new explicit
researcher authorization, and fail closed before any real case-level decision
is read. That gate must preserve the aggregate-only output and all limitations
above.

## Analysis execution gate

The one-shot execution gate is implemented in
`src/evicon/conformity_source_behavior_analysis_execution.py` and is bound by
the separate sidecar:

```text
configs/provenance_cascade/identification/conformity_source_behavior_analysis_execution_approval.v2.toml
```

The sidecar is accepted for this one-shot descriptive analysis. Its SHA-256 is
bound to the analysis lock, accepted analysis approval, qualification receipt
and audit, and the execution module. The offline preflight is:

```bash
.venv/bin/python -m evicon.conformity_source_behavior_analysis_execution
```

It does not parse case-level decisions or write output. After the sidecar is
independently accepted, a separate one-time command authorization is still
required:

```bash
.venv/bin/python -m evicon.conformity_source_behavior_analysis_execution \
  --allow-analysis \
  --confirm-run
```

That command may be run once only. A successful run would read the bound audit
and write exactly `descriptive_summary.json` and
`analysis_execution_receipt.json` under the new analysis output root. It would
not write case-level outcomes, prompts, raw responses, credentials, private
truth, historical joins, or inferential statistics.
