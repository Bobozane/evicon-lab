# H-G.1.1 Offline Pilot Analysis

This evaluator-only analysis is bound to the completed H-G.1.1 Pilot through the
locked config, protocol, controller, replay implementation, amendment, approval,
compatibility receipt, batch record, receipt, and all 48 safe run-record hashes.
It never constructs a provider or accesses a network.

All public run material, ledgers, coordinates, and three replay layers are validated
before evaluator-private truth fixtures are loaded for scoring. Private labels are
used only in memory and are excluded from the manifest and report. The report contains
only hashes, coordinates, metric values, denominators, operational counts, and safe
warnings.

The analysis preserves the nine preregistered 24A metrics. It reports scenario by
seed by condition units, descriptive condition summaries, and matched differences.
Logical requests and transport attempts remain separate: retry attempts are audit
facts, not additional experimental coordinates.

With three seeds, the output is a development-only descriptive Pilot report. No
significance threshold, Go/No-Go threshold, causal interpretation, or paper result is
created. H-G.1 and H-D.2.1, WVS, `002`, `003`, and calibration materials are excluded
from this analysis.
