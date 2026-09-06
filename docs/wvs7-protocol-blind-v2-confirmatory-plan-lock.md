# Protocol-Blind v2 Confirmatory Plan Lock

The formal plan at `configs/studies/wvs7_english_protocol_blind_v2_confirmatory_plan.toml` freezes only the intended design: one independently reviewed synthetic English test scenario, the audited English 23-item frozen ProbeSet, three preregistered new seeds, and four explicit protocol conditions per seed. The earlier template remains a blocked template and is not replaced by this plan.

The offline lock CLI validates the plan, signed scenario metadata, scenario SHA-256, frozen ProbeSet SHA-256, twelve unique run IDs, three unique matched groups, fixed conditions, fixed v2 probe template, and primary metrics. It writes only a safe receipt under `outputs/study-locks/`; it does not construct a provider, read environment variables, access the network, modify `results/`, or change any existing calibration/pilot artifact.

Cost fields are deliberately separate: each matched group plans 384 logical requests and reserves 51,200 completion tokens. Across three seeds this is 1,152 logical requests and 153,600 completion reservation. The `190,602 x 3 = 571,806` value is an observational total-token reference from calibration, not a hard cap, fee ceiling, or success guarantee.

This is an experiment-design lock, not execution or analysis. A later real run requires a separate explicit network, request, token, and researcher confirmation gate. The `002` v1 pilot and `003` v2 development calibration remain excluded from confirmatory statistics.
