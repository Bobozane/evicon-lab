# H-G.1 Pilot Runner

`evicon.provenance_cascade_hg1_pilot` is the only execution boundary for the H-G.1 public-content-identifiable Pilot. It is a sidecar and does not modify the historical H-G, H-D.2.1, WVS, calibration, or analysis runners.

The default mode is an offline final preflight. It validates the locked H-G.1 config, protocol/template, controller, replay validator, amendment, accepted approval, safe compatibility receipt, 48-run plan, 12 matched groups, 864 logical requests, 442,368 completion-token reservation, and the no-overwrite output root. This path does not inspect Provider environment variables or construct a Provider.

The real execution path remains blocked unless network access, run confirmation, and both exact caps are supplied. Only after every offline gate passes may it read Provider environment settings. Each registered run receives an independent append-only request ledger. A completed request fingerprint cannot be replayed. A failed transport request can continue only under explicit resume. Parser-invalid responses stop the run and have no automatic semantic recovery.

The runner injects the H-G.1 round-zero preload, registered propagation schedule, public-view-only controller, next-round directive application, and H-G.1 replay validators into the shared real Agent boundary. Directives cannot affect the round that created them. A Pilot receipt is written only after all 48 runs complete and cascade, application, and outcome replay all pass.

The FakeProvider smoke uses a temporary directory. It is an engineering check, not a Pilot result or paper evidence.
