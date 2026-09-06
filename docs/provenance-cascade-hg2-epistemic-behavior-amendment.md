# H-G.2 Epistemic-Behavior Separation Amendment

H-G.1.2 completed with full engineering integrity, but its main behavioral
effect remained unidentified. All conditions had zero false-claim endorsement,
and no run supplied an eligible initial-error-to-correction transition. The
four-field response conflated an epistemic assertion with a reversible public
decision. A safety-oriented model could therefore avoid both error and the
behavioral trade-off by returning `uncertain`.

H-G.2 is a new development-only design. It does not alter, resume, combine, or
reinterpret H-G.1.2. It separates an Agent's public evidential position from its
public behavior. An Agent may remain epistemically uncertain while sharing with
a caveat or supporting a limited, reversible action. Sharing never means that a
claim is true.

The strict output contains only an epistemic stance, a behavioral decision, and
IDs already visible in the round-start snapshot. It contains no rationale text,
truth label, source-independence label, controller verdict, or new provenance.
The controller, next-round directive, Exposure, and Replay boundaries are
unchanged.

Six public decision roles are fixed and assigned identically in every condition:
harm avoidance, verification first, rapid response, reversibility focused,
procedural fairness, and communication stability. They encode legitimate
decision priorities, not desired answers or private truth.

The CedarLine false-majority and independent-consensus tasks use the same public
decision framing, message count, timing, and surface statements. Their public
provenance differs: repeated messages resolve to one root in the first case and
to distinct roots in the second. A source-blind controller cannot use that
difference; a provenance-aware controller can. Neither controller receives an
evaluator label.

This amendment first authorizes only offline FakeProvider validation. A later
one-seed real eligibility calibration must show nonzero, nonsaturated provisional
behavior and nonzero correction-transition eligibility before any full Pilot is
designed. FakeProvider behavior is a contract fixture and is not evidence that an
intervention works. No significance threshold or effect threshold is introduced.

Safety flags are `development_only=true`, `calibration_only=true`,
`not_paper_result=true`, and `no_causal_conclusion=true`.
