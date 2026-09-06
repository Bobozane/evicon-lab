# H-G.2.3 Observable Adoption Amendment

H-G.2.2 is retained as an immutable development-only calibration. Its
engineering audit passed, but `false_cascade_adoption_rate` remained at zero
and both transition metrics lacked eligible units.

## Root cause

H-G.2.2 mapped `epistemic_stance=supported` to public endorsement. The same
protocol correctly encouraged an Agent to remain epistemically uncertain when
public evidence was incomplete. An Agent could therefore share a report or act
provisionally while never becoming an `endorses` observation. This conflated
evidential confidence with public adoption and made the primary behavioral
quantity poorly identifiable.

## Versioned change

H-G.2.3 separates three public outputs:

- epistemic confidence: supported, unsupported, or uncertain;
- adoption decision: adopt the claim as the current premise for the stated
  reversible decision, reject it, or withhold judgment;
- sharing decision: share, share with caveat, or do not share.

`adopt_claim` is an observable public behavioral commitment. It is explicitly
not a certification that the claim is objectively true. Public outcome stance
is derived from adoption decision. Confidence remains a separate audit field.

The nine preregistered metric names remain unchanged. Their H-G.2.3 outcome
contract is versioned: false adoption and transition eligibility use observable
adoption decisions rather than epistemic confidence. Denominators and N/A
rules remain explicit.

## Unchanged material

The four synthetic scenarios, public statements, provenance graphs, exposure
schedule, roles, controller policy, six-Agent ring, three rounds, and evaluator
truth fixtures are unchanged. Source-blind and provenance-aware controllers
still differ only in their permitted public provenance inputs. No private truth
enters an Agent or controller view.

The FakeProvider calibration checks only that the timing and denominator chain
can exist. Its behavior is condition-neutral before directives and it is not
evidence that an intervention works. H-G.2.3 is development-only and requires
new approval and a one-shot compatibility check before any real calibration.

