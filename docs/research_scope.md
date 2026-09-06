# EviCon-Lab Research Scope

## Research objective

EviCon-Lab studies social conformity and representational loss in LLM
multi-agent deliberation. Agents may begin with different value profiles. A
change after credible new evidence can be appropriate. Convergence caused by
majority opinion, authority labels, response order, or group pressure without
new factual support is the target phenomenon.

The project does not attempt to freeze values or manufacture disagreement. Its
goal is to measure diversity and representational coverage, attribute excess
loss to social influence, apply minimal selective mediation when warranted, and
evaluate task quality, factuality, safety, and cost alongside those outcomes.

## Questions

1. Does social exposure cause convergence on held-out value probes that were
   not shown during deliberation?
2. How can evidence-supported revision be separated from social-influence
   revision under controlled exposure?
3. Does evidence-aware selective mediation outperform no mediation, a generic
   mediator, fixed diversity prompts, and random intervention?
4. Does mediation degrade task quality, factuality, safety, or cost?

## Planned experimental conditions

The study protocol will compare Independent, Social-only, Evidence-only, and
Evidence+Social exposure. It will also compare Generic mediator, Fixed
diversity prompt, Random intervention, and EviCon mediator conditions. The
precise factorization, randomization, and sample size are future configuration
decisions; they are not fixed by this skeleton.

## Operational boundaries

- Exposure prompts and hidden value probes must remain distinct.
- Evidence must have machine-recorded provenance, timing, and visibility.
- Social messages must record source, order, authority treatment, and audience.
- Policy code, not an LLM, selects whether to intervene, who is targeted, and
  which intervention is used. An LLM may only render an already selected
  action.
- Every eventual run must retain its model identifier, parameters, seed,
  rendered prompts, configuration version, and relevant cache key.
- Results are generated artifacts, stored as JSONL and CSV, and never embedded
  as claimed outcomes in source or documentation.

## Phase 1 boundary

Phase 1 creates no agents, prompts, model clients, evidence objects, metrics,
mediators, experiment runners, or real-model API calls. It establishes only a
clean project boundary and implementation contract.

