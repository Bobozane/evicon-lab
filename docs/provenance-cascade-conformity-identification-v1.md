# Conformity Identification Study v1

This is a development-only, offline design for identifying social influence before evaluating any intervention. It does not reuse or merge H-G.2.3.2 observations, does not run a controller, and is not a paper result.

## Research question

Agreement is not treated as conformity. The design separates five possible reasons for a changed judgment: ordinary self-revision, repeated content without a social source, repeated messages sharing one provenance root, corroboration by independent public roots, and verified public evidence. The study asks whether these mechanisms are behaviorally distinguishable under matched public tasks.

The two frozen contrasts are:

```text
social_conformity_increment = same_root_social - source_free_repetition
evidence_receptivity_increment = verified_evidence - self_reflection
```

The statistical unit is `scenario x seed x agent`. These differences are descriptive identification contrasts, not causal conclusions by themselves.

## Conditions and timing

The six conditions are `private_baseline`, `self_reflection`, `source_free_repetition`, `same_root_social`, `independent_roots`, and `verified_evidence`.

Every run uses five ordered stages:

1. `initial_private`: the initial synthetic claim is visible before the first judgment.
2. `self_reflection`: a second judgment is collected without peer information.
3. `social_exposure`: matched repeated text is exposed with no source projection, one shared root, or two independent roots according to condition.
4. `correction_evidence`: the evidence arm receives a public correction and its authorized evidence.
5. `final_judgment`: ordinal judgment, public action, sharing decision, and confidence are recorded.

The source-free, same-root, and independent-root arms have identical repeated text IDs, text, count, and timing. Only their public source projection differs. The condition label itself is not rendered to the Agent.

## Response contract

The response uses a five-level ordinal judgment: `strongly_reject`, `lean_reject`, `uncertain`, `lean_adopt`, and `strongly_adopt`. Public adoption and sharing remain separate decisions. The strict parser rejects unknown fields, duplicated IDs, and references to unexposed content or evidence.

The frozen primary measurements are initial substantive judgment rate, ordinal judgment shift, public adoption rate, public share rate, social conformity increment, and evidence receptivity increment. No significance or Go/No-Go threshold is registered in this version.

## Safety and scope

All four scenarios are original synthetic development material. Runtime input contains no evaluator truth, source-independence annotation, WVS material, historical Pilot observation, private path, prompt log, model response log, API key, or provider metadata. Provenance IDs are public structural observations; they do not disclose correctness.

The qualification FakeProvider is deliberately condition-neutral at the initial stage. It establishes complete coordinates and non-empty eligibility denominators only. Its outputs cannot be used to claim a condition effect.

The offline design contains 4 scenarios, 3 seeds, 6 conditions, 6 Agents, and 5 observations per Agent. This yields 72 planned runs and 2,160 future logical requests. At 512 reserved completion tokens per request, the completion reservation is 1,105,920 tokens. This reservation is not a total-token or price limit.

Network execution remains blocked pending separate exact-hash approval, provider compatibility validation, budget confirmation, and a real-run implementation. A failed qualification gate requires a versioned amendment; it must not overwrite this design or prior Pilot artifacts.
