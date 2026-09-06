# Metrics

Stage 7 provides deterministic, pure functions in `evicon.metrics`. They do not call a model, read a run directory, modify `DialogueState`, write files, or generate experimental conclusions.

## Pairwise Diversity

For aligned normalized profile vectors `x` and `y` with `d` dimensions, the distance is normalized Euclidean distance:

```text
D(x, y) = sqrt(sum_i((x_i - y_i)^2)) / sqrt(sum_i(m_i^2))
```

For `ValueProfile` scores, `m_i = 1` because scores are in `[0, 1]`. Pairwise Diversity is the arithmetic mean of `D` over all valid unordered agent pairs. This is intentionally normalized Euclidean distance, not cosine distance or cosine similarity.

For item answers, `answer_distance` follows the referenced baseline behavior independently: it uses only question IDs held by both agents and present in `reference`. For each question `q`, `m_q = option_count[q] - 1` and answers are zero-based indices. The denominator is `sqrt(sum_q(m_q^2))`. No overlapping valid item raises `InsufficientDataError`; it is never silently treated as distance zero.

`profile_distance` is distinct from `answer_distance`: it aligns `ValueProfile` vectors by dimension name and compares normalized aggregate scores. It does not reproduce item-level option ranges.

## Structural Diversity

Structural Diversity builds the complete weighted agent graph from pairwise distances. A deterministic local Kruskal implementation selects a minimum spanning tree (MST):

```text
MST span = total MST edge length / (agent_count - 1)
```

The result includes the selected canonical edges, total length, and span. All-zero graphs have span zero. Strict mode requires every pair. Non-strict mode records missing-pair warnings; it can still return a valid MST when supplied edges connect the graph, otherwise it returns `valid=false` rather than inventing zero-length edges.

## Representation

`value_dimension_coverage` requires an explicit threshold. A dimension is expressed when at least one supplied profile reaches the threshold. Coverage is the fraction of declared dimensions that are expressed. The result also reports per-dimension minimum, maximum, and mean scores.

Coverage is not diversity: a group can cover many dimensions while every agent has the same profile, or be diverse while few dimensions cross a chosen threshold.

`minority_retention` does not infer a majority from counts or scores. The caller must explicitly pass `minority_dimensions`. A nominated dimension belongs to the initial minority set only if it is initially expressed at the same threshold. Retention is the fraction of those initially expressed, nominated dimensions still expressed in final profiles. This convention is an experimental input, not a cultural or normative definition of minority status.

`holdout_profile_drift` reports matched within-agent normalized profile distances, plus mean and maximum drift. It describes change only; it does not label a change as beneficial or harmful.

## Controlled Contrast

`social_influence_loss` computes:

```text
L_social = D_evidence_only - D_evidence_social
```

Both inputs must contain exactly the same agent IDs and matched value dimensions. A positive value has the structured label `social_diversity_lower`; a negative value is preserved and labeled `social_diversity_higher`; zero is `no_diversity_difference`. `holdout_social_influence_loss` applies the same calculation to profiles measured by hidden probes and uses `scope="holdout"`.

This is a matched counterfactual estimate, also called a controlled social-influence loss. It is not a proof of a strict causal effect. Valid interpretation still depends on matched scenarios, models, seeds, measurement schedules, evidence exposure, and other experimental controls.

## Missing Data And Limits

Strict mode raises on a non-computable pair or incomplete MST graph. Non-strict Pairwise Diversity skips only failing pairs and returns their reasons; no valid pair still raises. Non-strict Structural Diversity records missing edges and marks a disconnected graph invalid.

These metrics quantify behavior of configured language-model agents and probe instruments. They cannot establish that an Agent has human values, that a score represents a human population, or that an observed loss is socially harmful without additional study design and validation.

This stage does not implement `ConformityMonitor`, mediator policy, real-model providers, statistics, or CSV result export.
