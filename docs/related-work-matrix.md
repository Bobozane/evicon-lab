# EviCon Related-Work Matrix

Screening date: 2026-08-27. This is a positioning aid, not a systematic review or a claim that an unreported feature is absent. Preprints are marked because they are relevant to novelty assessment even when they have not completed peer review.

## Strict comparison rules

The matrix uses deliberately strict criteria. `Y` means the paper explicitly provides the feature; `P` means a partial proxy that does not satisfy the strict criterion; `N` means the feature is not part of the reported protocol; `NR` means it was not reported clearly enough in the screened source to classify as present.

| Column | Count as present only when... |
| --- | --- |
| `E` | External evidence/content and a peer's social expression are independently manipulated, rather than bundled into one peer message. |
| `T` | Per-agent, per-round *actual* exposure is recorded or reconstructable, not merely that a dialogue transcript exists. |
| `B` | The primary behavioral measurement is protocol/condition blind; a blind judge alone is only `P`. |
| `R` | A verifier/replayer checks the exposure/order contract, not merely that code can be re-run. |
| `I` | A mitigation/intervention is implemented and empirically evaluated. |

## Social influence and conformity

| Work | Status | What it establishes | E | T | B | R | I | Positioning implication |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [Asch (1956), *Studies of Independence and Conformity*](https://doi.org/10.1037/h0093718) | peer reviewed | Canonical majority-pressure paradigm. | N | N | N | N | N | The behavioral phenomenon is not new. |
| [Deutsch & Gerard (1955)](https://doi.org/10.1037/h0046408) | peer reviewed | Defines informational versus normative social influence. | P | N | N | N | N | EviCon inherits, rather than originates, this conceptual distinction. |
| [Cialdini & Goldstein (2004)](https://doi.org/10.1146/annurev.psych.55.090902.142015) | peer reviewed | Broad theory of compliance and conformity. | P | N | N | N | N | Do not claim a new social-influence theory. |
| [Zhu et al. (2025), *Conformity in Large Language Models*](https://aclanthology.org/2025.acl-long.195/) | ACL long paper | Adapts psychological conformity experiments to LLMs; studies uncertainty and majority-tone effects; evaluates Devil's Advocate and Question Distillation. | N | N | N | N | Y | Direct prior work on LLM conformity and mitigation. |
| [Weng et al. (2025), *Do as We Do, Not as You Think*](https://arxiv.org/abs/2501.13381) | ICLR paper | BenchForm uses five multi-agent interaction protocols and studies conformity, majority size, interaction time, personas, and reflection. | N | N | N | N | Y | Direct benchmark competitor; must be compared or sharply distinguished. |
| [Song et al. (2025), *LLMs Can't Handle Peer Pressure*](https://arxiv.org/abs/2508.18321) | preprint | KAIROS controls peer rapport, historical interaction, current peer behavior, and self-confidence; evaluates prompting, SFT, and GRPO. | N | P | N | N | Y | Strong controlled peer-influence benchmark; rapport is a distinct confound EviCon should acknowledge. |
| [Han et al. (2026), *Conformity Dynamics in LLM Multi-Agent Systems*](https://arxiv.org/abs/2601.05606) | preprint | Studies topology, centralization, social weighting, and wrong-but-sure cascades in misinformation detection. | N | N | N | N | N | Covers interaction topology and consensus failure, not an auditable exposure protocol. |
| [De Marzo et al. (2026), *Conformity Generates Collective Misalignment in AI Agent Societies*](https://arxiv.org/abs/2605.10721) | preprint | Studies majority-following, intrinsic bias, tipping points, and collective misalignment across nine models and 100 opinion pairs. | N | N | N | N | N | Directly challenges a broad "value drift under conformity" claim. |
| [Hao et al. (2026), *Not All Flips Are Conformity*](https://arxiv.org/abs/2606.00820) | preprint | Separates self-reflection instability, stance-induced conformity, and reasoning-induced persuasion through counterfactual conditions; tests risk-targeted intervention. | P | N | N | N | Y | The most direct threat: it already argues that observed convergence has multiple causes. EviCon must offer a genuinely different and verifiable treatment decomposition. |
| [Hu & Qu (2026), *Most LLM Conformity Needs No Speaker*](https://arxiv.org/abs/2607.05545) | preprint | Separates repeated candidate content from explicit speaker/source framing using a no-source control. | P | N | N | N | N | EviCon must include a content-only or source-free control; otherwise peer text repetition remains a confound. |
| [Sahrawat et al. (2026), *The Evaluator Is Part of the Experiment*](https://arxiv.org/abs/2608.04463) | preprint | Shows that open-ended conformity measurement depends on judge bias; distinguishes blind and informed evaluator conditions. | P | N | P | N | N | EviCon needs evaluator calibration and blind scoring, not only a blind probe. |
| [Hussain & Nielbo (2026), *Conformity Mitigations ... Resistance-Receptivity Frontier*](https://arxiv.org/abs/2608.11247) | preprint | Evaluates six mitigations over 23 models, 19 conditions, and three datasets; frames resistance versus beneficial receptivity. | N | N | N | N | Y | Any intervention paper must measure preservation of beneficial peer learning, not only reduced agreement. |
| [Qiu et al. (2026), *Belief Cascades Drive Persuasion in LLM Agent Networks*](https://arxiv.org/abs/2608.25152) | preprint | Studies persuasion across four backbones, five ego-network graphs, and 55 policy statements; relates direct exposure and peer relays to next-round stance change using belief probes, exposure provenance, and action logs. | P | Y | P | N | N | Exposure-level persuasion and provenance logging are no longer defensible as standalone novelty. EviCon must test source-root independence, matched repetition controls, and executable replay validation. |
| [Wu et al. (2026), *Group Perspective Matters*](https://arxiv.org/abs/2608.03648) | preprint | DEAR dynamically regulates debate relationships with RL agents to reduce blind conformity. | N | N | N | N | Y | Strong direct mitigation baseline for a later intervention paper. |
| [Okawa (2026), *Emergence of Biased Consensus*](https://arxiv.org/abs/2608.02827) | preprint | Models collective bias, noise-driven transitions, and agent heterogeneity in LLM debate. | N | N | N | N | N | Existing work already connects conformity to collective bias and heterogeneity. |
| [YS (2026), *Everyone Conforms, No One Believes*](https://arxiv.org/abs/2608.02758) | preprint | Benchmarks pluralistic ignorance in 100 scenarios and tests a public-dissent intervention. | N | N | N | N | P | Shows that public/private response separation is another established social-dynamics axis. |
| [Lin et al. (2025), *Towards Simulating Social Influence Dynamics with LLM-based Multi-agents*](https://arxiv.org/abs/2507.22467) | preprint | Studies conformity, polarization, and fragmentation by model scale and reasoning capability. | N | NR | N | N | N | Broad social-simulation context; it does not by itself establish a treatment-audit protocol. |
| [Gu et al. (2025), *LLM-driven Agents for Simulating Echo Chamber Formation*](https://arxiv.org/abs/2502.18138) | preprint | Simulates opinion updates and network rewiring; compares generated dynamics with social-media data. | N | NR | N | N | N | Relevant to polarization, but different from within-run exposure attribution. |
| [Singh (2026), *When Does Belief-Based Agent Memory Help?*](https://arxiv.org/abs/2606.22030) | preprint | Uses provenance-capped belief updating to limit volumetric poisoning in long-term Agent memory. | P | N | N | N | Y | Provenance-aware trust capping is adjacent prior art; EviCon must distinguish social source-root duplication and exposure-contract auditing from memory poisoning defense. |

## Multi-agent evaluation and interaction baselines

| Work | Status | What it establishes | E | T | B | R | I | Positioning implication |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [Du et al. (2023), *Improving Factuality and Reasoning through Multiagent Debate*](https://arxiv.org/abs/2305.14325) | preprint / widely used baseline | Peer debate can improve reasoning and factuality. | N | N | N | N | N | A no-intervention/debate baseline; agreement can be beneficial. |
| [Chan et al. (2023), *ChatEval*](https://arxiv.org/abs/2308.07201) | preprint | Uses multi-agent debate for LLM evaluation. | N | P | N | N | N | A transcript does not establish each agent's authorized exposure. |
| [Liu et al. (2024), *AgentBench*](https://arxiv.org/abs/2308.03688) | ICLR paper | Multi-environment benchmark for LLM agents. | N | N | N | N | N | General agent benchmark, not social-influence measurement. |
| [Zhou et al. (2024), *SOTOPIA*](https://arxiv.org/abs/2310.11667) | ICLR paper | Interactive social-intelligence environment and evaluator. | N | P | N | N | N | Social interaction evaluation is established; EviCon needs a narrower causal/audit claim. |
| [Jin et al. (2024), *AgentReview*](https://arxiv.org/abs/2406.12708) | EMNLP paper | LLM peer-review simulation that disentangles latent factors such as reviewer bias. | N | NR | N | N | N | Demonstrates simulation-based factor analysis, but not evidence-versus-peer exposure control. |
| [Borah & Mihalcea (2024), *Implicit Bias Detection and Mitigation in Multi-Agent LLM Interactions*](https://aclanthology.org/2024.findings-emnlp.545/) | Findings of EMNLP | Studies bias arising during multi-agent interaction and mitigation. | N | NR | N | N | Y | Adjacent safety baseline; inspect before claiming the first mitigation-aware audit. |

## Evaluation and audit methodology

| Work | Status | What it establishes | E | T | B | R | I | Positioning implication |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [Ribeiro et al. (2020), *CheckList*](https://aclanthology.org/2020.acl-main.442/) | ACL paper | Behavioral testing methodology for NLP models. | N | N | N | N | N | Supports the value of controlled behavioral tests, not EviCon's specific social treatment. |
| [Liang et al. (2023), *HELM*](https://doi.org/10.48550/arXiv.2211.09110) | TMLR paper | Broad, standardized and transparent LLM evaluation. | N | N | N | N | N | Useful precedent for framework papers, but does not audit multi-agent exposure. |
| [Mökander et al. (2023), *Auditing Large Language Models*](https://doi.org/10.1007/s43681-023-00289-2) | AI and Ethics | Proposes layered LLM-audit governance. | N | N | N | N | N | General auditing is not novel; EviCon must show execution-level audit, not use the word "audit" alone. |
| **EviCon target protocol** | current system | Factorial evidence/peer exposure, protocol-blind contextual probing, frozen materials, request ledger, and replay validator. | **Y** | **Y** | **Y** | **Y** | partial | This is a target position, not evidence of novelty. It must be demonstrated against the closest rows above. |

## Full-text verification of the five closest papers

The following five rows were checked against the supplied PDF, including their methods, controls, reported limits, and prompt/appendix material where relevant.

| Work | Verified design | What it makes unavailable as an EviCon claim | Remaining difference to test, not assert |
| --- | --- | --- | --- |
| Zhu et al. (ACL 2025) | Simulated majority-answer conversations; varying majority configuration and uncertainty; Devil's Advocate and Question Distillation interventions. | The existence of LLM conformity and basic prompt mitigation. | It does not report a self-reflection, source-free, or verified external-evidence control. |
| Weng et al. (ICLR 2025, BenchForm) | Raw, Correct Guidance, Wrong Guidance, Trust, and Doubt protocols; six scripted peers; repeated prior correct/wrong interactions; metrics for accuracy, conformity, and independence; persona and reflection mitigation. | A first multi-agent conformity benchmark, peer-history/trust manipulation, and lightweight mitigation. | Peers provide answers, not independently auditable external evidence. The study provides prompt templates but no per-agent exposure ledger, blind probe, or replay verifier. |
| Hao et al. (2026) | Five agents; parallel self-reflection, stance-only, and full-peer-reasoning counterfactuals; also an invalid-reasoning versus wrong-reasoning information gradient; risk-targeted intervention. | A generic claim that answer flips reveal conformity, or that self-reflection versus peer reasoning has not been decomposed. | Their decomposition is correctness-label QA based. It does not manipulate a provenance-verified external evidence card independently from a peer social expression, and it has no exposure-audit/replay contract. |
| Hu & Qu (2026) | Two-read deterministic protocol; plain re-ask and length controls; same asserted answer rendered as no-source, person, rich peers, or experts; repetition/source/format controls. | A source-free repeated-content control as a new idea. | It is single-model, single-pass, mostly scripted pressure. It measures a source-attributed increment, but does not establish multi-agent temporal exposure, verified evidence access, or replay audit. |
| Hussain & Nielbo (2026) | 23 open-weight models, scripted wrong/correct peers, five pressure conditions, six mitigations, and paired Resistance/Receptivity outcomes. Its stated limitations include no content-free placebo, single-pass interactions, and scripted peers. | A claim that reducing harmful peer adoption alone establishes a good intervention. | It explicitly predicts that independent verification or retrieval may escape the Resistance-Receptivity trade-off but does not test this. That is a potential empirical opening, not yet a novelty claim. |

## Public-opinion and platform-governance overlap

Do not claim that LLM-agent public-opinion governance, intervention-aware social simulation, or cyberbullying support is unexplored. Closely adjacent work includes [POSIM](https://arxiv.org/abs/2603.23884) (public-opinion evolution and governance interventions), [IntervenSim](https://arxiv.org/abs/2604.06600) (closed-loop source-side interventions), [MIDSim](https://arxiv.org/abs/2606.13140) (separate social and recommendation exposure streams), [DEBATE](https://arxiv.org/abs/2510.25110) (human-grounded opinion-dynamics benchmark), and [Upstanders' Practicum](https://arxiv.org/abs/2605.03287) (cyberbullying bystander practice in an LLM-agent social-media simulation). [TRAILS](https://arxiv.org/abs/2605.18890) further argues that social-simulation claims require explicit robustness audits.

An ethically defensible open question is narrower: whether a platform-policy simulator can *auditably* compare information-integrity interventions while preserving legitimate minority expression and beneficial correction. It must not be framed as autonomously steering real people or as deciding which political opinions should prevail. A credible study would choose one harm mechanism (for example, a false-claim cascade, harassment-norm escalation, or recommender-driven exposure concentration), use only offline/synthetic or properly authorized data, and report both harm reduction and minority-retention/false-positive costs.

## Immediate consequences for the paper

1. Do **not** claim discovery of LLM conformity, the first multi-agent conformity benchmark, the first decomposition of convergence causes, the first source-free control, or the first intervention.
2. The defensible hypothesis is narrower: a *verified exposure protocol* can estimate a social-expression effect separately from a provenance-verified external-evidence effect and can expose protocol violations that would otherwise invalidate this estimate.
3. Before scaling the current experiment, add at least two controls demanded by the new literature: a **source-free repeated-content** control and a **self-reflection/no-peer** control. These are necessary to distinguish social influence from text repetition and ordinary response instability.
4. The AAMAS paper needs an empirical "measurement failure" result: show a conventional answer-flip/similarity metric reaches a different conclusion from the verified, factorized protocol on matched tasks.
5. A later intervention paper must compare against the published/recent mitigation baselines and report harmful conformity reduction, beneficial receptivity, task quality, safety, and intervention errors.

## Review limits and next verification pass

The classifications outside the full-text section are based on the linked version's abstract. `NR` must be resolved by reading the full methods, appendices, released code, and artifacts before submission. The five supplied papers above have now received full-text review; the next high-priority checks are the code/artifacts for BenchForm, Hao et al., POSIM, IntervenSim, MIDSim, and DEBATE.
