# EviCon-Lab

EviCon-Lab is a research codebase for controlled experiments on social
conformity and representational loss in LLM multi-agent deliberation. The
project distinguishes evidence-driven belief revision from convergence caused
by social exposure, then evaluates selective evidence-aware mediation.

The project is inspired by ProMediate, MultiAgent-Diversity, ASVO,
Social-Conformity-in-Large-Language-Models,
Easier-to-Mislead-Than-to-Correct, CASCADE, and Agent-Kernel. These works are
research references, not bundled dependencies or copied source code.

## Status

The current tree contains versioned contracts, provenance-aware exposure and
replay components, offline protocol simulations, and a public-view controller
smoke.  An interview-ready, fully offline end-to-end showcase is available at
[`docs/interview-portfolio.zh-CN.md`](docs/interview-portfolio.zh-CN.md).

Run it from the repository root:

```bash
.venv/bin/python -m evicon.interview_portfolio
```

The showcase writes only redacted report, trace, and receipt artifacts under
`outputs/interview-portfolio-v1/`. It does not read credentials, construct a
provider, access evaluator-private truth, or claim model behavior or causal
effects. The generated report also records the implemented Agent runtime,
provider boundary, provenance, replay, testing, and integrity stack, while
separating those facts from future or unimplemented capabilities. Historical
H-G, H-D, WVS, and other pilot artifacts remain isolated.

## Layout

```text
src/evicon/  Python package
tests/       pytest suite
configs/     versioned experiment configurations
scripts/     command-line entry points
results/     generated JSONL, CSV, figures, and reports
docs/        research scope, provenance, and implementation plan
```

## Development

Create an environment with Python 3.11 or later, install the development
extra, then run:

```bash
pytest
```

The test suite covers the versioned contracts, replay checks, offline
simulations, and the interview portfolio boundary. The portfolio command is
the recommended first run when reviewing the current implementation.

## Provenance and Licensing

Do not copy code, prompts, datasets, or derived artifacts from
MultiAgent-Diversity or Easier-to-Mislead-Than-to-Correct into this project;
they are used only for local reading, replication planning, and methodological
reference unless their licensing status is independently confirmed. Any future
use of MIT- or Apache-2.0-licensed material must preserve required notices,
attribution, and a source record. See `docs/upstream_mapping.md`.
