# CognOS — Epistemic Integrity Layer for Agentic AI

An open-source operating system for decision-aware AI systems. CognOS combines epistemic and aleatoric uncertainty to help AI agents know when they know — and when they should ask for help.

## What is CognOS?

CognOS is **externalized metacognition** for AI. It answers a fundamental question:

> *When should an AI system act autonomously, and when should it escalate to humans?*

Instead of binary auto/escalate decisions, CognOS provides four nuanced decision types:
- **auto** — high confidence, act autonomously
- **synthesize** — conflicting but coherent perspectives, combine them
- **explore** — noise, gather more information
- **escalate** — too risky, require human judgment

## Core Formula

```
C = p × (1 - Ue - Ua)

where:
  p   = prediction probability [0, 1]
  Ue  = epistemic uncertainty (variance of MC samples)
  Ua  = aleatoric/semantic risk (ambiguity + irreversibility + blast_radius) / 3
  C   = decision confidence [0, 1]
```

## Installation

```bash
pip install cognos-ai
```

## Quick Start

```python
from cognos import compute_confidence

result = compute_confidence(
    prediction=0.85,
    mc_predictions=[0.84, 0.86, 0.85, 0.87],
)

print(result['decision'])  # 'auto', 'synthesize', 'explore', or 'escalate'
print(result['confidence'])  # 0.803
```

## Architecture

### Layer 1: Confidence Engine (`compute_confidence`)
Combines probabilistic and semantic uncertainty into a single confidence score.

```python
from cognos import compute_confidence

result = compute_confidence(
    prediction=0.85,
    mc_predictions=[0.84, 0.86, 0.85, 0.87],
    ambiguity=0.1,
    irreversibility=0.2,
    blast_radius=0.05,
)
```

### Layer 2: Divergence Semantics (`synthesize_reason`, `frame_transform`)
When perspectives conflict, extract *why* they differ and detect ill-posed questions.

```python
from cognos import synthesize_reason, frame_transform

# Extract underlying assumptions from divergent votes
divergence = synthesize_reason(
    question="Should AI systems be regulated?",
    alternatives=["Strict control", "Light touch", "Innovation first"],
    vote_distribution={"A": 2, "B": 2},
    confidence=0.45,
    is_multimodal=True,
)

# Detect if a question is well-formed
frame = frame_transform(question="Color of Tuesday?")
```

### Layer 3: Convergence Control (`convergence_check`)
Stop recursion when the system has converged.

```python
from cognos import convergence_check

# Check if additional iterations would help
result = convergence_check(
    iteration=3,
    confidence_history=[0.45, 0.50, 0.51],
    assumption_history=["Assumption A", "Assumption A"],
    threshold=0.05,
)
```

## Project Structure

```
cognos-standalone/
├── cognos/                    # Main package
│   ├── __init__.py           # Exports: compute_confidence, synthesize_reason, ...
│   ├── confidence.py         # Layer 1: Confidence engine
│   ├── divergence_semantics.py # Layer 2–3: Divergence + convergence
│   └── core/                 # Advanced implementations
│       ├── cognos_deep.py           # Five-layer recursive analysis
│       └── cognos_integration_demo.py # Working example
├── research/                 # Theoretical foundation
│   ├── HYPOTHESIS.md         # Operational definitions & falsification criteria
│   └── INSIGHTS.md           # Theoretical insights & empirical findings
├── experiments/              # Empirical validation
│   ├── eval_hypothesis.py    # Run CognOS on HYPOTHESIS
│   ├── test_cognos_*.py      # Test suites
│   └── analyze_*.py          # Analysis scripts
├── figures/                  # Results & visualizations
│   ├── figure1_pareto_curves.png
│   ├── figure2_operating_regime.png
│   └── ...
├── tests/                    # Unit tests
├── examples/                 # Usage examples
├── docs/                     # Documentation
├── README.md                 # This file
├── setup.py                  # PyPI setup
└── LICENSE                   # MIT License
```

## Research Background

For the theoretical foundation and empirical findings, see [research/INSIGHTS.md](research/INSIGHTS.md).

**Key Findings:**
- CognOS v2 achieves 40–60% safety gain on cost-constrained systems (40–55% escalation budget)
- Three separable uncertainty types: U_model (internal), U_prompt (format-induced), U_problem (ill-posed)
- Structured choice prompts reveal consensus geometry; free-form responses collapse to narrative similarity

## Experiments

Run the evaluation suite:

```bash
python experiments/eval_hypothesis.py
python experiments/test_cognos_comprehensive.py
```

Generate figures from empirical results:

```bash
python experiments/analyze_pareto.py
python experiments/analyze_ue_distribution.py
```

## License

MIT License — see [LICENSE](LICENSE) file

## Citation

```bibtex
@software{wikstrom2026cognos,
  title={CognOS: Epistemic Integrity Layer for Agentic AI},
  author={Wikström, Björn},
  year={2026},
  url={https://github.com/bjornshomelab/cognos},
}
```

## Contributing

Contributions welcome. Please file issues and PRs at [GitHub](https://github.com/bjornshomelab/cognos).
