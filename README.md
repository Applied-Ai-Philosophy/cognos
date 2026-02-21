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

## Three Layers

### Layer 1: Confidence Engine
Combines probabilistic and semantic uncertainty into a single confidence score.

### Layer 2: Divergence Semantics
When perspectives conflict, extract *why* they differ.

### Layer 3: Convergence Control
Stop recursion when the system has converged.

## License

MIT License — see LICENSE file

## Citation

```bibtex
@software{wikstrom2026cognos,
  title={CognOS: Epistemic Integrity Layer for Agentic AI},
  author={Wikström, Björn},
  year={2026},
  url={https://github.com/bjornshomelab/cognos},
}
```

See docs/ for research notes and full paper draft.
