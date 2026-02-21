# HYPOTHESIS — Operationalization of Uncertainty Sources

**Date:** February 21, 2026  
**Status:** Candidate hypothesis for empirical testing (10×3×3)  
**Source:** Björn Wikström with Claude AI

## Core Hypothesis

LLM uncertainty in decision questions is a mixture of three separable components:
- **U_model** (internal epistemic uncertainty)
- **U_prompt** (format-induced uncertainty)
- **U_problem** (intrinsic ill-posedness of the question)

CognOS should be able to distinguish these sufficiently well to improve decision gating compared to prediction probability alone.

## Operational Definitions

- **U_model:** Variance that persists when prompt format is held constant but sampling varies (temperature/seed). Measure: within-format variance in responses/confidence.

- **U_prompt:** Variance that arises when question + model are constant but format changes (narrative, forced binary, structured choice). Measure: between-format difference in majority choice and/or Ue.

- **U_problem:** Stable high uncertainty regardless of format. Measure: low confidence in all three formats + no robust majority.

## Predictions

1. The same question exhibits significantly larger between-format variance than within-format variance for a non-trivial proportion of questions (U_prompt exists).

2. Structured choice reduces measurement artifacts and yields more stable consensus geometry than free text.

3. Questions with genuine ambiguity display persistent low-confidence outcomes in all formats (U_problem).

## Falsification Criteria

The hypothesis is rejected wholly or partially if any of the following hold in the 10×3×3 experiment:

1. **No format sensitivity:** between-format variance ≈ within-format variance for nearly all questions.

2. **No choice instability:** majority choice remains unchanged across formats in nearly all cases.

3. **No robust U_problem signal:** low confidence does not appear consistently across all formats for ill-posed questions.

## Decision Rule for CognOS

- Dominant between-format variance ⇒ flag **U_prompt risk**.
- Persistent low-confidence outcomes across formats ⇒ flag **U_problem risk**.
- Low within-format stability under fixed format ⇒ flag **U_model risk**.

## Experiment Design

**10×3×3 Matrix:**

| Dimension | Values | Purpose |
|-----------|--------|---------|
| Questions | 10 diverse decision questions | Coverage across domains |
| Formats | Narrative, Forced Binary, Structured Choice | Isolate format effects |
| Repetitions | 3 samples per format | Measure within-format variance |

**Metrics:**

- `Ue` (epistemic uncertainty): variance of MC predictions
- `p_format_stability`: proportion of questions where majority choice stable across formats
- `C_consistency`: consistency of confidence scores across formats

**Success Criterion:**

- ≥70% of questions show p_format_stability = False
- ≥60% of ill-posed questions show Ue_problem > 2 × Ue_well-posed
