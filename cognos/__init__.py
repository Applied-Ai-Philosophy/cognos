"""
CognOS — Epistemic Integrity Layer for Agentic AI

The operating system for decision-aware AI systems.
"""

__version__ = "0.3.0"
__author__ = "Björn Wikström"
__license__ = "MIT"

from .confidence import compute_confidence
from .divergence_semantics import synthesize_reason, frame_transform, convergence_check
from .strong_synthesis import (
    synthesize_strong,
    extract_assumptions_and_geometry,
    generate_integration_strategy,
    generate_meta_alternatives,
    compute_epistemic_gain,
)
from .orchestrator import CognOSOrchestrator

__all__ = [
    "compute_confidence",
    "synthesize_reason",
    "frame_transform",
    "convergence_check",
    "synthesize_strong",
    "extract_assumptions_and_geometry",
    "generate_integration_strategy",
    "generate_meta_alternatives",
    "compute_epistemic_gain",
    "CognOSOrchestrator",
]
