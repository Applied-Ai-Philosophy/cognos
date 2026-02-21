"""
CognOS — Epistemic Integrity Layer for Agentic AI

The operating system for decision-aware AI systems.
"""

__version__ = "0.3.0"
__author__ = "Björn Wikström"
__license__ = "MIT"

from .confidence import compute_confidence
from .divergence_semantics import synthesize_reason, frame_transform, convergence_check
from .orchestrator import CognOSOrchestrator

__all__ = [
    "compute_confidence",
    "synthesize_reason",
    "frame_transform",
    "convergence_check",
    "CognOSOrchestrator",
]
