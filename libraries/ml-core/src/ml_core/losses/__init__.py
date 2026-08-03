"""
Loss functions for optimization and model evaluation in ml-core.

Provides loss metric implementations such as Binary Cross Entropy.
"""

from __future__ import annotations

from .binary_cross_entropy import EPSILON, binary_cross_entropy

__all__ = [
    "EPSILON",
    "binary_cross_entropy",
]
