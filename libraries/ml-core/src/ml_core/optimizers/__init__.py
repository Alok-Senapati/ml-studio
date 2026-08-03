"""
Optimization algorithms for training models in ml-core.

Provides gradient-based optimization utilities for parameter updating.
"""

from __future__ import annotations

from .gradient_descent import gradient_descent

__all__ = [
    "gradient_descent",
]
