"""
Typing definitions and array type aliases for ml-core.

Provides type hints for NumPy arrays to maintain strict type safety and code clarity
throughout the library.
"""

from __future__ import annotations

from .aliases import BoolArray, FloatArray, IntArray, NumericArray

__all__ = [
    "NumericArray",
    "FloatArray",
    "IntArray",
    "BoolArray",
]
