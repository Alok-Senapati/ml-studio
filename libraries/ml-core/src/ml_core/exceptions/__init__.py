"""
Custom exception classes for ml-core.

Provides centralized error types for contract violations, unfitted model calls,
and validation failures.
"""

from __future__ import annotations

from .base import MLCoreError
from .not_fitted import NotFittedError

__all__ = [
    "MLCoreError",
    "NotFittedError",
]
