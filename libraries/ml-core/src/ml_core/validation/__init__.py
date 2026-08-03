"""
Input validation utilities for ml-core.

Provides assertion functions for validating input matrix shapes, array dimensions,
and label subsets prior to model estimation or loss computation.
"""

from __future__ import annotations

from .input import (
    check_binary_labels,
    check_feature_matrix,
    check_matching_target_vectors,
    check_same_length,
    check_target_vector,
)

__all__ = [
    "check_feature_matrix",
    "check_target_vector",
    "check_same_length",
    "check_binary_labels",
    "check_matching_target_vectors",
]
