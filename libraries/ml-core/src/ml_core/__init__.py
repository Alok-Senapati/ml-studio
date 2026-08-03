"""
ml-core: An educational machine learning library built from scratch with NumPy.

This library provides foundational components for building, training, and evaluating
machine learning models from first principles.

Subpackages
-----------
base
    Abstract base classes and interfaces for estimators and classifiers.
exceptions
    Custom exception classes used across the library.
linear_models
    Linear modeling algorithms (e.g., Logistic Regression).
losses
    Loss functions for model optimization and evaluation.
mathematics
    Mathematical helper functions and activation operations.
metrics
    Evaluation metrics for assessing model performance.
optimizers
    Optimization algorithms used for parameter fitting.
typing
    Type aliases and annotations for NumPy arrays.
validation
    Input validation functions ensuring data consistency and validity.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__: list[str] = []
