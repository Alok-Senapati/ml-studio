"""
Base interfaces and abstract classes for ml-core estimators and classifiers.

Provides base abstract classes defining standard method contracts for all machine
learning algorithms implemented in the library.
"""

from __future__ import annotations

from .classifier import Classifier
from .estimator import Estimator

__all__ = [
    "Estimator",
    "Classifier",
]
