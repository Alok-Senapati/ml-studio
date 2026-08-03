from __future__ import annotations

from abc import abstractmethod

from ml_core.base.estimator import Estimator
from ml_core.typing import FloatArray, IntArray, NumericArray


class Classifier(Estimator):
    """Base interface for all classification estimators."""

    @abstractmethod
    def predict_proba(self, X: NumericArray) -> FloatArray:
        """Predict class probabilities for the given feature matrix."""
