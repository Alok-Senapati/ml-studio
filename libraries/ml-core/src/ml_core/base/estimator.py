from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

from ml_core.typing import NumericArray


class Estimator(ABC):
    """Base interface for all machine learning estimators."""

    @abstractmethod
    def fit(self, X: NumericArray, y: NumericArray) -> Self:
        """Fit the estimator to the training data."""

    @abstractmethod
    def predict(self, X: NumericArray) -> NumericArray:
        """Predict outputs for the given feature matrix."""