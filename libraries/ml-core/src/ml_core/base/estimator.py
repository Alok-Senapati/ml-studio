from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Estimator(ABC):
    """Base interface for all machine learning estimators."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Estimator:
        """Train the estimator"""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outputs for the given inputs."""