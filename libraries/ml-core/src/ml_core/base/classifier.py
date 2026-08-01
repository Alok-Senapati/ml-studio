from __future__ import annotations

from abc import abstractmethod

import numpy as np

from ml_core.base.estimator import Estimator


class Classifier(Estimator):
    """Base interface for all classifiers"""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities"""