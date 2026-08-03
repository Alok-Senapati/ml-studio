"""
Abstract base classifier interface for classification models.
"""

from __future__ import annotations

from abc import abstractmethod

from ml_core.base.estimator import Estimator
from ml_core.typing import FloatArray, NumericArray


class Classifier(Estimator):
    """
    Abstract base interface for all classification models in ml-core.

    Extends the base `Estimator` contract by adding probability prediction
    capabilities via `predict_proba`.

    Notes
    -----
    Subclasses must implement `fit`, `predict`, and `predict_proba`.
    """

    @abstractmethod
    def predict_proba(self, X: NumericArray) -> FloatArray:
        """
        Predict class probabilities for the given feature matrix.

        Parameters
        ----------
        X : NumericArray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        FloatArray
            Predicted class probabilities of shape (n_samples,) for binary
            classification or (n_samples, n_classes) for multi-class problems.
        """
