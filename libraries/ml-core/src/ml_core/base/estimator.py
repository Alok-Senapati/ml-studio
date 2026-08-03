"""
Abstract base estimator interface for machine learning models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

from ml_core.typing import NumericArray


class Estimator(ABC):
    """
    Abstract base interface for all machine learning estimators in ml-core.

    All supervised learning models, transformers, and estimators inherit from this
    class and implement its standard unified API design (`fit`, `predict`).

    Notes
    -----
    This abstraction guarantees API compatibility across different algorithms in
    the library, mirroring scikit-learn's estimator pattern.
    """

    @abstractmethod
    def fit(self, X: NumericArray, y: NumericArray) -> Self:
        """
        Fit the estimator to the training data.

        Parameters
        ----------
        X : NumericArray
            Training feature matrix of shape (n_samples, n_features).
        y : NumericArray
            Target labels or values of shape (n_samples,).

        Returns
        -------
        Self
            The fitted estimator instance.
        """

    @abstractmethod
    def predict(self, X: NumericArray) -> NumericArray:
        """
        Predict target values or class labels for the given feature matrix.

        Parameters
        ----------
        X : NumericArray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        NumericArray
            Predicted target values or discrete class labels of shape (n_samples,).
        """
