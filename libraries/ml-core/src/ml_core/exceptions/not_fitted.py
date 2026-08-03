"""
Exception raised when calling prediction methods before fitting an estimator.
"""

from __future__ import annotations

from ml_core.exceptions.base import MLCoreError


class NotFittedError(MLCoreError):
    """
    Exception raised when an unfitted estimator is used for inference.

    This exception is raised when methods requiring fitted parameters (e.g.,
    `predict`, `predict_proba`, `transform`) are called on an estimator before
    `fit` has been invoked.

    Notes
    -----
    This error ensures that model parameters are properly initialized and learned
    prior to executing forward propagation or generating inference predictions.
    """
