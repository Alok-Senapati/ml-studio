from __future__ import annotations

import numpy as np

from ml_core.typing import NumericArray


def check_feature_matrix(X: NumericArray) -> None:
    """
    Validates that x is a 2D feature matrix.

    Parameters
    ------------
    X : NumericArray
        Input Matrix
    """

    if X.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got array with {X.ndim} dimension.")


def check_target_vector(y: NumericArray) -> None:
    """
    Validates that y is a 1D target vector.

    Parameters
    ------------
    y : NumericArray
        Input Array
    """

    if y.ndim != 1:
        raise ValueError(f"Expected y to be a 1D array, got {y.ndim} dimensions")


def check_same_length(X: NumericArray, y: NumericArray) -> None:
    """
    Validates that X and y contain same number of samples.

    Parameters
    ------------
    X : NumericArray
        Input Feature Matrix
    y : NumericArray
        Target Vector
    """

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y contains different number of samples.")


def check_binary_labels(y: NumericArray) -> None:
    """
    Validates binary classification labels.

    Parameters
    ------------
    y : NumericArray
        Target Vector
    """

    labels = set(np.unique(y))

    if not labels.issubset({0, 1}):
        raise ValueError("Binary labels must contain only 0 and 1")


def check_matching_target_vectors(y_true: NumericArray, y_pred: NumericArray) -> None:
    """
    Validates the size of y_true and y_pred, and they are 1D arrays.

    Parameters
    -------------
    y_true: NumericArray
            Actual Labels
    y_pred: NumericArray
            Predicted Probabilities
    """
    check_target_vector(y_true)
    check_target_vector(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred contains different number of samples.")
