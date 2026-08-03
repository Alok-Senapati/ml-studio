"""
Input data validation utilities for array dimensionalities and label formats.
"""

from __future__ import annotations

import numpy as np

from ml_core.typing import NumericArray


def check_feature_matrix(X: NumericArray) -> None:
    """
    Validate that input `X` is a 2D feature matrix.

    Parameters
    ----------
    X : NumericArray
        Feature matrix array to validate.

    Raises
    ------
    ValueError
        If `X` does not have exactly 2 dimensions (i.e., `X.ndim != 2`).

    Examples
    --------
    >>> import numpy as np
    >>> from ml_core.validation.input import check_feature_matrix
    >>> X = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> check_feature_matrix(X)  # passes silently
    """
    # Ensure X is a 2D matrix (n_samples, n_features)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got array with {X.ndim} dimension.")


def check_target_vector(y: NumericArray) -> None:
    """
    Validate that target `y` is a 1D vector.

    Parameters
    ----------
    y : NumericArray
        Target array to validate.

    Raises
    ------
    ValueError
        If `y` does not have exactly 1 dimension (i.e., `y.ndim != 1`).

    Examples
    --------
    >>> import numpy as np
    >>> from ml_core.validation.input import check_target_vector
    >>> y = np.array([0, 1, 1])
    >>> check_target_vector(y)  # passes silently
    """
    # Ensure y is a 1D vector (n_samples,)
    if y.ndim != 1:
        raise ValueError(f"Expected y to be a 1D array, got {y.ndim} dimensions")


def check_same_length(X: NumericArray, y: NumericArray) -> None:
    """
    Validate that feature matrix `X` and target vector `y` have the same sample count.

    Parameters
    ----------
    X : NumericArray
        Feature matrix of shape (n_samples, n_features).
    y : NumericArray
        Target vector of shape (n_samples,).

    Raises
    ------
    ValueError
        If `X` and `y` have different numbers of rows/samples (`X.shape[0] != y.shape[0]`).

    Examples
    --------
    >>> import numpy as np
    >>> from ml_core.validation.input import check_same_length
    >>> X = np.array([[1.0], [2.0]])
    >>> y = np.array([0, 1])
    >>> check_same_length(X, y)  # passes silently
    """
    # Verify sample counts along the first axis match exactly
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y contains different number of samples.")


def check_binary_labels(y: NumericArray) -> None:
    """
    Validate that target vector `y` contains only binary classification labels (0 and 1).

    Parameters
    ----------
    y : NumericArray
        Target vector containing discrete class labels.

    Raises
    ------
    ValueError
        If `y` contains values other than 0 and 1.

    Examples
    --------
    >>> import numpy as np
    >>> from ml_core.validation.input import check_binary_labels
    >>> y = np.array([0, 1, 1, 0])
    >>> check_binary_labels(y)  # passes silently
    """
    # Extract set of unique elements present in y
    labels = set(np.unique(y))

    # Assert that all elements belong strictly to {0, 1}
    if not labels.issubset({0, 1}):
        raise ValueError("Binary labels must contain only 0 and 1")


def check_matching_target_vectors(y_true: NumericArray, y_pred: NumericArray) -> None:
    """
    Validate that `y_true` and `y_pred` are both 1D vectors with equal sample counts.

    Parameters
    ----------
    y_true : NumericArray
        Actual target labels or ground truth array of shape (n_samples,).
    y_pred : NumericArray
        Predicted target values or probabilities array of shape (n_samples,).

    Raises
    ------
    ValueError
        If either `y_true` or `y_pred` is not 1D, or if their sample counts mismatch.

    Examples
    --------
    >>> import numpy as np
    >>> from ml_core.validation.input import check_matching_target_vectors
    >>> y_true = np.array([1.0, 0.0])
    >>> y_pred = np.array([0.9, 0.1])
    >>> check_matching_target_vectors(y_true, y_pred)  # passes silently
    """
    # Verify both y_true and y_pred are 1D arrays
    check_target_vector(y_true)
    check_target_vector(y_pred)

    # Verify lengths along the primary axis match
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred contains different number of samples.")
