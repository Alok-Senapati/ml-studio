"""
Binary Cross Entropy (Log Loss) loss function implementation.
"""

from __future__ import annotations

import numpy as np

from ml_core.typing import NumericArray
from ml_core.validation.input import check_matching_target_vectors

#: Small floating-point constant used to prevent log(0) domain errors during loss calculation.
EPSILON: float = 1e-15


def binary_cross_entropy(y_true: NumericArray, y_pred: NumericArray) -> float:
    """
    Compute Binary Cross Entropy (Log Loss) between true labels and predicted probabilities.

    Binary Cross Entropy evaluates the performance of a classification model whose
    output is a probability value between 0 and 1. The loss increases as the predicted
    probability diverges from the actual label.

    Parameters
    ----------
    y_true : NumericArray
        Ground truth binary target labels (0.0 or 1.0) of shape (n_samples,).
    y_pred : NumericArray
        Predicted class probabilities in [0.0, 1.0] of shape (n_samples,).

    Returns
    -------
    float
        The scalar mean binary cross entropy loss value.

    Raises
    ------
    ValueError
        If `y_true` and `y_pred` are not 1D vectors or have mismatched lengths.

    Notes
    -----
    The Binary Cross Entropy loss formula over :math:`N` samples is given by:

    .. math::

        \\mathcal{L}(y, \\hat{y}) = -\\frac{1}{N} \\sum_{i=1}^{N}
        \\left[ y_i \\log(\\hat{y}_i) + (1 - y_i) \\log(1 - \\hat{y}_i) \\right]

    To prevent numerical instability where :math:`\\log(0) \\to -\\infty` yields ``NaN`` values,
    predicted probabilities are clipped to the interval :math:`[\\epsilon, 1 - \\epsilon]`,
    where :math:`\\epsilon = 10^{-15}`.

    Examples
    --------
    >>> import numpy as np
    >>> from ml_core.losses.binary_cross_entropy import binary_cross_entropy
    >>> y_true = np.array([1.0, 0.0, 1.0, 0.0])
    >>> y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    >>> round(binary_cross_entropy(y_true, y_pred), 4)
    0.1643
    """
    # Convert input arrays to float64 for uniform high-precision numerical operations
    y_true_float = np.asarray(y_true, dtype=np.float64)
    y_pred_float = np.asarray(y_pred, dtype=np.float64)

    # Validate that both y_true and y_pred are 1D arrays with matching dimensions
    check_matching_target_vectors(y_true_float, y_pred_float)

    # Prevent extreme probabilities (0.0 or 1.0) from causing log(0) = -inf,
    # which leads to NaN values in binary cross-entropy loss calculation.
    # Maps probabilities to range [1e-15, 1 - 1e-15]
    y_pred_clipped = np.clip(y_pred_float, EPSILON, 1.0 - EPSILON)

    # Compute element-wise loss across all samples
    loss = -(
        y_true_float * np.log(y_pred_clipped) + (1.0 - y_true_float) * np.log(1.0 - y_pred_clipped)
    )

    # Return scalar mean loss across the sample batch
    return float(np.mean(loss))
