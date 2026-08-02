from __future__ import annotations

import numpy as np

from ml_core.typing import NumericArray
from ml_core.validation.input import check_matching_target_vectors

EPSILON = 1e-15


def binary_cross_entropy(y_true: NumericArray, y_pred: NumericArray) -> float:
    """
    Computes Binary Cross Entropy Loss.

    Parameters
    -----------
    y_true : NumericArray
             Actual Target Vector
    y_pred : NumericArray
             Predicted Probabilities

    Returns
    --------
    float
        Binary Cross Entropy Loss
    """
    y_true_float = np.asarray(y_true, dtype=np.float64)
    y_pred_float = np.asarray(y_pred, dtype=np.float64)

    check_matching_target_vectors(y_true_float, y_pred_float)

    # Prevent extreme probabilities (0.0 or 1.0) from causing log(0) = -inf,
    # which leads to NaN values in binary cross-entropy loss calculation.
    # Example: Maps [0.0, 1.0] -> [1e-15, 0.999999999999999]
    y_pred_clipped = np.clip(y_pred_float, EPSILON, 1 - EPSILON)

    loss = -(
        y_true_float * np.log(y_pred_clipped) + (1.0 - y_true_float) * np.log(1.0 - y_pred_clipped)
    )

    return float(np.mean(loss))
