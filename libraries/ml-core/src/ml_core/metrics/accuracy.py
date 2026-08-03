"""
Classification accuracy evaluation metric implementation.
"""

from __future__ import annotations

import numpy as np

from ml_core.typing import NumericArray


def accuracy(y_true: NumericArray, y_pred: NumericArray) -> float:
    """
    Compute classification accuracy score.

    Accuracy measures the proportion of correctly predicted labels relative
    to total predictions made.

    Parameters
    ----------
    y_true : NumericArray
        Ground truth target labels of shape (n_samples,).
    y_pred : NumericArray
        Predicted target labels of shape (n_samples,).

    Returns
    -------
    float
        Fraction of correctly classified samples in range [0.0, 1.0].

    Notes
    -----
    Accuracy is defined mathematically as:

    .. math::

        \\text{Accuracy} = \\frac{1}{N} \\sum_{i=1}^{N} \\mathbb{I}(y_i = \\hat{y}_i)

    where :math:`\\mathbb{I}` is the indicator function equal to 1 if :math:`y_i = \\hat{y}_i`
    and 0 otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> from ml_core.metrics.accuracy import accuracy
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0, 1, 0, 0])
    >>> accuracy(y_true, y_pred)
    0.75
    """
    # Convert inputs to NumPy arrays for element-wise comparison
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    # Compute mean proportion of exact element matches
    return float(np.mean(y_true_arr == y_pred_arr))
