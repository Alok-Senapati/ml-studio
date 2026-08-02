from __future__ import annotations

import numpy as np

from ml_core.typing import NumericArray


def accuracy(y_true: NumericArray, y_pred: NumericArray) -> float:
    """
    Computes classification accuracy.

    Parameters
    ----------
    y_true: NumericArray
        Actual Labels
    y_pred: NumericArray
        Predicted Labels

    Returns
    -------
    float
        Accuracy
    """

    return float(np.mean(y_true == y_pred))
