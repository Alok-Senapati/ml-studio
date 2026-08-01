from __future__ import annotations  # noqa: I001

import numpy as np
from ml_core.typing import FloatArray, NumericArray


def sigmoid(x: NumericArray) -> FloatArray:
    """
    Compute the sigmoid activation function.

    Parameters
    -------------
    x: NumericArray
        Input Array.

    Returns
    -------------
    FloatArray
        Sigmoid of each element.

    """
    x = np.asarray(x, dtype=np.float64)

    return 1.0 / (1.0 + np.exp(-x))
