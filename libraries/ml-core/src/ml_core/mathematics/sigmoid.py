"""
Mathematical activation functions and numeric helper utilities.
"""

from __future__ import annotations

import numpy as np

from ml_core.typing import FloatArray, NumericArray


def sigmoid(x: NumericArray) -> FloatArray:
    """
    Compute the logistic sigmoid activation function element-wise.

    The sigmoid function maps any real-valued number into the open interval (0, 1),
    making it ideal for converting raw linear model outputs into probabilities.

    Parameters
    ----------
    x : NumericArray
        Input scalar, vector, or multi-dimensional array.

    Returns
    -------
    FloatArray
        Array of the same shape as `x`, with sigmoid applied element-wise.

    Notes
    -----
    The logistic sigmoid function is mathematically defined as:

    .. math::

        \\sigma(z) = \\frac{1}{1 + e^{-z}}

    Examples
    --------
    >>> import numpy as np
    >>> from ml_core.mathematics.sigmoid import sigmoid
    >>> sigmoid(np.array([0.0]))
    array([0.5])
    >>> sigmoid(np.array([-2.0, 2.0]))
    array([0.11920292, 0.88079708])
    """
    # Ensure input array is cast to float64 to prevent integer division issues
    x_float = np.asarray(x, dtype=np.float64)

    # Calculate element-wise logistic sigmoid: 1 / (1 + exp(-x))
    return 1.0 / (1.0 + np.exp(-x_float))
