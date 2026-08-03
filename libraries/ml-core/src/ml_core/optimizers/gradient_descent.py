"""
Gradient Descent optimization implementation.
"""

from __future__ import annotations

from ml_core.typing import FloatArray


def gradient_descent(
    weights: FloatArray, bias: float, dw: FloatArray, db: float, learning_rate: float
) -> tuple[FloatArray, float]:
    """
    Perform a single parameter update step using standard Gradient Descent.

    Updates model weight vector and scalar bias parameter in the opposite direction
    of their respective gradients to minimize the objective loss function.

    Parameters
    ----------
    weights : FloatArray
        Current model weights parameter vector of shape (n_features,).
    bias : float
        Current model scalar bias parameter.
    dw : FloatArray
        Gradient of the loss function with respect to weights of shape (n_features,).
    db : float
        Gradient of the loss function with respect to bias.
    learning_rate : float
        Learning rate hyperparameter (step size :math:`\\alpha > 0`).

    Returns
    -------
    tuple[FloatArray, float]
        Tuple containing:
        - weights_updated : FloatArray
            Updated weights vector of shape (n_features,).
        - bias_updated : float
            Updated scalar bias value.

    Notes
    -----
    The first-order parameter update equations for Gradient Descent are:

    .. math::

        w^{(t+1)} = w^{(t)} - \\alpha \\cdot \\frac{\\partial \\mathcal{L}}{\\partial w}

    .. math::

        b^{(t+1)} = b^{(t)} - \\alpha \\cdot \\frac{\\partial \\mathcal{L}}{\\partial b}

    where :math:`\\alpha` is the learning rate,
    :math:`\\frac{\\partial \\mathcal{L}}{\\partial w} = \\text{dw}`,
    and :math:`\\frac{\\partial \\mathcal{L}}{\\partial b} = \\text{db}`.

    Examples
    --------
    >>> import numpy as np
    >>> from ml_core.optimizers.gradient_descent import gradient_descent
    >>> w = np.array([1.0, 2.0])
    >>> b = 0.5
    >>> dw = np.array([0.1, 0.2])
    >>> db = 0.05
    >>> w_new, b_new = gradient_descent(w, b, dw, db, learning_rate=0.1)
    >>> w_new
    array([0.99, 1.98])
    >>> b_new
    0.495
    """
    # Perform vector parameter update step for weights: w = w - alpha * dw
    weights_updated = weights - (learning_rate * dw)

    # Perform scalar parameter update step for bias: b = b - alpha * db
    bias_updated = bias - (learning_rate * db)

    return weights_updated, bias_updated
