from __future__ import annotations

from ml_core.typing import FloatArray


def gradient_descent(
    weights: FloatArray, bias: float, dw: FloatArray, db: float, learning_rate: float
) -> tuple[FloatArray, float]:
    """
    Performs one gradient descent operation.

    Parameters
    -----------
    weights: FloatArray
        Weights array to be updated.
    bias: float
        Bias to be updated
    dw: FloatArray
        Derivative of loss with respect to weights.
    db: float
        Derivative of loss with respect to bias.
    learning_rate: float
        Learning Rate

    Returns
    ---------
    tuple[FloatArray, float]
        Tuple containing updated weights array and bias.
    """

    weights_updated = weights - (learning_rate * dw)
    bias_updated = bias - (learning_rate * db)

    return weights_updated, bias
