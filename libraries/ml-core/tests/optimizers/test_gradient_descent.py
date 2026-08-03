"""
Unit tests for the gradient descent optimization function.
"""

from __future__ import annotations

import numpy as np
from ml_core.optimizers.gradient_descent import gradient_descent


def test_gradient_descent_basic_update() -> None:
    """Test standard gradient descent parameter update equations."""
    weights = np.array([1.0, 2.0, 3.0])
    bias = 0.5
    dw = np.array([0.1, 0.2, 0.3])
    db = 0.05
    learning_rate = 0.01

    weights_updated, bias_updated = gradient_descent(weights, bias, dw, db, learning_rate)

    expected_weights = weights - (learning_rate * dw)
    expected_bias = bias - (learning_rate * db)

    np.testing.assert_array_almost_equal(weights_updated, expected_weights)
    assert np.isclose(bias_updated, expected_bias)


def test_gradient_descent_zero_gradients() -> None:
    """Test parameter stability when loss gradients are zero."""
    weights = np.array([1.0, 2.0])
    bias = 0.5
    dw = np.array([0.0, 0.0])
    db = 0.0
    learning_rate = 0.01

    weights_updated, bias_updated = gradient_descent(weights, bias, dw, db, learning_rate)

    np.testing.assert_array_almost_equal(weights_updated, weights)
    assert np.isclose(bias_updated, bias)


def test_gradient_descent_single_weight() -> None:
    """Test gradient descent update with a single 1D weight element."""
    weights = np.array([2.0])
    bias = 1.0
    dw = np.array([0.5])
    db = 0.2
    learning_rate = 0.1

    weights_updated, bias_updated = gradient_descent(weights, bias, dw, db, learning_rate)

    expected_w = 2.0 - (0.1 * 0.5)
    expected_b = 1.0 - (0.1 * 0.2)

    assert np.isclose(weights_updated[0], expected_w)
    assert np.isclose(bias_updated, expected_b)


def test_gradient_descent_high_learning_rate() -> None:
    """Test gradient descent step behavior under a large learning rate (alpha=0.5)."""
    weights = np.array([1.0, 1.0])
    bias = 1.0
    dw = np.array([1.0, 1.0])
    db = 1.0
    learning_rate = 0.5

    weights_updated, bias_updated = gradient_descent(weights, bias, dw, db, learning_rate)

    expected_weights = np.array([0.5, 0.5])
    expected_bias = 0.5

    np.testing.assert_array_almost_equal(weights_updated, expected_weights)
    assert np.isclose(bias_updated, expected_bias)


def test_gradient_descent_low_learning_rate() -> None:
    """Test gradient descent step behavior under a small learning rate (alpha=0.001)."""
    weights = np.array([1.0, 1.0])
    bias = 1.0
    dw = np.array([1.0, 1.0])
    db = 1.0
    learning_rate = 0.001

    weights_updated, bias_updated = gradient_descent(weights, bias, dw, db, learning_rate)

    expected_weights = np.array([0.999, 0.999])
    expected_bias = 0.999

    np.testing.assert_array_almost_equal(weights_updated, expected_weights)
    assert np.isclose(bias_updated, expected_bias)


def test_gradient_descent_zero_learning_rate() -> None:
    """Test that zero learning rate (alpha=0.0) leaves parameters unchanged."""
    weights = np.array([1.0, 2.0])
    bias = 0.5
    dw = np.array([0.1, 0.2])
    db = 0.05
    learning_rate = 0.0

    weights_updated, bias_updated = gradient_descent(weights, bias, dw, db, learning_rate)

    np.testing.assert_array_almost_equal(weights_updated, weights)
    assert np.isclose(bias_updated, bias)


def test_gradient_descent_positive_gradients() -> None:
    """Test that positive gradients cause parameters to decrease."""
    weights = np.array([1.0, 2.0])
    bias = 0.5
    dw = np.array([0.1, 0.2])
    db = 0.05
    learning_rate = 0.1

    weights_updated, bias_updated = gradient_descent(weights, bias, dw, db, learning_rate)

    assert all(weights_updated < weights)
    assert bias_updated < bias


def test_gradient_descent_negative_gradients() -> None:
    """Test that negative gradients cause parameters to increase."""
    weights = np.array([1.0, 2.0])
    bias = 0.5
    dw = np.array([-0.1, -0.2])
    db = -0.05
    learning_rate = 0.1

    weights_updated, bias_updated = gradient_descent(weights, bias, dw, db, learning_rate)

    assert all(weights_updated > weights)
    assert bias_updated > bias


def test_gradient_descent_mixed_gradients() -> None:
    """Test parameter update behavior with mixed positive and negative gradients."""
    weights = np.array([1.0, 2.0, 3.0])
    bias = 1.0
    dw = np.array([0.1, -0.2, 0.3])
    db = 0.1
    learning_rate = 0.1

    weights_updated, bias_updated = gradient_descent(weights, bias, dw, db, learning_rate)

    # First weight should decrease
    assert weights_updated[0] < weights[0]
    # Second weight should increase
    assert weights_updated[1] > weights[1]
    # Third weight should decrease
    assert weights_updated[2] < weights[2]


def test_gradient_descent_return_types() -> None:
    """Test that gradient descent returns NumPy array weights and float bias."""
    weights = np.array([1.0, 2.0])
    bias = 0.5
    dw = np.array([0.1, 0.2])
    db = 0.05
    learning_rate = 0.01

    weights_updated, bias_updated = gradient_descent(weights, bias, dw, db, learning_rate)

    assert isinstance(weights_updated, np.ndarray)
    assert isinstance(bias_updated, (float, np.floating))


def test_gradient_descent_weights_shape_preserved() -> None:
    """Test that gradient descent preserves input weight array shape."""
    weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    bias = 0.5
    dw = np.array([[0.1, 0.2], [0.3, 0.4]])
    db = 0.05
    learning_rate = 0.01

    weights_updated, _ = gradient_descent(weights, bias, dw, db, learning_rate)

    assert weights_updated.shape == weights.shape


def test_gradient_descent_multiple_iterations() -> None:
    """Test parameter trajectory over multiple iterative gradient descent steps."""
    weights = np.array([5.0, 5.0])
    bias = 5.0
    learning_rate = 0.1

    dw = np.array([1.0, 1.0])
    db = 1.0

    for _ in range(10):
        weights, bias = gradient_descent(weights, bias, dw, db, learning_rate)

    # After 10 iterations with gradient 1.0 and learning rate 0.1
    expected_weights = 5.0 - (10 * 0.1 * 1.0)  # = 4.0
    expected_bias = 5.0 - (10 * 0.1 * 1.0)  # = 4.0

    np.testing.assert_array_almost_equal(weights, [expected_weights, expected_weights])
    assert np.isclose(bias, expected_bias)


def test_gradient_descent_converges_to_minimum() -> None:
    """Test that iterative gradient descent reduces total parameter norm towards local minimum."""
    weights = np.array([10.0])
    bias = 10.0
    dw = np.array([2.0])
    db = 2.0
    learning_rate = 0.01

    initial_loss = weights[0] ** 2 + bias**2

    for _ in range(100):
        weights, bias = gradient_descent(weights, bias, dw, db, learning_rate)

    final_loss = weights[0] ** 2 + bias**2
    assert final_loss < initial_loss


def test_gradient_descent_large_weights() -> None:
    """Test gradient descent numerical stability when operating on large weight values."""
    weights = np.array([1e6, 1e6])
    bias = 1e6
    dw = np.array([1.0, 1.0])
    db = 1.0
    learning_rate = 1e-3

    weights_updated, bias_updated = gradient_descent(weights, bias, dw, db, learning_rate)

    assert np.isfinite(weights_updated).all()
    assert np.isfinite(bias_updated)


def test_gradient_descent_very_small_learning_rate() -> None:
    """Test gradient descent step behavior under extremely small learning rates."""
    weights = np.array([1.0])
    bias = 1.0
    dw = np.array([1.0])
    db = 1.0
    learning_rate = 1e-10

    weights_updated, bias_updated = gradient_descent(weights, bias, dw, db, learning_rate)

    assert np.isclose(weights_updated[0], weights[0], atol=1e-8)
    assert np.isclose(bias_updated, bias, atol=1e-8)


def test_gradient_descent_high_dimensional() -> None:
    """Test gradient descent execution on high-dimensional weight arrays."""
    weights = np.random.randn(100)
    bias = 1.0
    dw = np.random.randn(100)
    db = 0.1
    learning_rate = 0.01

    weights_updated, bias_updated = gradient_descent(weights, bias, dw, db, learning_rate)

    assert weights_updated.shape == weights.shape
    assert np.isfinite(weights_updated).all()
    assert np.isfinite(bias_updated)
