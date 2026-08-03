"""
Unit tests for the sigmoid activation function.
"""

from __future__ import annotations

import numpy as np
from ml_core.mathematics.sigmoid import sigmoid


def test_sigmoid_zero() -> None:
    """Test that sigmoid(0) equals 0.5."""
    result = sigmoid(np.array([0.0]))
    assert np.isclose(result[0], 0.5)


def test_sigmoid_positive() -> None:
    """Test sigmoid output for positive inputs."""
    result = sigmoid(np.array([1.0]))
    expected = 1.0 / (1.0 + np.exp(-1.0))
    assert np.isclose(result[0], expected)


def test_sigmoid_negative() -> None:
    """Test sigmoid output for negative inputs."""
    result = sigmoid(np.array([-1.0]))
    expected = 1.0 / (1.0 + np.exp(1.0))
    assert np.isclose(result[0], expected)


def test_sigmoid_multiple_values() -> None:
    """Test sigmoid evaluated element-wise across a 1D vector."""
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = sigmoid(x)
    expected = 1.0 / (1.0 + np.exp(-x))
    np.testing.assert_array_almost_equal(result, expected)


def test_sigmoid_2d_array() -> None:
    """Test sigmoid evaluated element-wise across a 2D matrix."""
    x = np.array([[0.0, 1.0], [-1.0, 2.0]])
    result = sigmoid(x)
    expected = 1.0 / (1.0 + np.exp(-x))
    np.testing.assert_array_almost_equal(result, expected)


def test_sigmoid_single_element() -> None:
    """Test sigmoid evaluated on a single-element 1D array."""
    result = sigmoid(np.array([0.5]))
    expected = 1.0 / (1.0 + np.exp(-0.5))
    assert np.isclose(result[0], expected)


def test_sigmoid_output_range() -> None:
    """Test that sigmoid outputs remain strictly within range [0, 1]."""
    x = np.linspace(-100, 100, 1000)
    result = sigmoid(x)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_sigmoid_symmetry() -> None:
    """Test mathematical symmetry property: sigmoid(x) + sigmoid(-x) = 1."""
    x = np.array([1.0, 2.0, 3.0])
    sig_x = sigmoid(x)
    sig_neg_x = sigmoid(-x)
    np.testing.assert_array_almost_equal(sig_x + sig_neg_x, 1.0)


def test_sigmoid_monotonicity() -> None:
    """Test that sigmoid is a strictly monotonically increasing function."""
    x = np.linspace(-5, 5, 100)
    result = sigmoid(x)
    differences = np.diff(result)
    assert np.all(differences >= 0)


def test_sigmoid_large_positive() -> None:
    """Test that sigmoid with large positive input approaches 1.0."""
    result = sigmoid(np.array([100.0]))
    assert np.isclose(result[0], 1.0)


def test_sigmoid_large_negative() -> None:
    """Test that sigmoid with large negative input approaches 0.0."""
    result = sigmoid(np.array([-100.0]))
    assert np.isclose(result[0], 0.0, atol=1e-10)


def test_sigmoid_list_input() -> None:
    """Test that sigmoid accepts list sequences and converts them to NumPy arrays."""
    result = sigmoid([0.0, 1.0, -1.0])
    assert len(result) == 3
    assert np.isclose(result[0], 0.5)
