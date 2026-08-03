"""
Unit tests for Binary Cross Entropy loss function.
"""

from __future__ import annotations

import numpy as np
import pytest
from ml_core.losses.binary_cross_entropy import binary_cross_entropy


def test_binary_cross_entropy_perfect_prediction() -> None:
    """Test BCE with perfect predictions (y_true == y_pred)."""
    y_true = np.array([0.0, 1.0, 1.0, 0.0])
    y_pred = np.array([0.0, 1.0, 1.0, 0.0])
    loss = binary_cross_entropy(y_true, y_pred)
    # With epsilon clipping, loss is very close to zero but non-negative
    assert loss < 1e-10


def test_binary_cross_entropy_worst_prediction() -> None:
    """Test BCE with worst possible predictions (y_true != y_pred)."""
    y_true = np.array([0.0, 1.0, 1.0, 0.0])
    y_pred = np.array([1.0, 0.0, 0.0, 1.0])
    loss = binary_cross_entropy(y_true, y_pred)
    # With epsilon clipping, this yields high log-loss values
    assert loss > 10.0


def test_binary_cross_entropy_single_sample() -> None:
    """Test BCE with a single sample vector."""
    y_true = np.array([1.0])
    y_pred = np.array([0.5])
    loss = binary_cross_entropy(y_true, y_pred)
    expected = -(1.0 * np.log(0.5) + 0.0 * np.log(0.5))
    assert np.isclose(loss, expected)


def test_binary_cross_entropy_range() -> None:
    """Test that BCE loss output is strictly non-negative."""
    y_true = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    loss = binary_cross_entropy(y_true, y_pred)
    assert loss >= 0.0


def test_binary_cross_entropy_symmetry_of_errors() -> None:
    """Test BCE loss symmetry when both ground truth and predictions are inverted."""
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.7, 0.3])
    loss1 = binary_cross_entropy(y_true, y_pred)

    y_true_flipped = np.array([1.0, 0.0])
    y_pred_flipped = np.array([0.3, 0.7])
    loss2 = binary_cross_entropy(y_true_flipped, y_pred_flipped)

    assert np.isclose(loss1, loss2)


def test_binary_cross_entropy_increases_with_error() -> None:
    """Test that BCE loss monotonically increases as prediction error increases."""
    y_true = np.array([1.0, 1.0, 1.0])
    y_pred_close = np.array([0.9, 0.9, 0.9])
    y_pred_far = np.array([0.5, 0.5, 0.5])

    loss_close = binary_cross_entropy(y_true, y_pred_close)
    loss_far = binary_cross_entropy(y_true, y_pred_far)

    assert loss_far > loss_close


def test_binary_cross_entropy_extreme_predictions() -> None:
    """Test that BCE handles extreme probabilities (0.0 or 1.0) without NaN or Infinity."""
    y_true = np.array([1.0, 0.0])
    y_pred = np.array([1.0, 0.0])
    # Clipping prevents log(0) domain errors
    loss = binary_cross_entropy(y_true, y_pred)
    assert not np.isnan(loss)
    assert not np.isinf(loss)


def test_binary_cross_entropy_mixed_predictions() -> None:
    """Test BCE calculation with mixed correct and incorrect predictions."""
    y_true = np.array([1.0, 1.0, 0.0, 0.0])
    y_pred = np.array([0.9, 0.2, 0.1, 0.8])
    loss = binary_cross_entropy(y_true, y_pred)
    assert not np.isnan(loss)
    assert loss > 0.0


def test_binary_cross_entropy_probability_predictions() -> None:
    """Test BCE calculation with standard continuous probability predictions."""
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_pred = np.array([0.8, 0.2, 0.6, 0.3])
    loss = binary_cross_entropy(y_true, y_pred)
    assert 0.0 <= loss < np.inf


def test_binary_cross_entropy_mismatched_lengths() -> None:
    """Test that BCE raises ValueError when target array lengths mismatch."""
    y_true = np.array([1.0, 0.0, 1.0])
    y_pred = np.array([0.8, 0.2])
    with pytest.raises(ValueError):
        binary_cross_entropy(y_true, y_pred)


def test_binary_cross_entropy_2d_arrays() -> None:
    """Test that BCE raises ValueError when inputs are 2D matrices instead of 1D vectors."""
    y_true = np.array([[1.0, 0.0]])
    y_pred = np.array([[0.8, 0.2]])
    with pytest.raises(ValueError):
        binary_cross_entropy(y_true, y_pred)


def test_binary_cross_entropy_scalar_inputs() -> None:
    """Test that BCE with scalar array inputs returns a Python float."""
    y_true = np.array([1.0])
    y_pred = np.array([0.7])
    loss = binary_cross_entropy(y_true, y_pred)
    assert isinstance(loss, float)
