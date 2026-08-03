"""
Unit tests for the accuracy metric evaluation function.
"""

from __future__ import annotations

import numpy as np
from ml_core.metrics.accuracy import accuracy


def test_accuracy_perfect_predictions() -> None:
    """Test accuracy calculation with 100% correct predictions."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    acc = accuracy(y_true, y_pred)
    assert np.isclose(acc, 1.0)


def test_accuracy_all_wrong() -> None:
    """Test accuracy calculation with 0% correct predictions."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 0])
    acc = accuracy(y_true, y_pred)
    assert np.isclose(acc, 0.0)


def test_accuracy_half_correct() -> None:
    """Test accuracy calculation with 50% correct predictions."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    acc = accuracy(y_true, y_pred)
    assert np.isclose(acc, 0.5)


def test_accuracy_single_sample() -> None:
    """Test accuracy calculation with a single sample array."""
    y_true = np.array([1])
    y_pred = np.array([1])
    acc = accuracy(y_true, y_pred)
    assert np.isclose(acc, 1.0)


def test_accuracy_multiclass_perfect() -> None:
    """Test accuracy calculation with perfect multiclass predictions."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    acc = accuracy(y_true, y_pred)
    assert np.isclose(acc, 1.0)


def test_accuracy_multiclass_partial() -> None:
    """Test accuracy calculation with partial multiclass predictions."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 1, 1, 0])
    acc = accuracy(y_true, y_pred)
    expected = 4.0 / 6.0  # 4 correct out of 6
    assert np.isclose(acc, expected)


def test_accuracy_multiclass_all_wrong() -> None:
    """Test accuracy calculation with completely wrong multiclass predictions."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([1, 2, 0, 2, 0, 1])
    acc = accuracy(y_true, y_pred)
    assert np.isclose(acc, 0.0)


def test_accuracy_range() -> None:
    """Test that accuracy output is always bounded in range [0.0, 1.0]."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    acc = accuracy(y_true, y_pred)
    assert 0.0 <= acc <= 1.0


def test_accuracy_symmetry() -> None:
    """Test accuracy symmetry for binary cases."""
    y_true = np.array([0, 1])
    y_pred1 = np.array([1, 0])
    y_pred2 = np.array([0, 1])
    # First predictions are both wrong
    acc1 = accuracy(y_true, y_pred1)
    # Second predictions are both correct
    acc2 = accuracy(y_true, y_pred2)
    assert np.isclose(acc1, 0.0)
    assert np.isclose(acc2, 1.0)


def test_accuracy_large_dataset() -> None:
    """Test accuracy calculation on a larger randomized dataset."""
    n_samples = 1000
    y_true = np.random.randint(0, 3, n_samples)
    y_pred = y_true.copy()
    y_pred[0:100] = (y_pred[0:100] + 1) % 3  # Intentionally flip 100 predictions
    acc = accuracy(y_true, y_pred)
    expected = 900.0 / 1000.0
    assert np.isclose(acc, expected)


def test_accuracy_float_inputs() -> None:
    """Test accuracy calculation with floating-point label arrays."""
    y_true = np.array([0.0, 1.0, 1.0, 0.0])
    y_pred = np.array([0.0, 1.0, 0.0, 0.0])
    acc = accuracy(y_true, y_pred)
    expected = 3.0 / 4.0
    assert np.isclose(acc, expected)


def test_accuracy_list_inputs() -> None:
    """Test accuracy calculation with Python list inputs."""
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    acc = accuracy(y_true, y_pred)
    expected = 3.0 / 4.0
    assert np.isclose(acc, expected)


def test_accuracy_return_type() -> None:
    """Test that accuracy returns a standard Python float."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    acc = accuracy(y_true, y_pred)
    assert isinstance(acc, float)


def test_accuracy_single_class() -> None:
    """Test accuracy calculation when all samples belong to the same class."""
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1])
    acc = accuracy(y_true, y_pred)
    assert np.isclose(acc, 1.0)


def test_accuracy_many_classes() -> None:
    """Test accuracy calculation with many distinct class labels."""
    y_true = np.arange(100)
    y_pred = np.arange(100)
    acc = accuracy(y_true, y_pred)
    assert np.isclose(acc, 1.0)


def test_accuracy_boolean_labels() -> None:
    """Test accuracy calculation with boolean arrays."""
    y_true = np.array([True, False, True, False])
    y_pred = np.array([True, False, False, False])
    acc = accuracy(y_true, y_pred)
    expected = 3.0 / 4.0
    assert np.isclose(acc, expected)
