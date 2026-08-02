import numpy as np
from ml_core.metrics.accuracy import accuracy


def test_accuracy_perfect_predictions():
    """Test accuracy with perfect predictions."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    acc = accuracy(y_true, y_pred)
    assert np.isclose(acc, 1.0)


def test_accuracy_all_wrong():
    """Test accuracy with all wrong predictions."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 0])
    acc = accuracy(y_true, y_pred)
    assert np.isclose(acc, 0.0)


def test_accuracy_half_correct():
    """Test accuracy with half correct predictions."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    acc = accuracy(y_true, y_pred)
    assert np.isclose(acc, 0.5)


def test_accuracy_single_sample():
    """Test accuracy with single sample."""
    y_true = np.array([1])
    y_pred = np.array([1])
    acc = accuracy(y_true, y_pred)
    assert np.isclose(acc, 1.0)


def test_accuracy_multiclass_perfect():
    """Test accuracy with perfect multiclass predictions."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    acc = accuracy(y_true, y_pred)
    assert np.isclose(acc, 1.0)


def test_accuracy_multiclass_partial():
    """Test accuracy with partial multiclass predictions."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 1, 1, 0])
    acc = accuracy(y_true, y_pred)
    expected = 4 / 6  # 4 correct out of 6
    assert np.isclose(acc, expected)


def test_accuracy_multiclass_all_wrong():
    """Test accuracy with all wrong multiclass predictions."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([1, 2, 0, 2, 0, 1])
    acc = accuracy(y_true, y_pred)
    assert np.isclose(acc, 0.0)


def test_accuracy_range():
    """Test accuracy is always between 0 and 1."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    acc = accuracy(y_true, y_pred)
    assert 0.0 <= acc <= 1.0


def test_accuracy_symmetry():
    """Test accuracy is symmetric for binary case."""
    y_true = np.array([0, 1])
    y_pred1 = np.array([1, 0])
    y_pred2 = np.array([0, 1])
    # First predictions are both wrong
    acc1 = accuracy(y_true, y_pred1)
    # Second predictions are both correct
    acc2 = accuracy(y_true, y_pred2)
    assert np.isclose(acc1, 0.0)
    assert np.isclose(acc2, 1.0)


def test_accuracy_large_dataset():
    """Test accuracy with larger dataset."""
    n_samples = 1000
    y_true = np.random.randint(0, 3, n_samples)
    y_pred = y_true.copy()
    y_pred[0:100] = (y_pred[0:100] + 1) % 3  # Flip 100 predictions
    acc = accuracy(y_true, y_pred)
    expected = 900 / 1000
    assert np.isclose(acc, expected)


def test_accuracy_float_inputs():
    """Test accuracy with float inputs."""
    y_true = np.array([0.0, 1.0, 1.0, 0.0])
    y_pred = np.array([0.0, 1.0, 0.0, 0.0])
    acc = accuracy(y_true, y_pred)
    expected = 3 / 4
    assert np.isclose(acc, expected)


def test_accuracy_list_inputs():
    """Test accuracy with list inputs."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    acc = accuracy(y_true, y_pred)
    expected = 3 / 4
    assert np.isclose(acc, expected)


def test_accuracy_return_type():
    """Test accuracy returns float."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    acc = accuracy(y_true, y_pred)
    assert isinstance(acc, float)


def test_accuracy_single_class():
    """Test accuracy with all same labels."""
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1])
    acc = accuracy(y_true, y_pred)
    assert np.isclose(acc, 1.0)


def test_accuracy_many_classes():
    """Test accuracy with many different classes."""
    y_true = np.arange(100)
    y_pred = np.arange(100)
    acc = accuracy(y_true, y_pred)
    assert np.isclose(acc, 1.0)


def test_accuracy_boolean_labels():
    """Test accuracy with boolean labels."""
    y_true = np.array([True, False, True, False])
    y_pred = np.array([True, False, False, False])
    acc = accuracy(y_true, y_pred)
    expected = 3 / 4
    assert np.isclose(acc, expected)
