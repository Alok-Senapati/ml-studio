import numpy as np
import pytest
from ml_core.losses.binary_cross_entropy import EPSILON, binary_cross_entropy


def test_binary_cross_entropy_perfect_prediction():
    """Test BCE with perfect predictions (y_true == y_pred)."""
    y_true = np.array([0.0, 1.0, 1.0, 0.0])
    y_pred = np.array([0.0, 1.0, 1.0, 0.0])
    loss = binary_cross_entropy(y_true, y_pred)
    # With clipping, very small loss but not exactly 0
    assert loss < 1e-10


def test_binary_cross_entropy_worst_prediction():
    """Test BCE with worst predictions (y_true != y_pred)."""
    y_true = np.array([0.0, 1.0, 1.0, 0.0])
    y_pred = np.array([1.0, 0.0, 0.0, 1.0])
    loss = binary_cross_entropy(y_true, y_pred)
    # With clipping, this should be high loss
    assert loss > 10.0


def test_binary_cross_entropy_single_sample():
    """Test BCE with single sample."""
    y_true = np.array([1.0])
    y_pred = np.array([0.5])
    loss = binary_cross_entropy(y_true, y_pred)
    expected = -(1.0 * np.log(0.5) + 0.0 * np.log(0.5))
    assert np.isclose(loss, expected)


def test_binary_cross_entropy_range():
    """Test BCE output is always non-negative."""
    y_true = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    loss = binary_cross_entropy(y_true, y_pred)
    assert loss >= 0.0


def test_binary_cross_entropy_symmetry_of_errors():
    """Test BCE with flipped labels should be same."""
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.7, 0.3])
    loss1 = binary_cross_entropy(y_true, y_pred)

    y_true_flipped = np.array([1.0, 0.0])
    y_pred_flipped = np.array([0.3, 0.7])
    loss2 = binary_cross_entropy(y_true_flipped, y_pred_flipped)

    assert np.isclose(loss1, loss2)


def test_binary_cross_entropy_increases_with_error():
    """Test BCE increases as prediction error increases."""
    y_true = np.array([1.0, 1.0, 1.0])
    y_pred_close = np.array([0.9, 0.9, 0.9])
    y_pred_far = np.array([0.5, 0.5, 0.5])

    loss_close = binary_cross_entropy(y_true, y_pred_close)
    loss_far = binary_cross_entropy(y_true, y_pred_far)

    assert loss_far > loss_close


def test_binary_cross_entropy_extreme_predictions():
    """Test BCE handles extreme predictions with clipping."""
    y_true = np.array([1.0, 0.0])
    y_pred = np.array([1.0, 0.0])
    # Should not raise due to clipping
    loss = binary_cross_entropy(y_true, y_pred)
    assert not np.isnan(loss)
    assert not np.isinf(loss)


def test_binary_cross_entropy_mixed_predictions():
    """Test BCE with mixed correct and incorrect predictions."""
    y_true = np.array([1.0, 1.0, 0.0, 0.0])
    y_pred = np.array([0.9, 0.2, 0.1, 0.8])
    loss = binary_cross_entropy(y_true, y_pred)
    assert not np.isnan(loss)
    assert loss > 0.0


def test_binary_cross_entropy_probability_predictions():
    """Test BCE with valid probability predictions."""
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_pred = np.array([0.8, 0.2, 0.6, 0.3])
    loss = binary_cross_entropy(y_true, y_pred)
    assert 0.0 <= loss < np.inf


def test_binary_cross_entropy_mismatched_lengths():
    """Test BCE raises error for mismatched lengths."""
    y_true = np.array([1.0, 0.0, 1.0])
    y_pred = np.array([0.8, 0.2])
    with pytest.raises(ValueError):
        binary_cross_entropy(y_true, y_pred)


def test_binary_cross_entropy_2d_arrays():
    """Test BCE raises error for 2D arrays."""
    y_true = np.array([[1.0, 0.0]])
    y_pred = np.array([[0.8, 0.2]])
    with pytest.raises(ValueError):
        binary_cross_entropy(y_true, y_pred)


def test_binary_cross_entropy_scalar_inputs():
    """Test BCE with scalar inputs converted to arrays."""
    y_true = np.array([1.0])
    y_pred = np.array([0.7])
    loss = binary_cross_entropy(y_true, y_pred)
    assert isinstance(loss, float)
