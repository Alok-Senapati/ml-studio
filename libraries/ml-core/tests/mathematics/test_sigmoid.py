import numpy as np
from ml_core.mathematics.sigmoid import sigmoid


def test_sigmoid_zero():
    """Test sigmoid(0) = 0.5."""
    result = sigmoid(np.array([0.0]))
    assert np.isclose(result[0], 0.5)


def test_sigmoid_positive():
    """Test sigmoid with positive values."""
    result = sigmoid(np.array([1.0]))
    expected = 1.0 / (1.0 + np.exp(-1.0))
    assert np.isclose(result[0], expected)


def test_sigmoid_negative():
    """Test sigmoid with negative values."""
    result = sigmoid(np.array([-1.0]))
    expected = 1.0 / (1.0 + np.exp(1.0))
    assert np.isclose(result[0], expected)


def test_sigmoid_multiple_values():
    """Test sigmoid with multiple values."""
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = sigmoid(x)
    expected = 1.0 / (1.0 + np.exp(-x))
    np.testing.assert_array_almost_equal(result, expected)


def test_sigmoid_2d_array():
    """Test sigmoid with 2D array."""
    x = np.array([[0.0, 1.0], [-1.0, 2.0]])
    result = sigmoid(x)
    expected = 1.0 / (1.0 + np.exp(-x))
    np.testing.assert_array_almost_equal(result, expected)


def test_sigmoid_single_element():
    """Test sigmoid with single element."""
    result = sigmoid(np.array([0.5]))
    expected = 1.0 / (1.0 + np.exp(-0.5))
    assert np.isclose(result[0], expected)


def test_sigmoid_output_range():
    """Test sigmoid output is always between 0 and 1."""
    x = np.linspace(-100, 100, 1000)
    result = sigmoid(x)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_sigmoid_symmetry():
    """Test sigmoid symmetry: sigmoid(x) + sigmoid(-x) = 1."""
    x = np.array([1.0, 2.0, 3.0])
    sig_x = sigmoid(x)
    sig_neg_x = sigmoid(-x)
    np.testing.assert_array_almost_equal(sig_x + sig_neg_x, 1.0)


def test_sigmoid_monotonicity():
    """Test sigmoid is monotonically increasing."""
    x = np.linspace(-5, 5, 100)
    result = sigmoid(x)
    differences = np.diff(result)
    assert np.all(differences >= 0)


def test_sigmoid_large_positive():
    """Test sigmoid with large positive values approaches 1."""
    result = sigmoid(np.array([100.0]))
    assert np.isclose(result[0], 1.0)


def test_sigmoid_large_negative():
    """Test sigmoid with large negative values approaches 0."""
    result = sigmoid(np.array([-100.0]))
    assert np.isclose(result[0], 0.0, atol=1e-10)


def test_sigmoid_list_input():
    """Test sigmoid accepts list input."""
    result = sigmoid([0.0, 1.0, -1.0])
    assert len(result) == 3
    assert np.isclose(result[0], 0.5)
