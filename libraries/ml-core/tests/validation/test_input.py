import numpy as np  # noqa: I001
import pytest

from ml_core.validation.input import (
    check_feature_matrix,
    check_target_vector,
    check_same_length,
    check_binary_labels,
    check_matching_target_vectors,
)


def test_check_feature_matrix_valid_2d():
    """Test valid 2D feature matrix passes."""
    X = np.array([[1, 2, 3], [4, 5, 6]])
    # Should not raise
    check_feature_matrix(X)


def test_check_feature_matrix_valid_2d_single_sample():
    """Test valid 2D matrix with single sample."""
    X = np.array([[1, 2, 3]])
    check_feature_matrix(X)


def test_check_feature_matrix_1d_array():
    """Test 1D array raises error."""
    X = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="Expected a 2D feature matrix"):
        check_feature_matrix(X)


def test_check_feature_matrix_3d_array():
    """Test 3D array raises error."""
    X = np.array([[[1, 2], [3, 4]]])
    with pytest.raises(ValueError, match="Expected a 2D feature matrix"):
        check_feature_matrix(X)


def test_check_feature_matrix_scalar():
    """Test scalar raises error."""
    X = np.array(5)
    with pytest.raises(ValueError, match="Expected a 2D feature matrix"):
        check_feature_matrix(X)


def test_check_target_vector_valid_1d():
    """Test valid 1D target vector passes."""
    y = np.array([0, 1, 1, 0])
    check_target_vector(y)


def test_check_target_vector_float_1d():
    """Test valid 1D float vector passes."""
    y = np.array([0.0, 1.0, 0.5])
    check_target_vector(y)


def test_check_target_vector_2d_array():
    """Test 2D array raises error."""
    y = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="Expected y to be a 1D array"):
        check_target_vector(y)


def test_check_target_vector_scalar():
    """Test scalar raises error."""
    y = np.array(5)
    with pytest.raises(ValueError, match="Expected y to be a 1D array"):
        check_target_vector(y)


def test_check_target_vector_3d_array():
    """Test 3D array raises error."""
    y = np.array([[[1], [2]]])
    with pytest.raises(ValueError, match="Expected y to be a 1D array"):
        check_target_vector(y)


def test_check_same_length_matching():
    """Test matching lengths pass."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 1])
    check_same_length(X, y)


def test_check_same_length_single_sample():
    """Test matching single sample passes."""
    X = np.array([[1, 2, 3]])
    y = np.array([1])
    check_same_length(X, y)


def test_check_same_length_mismatched():
    """Test mismatched lengths raise error."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1, 1])
    with pytest.raises(ValueError, match="different number of samples"):
        check_same_length(X, y)


def test_check_same_length_more_features():
    """Test more features than samples passes."""
    X = np.array([[1, 2, 3, 4]])
    y = np.array([1])
    check_same_length(X, y)


def test_check_binary_labels_valid_0_1():
    """Test valid binary labels with 0 and 1."""
    y = np.array([0, 1, 1, 0, 1])
    check_binary_labels(y)


def test_check_binary_labels_single_class():
    """Test binary labels with single class passes."""
    y = np.array([1, 1, 1])
    check_binary_labels(y)


def test_check_binary_labels_only_zeros():
    """Test binary labels with only zeros."""
    y = np.array([0, 0, 0])
    check_binary_labels(y)


def test_check_binary_labels_invalid_negative():
    """Test invalid labels with negative values."""
    y = np.array([-1, 0, 1])
    with pytest.raises(ValueError, match="Binary labels must contain only 0 and 1"):
        check_binary_labels(y)


def test_check_binary_labels_invalid_2():
    """Test invalid labels with value 2."""
    y = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="Binary labels must contain only 0 and 1"):
        check_binary_labels(y)


def test_check_binary_labels_multiclass():
    """Test multiclass labels raise error."""
    y = np.array([0, 1, 2, 0, 1])
    with pytest.raises(ValueError, match="Binary labels must contain only 0 and 1"):
        check_binary_labels(y)


def test_check_binary_labels_float():
    """Test float binary labels."""
    y = np.array([0.0, 1.0, 1.0, 0.0])
    check_binary_labels(y)


def test_check_matching_target_vectors_valid():
    """Test matching valid vectors pass."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.8, 0.2])
    check_matching_target_vectors(y_true, y_pred)


def test_check_matching_target_vectors_single_sample():
    """Test matching single sample passes."""
    y_true = np.array([1])
    y_pred = np.array([0.7])
    check_matching_target_vectors(y_true, y_pred)


def test_check_matching_target_vectors_different_length():
    """Test different lengths raise error."""
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0.1, 0.9])
    with pytest.raises(ValueError, match="different number of samples"):
        check_matching_target_vectors(y_true, y_pred)


def test_check_matching_target_vectors_y_true_2d():
    """Test 2D y_true raises error."""
    y_true = np.array([[0, 1]])
    y_pred = np.array([0.1, 0.9])
    with pytest.raises(ValueError, match="Expected y to be a 1D array"):
        check_matching_target_vectors(y_true, y_pred)


def test_check_matching_target_vectors_y_pred_2d():
    """Test 2D y_pred raises error."""
    y_true = np.array([0, 1])
    y_pred = np.array([[0.1, 0.9]])
    with pytest.raises(ValueError, match="Expected y to be a 1D array"):
        check_matching_target_vectors(y_true, y_pred)


def test_check_matching_target_vectors_both_2d():
    """Test both 2D raises error."""
    y_true = np.array([[0, 1]])
    y_pred = np.array([[0.1, 0.9]])
    with pytest.raises(ValueError, match="Expected y to be a 1D array"):
        check_matching_target_vectors(y_true, y_pred)
