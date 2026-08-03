"""
Unit tests for input matrix and vector validation routines.
"""

from __future__ import annotations

import numpy as np
import pytest
from ml_core.validation.input import (
    check_binary_labels,
    check_feature_matrix,
    check_matching_target_vectors,
    check_same_length,
    check_target_vector,
)


def test_check_feature_matrix_valid_2d() -> None:
    """Test that a valid 2D feature matrix passes validation silently."""
    X = np.array([[1, 2, 3], [4, 5, 6]])
    check_feature_matrix(X)


def test_check_feature_matrix_valid_2d_single_sample() -> None:
    """Test that a valid 2D feature matrix with a single sample (1, n_features) passes."""
    X = np.array([[1, 2, 3]])
    check_feature_matrix(X)


def test_check_feature_matrix_1d_array() -> None:
    """Test that a 1D array triggers ValueError for check_feature_matrix."""
    X = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="Expected a 2D feature matrix"):
        check_feature_matrix(X)


def test_check_feature_matrix_3d_array() -> None:
    """Test that a 3D array triggers ValueError for check_feature_matrix."""
    X = np.array([[[1, 2], [3, 4]]])
    with pytest.raises(ValueError, match="Expected a 2D feature matrix"):
        check_feature_matrix(X)


def test_check_feature_matrix_scalar() -> None:
    """Test that a 0D scalar array triggers ValueError for check_feature_matrix."""
    X = np.array(5)
    with pytest.raises(ValueError, match="Expected a 2D feature matrix"):
        check_feature_matrix(X)


def test_check_target_vector_valid_1d() -> None:
    """Test that a valid 1D integer target vector passes validation silently."""
    y = np.array([0, 1, 1, 0])
    check_target_vector(y)


def test_check_target_vector_float_1d() -> None:
    """Test that a valid 1D floating-point target vector passes validation silently."""
    y = np.array([0.0, 1.0, 0.5])
    check_target_vector(y)


def test_check_target_vector_2d_array() -> None:
    """Test that a 2D array triggers ValueError for check_target_vector."""
    y = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="Expected y to be a 1D array"):
        check_target_vector(y)


def test_check_target_vector_scalar() -> None:
    """Test that a 0D scalar triggers ValueError for check_target_vector."""
    y = np.array(5)
    with pytest.raises(ValueError, match="Expected y to be a 1D array"):
        check_target_vector(y)


def test_check_target_vector_3d_array() -> None:
    """Test that a 3D array triggers ValueError for check_target_vector."""
    y = np.array([[[1], [2]]])
    with pytest.raises(ValueError, match="Expected y to be a 1D array"):
        check_target_vector(y)


def test_check_same_length_matching() -> None:
    """Test that matching sample lengths in X and y pass validation silently."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 1])
    check_same_length(X, y)


def test_check_same_length_single_sample() -> None:
    """Test that matching single sample in X and y passes validation silently."""
    X = np.array([[1, 2, 3]])
    y = np.array([1])
    check_same_length(X, y)


def test_check_same_length_mismatched() -> None:
    """Test that mismatched sample counts in X and y trigger ValueError."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1, 1])
    with pytest.raises(ValueError, match="different number of samples"):
        check_same_length(X, y)


def test_check_same_length_more_features() -> None:
    """Test that feature count exceeding sample count passes if row counts match."""
    X = np.array([[1, 2, 3, 4]])
    y = np.array([1])
    check_same_length(X, y)


def test_check_binary_labels_valid_0_1() -> None:
    """Test that binary labels consisting of 0 and 1 pass validation silently."""
    y = np.array([0, 1, 1, 0, 1])
    check_binary_labels(y)


def test_check_binary_labels_single_class() -> None:
    """Test that single class subset {1} passes binary label validation."""
    y = np.array([1, 1, 1.0])
    check_binary_labels(y)


def test_check_binary_labels_only_zeros() -> None:
    """Test that single class subset {0} passes binary label validation."""
    y = np.array([0, 0, 0.0])
    check_binary_labels(y)


def test_check_binary_labels_invalid_negative() -> None:
    """Test that target containing negative values triggers ValueError."""
    y = np.array([-1, 0, 1])
    with pytest.raises(ValueError, match="Binary labels must contain only 0 and 1"):
        check_binary_labels(y)


def test_check_binary_labels_invalid_2() -> None:
    """Test that target containing value 2 triggers ValueError."""
    y = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="Binary labels must contain only 0 and 1"):
        check_binary_labels(y)


def test_check_binary_labels_multiclass() -> None:
    """Test that target containing multiple class values triggers ValueError."""
    y = np.array([0, 1, 2, 0, 1])
    with pytest.raises(ValueError, match="Binary labels must contain only 0 and 1"):
        check_binary_labels(y)


def test_check_binary_labels_float() -> None:
    """Test that floating point representations of 0.0 and 1.0 pass binary validation."""
    y = np.array([0.0, 1.0, 1.0, 0.0])
    check_binary_labels(y)


def test_check_matching_target_vectors_valid() -> None:
    """Test that matching 1D target vectors pass validation silently."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.8, 0.2])
    check_matching_target_vectors(y_true, y_pred)


def test_check_matching_target_vectors_single_sample() -> None:
    """Test that matching single sample 1D target vectors pass validation silently."""
    y_true = np.array([1])
    y_pred = np.array([0.7])
    check_matching_target_vectors(y_true, y_pred)


def test_check_matching_target_vectors_different_length() -> None:
    """Test that mismatched vector lengths trigger ValueError."""
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0.1, 0.9])
    with pytest.raises(ValueError, match="different number of samples"):
        check_matching_target_vectors(y_true, y_pred)


def test_check_matching_target_vectors_y_true_2d() -> None:
    """Test that 2D y_true vector triggers ValueError."""
    y_true = np.array([[0, 1]])
    y_pred = np.array([0.1, 0.9])
    with pytest.raises(ValueError, match="Expected y to be a 1D array"):
        check_matching_target_vectors(y_true, y_pred)


def test_check_matching_target_vectors_y_pred_2d() -> None:
    """Test that 2D y_pred vector triggers ValueError."""
    y_true = np.array([0, 1])
    y_pred = np.array([[0.1, 0.9]])
    with pytest.raises(ValueError, match="Expected y to be a 1D array"):
        check_matching_target_vectors(y_true, y_pred)


def test_check_matching_target_vectors_both_2d() -> None:
    """Test that both 2D y_true and y_pred trigger ValueError."""
    y_true = np.array([[0, 1]])
    y_pred = np.array([[0.1, 0.9]])
    with pytest.raises(ValueError, match="Expected y to be a 1D array"):
        check_matching_target_vectors(y_true, y_pred)
