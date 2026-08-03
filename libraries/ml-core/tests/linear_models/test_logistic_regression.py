"""
Unit tests for the LogisticRegression classifier implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
from ml_core.exceptions.not_fitted import NotFittedError
from ml_core.linear_models.logistic_regression import LogisticRegression
from ml_core.typing import NumericArray


@pytest.fixture
def sample_dataset() -> tuple[NumericArray, NumericArray]:
    """
    Create a deterministic binary classification dataset for testing.

    Returns
    -------
    tuple[NumericArray, NumericArray]
        Tuple containing sample feature matrix `X` and binary target vector `y`.
    """
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )

    y = np.array([0, 0, 0, 1], dtype=np.int64)

    return X, y


def test_constructor_initializes_hyperparameters() -> None:
    """Verify that constructor arguments are correctly stored."""
    model = LogisticRegression(learning_rate=0.05, epochs=500)

    assert model.learning_rate == 0.05
    assert model.epochs == 500
    assert model.weights is None
    assert model.bias is None


def test_fit_returns_self(sample_dataset: tuple[NumericArray, NumericArray]) -> None:
    """Verify that fit() returns the estimator instance itself."""
    X, y = sample_dataset
    model = LogisticRegression()
    fitted_model = model.fit(X, y)

    assert fitted_model is model


def test_fit_marks_model_as_fitted(sample_dataset: tuple[NumericArray, NumericArray]) -> None:
    """Verify that model is marked as fitted and parameters are non-null after training."""
    X, y = sample_dataset

    model = LogisticRegression()
    model.fit(X, y)

    assert model.is_fitted is True
    assert model.weights is not None
    assert model.bias is not None


def test_predict_before_fit_raises_not_fitted_error() -> None:
    """Verify that predict() before training raises NotFittedError."""
    model = LogisticRegression()
    X = np.array([[1.0, 2.0]])

    with pytest.raises(NotFittedError):
        model.predict(X)


def test_predict_proba_before_fit_raises_not_fitted_error() -> None:
    """Verify that predict_proba() before training raises NotFittedError."""
    model = LogisticRegression()
    with pytest.raises(NotFittedError):
        model.predict_proba(np.array([[0.0, 0.0]]))


def test_predict_proba_returns_valid_probabilities(
    sample_dataset: tuple[NumericArray, NumericArray],
) -> None:
    """Verify that predicted probabilities lie within the valid range [0, 1]."""
    X, y = sample_dataset

    model = LogisticRegression()
    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == y.shape
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_predict_returns_binary_labels(
    sample_dataset: tuple[NumericArray, NumericArray],
) -> None:
    """Verify that predict() returns only binary labels (0 or 1)."""
    X, y = sample_dataset
    model = LogisticRegression()
    model.fit(X, y)
    predicted_labels = model.predict(X)

    assert set(np.unique(predicted_labels)).issubset({1, 0})


def test_logistic_regression_learns_simple_and() -> None:
    """Verify that Logistic Regression successfully learns the logical AND function."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 0, 1], dtype=float)

    model = LogisticRegression(learning_rate=0.1, epochs=2000)
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (4,)
    assert (preds == y).all()

    probs = model.predict_proba(X)
    assert probs.shape == (4,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Positive sample should have higher probability than a negative sample
    assert probs[3] > probs[0]


def test_init_invalid_params() -> None:
    """Verify constructor raises ValueError when given non-positive hyperparameters."""
    with pytest.raises(ValueError):
        LogisticRegression(learning_rate=0.0)
    with pytest.raises(ValueError):
        LogisticRegression(learning_rate=-0.1)
    with pytest.raises(ValueError):
        LogisticRegression(epochs=0)
    with pytest.raises(ValueError):
        LogisticRegression(epochs=-10)


def test_predict_threshold_validation() -> None:
    """Verify threshold boundary validation in predict()."""
    model = LogisticRegression()
    with pytest.raises(ValueError):
        model.predict(np.array([[0.0, 0.0]]), threshold=-0.1)
    with pytest.raises(ValueError):
        model.predict(np.array([[0.0, 0.0]]), threshold=1.1)


def test_forward_and_update_parameters_not_initialized_raise() -> None:
    """Verify internal forward/update methods before initialization raise NotFittedError."""
    model = LogisticRegression()
    X = np.array([[0.0, 1.0]])
    with pytest.raises(NotFittedError):
        model._forward(X)

    dw = np.zeros(1, dtype=float)
    with pytest.raises(NotFittedError):
        model._update_parameters(dw=dw, db=0.0)


def test_predict_respects_threshold() -> None:
    """Verify that decision threshold parameter alters binary predictions."""
    model = LogisticRegression()
    model.weights = np.array([1.0, -1.0], dtype=float)
    model.bias = 0.0
    model.is_fitted = True

    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    preds_05 = model.predict(X, threshold=0.5)
    preds_09 = model.predict(X, threshold=0.9)

    assert preds_05.shape == preds_09.shape
    assert np.any(preds_05 != preds_09)


def test_invalid_thresholds_raise() -> None:
    """Verify that invalid threshold values trigger ValueError even before fit check."""
    model = LogisticRegression()
    with pytest.raises(ValueError):
        model.predict(np.array([[0.0, 0.0]]), threshold=-0.1)
    with pytest.raises(ValueError):
        model.predict(np.array([[0.0, 0.0]]), threshold=1.1)


def test_validation_failures_on_fit() -> None:
    """Verify input shape and length validation during model fitting."""
    model = LogisticRegression()

    X = np.array([[1, 2], [3, 4]], dtype=float)
    y = np.array([0, 1], dtype=float)

    # 1D feature matrix should fail
    X_bad = X.reshape(-1)
    with pytest.raises(ValueError):
        model.fit(X_bad, y)

    # 2D target vector should fail
    y_bad = y.reshape(-1, 1)
    with pytest.raises(ValueError):
        model.fit(X, y_bad)

    # Mismatched sample count should fail
    y_short = np.array([0], dtype=float)
    with pytest.raises(ValueError):
        model.fit(X, y_short)


def test_learning_linearly_separable_dataset() -> None:
    """Verify convergence and correct classification on a linearly separable dataset."""
    X = np.array(
        [
            [1, 1],
            [2, 2],
            [3, 3],
            [8, 8],
            [9, 9],
            [10, 10],
        ],
        dtype=np.float64,
    )

    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)

    model = LogisticRegression(
        learning_rate=0.1,
        epochs=5000,
    )

    model.fit(X, y)

    predictions = model.predict(X)

    assert np.array_equal(predictions, y.astype(np.int64))
