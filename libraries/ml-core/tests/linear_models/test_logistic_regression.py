import numpy as np
import pytest

from ml_core.linear_models.logistic_regression import LogisticRegression
from ml_core.exceptions.not_fitted import NotFittedError


def test_predict_proba_raises_not_fitted():
    model = LogisticRegression()
    with pytest.raises(NotFittedError):
        model.predict_proba(np.array([[0.0, 0.0]]))


def test_logistic_regression_learns_simple_and():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 0, 1], dtype=float)

    model = LogisticRegression(learning_rate=0.1, epochs=2000)
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (4,)
    assert (preds == y).all()

    probs = model.predict_proba(X)
    assert probs.shape == (4,)
    assert (probs >= 0).all() and (probs <= 1).all()

    # positive sample should have higher probability than a negative sample
    assert probs[3] > probs[0]
