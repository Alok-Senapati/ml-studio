from __future__ import annotations

from typing import Self, override

import numpy as np

from ml_core.base import Classifier
from ml_core.exceptions.not_fitted import NotFittedError
from ml_core.mathematics.sigmoid import sigmoid
from ml_core.optimizers.gradient_descent import gradient_descent
from ml_core.typing import FloatArray, IntArray, NumericArray
from ml_core.validation.input import (
    check_binary_labels,
    check_feature_matrix,
    check_same_length,
    check_target_vector,
)


class LogisticRegression(Classifier):
    """
    Binary Logistic Regression classifier.

    Logistic Regression models the probability that an input belongs to the
    positive class using the sigmoid activation function.

    The model learns a linear decision boundary by minimizing the Binary
    Cross Entropy (BCE) loss using Gradient Descent.

    Notes
    -----
    This implementation is educational and intentionally built from scratch
    using NumPy to understand the internal workings of Logistic Regression.
    """

    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000) -> None:
        """
        Initializes a Logistic Regression Model.

        Parameters
        ----------
        learning_rate: float, default=0.01
            Learning Rate (Step size used by Gradient Descent).
        epochs: int, default=1000
            Number of Optimization Iterations
        """
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0.")

        if epochs <= 0:
            raise ValueError("epochs must be greater than 0.")

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights: FloatArray | None = None
        self.bias: float | None = None

        self.is_fitted = False

    @override
    def fit(self, X: NumericArray, y: NumericArray) -> Self:
        """
        Train the Logistic Regression model.

        Parameters
        ----------
        X: NumericArray
            Training feature matrix.
        y: NumericArray
            Binary target labels.

        Returns
        -------
        Self
            The fitted Estimator
        """
        X_float = np.asarray(X, dtype=np.float64)
        y_float = np.asarray(y, dtype=np.float64)

        check_feature_matrix(X_float)
        check_target_vector(y_float)
        check_same_length(X_float, y_float)
        check_binary_labels(y_float)

        _, n_features = X_float.shape

        self._initialize_parameters(n_features)

        for _ in range(self.epochs):
            y_pred = self._forward(X_float)

            dw, db = self._compute_gradients(X=X_float, y=y_float, y_pred=y_pred)

            self._update_parameters(dw=dw, db=db)

        self.is_fitted = True

        return self

    @override
    def predict(self, X: NumericArray, threshold: float = 0.5) -> IntArray:
        """
        Predict binary class labels.

        Parameters
        ----------
        X: NumericArray
            Feature matrix.
        threshold: float, default=0.5
            Decision threshold use to convert predicted probabilities
            into binary class labels, Must be in range [0.0, 1.0]

        Returns
        -------
        IntArray
            Predicted class labels.
        """

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 to 1.0.")

        probabilities = self.predict_proba(X)

        return (probabilities >= threshold).astype(np.int64)

    @override
    def predict_proba(self, X: NumericArray) -> FloatArray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X: NumericArray
            Input features matrix.

        Returns
        -------
        FloatArray
            Probability of the positive class for each sample.
        """

        if not self.is_fitted:
            raise NotFittedError("The LogisticRegression instance is not fitted yet.")

        X_float = np.asarray(X, dtype=np.float64)

        check_feature_matrix(X_float)

        probabilities = self._forward(X_float)

        return probabilities

    def _initialize_parameters(self, n_features: int) -> None:
        """
        Initialize the model parameters.

        Parameters
        ----------
        n_features: int
            Number of input features in the training data.

        Notes
        -----
        Logistic Regression starts with all weights and bias initialized
        to zero. Gradient Descent iteratively updates these parameters
        during training.
        """

        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0

    def _forward(self, X: NumericArray) -> FloatArray:
        """
        Predicted probability of the positive class for each sample.

        Parameters
        ----------
        X: NumericArray
           Feature Matrix with shape (n_samples, n_features).

        Returns
        -------
        FloatArray
            Predictive probabilities for the predictive class

        Notes
        -------
        Forward propagation computes the linear combination of the
        input features and model parameters, followed by the sigmoid
        activation function.

            z = XW + b
            ŷ = sigmoid(z)
        """

        if self.weights is None or self.bias is None:
            raise NotFittedError("Model parameters have not been initialized.")

        z = X @ self.weights + self.bias

        return sigmoid(z)

    def _compute_gradients(
        self, X: NumericArray, y: NumericArray, y_pred: FloatArray
    ) -> tuple[FloatArray, float]:
        """
        Compute the gradients of Binary Cross Entropy loss.

        Parameters
        ----------
        X: NumericArray
            Training feature matrix.
        y: NumericArray
            Ground truth labels.
        y_pred: FloatArray
            Predicted probabilities.

        Returns
        -------
        tuple[FloatArray, float]
            Weights gradient and bias gradient.

        Notes
        -----
        The gradients are computed using the analytical derivatives of
        Binary Cross Entropy combined with the sigmoid activation.

            dw = (1 / m) * Xᵀ (ŷ - y)

            db = (1 / m) * Σ(ŷ - y)
        """

        n_samples = X.shape[0]

        error = y_pred - y

        dw = (X.T @ error) / n_samples
        db = float(np.mean(error))

        return dw, db

    def _update_parameters(self, dw: FloatArray, db: float) -> None:
        """
        Updates the model parameters using Gradient Descent.

        Parameters
        ----------
        dw: FloatArray
            Gradient of the loss with respect to the weights.
        db: float
            Gradient of the loss with respect to the bias.
        """

        if self.weights is None or self.bias is None:
            raise NotFittedError("Model parameters have not been initialized.")

        self.weights, self.bias = gradient_descent(
            weights=self.weights, bias=self.bias, dw=dw, db=db, learning_rate=self.learning_rate
        )
