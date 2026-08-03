"""
Binary Logistic Regression classifier implementation using Gradient Descent.
"""

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

    Logistic Regression models the probability that a given input sample belongs to the
    positive class (1) using the logistic sigmoid activation function applied to a linear
    combination of features.

    The model learns decision boundaries by minimizing Binary Cross Entropy (BCE) loss
    via Gradient Descent optimization.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Optimization step size :math:`\\alpha > 0` used by Gradient Descent.
    epochs : int, default=1000
        Total number of optimization iterations performed during training.

    Attributes
    ----------
    weights : FloatArray or None
        Learned feature weight vector of shape (n_features,) after fitting.
        Initializes to ``None`` prior to training.
    bias : float or None
        Learned scalar bias/intercept parameter after fitting.
        Initializes to ``None`` prior to training.
    is_fitted : bool
        Boolean flag indicating whether the model has completed fitting.

    Raises
    ------
    ValueError
        If `learning_rate` <= 0 or `epochs` <= 0 during initialization.

    Notes
    -----
    The hypothesis function for Logistic Regression is defined as:

    .. math::

        z = X w + b

    .. math::

        \\hat{y} = \\sigma(z) = \\frac{1}{1 + e^{-z}}

    where :math:`X` is the feature matrix, :math:`w` is the weight vector,
    :math:`b` is the scalar bias, and :math:`\\sigma` is the logistic sigmoid function.

    The model updates parameters by minimizing Binary Cross Entropy loss:

    .. math::

        \\mathcal{L}(w, b) = -\\frac{1}{N} \\sum_{i=1}^{N}
        \\left[ y_i \\log(\\hat{y}_i) + (1 - y_i) \\log(1 - \\hat{y}_i) \\right]

    The analytical gradients with respect to parameters are given by:

    .. math::

        \\frac{\\partial \\mathcal{L}}{\\partial w} = \\frac{1}{N} X^T (\\hat{y} - y)

    .. math::

        \\frac{\\partial \\mathcal{L}}{\\partial b} = \\frac{1}{N}
        \\sum_{i=1}^{N} (\\hat{y}_i - y_i)

    Examples
    --------
    >>> import numpy as np
    >>> from ml_core.linear_models import LogisticRegression
    >>> X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    >>> y = np.array([0, 0, 0, 1])
    >>> clf = LogisticRegression(learning_rate=0.1, epochs=2000)
    >>> _ = clf.fit(X, y)
    >>> clf.predict(X)
    array([0, 0, 0, 1])
    """

    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000) -> None:
        """
        Initialize a Logistic Regression classifier instance.

        Parameters
        ----------
        learning_rate : float, default=0.01
            Step size used by the Gradient Descent optimizer. Must be strictly positive (> 0).
        epochs : int, default=1000
            Number of optimization passes over the training set. Must be strictly positive (> 0).

        Raises
        ------
        ValueError
            If `learning_rate` <= 0 or `epochs` <= 0.
        """
        # Validate learning rate hyperparameter
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0.")

        # Validate epochs hyperparameter
        if epochs <= 0:
            raise ValueError("epochs must be greater than 0.")

        self.learning_rate = learning_rate
        self.epochs = epochs

        # Model parameters initialized to None before training
        self.weights: FloatArray | None = None
        self.bias: float | None = None

        # Tracking attribute for fit status check
        self.is_fitted = False

    @override
    def fit(self, X: NumericArray, y: NumericArray) -> Self:
        """
        Fit the Logistic Regression model using Gradient Descent.

        Validates input arrays, initializes model parameters, and iteratively updates
        weights and bias parameters over the specified number of epochs.

        Parameters
        ----------
        X : NumericArray
            Training feature matrix of shape (n_samples, n_features).
        y : NumericArray
            Binary target label vector of shape (n_samples,). Must contain only 0 and 1.

        Returns
        -------
        Self
            The fitted classifier instance.

        Raises
        ------
        ValueError
            If `X` or `y` fail input shape validation, contain mismatched sample counts,
            or if `y` contains non-binary labels.
        """
        # Ensure input arrays are represented as 64-bit floating point NumPy arrays
        X_float = np.asarray(X, dtype=np.float64)
        y_float = np.asarray(y, dtype=np.float64)

        # Validate feature matrix shape (2D), target vector shape (1D),
        # sample count alignment, and binary labels
        check_feature_matrix(X_float)
        check_target_vector(y_float)
        check_same_length(X_float, y_float)
        check_binary_labels(y_float)

        # Extract total number of input features
        _, n_features = X_float.shape

        # Initialize weights to zero vector and bias to scalar zero
        self._initialize_parameters(n_features)

        # Gradient descent optimization loop
        for _ in range(self.epochs):
            # Compute predicted probabilities via forward propagation
            y_pred = self._forward(X_float)

            # Compute analytical loss gradients w.r.t weights and bias
            dw, db = self._compute_gradients(X=X_float, y=y_float, y_pred=y_pred)

            # Apply gradient update to model parameters
            self._update_parameters(dw=dw, db=db)

        # Mark model as fitted after completing training loop
        self.is_fitted = True

        return self

    @override
    def predict(self, X: NumericArray, threshold: float = 0.5) -> IntArray:
        """
        Predict binary class labels for the given input features.

        Converts predicted class probabilities into discrete binary class predictions (0 or 1)
        using the specified decision threshold.

        Parameters
        ----------
        X : NumericArray
            Input feature matrix of shape (n_samples, n_features).
        threshold : float, default=0.5
            Decision boundary probability threshold in range [0.0, 1.0].
            Probabilities >= `threshold` are mapped to class 1, otherwise class 0.

        Returns
        -------
        IntArray
            Predicted binary class labels (0 or 1) of shape (n_samples,).

        Raises
        ------
        ValueError
            If `threshold` is outside the range [0.0, 1.0].
        NotFittedError
            If called before fitting the model.
        """
        # Validate decision threshold boundary
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 to 1.0.")

        # Obtain continuous probability predictions for positive class
        probabilities = self.predict_proba(X)

        # Apply threshold rule to output 64-bit integer class labels
        return (probabilities >= threshold).astype(np.int64)

    @override
    def predict_proba(self, X: NumericArray) -> FloatArray:
        """
        Predict positive class probabilities for input samples.

        Parameters
        ----------
        X : NumericArray
            Input feature matrix of shape (n_samples, n_features).

        Returns
        -------
        FloatArray
            Predicted positive class probabilities in range [0.0, 1.0] of shape (n_samples,).

        Raises
        ------
        NotFittedError
            If called before fitting the model.
        ValueError
            If `X` is not a 2D feature matrix.
        """
        # Assert that the model instance has been fitted
        if not self.is_fitted:
            raise NotFittedError("The LogisticRegression instance is not fitted yet.")

        # Cast feature input to float64 precision
        X_float = np.asarray(X, dtype=np.float64)

        # Validate feature matrix 2D shape requirement
        check_feature_matrix(X_float)

        # Compute probabilities via forward propagation z = X w + b; y_hat = sigmoid(z)
        probabilities = self._forward(X_float)

        return probabilities

    def _initialize_parameters(self, n_features: int) -> None:
        """
        Initialize weights to zeros and scalar bias to zero.

        Parameters
        ----------
        n_features : int
            Number of features in the input feature matrix.

        Notes
        -----
        For Logistic Regression, initializing weights and bias to zeros is a standard,
        convex-optimization compatible initialization strategy.
        """
        # Initialize weight vector to zeros for each feature
        self.weights = np.zeros(n_features, dtype=np.float64)
        # Initialize scalar intercept/bias to zero
        self.bias = 0.0

    def _forward(self, X: NumericArray) -> FloatArray:
        """
        Perform forward pass computation to calculate positive class probabilities.

        Parameters
        ----------
        X : NumericArray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        FloatArray
            Predicted probability vector of shape (n_samples,).

        Raises
        ------
        NotFittedError
            If model parameters (`weights` or `bias`) have not been initialized.

        Notes
        -----
        Forward propagation computes linear combinations followed by sigmoid activation:

        .. math::

            z = X w + b

        .. math::

            \\hat{y} = \\sigma(z)
        """
        # Check parameter initialization state
        if self.weights is None or self.bias is None:
            raise NotFittedError("Model parameters have not been initialized.")

        # Compute linear logit values: z = X @ w + b
        z = X @ self.weights + self.bias

        # Apply sigmoid activation function to map logits to probabilities in (0, 1)
        return sigmoid(z)

    def _compute_gradients(
        self, X: NumericArray, y: NumericArray, y_pred: FloatArray
    ) -> tuple[FloatArray, float]:
        """
        Compute gradients of Binary Cross Entropy loss with respect to parameters.

        Parameters
        ----------
        X : NumericArray
            Feature matrix of shape (n_samples, n_features).
        y : NumericArray
            Ground truth binary target vector of shape (n_samples,).
        y_pred : FloatArray
            Predicted probabilities vector of shape (n_samples,).

        Returns
        -------
        tuple[FloatArray, float]
            Tuple containing:
            - dw : FloatArray
                Gradient of loss with respect to weights vector of shape (n_features,).
            - db : float
                Gradient of loss with respect to scalar bias parameter.

        Notes
        -----
        The analytical gradients are computed as:

        .. math::

            \\text{error} = \\hat{y} - y

        .. math::

            \\text{dw} = \\frac{1}{m} X^T (\\hat{y} - y)

        .. math::

            \\text{db} = \\frac{1}{m} \\sum_{i=1}^{m} (\\hat{y}_i - y_i)
        """
        # Total number of training samples m
        n_samples = X.shape[0]

        # Compute error vector: (y_pred - y)
        error = y_pred - y

        # Compute weight gradient vector: (1 / m) * X^T @ error
        dw = (X.T @ error) / n_samples

        # Compute scalar bias gradient: mean(error)
        db = float(np.mean(error))

        return dw, db

    def _update_parameters(self, dw: FloatArray, db: float) -> None:
        """
        Update weights and bias using the Gradient Descent optimizer step.

        Parameters
        ----------
        dw : FloatArray
            Gradient of loss with respect to weights of shape (n_features,).
        db : float
            Gradient of loss with respect to bias.

        Raises
        ------
        NotFittedError
            If model parameters (`weights` or `bias`) have not been initialized.
        """
        # Ensure parameters are initialized before update
        if self.weights is None or self.bias is None:
            raise NotFittedError("Model parameters have not been initialized.")

        # Delegate parameter update step to gradient_descent optimizer function
        self.weights, self.bias = gradient_descent(
            weights=self.weights, bias=self.bias, dw=dw, db=db, learning_rate=self.learning_rate
        )
