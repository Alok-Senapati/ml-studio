"""
Configured scikit-learn Logistic Regression model instance for titanic_survival.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from titanic_survival.config import settings

# Extract logistic regression settings configuration sub-object
config = settings.logistic_regression

#: Pre-configured scikit-learn LogisticRegression model instance
logistic_regression = LogisticRegression(
    max_iter=config.max_iter,
    solver=config.solver,
    C=config.C,
    l1_ratio=config.l1_ratio,
    random_state=settings.common.random_seed,
)
