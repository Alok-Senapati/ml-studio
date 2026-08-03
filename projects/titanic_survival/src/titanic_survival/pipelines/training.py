"""
End-to-end training pipeline combining preprocessing and classifier estimator.
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from titanic_survival.models.sklearn_models.logistic_regression import logistic_regression
from titanic_survival.pipelines.preprocessing import training_preprocessor

#: Complete end-to-end ML pipeline combining feature engineering, preprocessing, and classification
training_pipeline = Pipeline(
    steps=[("preprocessing", training_preprocessor), ("classifier", logistic_regression)]
)
