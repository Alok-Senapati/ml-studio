"""
Column transformation and data preprocessing pipelines for numeric and categorical features.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from titanic_survival.pipelines.engineering import feature_engineering_pipeline

#: Preprocessing sub-pipeline for numeric features (median imputation)
numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

#: Preprocessing sub-pipeline for categorical features (most frequent imputation + one-hot encoding)
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ]
)

#: List of numerical feature column names
numeric_features = ["Pclass", "Age", "Fare", "SibSp", "Parch", "FamilySize", "TicketGroupSize"]

#: List of categorical feature column names
categorical_features = ["Sex", "Embarked", "Title", "Deck", "TicketPrefix", "IsAlone"]

#: ColumnTransformer applying numeric and categorical pipelines to respective features
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ],
    remainder="drop",
)

#: Combined preprocessing pipeline chaining feature engineering and column transformation
training_preprocessor = Pipeline(
    steps=[("feature_engineering", feature_engineering_pipeline), ("preprocessing", preprocessor)]
)
