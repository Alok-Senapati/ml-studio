from sklearn.pipeline import Pipeline

from titanic_survival.pipelines.preprocessing import training_preprocessor
from titanic_survival.models.sklearn_models.logistic_regression import logistic_regression

training_pipeline = Pipeline(
    steps=[("preprocessing", training_preprocessor), ("classifier", logistic_regression)]
)
