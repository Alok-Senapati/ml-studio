import numpy as np

from sklearn.model_selection import train_test_split
from titanic_survival.pipelines.training import training_pipeline
from titanic_survival.config import settings


def test_training_pipeline_fit(sample_training_data, sample_training_labels):
    """Test that pipeline can fit and transform data."""
    # Should not raise error
    result = training_pipeline.fit(sample_training_data, sample_training_labels)
    assert result is not None


def test_training_pipeline_predict(sample_training_data, sample_training_labels):
    """Test that pipeline can make predictions."""
    X_train, X_test, y_train, y_test = train_test_split(
        sample_training_data,
        sample_training_labels,
        test_size=settings.training.test_size,
        random_state=settings.common.random_seed,
        stratify=sample_training_labels
    )

    training_pipeline.fit(X_train, y_train)
    predictions = training_pipeline.predict(X_test)

    # Predictions should be binary (0 or 1)
    assert all(pred in [0, 1] for pred in predictions)
    assert len(predictions) == len(X_test)


def test_training_pipeline_predict_proba(sample_training_data, sample_training_labels):
    """Test that pipeline can provide probability predictions."""
    X_train, X_test, y_train, y_test = train_test_split(
        sample_training_data,
        sample_training_labels,
        test_size=settings.training.test_size,
        random_state=settings.common.random_seed,
        stratify=sample_training_labels
    )

    training_pipeline.fit(X_train, y_train)
    probabilities = training_pipeline.predict_proba(X_test)

    # Probabilities should be between 0 and 1
    assert (probabilities >= 0).all() and (probabilities <= 1).all()
    # Sum of probabilities for each sample should be close to 1
    np.testing.assert_array_almost_equal(probabilities.sum(axis=1), 1.0)

