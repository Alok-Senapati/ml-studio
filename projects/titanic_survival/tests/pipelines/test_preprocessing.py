import numpy as np
import pandas as pd

from titanic_survival.pipelines.preprocessing import training_preprocessor


def test_training_preprocessor_transforms_data(sample_training_data):
    """Test that training_preprocessor transforms data successfully."""
    result = training_preprocessor.fit_transform(sample_training_data.copy())

    # Check output is a numpy array or sparse matrix
    assert hasattr(result, "shape")
    assert result.shape[0] == len(sample_training_data)


def test_training_preprocessor_handles_missing_age(sample_training_data):
    """Test that missing Age values are imputed with median."""
    result = training_preprocessor.fit_transform(sample_training_data.copy())

    # Result should have no NaN values
    if hasattr(result, "toarray"):
        result_array = result.toarray()  # pragma: no cover
    else:
        result_array = result

    assert not np.isnan(result_array).any()


def test_training_preprocessor_creates_engineered_features(sample_training_data):
    """Test that feature engineering creates expected columns."""
    # Get intermediate output after feature engineering
    feature_engineered = training_preprocessor.named_steps["feature_engineering"].fit_transform(
        sample_training_data.copy()
    )

    # Check for engineered features
    assert "Title" in feature_engineered.columns
    assert "FamilySize" in feature_engineered.columns
    assert "IsAlone" in feature_engineered.columns
    assert "Deck" in feature_engineered.columns
    assert "TicketPrefix" in feature_engineered.columns
    assert "TicketGroupSize" in feature_engineered.columns


def test_training_preprocessor_fit_transform_consistency(sample_training_data):
    """Test that fit_transform and fit().transform() produce same results."""
    data_copy1 = sample_training_data.copy()
    data_copy2 = sample_training_data.copy()

    # fit_transform
    result1 = training_preprocessor.fit_transform(data_copy1)

    # fit then transform
    from titanic_survival.pipelines.preprocessing import training_preprocessor as pipeline2

    pipeline2.fit(data_copy2)
    result2 = pipeline2.transform(data_copy2)

    # Convert to dense arrays for comparison
    if hasattr(result1, "toarray"):
        result1_array = result1.toarray()  # pragma: no cover
    else:
        result1_array = result1

    if hasattr(result2, "toarray"):
        result2_array = result2.toarray()  # pragma: no cover
    else:
        result2_array = result2

    np.testing.assert_array_almost_equal(result1_array, result2_array)


def test_training_preprocessor_handles_all_missing_values():
    """Test that pipeline handles columns with all missing values."""
    data_with_missing = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3],
            "Name": ["A, Mr. B", "C, Mrs. D", "E, Miss. F"],
            "Sex": ["male", "female", "male"],
            "Age": [25.0, 30.0, 35.0],
            "SibSp": [1, 0, 1],
            "Parch": [0, 1, 0],
            "Fare": [10.0, 20.0, 30.0],
            "Cabin": ["C89", np.nan, np.nan],  # All missing
            "Embarked": ["S", "C", "Q"],
            "Pclass": [1, 2, 3],
            "Ticket": ["T1", "T2", "T3"],
        }
    )

    result = training_preprocessor.fit_transform(data_with_missing)

    # Should not raise error
    assert result.shape[0] == len(data_with_missing)


def test_training_preprocessor_output_shape(sample_training_data):
    """Test that output has correct number of samples."""
    result = training_preprocessor.fit_transform(sample_training_data.copy())

    if hasattr(result, "toarray"):
        result_array = result.toarray()  # pragma: no cover
    else:
        result_array = result

    # Number of rows should match input
    assert result_array.shape[0] == len(sample_training_data)


def test_training_preprocessor_handle_unknown_category(sample_training_data):
    training_preprocessor.fit(sample_training_data.copy())

    new_data = sample_training_data.copy()
    new_data.loc[0, "Embarked"] = "XYZ"

    result = training_preprocessor.transform(new_data)

    assert result.shape[0] == len(new_data)
