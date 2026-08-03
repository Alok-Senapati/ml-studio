"""
Unit tests for data preprocessing and column transformation pipelines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from titanic_survival.pipelines.preprocessing import training_preprocessor


def test_training_preprocessor_transforms_data(sample_training_data: pd.DataFrame) -> None:
    """Test that training_preprocessor transforms DataFrame inputs into a numeric matrix."""
    result = training_preprocessor.fit_transform(sample_training_data.copy())

    assert hasattr(result, "shape")
    assert result.shape[0] == len(sample_training_data)


def test_training_preprocessor_handles_missing_age(sample_training_data: pd.DataFrame) -> None:
    """Test median imputation on missing numerical Age values."""
    result = training_preprocessor.fit_transform(sample_training_data.copy())

    if hasattr(result, "toarray"):
        result_array = result.toarray()
    else:
        result_array = result

    # Assert no NaN elements remain in transformed output
    assert not np.isnan(result_array).any()


def test_training_preprocessor_creates_engineered_features(
    sample_training_data: pd.DataFrame,
) -> None:
    """Test intermediate feature engineering column generation step."""
    feature_engineered = training_preprocessor.named_steps["feature_engineering"].fit_transform(
        sample_training_data.copy()
    )

    assert "Title" in feature_engineered.columns
    assert "FamilySize" in feature_engineered.columns
    assert "IsAlone" in feature_engineered.columns
    assert "Deck" in feature_engineered.columns
    assert "TicketPrefix" in feature_engineered.columns
    assert "TicketGroupSize" in feature_engineered.columns


def test_training_preprocessor_fit_transform_consistency(
    sample_training_data: pd.DataFrame,
) -> None:
    """Test that fit_transform and fit().transform() yield identical matrix representations."""
    data_copy1 = sample_training_data.copy()
    data_copy2 = sample_training_data.copy()

    result1 = training_preprocessor.fit_transform(data_copy1)

    from titanic_survival.pipelines.preprocessing import training_preprocessor as pipeline2

    pipeline2.fit(data_copy2)
    result2 = pipeline2.transform(data_copy2)

    if hasattr(result1, "toarray"):
        result1_array = result1.toarray()
    else:
        result1_array = result1

    if hasattr(result2, "toarray"):
        result2_array = result2.toarray()
    else:
        result2_array = result2

    np.testing.assert_array_almost_equal(result1_array, result2_array)


def test_training_preprocessor_handles_all_missing_values() -> None:
    """Test preprocessor robustness when categorical columns contain missing values."""
    data_with_missing = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3],
            "Name": ["A, Mr. B", "C, Mrs. D", "E, Miss. F"],
            "Sex": ["male", "female", "male"],
            "Age": [25.0, 30.0, 35.0],
            "SibSp": [1, 0, 1],
            "Parch": [0, 1, 0],
            "Fare": [10.0, 20.0, 30.0],
            "Cabin": ["C89", np.nan, np.nan],
            "Embarked": ["S", "C", "Q"],
            "Pclass": [1, 2, 3],
            "Ticket": ["T1", "T2", "T3"],
        }
    )

    result = training_preprocessor.fit_transform(data_with_missing)

    assert result.shape[0] == len(data_with_missing)


def test_training_preprocessor_output_shape(sample_training_data: pd.DataFrame) -> None:
    """Test that preprocessor preserves row counts."""
    result = training_preprocessor.fit_transform(sample_training_data.copy())

    if hasattr(result, "toarray"):
        result_array = result.toarray()
    else:
        result_array = result

    assert result_array.shape[0] == len(sample_training_data)


def test_training_preprocessor_handle_unknown_category(
    sample_training_data: pd.DataFrame,
) -> None:
    """Test OneHotEncoder handle_unknown='ignore' behavior for novel category values."""
    training_preprocessor.fit(sample_training_data.copy())

    new_data = sample_training_data.copy()
    new_data.loc[0, "Embarked"] = "XYZ"

    result = training_preprocessor.transform(new_data)

    assert result.shape[0] == len(new_data)
