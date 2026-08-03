"""
Pytest fixtures and sample data generators for titanic_survival test suites.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_training_data() -> pd.DataFrame:
    """
    Generate a representative DataFrame containing raw sample Titanic passenger data.

    Returns
    -------
    pd.DataFrame
        DataFrame with standard Titanic dataset schema columns.
    """
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4, 5, 6],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley",
                "Heikkinen, Miss. Laina",
                "Futrelle, Mr. Jacques Heath",
                "Allen, Mr. William Henry",
                "Moran, Mr. James",
            ],
            "Pclass": [3, 1, 3, 1, 3, 3],
            "Sex": [
                "male",
                "female",
                "female",
                "male",
                "male",
                "male",
            ],
            "Age": [
                22.0,
                38.0,
                np.nan,
                35.0,
                35.0,
                28.0,
            ],
            "SibSp": [1, 1, 0, 1, 0, 0],
            "Parch": [0, 0, 0, 0, 0, 0],
            "Ticket": [
                "A/5 21171",
                "PC 17599",
                "STON/O2. 3101282",
                "113803",
                "373450",
                "330877",
            ],
            "Fare": [
                7.25,
                71.2833,
                7.925,
                53.10,
                8.05,
                8.4583,
            ],
            "Cabin": [
                "C85",
                "C123",
                np.nan,
                "C123",
                np.nan,
                np.nan,
            ],
            "Embarked": [
                "S",
                "C",
                "S",
                "S",
                "S",
                "Q",
            ],
        }
    )


@pytest.fixture
def sample_training_labels() -> np.ndarray:
    """
    Generate ground truth binary survival target labels matching sample dataset.

    Returns
    -------
    np.ndarray
        Array of binary target values (0 = did not survive, 1 = survived).
    """
    return np.array([0, 1, 1, 1, 0, 0], dtype=np.int64)
