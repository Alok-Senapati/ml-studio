import numpy as np
import pandas as pd

from titanic_survival.pipelines.preprocessing import feature_engineering_pipeline


def test_feature_engineering_pipeline():
    """Test the complete feature engineering pipeline with all transformers."""
    df = pd.DataFrame(
        {
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley",
                "Heikkinen, Miss. Laina",
                "Futrelle, Mr. Jacques Heath",
            ],
            "SibSp": [1, 1, 0, 1],
            "Parch": [0, 0, 0, 0],
            "Cabin": ["A85", "C85", np.nan, "C23"],
            "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803"],
        }
    )

    result = feature_engineering_pipeline.fit_transform(df)

    # Verify Title extraction
    assert list(result["Title"]) == ["Mr", "Mrs", "Miss", "Mr"]

    # Verify FamilySize creation
    assert list(result["FamilySize"]) == [2, 2, 1, 2]

    # Verify IsAlone creation
    assert list(result["IsAlone"]) == [0, 0, 1, 0]

    # Verify Deck extraction
    assert list(result["Deck"]) == ["A", "C", "Unknown", "C"]

    # Verify TicketPrefix extraction
    assert list(result["TicketPrefix"]) == ["A5", "PC", "STONO2", "NONE"]

    # Verify TicketGroupSize creation
    assert list(result["TicketGroupSize"]) == [1, 1, 1, 1]

    # Verify original columns are still present
    assert "Name" in result.columns
    assert "SibSp" in result.columns
    assert "Parch" in result.columns
    assert "Cabin" in result.columns
    assert "Ticket" in result.columns


def test_feature_engineering_pipeline_fit_transform_consistency():
    """Test that fit_transform and fit().transform() produce the same results."""
    df = pd.DataFrame(
        {
            "Name": ["Braund, Mr. Owen Harris", "Cumings, Mrs. John Bradley"],
            "SibSp": [1, 1],
            "Parch": [0, 0],
            "Cabin": ["A85", "C85"],
            "Ticket": ["A/5 21171", "PC 17599"],
        }
    )

    # fit_transform
    result1 = feature_engineering_pipeline.fit_transform(df.copy())

    # fit then transform on new pipeline instance
    from titanic_survival.pipelines.preprocessing import feature_engineering_pipeline as pipeline2
    pipeline2.fit(df.copy())
    result2 = pipeline2.transform(df.copy())

    # Compare Title outputs
    assert list(result1["Title"]) == list(result2["Title"])
    assert list(result1["FamilySize"]) == list(result2["FamilySize"])
