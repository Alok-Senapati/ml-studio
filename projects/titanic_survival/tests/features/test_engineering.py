import pandas as pd

from titanic_survival.features.engineering import TitleExtractor

def test_title_extraction():
    df = pd.DataFrame(
        {
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley",
                "Heikkinen, Miss. Laina",
            ]
        }
    )

    transformer = TitleExtractor()
    result = transformer.fit_transform(df)

    assert list(result["Title"]) == [
        "Mr",
        "Mrs",
        "Miss"
    ]