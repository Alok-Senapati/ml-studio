import numpy as np
import pandas as pd

from titanic_survival.features.engineering import TitleExtractor, TicketGroupSizeCreator
from titanic_survival.features.engineering import FamilySizeCreator
from titanic_survival.features.engineering import DeckExtractor
from titanic_survival.features.engineering import TicketPrefixExtractor
from titanic_survival.features.engineering import IsAloneCreator

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

    transformer = TitleExtractor(name_column="Name", output_column="Title")
    result = transformer.fit_transform(df)

    assert list(result["Title"]) == [
        "Mr",
        "Mrs",
        "Miss"
    ]


def test_family_size_creator():
    df = pd.DataFrame(
        {
            "SibSp": [1, 0, 2, 0],
            "Parch": [0, 5, 3, 0]
        }
    )

    transformer = FamilySizeCreator(
        sibsp_column="SibSp",
        parch_column="Parch",
        output_column="FamilySize"
    )
    result = transformer.fit_transform(df)

    assert list(result["FamilySize"]) == [2, 6, 6, 1]


def test_is_alone_creator():
    df = pd.DataFrame(
        {
            "FamilySize": [1, 2, 3]
        }
    )

    transformer = IsAloneCreator(
        family_size_column="FamilySize",
        output_column="IsAlone"
    )
    result = transformer.fit_transform(df)

    assert list(result["IsAlone"]) == [1, 0, 0]


def test_deck_extractor():
    df = pd.DataFrame(
        {
            "Cabin": ["C85", "B23", "B57", np.nan]
        }
    )

    transformer = DeckExtractor(
        cabin_column="Cabin",
        output_column="Deck",
        unknown_value="Unknown"
    )
    result = transformer.fit_transform(df)

    assert list(result["Deck"]) == ["C", "B", "B", "Unknown"]


def test_ticket_prefix_extractor():
    df = pd.DataFrame(
        {
            "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803"]
        }
    )

    transformer = TicketPrefixExtractor(
        ticket_column="Ticket",
        output_column="TicketPrefix",
        no_prefix_value="NONE"
    )
    result = transformer.fit_transform(df)

    assert list(result["TicketPrefix"]) == ["A5", "PC", "STONO2", "NONE"]


def test_ticket_group_size_creator():
    df = pd.DataFrame(
        {
            "Ticket": [
                "PC17599",
                "PC17599",
                "113803",
                "347082",
                "347082",
                "347082",
            ]
        }
    )

    transformer = TicketGroupSizeCreator(
        ticket_column="Ticket",
        output_column="TicketGroupSize",
    )

    result = transformer.fit_transform(df)

    assert list(result["TicketGroupSize"]) == [
        2,
        2,
        1,
        3,
        3,
        3,
    ]


def test_ticket_group_size_unseen_ticket():
    train_df = pd.DataFrame(
        {
            "Ticket": [
                "A",
                "A",
                "B",
            ]
        }
    )

    test_df = pd.DataFrame(
        {
            "Ticket": [
                "A",
                "C",
            ]
        }
    )

    transformer = TicketGroupSizeCreator()
    transformer.fit(train_df)

    result = transformer.transform(test_df)

    assert list(result["TicketGroupSize"]) == [2, 1]
