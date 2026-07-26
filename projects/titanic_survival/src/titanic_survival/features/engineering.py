from __future__ import annotations

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class TitleExtractor(BaseEstimator, TransformerMixin):
    """Extract passenger title from the Name column."""
    def __init__(self, name_column: str = "Name", output_column: str = "Title") -> None:
        self.name_column = name_column
        self.output_column = output_column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TitleExtractor:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X[self.output_column] = (
            X[self.name_column]
            .str.extract(r",\s*([^\.]+)\.")
            .iloc[:, 0]
            .str.strip()
        )

        return X


class FamilySizeCreator(BaseEstimator, TransformerMixin):
    """Calculate FamilySize from SibSp and ParCh"""
    def __init__(
        self,
        sibsp_column: str = "SibSp",
        parch_column: str = "Parch",
        output_column: str = "FamilySize"
    ) -> None:
        self.sibsp_column = sibsp_column
        self.parch_column = parch_column
        self.output_column = output_column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FamilySizeCreator:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.output_column] = (
            X[self.sibsp_column] + X[self.parch_column] + 1
        )
        return X


class IsAloneCreator(BaseEstimator, TransformerMixin):
    """Checks if the passenger is traveling alone based on FamilySize"""
    def __init__(self, family_size_column: str = "FamilySize", output_column: str = "IsAlone") -> None:
        self.family_size_column = family_size_column
        self.output_column = output_column


    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> IsAloneCreator:
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.output_column] = (X[self.family_size_column] == 1).astype(int)
        return X


class DeckExtractor(BaseEstimator, TransformerMixin):
    """Extract Deck from Cabin"""
    def __init__(self, cabin_column: str = "Cabin", output_column: str = "Deck", unknown_value: str = "Unknown") -> None:
        self.cabin_column = cabin_column
        self.output_column = output_column
        self.unknown_value = unknown_value


    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> DeckExtractor:
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        cabin = (
            X[self.cabin_column]
            .fillna("")
            .str.strip()
        )

        deck = cabin.str[0].str.upper()

        X[self.output_column] = deck.replace("", pd.NA).fillna(self.unknown_value)

        return X


class TicketPrefixExtractor(BaseEstimator, TransformerMixin):
    """Extract Prefix from Ticket No"""
    def __init__(self, ticket_column: str = "Ticket", output_column: str = "TicketPrefix", no_prefix_value: str = "NONE") -> None:
        self.ticket_column = ticket_column
        self.output_column = output_column
        self.no_prefix_value = no_prefix_value


    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TicketPrefixExtractor:
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X[self.output_column] = (
            X[self.ticket_column]
            .str.upper()
            .str.replace(".", "", regex=False)
            .str.replace("/", "", regex=False)
            .str.split(" ").str[0]
            .str.strip()
        )

        X[self.output_column] = X[self.output_column].where(~X[self.output_column].str.isnumeric(), self.no_prefix_value)

        return X