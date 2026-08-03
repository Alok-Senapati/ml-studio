"""
Custom scikit-learn feature engineering transformers for the Titanic dataset.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class TitleExtractor(BaseEstimator, TransformerMixin):
    """
    Extract honorific titles (e.g., Mr, Mrs, Miss, Master) from passenger name strings.

    Parameters
    ----------
    name_column : str, default="Name"
        Name of the input column containing passenger full names.
    output_column : str, default="Title"
        Name of the output column to store extracted titles.
    """

    def __init__(self, name_column: str = "Name", output_column: str = "Title") -> None:
        self.name_column = name_column
        self.output_column = output_column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TitleExtractor:
        """
        Fit transformer on dataset (no-op).

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame containing passenger feature columns.
        y : pd.Series or None, default=None
            Target vector (unused).

        Returns
        -------
        TitleExtractor
            The fitted transformer instance.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extract titles from the name column into a new column.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with extracted title column appended.
        """
        X = X.copy()
        # Extract substring following comma and preceding period (e.g., "Smith, Mr. John" -> "Mr")
        X[self.output_column] = (
            X[self.name_column].str.extract(r",\s*([^\.]+)\.").iloc[:, 0].str.strip()
        )
        return X


class FamilySizeCreator(BaseEstimator, TransformerMixin):
    """
    Compute total family size from sibling/spouse and parent/child counts.

    Parameters
    ----------
    sibsp_column : str, default="SibSp"
        Column name for number of siblings/spouses aboard.
    parch_column : str, default="Parch"
        Column name for number of parents/children aboard.
    output_column : str, default="FamilySize"
        Column name for total family size (`SibSp` + `Parch` + 1).
    """

    def __init__(
        self,
        sibsp_column: str = "SibSp",
        parch_column: str = "Parch",
        output_column: str = "FamilySize",
    ) -> None:
        self.sibsp_column = sibsp_column
        self.parch_column = parch_column
        self.output_column = output_column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FamilySizeCreator:
        """
        Fit transformer on dataset (no-op).

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.
        y : pd.Series or None, default=None
            Target vector (unused).

        Returns
        -------
        FamilySizeCreator
            The fitted transformer instance.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate and add FamilySize column.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with FamilySize column added.
        """
        X = X.copy()
        # Family size includes siblings, spouses, parents, children, plus passenger (+1)
        X[self.output_column] = X[self.sibsp_column] + X[self.parch_column] + 1
        return X


class IsAloneCreator(BaseEstimator, TransformerMixin):
    """
    Create a binary indicator specifying whether a passenger is traveling alone.

    Parameters
    ----------
    family_size_column : str, default="FamilySize"
        Input column containing calculated family size.
    output_column : str, default="IsAlone"
        Output column containing binary flag (1 if FamilySize == 1 else 0).
    """

    def __init__(
        self, family_size_column: str = "FamilySize", output_column: str = "IsAlone"
    ) -> None:
        self.family_size_column = family_size_column
        self.output_column = output_column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> IsAloneCreator:
        """
        Fit transformer on dataset (no-op).

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.
        y : pd.Series or None, default=None
            Target vector (unused).

        Returns
        -------
        IsAloneCreator
            The fitted transformer instance.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add IsAlone binary column based on FamilySize.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with IsAlone column added.
        """
        X = X.copy()
        # Set 1 if FamilySize equals 1, else 0
        X[self.output_column] = (X[self.family_size_column] == 1).astype(int)
        return X


class DeckExtractor(BaseEstimator, TransformerMixin):
    """
    Extract the deck letter code from Cabin allocation numbers.

    Parameters
    ----------
    cabin_column : str, default="Cabin"
        Input cabin string column.
    output_column : str, default="Deck"
        Output deck letter column.
    unknown_value : str, default="Unknown"
        Value assigned to missing or empty cabin records.
    """

    def __init__(
        self,
        cabin_column: str = "Cabin",
        output_column: str = "Deck",
        unknown_value: str = "Unknown",
    ) -> None:
        self.cabin_column = cabin_column
        self.output_column = output_column
        self.unknown_value = unknown_value

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> DeckExtractor:
        """
        Fit transformer on dataset (no-op).

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.
        y : pd.Series or None, default=None
            Target vector (unused).

        Returns
        -------
        DeckExtractor
            The fitted transformer instance.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extract first character of Cabin number as Deck.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with Deck column added.
        """
        X = X.copy()
        cabin = X[self.cabin_column].fillna("").str.strip()
        deck = cabin.str[0].str.upper()
        # Fill missing cabins with default unknown_value string
        X[self.output_column] = deck.replace("", pd.NA).fillna(self.unknown_value)
        return X


class TicketPrefixExtractor(BaseEstimator, TransformerMixin):
    """
    Extract text prefix codes from passenger ticket numbers.

    Parameters
    ----------
    ticket_column : str, default="Ticket"
        Input ticket identifier column.
    output_column : str, default="TicketPrefix"
        Output extracted ticket prefix column.
    no_prefix_value : str, default="NONE"
        Default code assigned when no text prefix is present.
    """

    def __init__(
        self,
        ticket_column: str = "Ticket",
        output_column: str = "TicketPrefix",
        no_prefix_value: str = "NONE",
    ) -> None:
        self.ticket_column = ticket_column
        self.output_column = output_column
        self.no_prefix_value = no_prefix_value

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TicketPrefixExtractor:
        """
        Fit transformer on dataset (no-op).

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.
        y : pd.Series or None, default=None
            Target vector (unused).

        Returns
        -------
        TicketPrefixExtractor
            The fitted transformer instance.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and extract ticket text prefix strings.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with TicketPrefix column added.
        """
        X = X.copy()
        # Remove punctuation marks and split on whitespace
        X[self.output_column] = (
            X[self.ticket_column]
            .str.upper()
            .str.replace(".", "", regex=False)
            .str.replace("/", "", regex=False)
            .str.split(" ")
            .str[0]
            .str.strip()
        )

        # Assign default fallback code when token is purely numeric
        X[self.output_column] = X[self.output_column].where(
            ~X[self.output_column].str.isnumeric(), self.no_prefix_value
        )
        return X


class TicketGroupSizeCreator(BaseEstimator, TransformerMixin):
    """
    Calculate the total number of passengers sharing the exact same ticket number.

    Parameters
    ----------
    ticket_column : str, default="Ticket"
        Input ticket number column.
    output_column : str, default="TicketGroupSize"
        Output column containing count of passengers sharing the ticket.
    """

    def __init__(
        self, ticket_column: str = "Ticket", output_column: str = "TicketGroupSize"
    ) -> None:
        self.ticket_column = ticket_column
        self.output_column = output_column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TicketGroupSizeCreator:
        """
        Calculate ticket occurrence frequencies across the training dataset.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.
        y : pd.Series or None, default=None
            Target vector (unused).

        Returns
        -------
        TicketGroupSizeCreator
            The fitted transformer instance.
        """
        # Store dictionary mapping ticket strings to frequency counts
        self.ticket_counts_ = X[self.ticket_column].value_counts().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Map ticket frequency counts to construct TicketGroupSize.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with TicketGroupSize column added.

        Raises
        ------
        NotFittedError
            If transformer has not been fitted prior to transform.
        """
        check_is_fitted(self, "ticket_counts_")
        X = X.copy()
        # Map ticket count frequencies, falling back to 1 for unseen tickets
        X[self.output_column] = (
            X[self.ticket_column].map(self.ticket_counts_.get).fillna(1).astype(int)
        )
        return X
