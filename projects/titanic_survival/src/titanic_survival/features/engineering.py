from __future__ import annotations

import re
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class TitleExtractor(BaseEstimator, TransformerMixin):
    """Extract passenger title from the Name column."""
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()

        X["Title"] = (
            X["Name"]
            .str.extract(r",\s*([^\.]+)\.")
            .iloc[:, 0]
            .str.strip()
        )

        return X