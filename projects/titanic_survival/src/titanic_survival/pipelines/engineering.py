"""
Feature engineering scikit-learn pipeline for titanic_survival.
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from titanic_survival.features.engineering import (
    DeckExtractor,
    FamilySizeCreator,
    IsAloneCreator,
    TicketGroupSizeCreator,
    TicketPrefixExtractor,
    TitleExtractor,
)

#: Sequential pipeline executing custom feature extraction and transformation steps
feature_engineering_pipeline = Pipeline(
    steps=[
        ("title_extractor", TitleExtractor()),
        ("family_size_creator", FamilySizeCreator()),
        ("is_alone_creator", IsAloneCreator()),
        ("deck_extractor", DeckExtractor()),
        ("ticket_prefix_extractor", TicketPrefixExtractor()),
        ("ticket_group_size_creator", TicketGroupSizeCreator()),
    ]
)
