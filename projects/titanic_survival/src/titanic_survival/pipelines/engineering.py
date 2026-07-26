from sklearn.pipeline import Pipeline

from titanic_survival.features.engineering import (
    TitleExtractor,
    FamilySizeCreator,
    IsAloneCreator,
    DeckExtractor,
    TicketGroupSizeCreator,
    TicketPrefixExtractor
)

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

