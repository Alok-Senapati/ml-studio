from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from titanic_survival.pipelines.engineering import feature_engineering_pipeline

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

numeric_features = ["Pclass", "Age", "Fare", "SibSp", "Parch", "FamilySize", "TicketGroupSize"]
categorical_features = ["Sex", "Embarked", "Title", "Deck", "TicketPrefix", "IsAlone"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ],
    remainder="drop"
)

training_preprocessor = Pipeline(
    steps=[
        ("feature_engineering", feature_engineering_pipeline),
        ("preprocessing", preprocessor)
    ]
)