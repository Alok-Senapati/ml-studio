from pydantic import BaseModel
from pydantic_settings import BaseSettings

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class CommonConfig(BaseModel):
    random_seed: int = 42

class DataConfig(BaseModel):
    project_root: Path = PROJECT_ROOT
    raw_data_dir: Path = project_root / "data" / "raw"
    processed_data_dir: Path = project_root / "data" / "processed"
    interim_data_dir: Path = project_root / "data" / "interim"
    external_data_dir: Path = project_root / "data" / "external"

class TrainingConfig(BaseModel):
    test_size: float = 0.2
    stratify: bool = True

class LogisticRegressionConfig(BaseModel):
    max_iter: int = 1000
    solver: str = "lbfgs"
    penalty: str = "l2"
    C: float = 1.0

class DecisionTreeConfig(BaseModel):
    criterion: str = "gini"
    max_depth: int | None = None
    min_samples_split: int = 2

class RandomForestConfig(BaseModel):
    n_estimators: int = 100
    criterion: str = "gini"
    max_depth: int | None = None

class Settings(BaseSettings):
    common: CommonConfig = CommonConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    logistic_regression: LogisticRegressionConfig = LogisticRegressionConfig()
    decision_tree: DecisionTreeConfig = DecisionTreeConfig()
    random_forest: RandomForestConfig = RandomForestConfig()


settings = Settings()

