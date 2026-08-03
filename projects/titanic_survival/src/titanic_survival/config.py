"""
Project configuration schemas and settings management for titanic_survival.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings

#: Absolute path to project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CommonConfig(BaseModel):
    """
    Common project configuration parameters.

    Attributes
    ----------
    random_seed : int, default=42
        Global pseudo-random number generator seed.
    """

    random_seed: int = 42


class DataConfig(BaseModel):
    """
    Directory path configurations for data artifacts.

    Attributes
    ----------
    project_root : Path
        Root path of the project.
    raw_data_dir : Path
        Path to raw data directory.
    processed_data_dir : Path
        Path to processed data directory.
    interim_data_dir : Path
        Path to interim data directory.
    external_data_dir : Path
        Path to external data directory.
    """

    project_root: Path = PROJECT_ROOT
    raw_data_dir: Path = project_root / "data" / "raw"
    processed_data_dir: Path = project_root / "data" / "processed"
    interim_data_dir: Path = project_root / "data" / "interim"
    external_data_dir: Path = project_root / "data" / "external"


class TrainingConfig(BaseModel):
    """
    Model training configuration settings.

    Attributes
    ----------
    test_size : float, default=0.2
        Fraction of dataset to allocate to the test split.
    stratify : bool, default=True
        Whether to perform stratified sampling on target labels.
    """

    test_size: float = 0.2
    stratify: bool = True


class LogisticRegressionConfig(BaseModel):
    """
    Hyperparameters for Logistic Regression classifier.

    Attributes
    ----------
    max_iter : int, default=1000
        Maximum number of optimization iterations.
    solver : str, default="lbfgs"
        Optimization solver algorithm name.
    l1_ratio : float, default=0.0
        ElasticNet mixing parameter (0 for L2, 1 for L1).
    C : float, default=1.0
        Inverse of regularization strength.
    """

    max_iter: int = 1000
    solver: str = "lbfgs"
    l1_ratio: float = 0.0
    C: float = 1.0


class DecisionTreeConfig(BaseModel):
    """
    Hyperparameters for Decision Tree classifier.

    Attributes
    ----------
    criterion : str, default="gini"
        Split quality criterion function.
    max_depth : int or None, default=None
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    """

    criterion: str = "gini"
    max_depth: int | None = None
    min_samples_split: int = 2


class RandomForestConfig(BaseModel):
    """
    Hyperparameters for Random Forest classifier.

    Attributes
    ----------
    n_estimators : int, default=100
        Number of decision trees in the forest.
    criterion : str, default="gini"
        Split quality criterion function.
    max_depth : int or None, default=None
        Maximum depth of individual trees.
    """

    n_estimators: int = 100
    criterion: str = "gini"
    max_depth: int | None = None


class Settings(BaseSettings):
    """
    Global application settings container.

    Combines common, data, training, and model-specific configuration objects.
    """

    common: CommonConfig = CommonConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    logistic_regression: LogisticRegressionConfig = LogisticRegressionConfig()
    decision_tree: DecisionTreeConfig = DecisionTreeConfig()
    random_forest: RandomForestConfig = RandomForestConfig()


#: Singleton instance of application settings
settings = Settings()
