from sklearn.linear_model import LogisticRegression
from titanic_survival.config import settings

config = settings.logistic_regression

logistic_regression = LogisticRegression(
    max_iter=config.max_iter,
    solver=config.solver,
    penalty=config.penalty,
    C=config.C,
    random_state=settings.common.random_seed
)