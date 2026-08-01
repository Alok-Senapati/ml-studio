from ml_core.exceptions.base import MLCoreError


class NotFittedError(MLCoreError):
    """Raised when a model is user for inference before fitting."""