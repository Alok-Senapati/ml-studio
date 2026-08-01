import numpy as np
from ml_core.math.sigmoid import sigmoid


def test_sigmoid_zero():
    assert sigmoid(np.array([0.0])) == np.array([0.5])
