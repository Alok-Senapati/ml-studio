import numpy as np
from numpy.typing import NDArray

type NumericArray = NDArray[np.number]
type FloatArray = NDArray[np.floating]
type IntArray = NDArray[np.integer]
type BoolArray = NDArray[np.bool_]

__all__ = [
    "NumericArray",
    "FloatArray",
    "IntArray",
    "BoolArray",
]