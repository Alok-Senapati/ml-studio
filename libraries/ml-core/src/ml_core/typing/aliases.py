from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

NumericArray = NDArray[np.number]
FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
BoolArray = NDArray[np.bool_]

__all__ = [
    "NumericArray",
    "FloatArray",
    "IntArray",
    "BoolArray",
]
