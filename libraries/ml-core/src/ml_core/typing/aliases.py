"""
Type aliases for NumPy array annotations across ml-core.

This module defines standardized type aliases for n-dimensional NumPy arrays
restricted to specific numeric scalar dtypes (e.g., float, integer, boolean).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Type alias representing any numeric (floating-point, integer, or complex) NumPy array.
NumericArray = NDArray[np.number]

# Type alias representing floating-point NumPy arrays (e.g., np.float64, np.float32).
FloatArray = NDArray[np.floating]

# Type alias representing integer NumPy arrays (e.g., np.int64, np.int32).
IntArray = NDArray[np.integer]

# Type alias representing boolean NumPy arrays (e.g., np.bool_).
BoolArray = NDArray[np.bool_]

__all__ = [
    "NumericArray",
    "FloatArray",
    "IntArray",
    "BoolArray",
]
