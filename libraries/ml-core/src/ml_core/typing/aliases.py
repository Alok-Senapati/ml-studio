import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
type BoolArray = NDArray[np.bool]

type FeatureMatrix = FloatArray
type TargetVector = IntArray
type ProbabilityVector = FloatArray