from typing import Any, Dict, Tuple, List

import torch
import numpy as np


def repeat_to_shape(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Repeat y to match the shape of x."""
    assert y.ndim <= x.ndim

    required_ndims = np.arange(x.ndim - y.ndim)
    y = np.expand_dims(y, axis=tuple(required_ndims))
    assert x.ndim == y.ndim

    reps = tuple(c // p for c, p in zip(x.shape, y.shape))
    y = np.tile(y, reps)
    assert x.shape == y.shape

    return y
