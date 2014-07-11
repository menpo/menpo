import numpy as np
from .base import Transform


class AppendNDims(Transform):
    r"""
    Adds n dims to a shape

    Parameters
    ----------

    n : `int`
        The number of dimensions to add

    value : `float`, optional
        The value that the extra dims should be given

        Default: 0

    """
    def __init__(self, n, value=0):
        self.n = n
        self.value = value

    @property
    def n_dims(self):
        # can operate on any dimensional shape
        return None

    @property
    def n_dims_output(self):
        # output is unknown (we don't in general know the input!)
        return None

    def _apply(self, x, **kwargs):
        return np.hstack([x, np.ones([x.shape[0], self.n]) * self.value]).copy()


class ExtractNDims(Transform):
    r"""
    Extracts out the first ``n`` dimensions of a shape.

    Parameters
    ----------

    n : `int`
        The number of dimensions to extract

    """
    def __init__(self, n):
        self.n = n

    @property
    def n_dims(self):
        # can operate on any dimensional shape
        return None

    @property
    def n_dims_output(self):
        # output is just n
        return self.n

    def _apply(self, x, **kwargs):
        return x[:, :self.n].copy()
