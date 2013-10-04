import abc
import numpy as np
from pybug.shape import PointCloud


class MultipleAlignment(object):
    r"""
    Abstract base class for aligning multiple sources to a target.

    Parameters
    ----------
    sources : (N, D) list of ndarrays
        List of points to be aligned
    target : (N, D) ndarray, optional
        The target to align to.

        Default: The ``mean`` of sources
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, sources, target=None):
        if len(sources) < 2 and target is None:
            raise Exception("Need at least two sources to align")
        self.n_sources = len(sources)
        self.n_points, self.n_dims = sources[0].n_points, sources[0].n_dims
        self.sources = sources
        if target is None:
            # set the target to the mean source position
            self.target = PointCloud(
                sum([s.points for s in self.sources]) / self.n_sources)
        else:
            assert self.n_dims, self.n_points == target.shape
            self.target = target
