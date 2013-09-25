import abc
import numpy as np


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
        self.n_landmarks, self.n_dims = sources[0].shape
        self.sources = sources
        if target is None:
            # set the target to the mean source position
            self.target = sum(self.sources) / self.n_sources
        else:
            assert self.n_dims, self.n_landmarks == target.shape
            self.target = target

    @abc.abstractproperty
    def transforms(self):
        r"""
        Returns a list of transforms, one for each source, which aligns it to
        the target.

        :type: list of :class:`pybug.transform.base.Transform`
        """
        pass
