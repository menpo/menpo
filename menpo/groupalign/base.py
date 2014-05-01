import abc
from menpo.shape import PointCloud


class MultipleAlignment(object):
    r"""
    Abstract base class for aligning multiple source shapes to a target shape.

    Parameters
    ----------
    sources : list of (N, D) ndarrays
        List of pointclouds to be aligned.
    target : (N, D) ndarray, optional
        The target pointcloud to align to. If None, then the mean of the
        sources is used.

        Default: None

    Raises
    -------
    Exception
        Need at least two sources to align
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, sources, target=None):
        if len(sources) < 2 and target is None:
            raise Exception("Need at least two sources to align")
        self.n_sources = len(sources)
        self.n_points, self.n_dims = sources[0].n_points, sources[0].n_dims
        self.sources = sources
        print type(sources[0])
        if target is None:
            # set the target to the mean source position
            self.target = PointCloud(
                sum([s.points for s in self.sources]) / self.n_sources)
        else:
            assert self.n_dims, self.n_points == target.shape
            self.target = target
        print type(self.target)
