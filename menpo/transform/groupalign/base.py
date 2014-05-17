import abc


class MultipleAlignment(object):
    r"""
    Abstract base class for aligning multiple source shapes to a target shape.

    Parameters
    ----------
    sources : list of :map:`PointCloud`
        List of pointclouds to be aligned.

    target : :map:`PointCloud`
        The target :map:`PointCloud` to align each source to.
        If None, then the mean of the sources is used.

        Default: None

    Raises
    -------
    ValueError
        Need at least two sources to align

    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, sources, target=None):
        from menpo.shape import PointCloud
        if len(sources) < 2 and target is None:
            raise ValueError("Need at least two sources to align")
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
