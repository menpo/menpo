import numpy as np

from menpo.base import Targetable


class Alignment(Targetable):
    r"""
    Mixin for Transforms that have been constructed from an
    optimisation aligning a source PointCloud to a target PointCloud.

    This is naturally an extension of the Targetable interface - we just
    augment Targetable with the concept of a source, and related methods to
    construct alignments between a source and a target.

    Construction from the align() class method enables certain features of the
    class, like the from_target() and update_from_target() method. If the
    instance is just constructed with it's regular constructor, it functions
    as a normal Transform - attempting to call alignment methods listed here
    will simply yield an Exception.
    """

    def __init__(self, source, target):
        self._verify_source_and_target(source, target)
        self._source = source
        self._target = target

    @staticmethod
    def _verify_source_and_target(source, target):
        if source.n_dims != target.n_dims:
            raise ValueError("Source and target must have the same "
                             "dimensionality")
        elif source.n_points != target.n_points:
            raise ValueError("Source and target must have the same number of"
                             " points")

    @property
    def source(self):
        return self._source

    @property
    def target(self):
        return self._target

    def _target_setter(self, new_target):
        self._target = new_target

    def _new_target_from_state(self):
        return self.aligned_source

    @property
    def aligned_source(self):
        return self.apply(self.source)

    @property
    def alignment_error(self):
        r"""
        The Frobenius Norm of the difference between the target and
        the aligned source.

        :type: float
        """
        return np.linalg.norm(self.target.points - self.aligned_source.points)
