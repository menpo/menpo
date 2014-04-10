import numpy as np

from menpo.base import Targetable
from menpo.visualize.base import Viewable, AlignmentViewer2d


class Alignment(Targetable, Viewable):
    r"""
    Mixin for :class:`Transforms` that have been constructed from an
    optimisation aligning a source :class:`PointCloud` to a target
    :class:`PointCloud`.

    This is naturally an extension of the :class:`Targetable` interface - we
    just augment Targetable with the concept of a source, and related methods to
    construct alignments between a source and a target.

    Note: To inherit from Alignment, you have to be a Transform subclass first.

    Parameters
    ----------

    source: :class:`PointCloud`
        A PointCloud that the alignment will be based from

    target: :class:`PointCloud`
        A PointCloud that the alignment is targeted towards

    """
    def __init__(self, source, target):
        self._verify_source_and_target(source, target)
        self._source = source
        self._target = target

    @staticmethod
    def _verify_source_and_target(source, target):
        r"""
        Checks that the dimensions and number of points match up of the source
        and the target.

        """
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
    def aligned_source(self):
        # note here we require Alignment
        return self.apply(self.source)

    @property
    def alignment_error(self):
        r"""
        The Frobenius Norm of the difference between the target and
        the aligned source.

        :type: float
        """
        return np.linalg.norm(self.target.points - self.aligned_source.points)

    @property
    def target(self):
        return self._target

    def _target_setter(self, new_target):
        r"""
        Fulfils the Transformable _target_setter interface for all
        Alignments. This method should purely set the target - we know how to do
         that for all Alignments.
        """
        self._target = new_target

    def _new_target_from_state(self):
        r"""
        Fulfils the Transformable _new_target_from_state interface for all
        Alignments. This method should purely return the new target to be set -
        for all Alignments that is just the aligned source
        """
        return self.aligned_source

    def _view(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        View the PureAlignmentTransform. This plots the source points and
        vectors that represent the shift from source to target.

        Parameters
        ----------
        image : bool, optional
            If ``True`` the vectors are plotted on top of an image

            Default: ``False``
        """
        if self.n_dims == 2:
            return AlignmentViewer2d(figure_id, new_figure, self).render(
                **kwargs)
        else:
            raise ValueError("Only 2D alignments can be viewed currently.")
