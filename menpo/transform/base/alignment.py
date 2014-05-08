import numpy as np

from menpo.base import Targetable
from menpo.visualize.base import Viewable, AlignmentViewer2d


class Alignment(Targetable, Viewable):
    r"""
    Mixin for :map:`Transforms` that have been constructed from an
    optimisation aligning a source :map:`PointCloud` to a target
    :map:`PointCloud`.

    This is naturally an extension of the :map:`Targetable` interface - we
    just augment :map:`Targetable` with the concept of a source, and related
    methods to construct alignments between a source and a target.

    ..note: To inherit from :map:`Alignment`, you have to be a
            :map:`Transform` subclass first.

    Parameters
    ----------
    source : :map:`PointCloud`
        A PointCloud that the alignment will be based from

    target : :map:`PointCloud`
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

        Parameters
        -----------
        source : (N, D) :map:`PointCloud`
            The source of the alignment.
        target : (N, D) :map:`PointCloud`
            The target of the alignment

        Raises
        -------
        ValueError:
            If `n_dims` or `n_points` don't match on the source and target.
        """
        if source.n_dims != target.n_dims:
            raise ValueError("Source and target must have the same "
                             "dimensionality")
        elif source.n_points != target.n_points:
            raise ValueError("Source and target must have the same number of"
                             " points")

    @property
    def source(self):
        r"""
        The source PointCloud.

        :type: :map:`PointCloud`
        """
        return self._source

    @property
    def aligned_source(self):
        r"""
        The source after having been aligned using the best possible
        transformation between source and target.

        :type: :map:`PointCloud`
        """
        # note here we require ourselves to be a subclass of Transform
        return self.apply(self.source)

    @property
    def alignment_error(self):
        r"""
        The Frobenius norm of the difference between the target and
        the aligned source.

        :type: float
        """
        return np.linalg.norm(self.target.points - self.aligned_source.points)

    @property
    def target(self):
        r"""
        The target PointCloud.

        :type: :map:`PointCloud`
        """
        return self._target

    def _target_setter(self, new_target):
        r"""
        Fulfils the Transformable `_target_setter` interface for all
        :map:`Alignment`s. This method should purely set the target - we know
        how to do that for all :map:`Alignment`s.

        Parameters
        ----------
        new_target : (N, D) :map:`PointCloud`
            The new target to set.
        """
        self._target = new_target

    def _new_target_from_state(self):
        r"""
        Fulfils the :map:`Transformable` `_new_target_from_state` interface for
        all :map:`Alignment`s. This method should purely return the new target
        to be set - for all :map:`Alignment`s that is just the aligned source.

        Returns
        -------
        new_target : (N, D) :map:`PointCloud`
            The new target to be set, by default this is the aligned source.
        """
        return self.aligned_source

    def _view(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        View the :map:`Alignment`. This plots the source points and
        vectors that represent the shift from source to target.

        Parameters
        ----------
        image : bool, optional
            If `True` the vectors are plotted on top of an image

            Default: `False`
        """
        if self.n_dims == 2:
            return AlignmentViewer2d(figure_id, new_figure, self).render(
                **kwargs)
        else:
            raise ValueError("Only 2D alignments can be viewed currently.")
