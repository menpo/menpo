import numpy as np

from menpo.base import Targetable
from menpo.visualize.base import Viewable


class Alignment(Targetable, Viewable):
    r"""
    Mix-in for :map:`Transform` that have been constructed from an optimisation
    aligning a source :map:`PointCloud` to a target :map:`PointCloud`.

    This is naturally an extension of the :map:`Targetable` interface - we just
    augment :map:`Targetable` with the concept of a source, and related methods
    to construct alignments between a source and a target.

    Note that to inherit from :map:`Alignment`, you have to be a
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
        ----------
        source : :map:`PointCloud`
            A PointCloud that the alignment will be based from
        target : :map:`PointCloud`
            A PointCloud that the alignment is targeted towards

        Raises
        ------
        ValueError
            Source and target must have the same dimensionality
        ValueError
            Source and target must have the same number of points
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
        The source :map:`PointCloud` that is used in the alignment.

        The source is not mutable.

        :type: :map:`PointCloud`
        """
        return self._source

    def aligned_source(self):
        r"""
        The result of applying ``self`` to :attr:`source`

        :type: :map:`PointCloud`
        """
        # Note that here we have the dependency that we are a Transform
        return self.apply(self.source)

    def alignment_error(self):
        r"""
        The Frobenius Norm of the difference between the target and the aligned
        source.

        :type: `float`
        """
        return np.linalg.norm(self.target.points - self.aligned_source().points)

    @property
    def target(self):
        r"""
        The current :map:`PointCloud` that this object produces.

        To change the target, use :meth:`set_target`.

        :type: :map:`PointCloud`
        """
        return self._target

    def _target_setter(self, new_target):
        r"""
        Fulfils the :map:`Targetable` `_target_setter` interface for all
        Alignments. This method should purely set the target - we know how to do
        that for all :map:`Alignment` instances.

        Parameters
        ----------
        new_target : :map:`PointCloud`
            The new PointCloud target
        """
        self._target = new_target

    def _new_target_from_state(self):
        r"""
        Fulfils the :map:`Targetable` :meth:`_new_target_from_state` interface
        for all Alignments.

        This method should purely return the new target to be set - for all
        :map:`Alignment` instances this is just the aligned source.
        """
        return self.aligned_source()

    def _view_2d(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        Plots the source points and vectors that represent the shift from
        source to target.

        Parameters
        ----------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        kwargs : `dict`
            The options passed to the rendered
        """
        from menpo.visualize import AlignmentViewer2d
        return AlignmentViewer2d(figure_id, new_figure, self).render(**kwargs)
