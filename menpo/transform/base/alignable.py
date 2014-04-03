import abc
from copy import deepcopy
import numpy as np

from menpo.visualize import AlignmentViewer2d
from menpo.visualize.base import Viewable


class Targetable(object):

    __metaclass__ = abc.ABCMeta

    @property
    def n_dims(self):
        return self.target.n_dims

    @property
    def n_points(self):
        return self.target.n_points


    @abc.abstractproperty
    def target(self):
        pass

    @target.setter
    def target(self, value):
        r"""
        Updates this alignment transform to point to a new target.
        """
        if value.n_dims != self.target.n_dims:
            raise ValueError(
                "The current target is {}D, the new target is {}D - new "
                "target has to have the same dimensionality as the "
                "old".format(self.target.n_dims, value.n_dims))
        elif value.n_points != self.target.n_points:
            raise ValueError(
                "The current target has {} points, the new target has {} "
                "- new target has to have the same number of points as the"
                " old".format(self.target.n_points, value.n_points))
        else:
            self._target_setter(value)

    @abc.abstractmethod
    def _target_setter(self, new_target):
        r"""
        Updates this alignment transform based on the new target.

        It is the responsibility of this method to leave the object in the
        updated state, including setting new_target to self._target as
        appropriate. Note that this method is called by the target setter,
        so this behavior must be respected.
        """
        pass

    @abc.abstractmethod
    def _sync_target(self):
        r"""
        Synchronizes the target to be correct after changes to
        AlignableTransforms.

        Needs to be called after any operation that may change the state of
        the transform (principally an issue on Vectorizable subclasses)

        This is pretty nasty, and will be removed when from_vector is made an
        underscore interface (in the same vein as _apply() or _view() ).
        """
        pass


class Alignable(Targetable):
    r"""
    Mixin for Transforms that can be constructed from an
    optimisation aligning a source PointCloud to a target PointCloud.

    Construction from the align() class method enables certain features of the
    class, like the from_target() and update_from_target() method. If the
    instance is just constructed with it's regular constructor, it functions
    as a normal Transform - attempting to call alignment methods listed here
    will simply yield an Exception.
    """

    def __init__(self):
        self._target = None
        self._source = None

    @classmethod
    def _align(cls, source, target, **kwargs):
        r"""
        Alternative Transform constructor. Constructs a Transform by finding
        the optimal transform to align source to target.

        Parameters
        ----------

        source: :class:`menpo.shape.PointCloud`
            The source pointcloud instance used in the alignment

        target: :class:`menpo.shape.PointCloud`
            The target pointcloud instance used in the alignment

        This is called automatically by align once verification of source and
        target is performed.

        Returns
        -------

        alignment_transform: :class:`menpo.transform.AlignableTransform`
            A Transform object that is_alignment.
        """
        pass

    @classmethod
    def align(cls, source, target, **kwargs):
        r"""
        Alternative Transform constructor. Constructs a Transform by finding
        the optimal transform to align source to target.

        Parameters
        ----------

        source: :class:`menpo.shape.PointCloud`
            The source pointcloud instance used in the alignment

        target: :class:`menpo.shape.PointCloud`
            The target pointcloud instance used in the alignment

        Returns
        -------

        alignment_transform: :class:`menpo.transform.AlignableTransform`
            A Transform object that is_alignment.
        """
        cls._verify_source_and_target(source, target)
        return cls._align(source, target, **kwargs)

    @property
    def is_alignment_transform(self):
        r"""
        True if this transform was constructed using a source and target.
        """
        return self.source is not None and self.target is not None

    @property
    def source(self):
        return self._source

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        r"""
        Updates this alignment transform to point to a new target.
        """
        if not self.is_alignment_transform:
            raise NotImplementedError("Cannot update target for Transforms "
                                      "not built with the align constructor")
        else:
            super(Alignable, self).target = value

    @property
    def aligned_source(self):
        if not self.is_alignment_transform:
            raise ValueError("This is not an alignment transform")
        else:
            return self.apply(self.source)

    @property
    def alignment_error(self):
        r"""
        The Frobenius Norm of the difference between the target and
        the aligned source.

        :type: float
        """
        return np.linalg.norm(self.target.points - self.aligned_source.points)

    @staticmethod
    def _verify_source_and_target(source, target):
        if source.n_dims != target.n_dims:
            raise ValueError("Source and target must have the same "
                             "dimensionality")
        elif source.n_points != target.n_points:
            raise ValueError("Source and target must have the same number of"
                             " points")
        else:
            return True


class PureAlignment(Alignable, Viewable):
    r"""
    :class:`AlignableTransform`s that are solely defined in terms of a source
    and target alignment.

    All transforms include support for alignments - all have a source and
    target property the alignment constructor, and methods like
    from_target(). However, for most transforms this is an optional
    interface - if the alignment constructor is not used, is_alignment is
    false, and all alignment methods will fail.

    This class is for transforms that solely make sense as alignments. It
    simplifies the interface down, so that :class:`PureAlignmentTransform`
    subclasses only have to override :meth:`_target_setter()`
    to satisfy the AlignableTransform interface.

    Note that if you are inheriting this class and Transform it is suggested
    that you inherit this class first, so as to get it's definitions of n_dims.
    """

    def __init__(self, source, target):
        Alignable.__init__(self)
        if self._verify_source_and_target(source, target):
            self._source = source
            self._target = target

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

    @classmethod
    def align(cls, source, target, **kwargs):
        r"""
        Alternative Transform constructor. Constructs a Transform by finding
        the optimal transform to align source to target. Note that for
        PureAlignmentTransform's we know that align == __init__. To save
        repetition we share the align method here.

        Parameters
        ----------

        source: :class:`menpo.shape.PointCloud`
            The source pointcloud instance used in the alignment

        target: :class:`menpo.shape.PointCloud`
            The target pointcloud instance used in the alignment

        Returns
        -------

        alignment_transform: :class:`menpo.transform.AlignableTransform`
            A Transform object that is_alignment.
        """
        return cls(source, target, **kwargs)
