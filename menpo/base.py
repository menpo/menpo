import abc
from copy import deepcopy
import os.path


class Vectorizable(object):
    """
    Interface that provides methods for 'flattening' an object into a
    vector, and restoring from the same vectorized form. Useful for
    statistical analysis of objects, which commonly requires the data
    to be provided as a single vector.
    """

    __metaclass__ = abc.ABCMeta

    @property
    def n_parameters(self):
        r"""
        The length of the vector that this Vectorizable object produces.

        type: int
        """
        return (self.as_vector()).shape[0]

    @abc.abstractmethod
    def as_vector(self):
        """
        Returns a flattened representation of the object as a single
        vector.

        Returns
        -------
        vector : (N,) ndarray
            The core representation of the object, flattened into a
            single vector.
        """
        pass

    @abc.abstractmethod
    def from_vector_inplace(self, vector):
        """
        Update the state of this object from it's vectorized state

        Parameters
        ----------
        vector : (N,) ndarray
            Flattened representation of this object.
        """
        pass

    def from_vector(self, vector):
        """
        Build a new instance of the object from it's vectorized state.


        `self` is used to fill out the missing state required to
        rebuild a full object from it's standardized flattened state. This
        is the default implementation, which is which is a
        `deepcopy` of the object followed by a call to
        :meth:`from_vector_inplace()`. This method can be overridden for a
        performance benefit if desired.

        Parameters
        ----------
        vector : (N,) ndarray
            Flattened representation of the object.

        Returns
        -------
        object : :class:`Vectorizable`
            An instance of the class.
        """
        self_copy = deepcopy(self)
        self_copy.from_vector_inplace(vector)
        return self_copy


class Targetable(object):
    r"""
    Interface for objects that can produce a 'target' PointCloud - which
    could for instance be the result of an alignment or a generation of a
    PointCloud instance from a shape model.

    Implementations must define sensible behavior for:

     - what a target is: target property
     - how to set a target: _target_setter
     - how to update the object after a target is set: _sync_state_from_target
     - how to produce a new target after the changes: _new_target_from_state

    Note that _sync_target_from_state() needs to be triggered as appropriate by
    subclasses e.g. when from_vector_inplace is called. This will in turn
    trigger _new_target_from_state(), which each subclass must implement.
    """
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

    def set_target(self, value):
        r"""
        Updates this alignment transform to point to a new target.
        """
        self._target_setter_with_verification(value)  # trigger the update
        self._sync_state_from_target()  # and a sync

    def _target_setter_with_verification(self, value):
        r"""
        Updates the target, checking it is sensible, without triggering a sync.

        Should be called by _sync_target_from_state with the new target value.
        """
        self._verify_target(value)
        self._target_setter(value)

    def _verify_target(self, new_target):
        # If the target is None (i.e. on construction) then dodge the
        # verification
        if self.target is None:
            return
        if new_target.n_dims != self.target.n_dims:
            raise ValueError(
                "The current target is {}D, the new target is {}D - new "
                "target has to have the same dimensionality as the "
                "old".format(self.target.n_dims, new_target.n_dims))
        elif new_target.n_points != self.target.n_points:
            raise ValueError(
                "The current target has {} points, the new target has {} "
                "- new target has to have the same number of points as the"
                " old".format(self.target.n_points, new_target.n_points))

    @abc.abstractmethod
    def _target_setter(self, new_target):
        r"""
        Sets the target to the new value. Does no synchronization. Note that
        it is advisable that _target_setter_with_verification is called from
        subclasses instead of this.
        """
        pass

    def _sync_target_from_state(self):
        new_target = self._new_target_from_state()
        self._target_setter_with_verification(new_target)

    @abc.abstractmethod
    def _new_target_from_state(self):
        r"""
        Returns a new target that is correct after changes to the object.
        """
        pass

    @abc.abstractmethod
    def _sync_state_from_target(self):
        r"""
        Synchronizes the object state to be correct after changes to the target.

        Called automatically from the target setter. This is called after the
        target is updated - only handle synchronization here.
        """
        pass


class DP(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def d_dp(self, points):
        r"""
        The derivative of this spatial object wrt parametrization changes
        evaluated at points.

        Parameters
        ----------

        points: ndarray of shape (n_points, n_dims)
            The spatial points at which the derivative should be evaluated.

        Returns
        -------

        ndarray of shape (n_points, n_params, n_dims)
            The jacobian wrt parameterization

        """


class DX(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def d_dx(self, points):
        r"""
        The first order derivative of this spatial object wrt spatial changes
        evaluated at points.

        Parameters
        ----------

        points: ndarray shape (n_points, n_dims)
            The spatial points at which the derivative should be evaluated.

        Returns
        -------

        d_dx: ndarray shape (n_points, n_dims, n_dims)
            The jacobian wrt spatial changes.

            d_dx[i, j, k] is the scalar differential change that the
            j'th dimension of the i'th point experiences due to a first order
            change in the k'th dimension.

            It may be the case that the jacobian is constant across space -
            in this case axis zero may have shape 1 to allow for broadcasting.

        """


class DL(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def d_dl(self, points):
        r"""
        The derivative of this spatial object wrt spatial changes in anchor
        landmark points or centres, evaluated at points.

        Parameters
        ----------

        points: ndarray shape (n_points, n_dims)
            The spatial points at which the derivative should be evaluated.

        Returns
        -------

        d_dl: ndarray shape (n_points, n_centres, n_dims)
            The jacobian wrt landmark changes.

            d_dl[i, k, m] is the scalar differential change that the
            any dimension of the i'th point experiences due to a first order
            change in the m'th dimension of the k'th landmark point.

            Note that at present this assumes that the change in every dimension
            is equal.
        """


def menpo_src_dir_path():
    return os.path.split(os.path.abspath(__file__))[0]
