import abc
import os.path

# To debug the Copyable interface, simply uncomment lines 11-23 below and the
# four lines in the copy() method.
# Then you can call print_copyable_log() to see exactly what types have been
# skipped in copying and why.

# from collections import defaultdict
# alien_copies = defaultdict(set)
# non_copies = defaultdict(set)
#
#
# def print_copyable_log():
#     print('Has .copy() but not Copyable:')
#     for k, v in alien_copies.iteritems():
#         print('  {:15}|  {}'.format(k, ', '.join(v)))
#
#     print('\nNo .copy() (shallow copied):')
#     for k, v in non_copies.iteritems():
#         print('  {:15}|  {}'.format(k, ', '.join(v)))


class Copyable(object):
    """
    Efficient copying of classes containing numpy arrays.

    Interface that provides a single method for copying classes very
    efficiently.
    """

    def copy(self):
        r"""
        Generate an efficient copy of this object.

        Note that Numpy arrays and other :map:`Copyable` objects on ``self``
        will be deeply copied. Dictionaries and sets will be shallow copied,
        and everything else will be assigned (no copy will be made).

        Classes that store state other than numpy arrays and immutable types
        should overwrite this method to ensure all state is copied.

        Returns
        -------

        ``type(self)``
            A copy of this object

        """
        # print('copy called on {}'.format(type(self).__name__))
        new = self.__class__.__new__(self.__class__)
        for k, v in self.__dict__.iteritems():
            try:
                new.__dict__[k] = v.copy()
                # if not isinstance(v, Copyable):
                #     alien_copies[type(v).__name__].add(type(self).__name__)
            except AttributeError:
                new.__dict__[k] = v
                # non_copies[type(v).__name__].add(type(self).__name__)
        return new


class Vectorizable(Copyable):
    """
    Flattening of rich objects to vectors and rebuilding them back.

    Interface that provides methods for 'flattening' an object into a
    vector, and restoring from the same vectorized form. Useful for
    statistical analysis of objects, which commonly requires the data
    to be provided as a single vector.
    """

    __metaclass__ = abc.ABCMeta

    @property
    def n_parameters(self):
        r"""The length of the vector that this object produces.

        :type: `int`
        """
        return (self.as_vector()).shape[0]

    def as_vector(self, **kwargs):
        """
        Returns a flattened representation of the object as a single
        vector.

        Returns
        -------
        vector : (N,) ndarray
            The core representation of the object, flattened into a
            single vector. Note that this is always a view back on to the
            original object, but is not writable.
        """
        v = self._as_vector(**kwargs)
        v.flags.writeable = False
        return v

    @abc.abstractmethod
    def _as_vector(self, **kwargs):
        """
        Returns a flattened representation of the object as a single
        vector.

        Returns
        -------
        vector : ``(n_parameters,)`` `ndarray`
            The core representation of the object, flattened into a
            single vector.
        """

    @abc.abstractmethod
    def from_vector_inplace(self, vector):
        """Update the state of this object from a vector form.

        Parameters
        ----------
        vector : ``(n_parameters,)`` `ndarray`
            Flattened representation of this object
        """

    def from_vector(self, vector):
        """Build a new instance of the object from it's vectorized state.

        ``self`` is used to fill out the missing state required to
        rebuild a full object from it's standardized flattened state. This
        is the default implementation, which is which is a ``deepcopy`` of the
        object followed by a call to :meth:`from_vector_inplace()`. This method
        can be overridden for a performance benefit if desired.

        Parameters
        ----------
        vector : ``(n_parameters,)`` `ndarray`
            Flattened representation of the object.

        Returns
        -------
        object : ``type(self)``
            An new instance of this class.
        """
        new = self.copy()
        new.from_vector_inplace(vector)
        return new


class Targetable(Copyable):
    """Interface for objects that can produce a target :map:`PointCloud`.

    This could for instance be the result of an alignment or a generation of a
    :map:`PointCloud` instance from a shape model.

    Implementations must define sensible behavior for:

     - what a target is: see :attr:`target`
     - how to set a target: see :meth:`set_target`
     - how to update the object after a target is set:
       see :meth:`_sync_state_from_target`
     - how to produce a new target after the changes:
       see :meth:`_new_target_from_state`

    Note that :meth:`_sync_target_from_state` needs to be triggered as
    appropriate by subclasses e.g. when :map:`from_vector_inplace` is
    called. This will in turn trigger :meth:`_new_target_from_state`, which each
    subclass must implement.
    """
    __metaclass__ = abc.ABCMeta

    @property
    def n_dims(self):
        r"""The number of dimensions of the :attr:`target`.

        :type: `int`
        """
        return self.target.n_dims

    @property
    def n_points(self):
        r"""The number of points on the :attr:`target`.

        :type: `int`
        """
        return self.target.n_points

    @abc.abstractproperty
    def target(self):
        r"""The current :map:`PointCloud` that this object produces.

        :type: :map:`PointCloud`
        """

    def set_target(self, new_target):
        r"""
        Update this object so that it attempts to recreate the ``new_target``.

        Parameters
        ----------
        new_target : :map:`PointCloud`
            The new target that this object should try and regenerate.
        """
        self._target_setter_with_verification(new_target)  # trigger the update
        self._sync_state_from_target()  # and a sync

    def _target_setter_with_verification(self, new_target):
        r"""Updates the target, checking it is sensible, without triggering a
        sync.

        Should be called by :meth:`_sync_target_from_state` once it has
        generated a suitable target representation.

        Parameters
        ----------
        new_target : :map:`PointCloud`
            The new target that should be set.
        """
        self._verify_target(new_target)
        self._target_setter(new_target)

    def _verify_target(self, new_target):
        r"""Performs sanity checks to ensure that the new target is valid.

        This includes checking the dimensionality matches and the number of
        points matches the current target's values.

        Parameters
        ----------
        new_target : :map:`PointCloud`
            The target that needs to be verified.

        Raises
        ------
        ValueError
            If the ``new_target`` has differing ``n_points`` or ``n_dims`` to
            ``self``.
        """
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
        r"""Sets the target to the new value.

        Does no synchronization. Note that it is advisable that
        :meth:`_target_setter_with_verification` is called from
        subclasses instead of this.

        Parameters
        ----------
        new_target : :map:`PointCloud`
            The new target that will be set.
        """

    def _sync_target_from_state(self):
        new_target = self._new_target_from_state()
        self._target_setter_with_verification(new_target)

    @abc.abstractmethod
    def _new_target_from_state(self):
        r"""Generate a new target that is correct after changes to the object.

        Returns
        -------
        object : ``type(self)``
        """
        pass

    @abc.abstractmethod
    def _sync_state_from_target(self):
        r"""Synchronizes the object state to be correct after changes to the
        target.

        Called automatically from the target setter. This is called after the
        target is updated - only handle synchronization here.
        """
        pass


def menpo_src_dir_path():
    r"""The path to the top of the menpo Python package.

    Useful for locating where the data folder is stored.

    Returns
    -------
    path : ``pathlib.Path``
        The full path to the top of the Menpo package
    """
    from pathlib import Path  # to avoid cluttering the menpo.base namespace
    return Path(os.path.abspath(__file__)).parent


class MenpoDeprecationWarning(Warning):
    r"""
    A warning that functionality in Menpo will be deprecated in a future major
    release.
    """
    pass
