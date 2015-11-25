from functools import partial, wraps
import os.path
import warnings


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
        new = self.__class__.__new__(self.__class__)
        for k, v in self.__dict__.items():
            try:
                new.__dict__[k] = v.copy()
            except AttributeError:
                new.__dict__[k] = v
        return new


class Vectorizable(Copyable):
    """
    Flattening of rich objects to vectors and rebuilding them back.

    Interface that provides methods for 'flattening' an object into a
    vector, and restoring from the same vectorized form. Useful for
    statistical analysis of objects, which commonly requires the data
    to be provided as a single vector.
    """

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
        raise NotImplementedError()

    def from_vector_inplace(self, vector):
        """
        Deprecated. Use the non-mutating API, :map:`from_vector`.

        For internal usage in performance-sensitive spots,
        see `_from_vector_inplace()`

        Parameters
        ----------
        vector : ``(n_parameters,)`` `ndarray`
            Flattened representation of this object
        """
        warnings.warn('the public API for inplace operations is deprecated '
                      'and will be removed in a future version of Menpo. '
                      'Use .from_vector() instead.', MenpoDeprecationWarning)
        return self._from_vector_inplace(vector)

    def _from_vector_inplace(self, vector):
        """
        Update the state of this object from a vector form.

        Parameters
        ----------
        vector : ``(n_parameters,)`` `ndarray`
            Flattened representation of this object
        """
        raise NotImplementedError()

    def from_vector(self, vector):
        """
        Build a new instance of the object from it's vectorized state.

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
        new._from_vector_inplace(vector)
        return new

    def has_nan_values(self):
        """
        Tests if the vectorized form of the object contains ``nan`` values or
        not. This is particularly useful for objects with unknown values that
        have been mapped to ``nan`` values.

        Returns
        -------
        has_nan_values : `bool`
            If the vectorized object contains ``nan`` values.
        """
        import numpy as np
        return np.any(np.isnan(self.as_vector()))


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

    @property
    def target(self):
        r"""The current :map:`PointCloud` that this object produces.

        :type: :map:`PointCloud`
        """
        raise NotImplementedError()

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
        raise NotImplementedError()

    def _sync_target_from_state(self):
        new_target = self._new_target_from_state()
        self._target_setter_with_verification(new_target)

    def _new_target_from_state(self):
        r"""Generate a new target that is correct after changes to the object.

        Returns
        -------
        object : ``type(self)``
        """
        raise NotImplementedError()

    def _sync_state_from_target(self):
        r"""Synchronizes the object state to be correct after changes to the
        target.

        Called automatically from the target setter. This is called after the
        target is updated - only handle synchronization here.
        """
        raise NotImplementedError()


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


class MenpoMissingDependencyError(Exception):
    r"""
    An exception that a dependency required for the requested functionality
    was not detected.
    """
    def __init__(self, package_name):
        super(MenpoMissingDependencyError, self).__init__()
        self.message = "You need to install the '{pname}' package in order " \
                       "to use this functionality. We recommend that you " \
                       "use conda to achieve this - try the command " \
                       "'conda install -c menpo {pname}' " \
                       "in your terminal.".format(pname=package_name)

    def __str__(self):
        return self.message


def name_of_callable(c):
    r"""
    Return the name of a callable (function or callable class) as a string.
    Recurses on partial function to attempt to find the wrapped
    methods actual name.

    Parameters
    ----------
    c : `callable`
        A callable class or function, or any valid Python object that can
        be wrapped with partial.

    Returns
    -------
    name : `str`
        The name of the passed object.
    """
    try:
        if isinstance(c, partial):  # partial
            # Recursively call as partial may be wrapping either a callable
            # or a function (or another partial for some reason!)
            return name_of_callable(c.func)
        else:
            return c.__name__  # function
    except AttributeError:
        return c.__class__.__name__  # callable class


class doc_inherit(object):
    """
    Docstring inheriting method descriptor.

    This uses some Python magic in order to create a decorator that implements
    the descriptor protocol that allows functions to inherit documentation.
    This is particularly useful for methods that directly override methods
    on their base class and simply alter the implementation but not the
    effective behaviour. Usage of this decorator is as follows:

        @doc_inherit()
        def foo():
            # Do something, but inherit the documentation from the method
            # called 'foo' found on the super() chain.

        @doc_inherit(name="foo2")
        def foo():
            # Do something, but inherit the documentation from the method
            # called 'foo2' found on the super() chain.

    When no argument is passed the name of the method being decorated is
    looked up on the ``super`` call chain.

    Parameters
    ----------
    name : `str`
        The name of the method to copy documentation from that exists somewhere
        on the ``super`` inheritance hierarchy.
    """

    def __init__(self, name=None):
        self.name = name

    def __call__(self, mthd):
        # Implementing the call method on a decorator allows the decorator
        # to recieve arguments in the constructor (__init__). Therefore,
        # the argument to the call method is always the method being wrapped.
        self.mthd = mthd
        # If name is None then default to the name of the method being wrapped.
        if self.name is None:
            self.name = self.mthd.__name__
        return self

    def __get__(self, obj, cls):
        # Implement the descriptor protocol. There are two different calling
        # strategies that involve whether the wrapped method has been passed
        # an instance or not.
        if obj:
            return self._get_with_instance(obj, cls)
        else:
            return self._get_with_no_instance(cls)

    def _get_with_instance(self, obj, cls):
        # An instance was passed, so lookup the name on the super chain
        overridden = getattr(super(cls, obj), self.name, None)

        # Return the wrapped method, passing through the arguments and the
        # object instance.
        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(obj, *args, **kwargs)

        return self._use_parent_doc(f, overridden)

    def _get_with_no_instance(self, cls):

        # This case is more complicated (than when an instance is passed). Here
        # we use reflection to try and lookup the method. When found, we drop
        # out the loop.
        for parent in cls.__mro__[1:]:
            overridden = getattr(parent, self.name, None)
            if overridden:
                break

        # Return the wrapped method, passing through the arguments and the
        # object instance.
        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(*args, **kwargs)

        return self._use_parent_doc(f, overridden)

    def _use_parent_doc(self, func, source):
        # Attach the documentation (unless the method was not found on the
        # super chain).
        if source is None:
            raise NameError("Can't find '{}' in parents".format(self.name))
        func.__doc__ = source.__doc__
        return func
