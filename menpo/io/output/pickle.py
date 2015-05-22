from contextlib import contextmanager
from pathlib import Path, PurePath
try:
    import cPickle as pickle  # request cPickle manually on Py2
except ImportError:  # Py3
    import pickle


# -------------- Custom pickle behavior for pathlib.Path objects ------------ #
#
# We make heavy use of pathlib.Path throughout Menpo - most obviously, any
# imported item has it's path automatically duck-typed on at self.path.
#
# This causes issues with serialization however. pathlib has two families of
# paths - PurePaths, which you can perform manipulation on but have no
# connection to the file system, and concrete paths - paths which additionally
# have functional methods on them like .is_dir() and .mkdir(). You are not
# allowed to instantiate a concrete path for one platform (e.g. a WindowsPath)
# on a different platform (e.g. Linux - which is a PosixPath) as the additional
# filesystem methods will not make sense.
#
# As we attach concrete paths liberally in Menpo, many Menpo objects
# serialized on one platform (e.g. OS X) will not be unpickable on another
# (e.g. Windows).
#
# To alleviate this issue, we override the pickling behavior of concrete paths
# so that they pickle out as PurePaths. This ensures you can always open
# pickled objects on other platforms. We do this using a context manager, so
# don't effect the pickle behavior of these objects globally.
#
def _pure_path_reduce(self):
    # Pickled paths should go to pure paths so pickles are
    # useful across different OSes
    return PurePath(self).__class__, tuple(self.parts)


@contextmanager
def pickle_paths_as_pure():
    r"""
    Pickle pathlib.Path subclasses as their corresponding pathlib.PurePath
    """
    # save out the original method
    default_reduce = Path.__reduce__
    # switch out to our PurePath varient
    Path.__reduce__ = _pure_path_reduce
    try:
        yield
    finally:
        # always clean up - restore to default behavior
        Path.__reduce__ = default_reduce


def pickle_export(obj, file_handle):
    with pickle_paths_as_pure():
        pickle.dump(obj, file_handle, protocol=2)
