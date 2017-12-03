import sys
try:
    import cPickle as pickle
except ImportError:
    import pickle
import gzip


def _unpickle_with_encoding(f, encoding=None):
    # Support the encoding kwarg on Python 3.x only.
    if encoding is not None and sys.version_info.major > 2:
        return pickle.load(f, encoding=encoding)
    else:
        return pickle.load(f)


def pickle_importer(filepath, asset=None, **kwargs):
    r"""Import a pickle file.

    Parameters
    ----------
    filepath : `Path`
        Absolute filepath of the file.
    asset : `object`, optional
        An optional asset that may help with loading. This is unused for this
        implementation.
    \**kwargs : `dict`, optional
        Any other keyword arguments.

    Returns
    -------
    object : `object`
        The pickled objects.
    """
    with filepath.open('rb') as f:
        x = _unpickle_with_encoding(f, encoding=kwargs.get('encoding'))
    return x


def pickle_gzip_importer(filepath, asset=None, **kwargs):
    r"""Import a pickle file that has been compressed with GZip compression.

    Parameters
    ----------
    filepath : `Path`
        Absolute filepath of the file.
    asset : `object`, optional
        An optional asset that may help with loading. This is unused for this
        implementation.
    \**kwargs : `dict`, optional
        Any other keyword arguments.

    Returns
    -------
    object : `object`
        The pickled objects.
    """
    with gzip.open(str(filepath), 'rb') as f:
        x = _unpickle_with_encoding(f, encoding=kwargs.get('encoding'))
    return x
