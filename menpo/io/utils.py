import os
from pathlib import Path


def _norm_path(filepath):
    r"""
    Uses all the tricks in the book to expand a path out to an absolute one.
    """
    return Path(os.path.abspath(os.path.normpath(
        os.path.expandvars(os.path.expanduser(str(filepath))))))


def _possible_extensions_from_filepath(filepath):
    r"""
    Generate a list of possible extensions from the given filepath. Since
    filenames can contain '.' characters and some extensions are compound
    (e.g. '.pkl.gz'), there may be many possible extensions for a given
    path. Generate a list possible extensions, preferring longer extensions.

    Parameters
    ----------
    filepath : `Path`
        A pathlib Path.

    Returns
    -------
    possible_extensions : `list` of `str`
        A list of extensions **with** leading '.' characters and converted
        to lowercase.
    """
    suffixes = filepath.suffixes
    return [''.join(suffixes[i:]).lower() for i in range(len(suffixes))]


def _normalize_extension(extension):
    r"""
    Simple function that takes a given extension string and ensures that it
    is lower case and contains the leading period e.g. ('.jpg')

    Parameters
    ----------
    extension : `str`
        The string extension.

    Returns
    -------
    norm_extension : `str`
        The normalised extension, lower case with '.' prefix.
    """
    # Account for the fact the user may only have passed the extension
    # without the proceeding period
    if extension[0] is not '.':
        extension = '.' + extension
    return extension.lower()
