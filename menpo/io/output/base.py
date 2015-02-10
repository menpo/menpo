import gzip
from functools import partial
from pathlib import Path

from .extensions import landmark_types, image_types, pickle_types
from ..utils import _norm_path

# an open file handle that uses a small fast level of compression
gzip_open = partial(gzip.open, compresslevel=3)


def export_landmark_file(landmark_group, fp, extension=None, overwrite=False):
    r"""
    Exports a given landmark group. The ``fp`` argument can be either
    or a `str` or any Python type that acts like a file. If a file is provided,
    the ``extension`` kwarg **must** be provided. If no
    ``extension`` is provided and a `str` filepath is provided, then
    the export type is calculated based on the filepath extension.

    Due to the mix in string and file types, an explicit overwrite argument is
    used which is ``False`` by default.

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to export.
    fp : `str` or `file`-like object
        The string path or file-like object to save the object at/into.
    extension : `str` or None, optional
        The extension to use, this must match the file path if the file
        path is a string. Determines the type of exporter that is used.
    overwrite : `bool`, optional
        Whether or not to overwrite a file if it already exists.

    Raises
    ------
    ValueError
        File already exists and ``overwrite`` != ``True``
    ValueError
        ``fp`` is a `str` and the ``extension`` is not ``None``
        and the two extensions do not match
    ValueError
        ``fp`` is a `file`-like object and ``extension`` is
        ``None``
    ValueError
        The provided extension does not match to an existing exporter type
        (the output type is not supported).
    """
    _export(landmark_group, fp, landmark_types, extension, overwrite)


def export_image(image, fp, extension=None, overwrite=False):
    r"""
    Exports a given image. The ``fp`` argument can be either
    a `str` or any Python type that acts like a file. If a file is provided,
    the ``extension`` kwarg **must** be provided. If no
    ``extension`` is provided and a `str` filepath is provided, then
    the export type is calculated based on the filepath extension.

    Due to the mix of string and file types, an explicit overwrite argument is
    used which is ``False`` by default.

    Parameters
    ----------
    image : :map:`Image`
        The image to export.
    fp : `str` or `file`-like object
        The string path or file-like object to save the object at/into.
    extension : `str` or None, optional
        The extension to use, this must match the file path if the file
        path is a string. Determines the type of exporter that is used.
    overwrite : `bool`, optional
        Whether or not to overwrite a file if it already exists.

    Raises
    ------
    ValueError
        File already exists and ``overwrite`` != ``True``
    ValueError
        ``fp`` is a `str` and the ``extension`` is not ``None``
        and the two extensions do not match
    ValueError
        ``fp`` is a `file`-like object and ``extension`` is
        ``None``
    ValueError
        The provided extension does not match to an existing exporter type
        (the output type is not supported).
    """
    _export(image, fp, image_types, extension, overwrite)


def export_pickle(obj, fp, overwrite=False):
    r"""
    Exports a given collection of Python objects with Pickle.

    The ``fp`` argument can be either a `str` or any Python type that acts like
    a file.
    If ``fp`` is a path, it must have the suffix `.pkl` or `.pkl.gz`. If
    `.pkl`, the object will be pickled using Pickle protocol 2 without
    compression. If `.pkl.gz` the object will be pickled using Pickle protocol
    2 with gzip compression (at a fixed compression level of 3).

    Parameters
    ----------
    obj : ``object``
        The object to export.
    fp : `str` or `file`-like object
        The string path or file-like object to save the object at/into.
    overwrite : `bool`, optional
        Whether or not to overwrite a file if it already exists.

    Raises
    ------
    ValueError
        File already exists and ``overwrite`` != ``True``
    ValueError
        ``fp`` is a `file`-like object and ``extension`` is
        ``None``
    ValueError
        The provided extension does not match to an existing exporter type
        (the output type is not supported).
    """
    if isinstance(fp, Path):
        fp = str(fp)  # cheeky conversion to string to reuse existing code
    if isinstance(fp, basestring):
        # user provided a path - if it ended .gz we will compress
        path_filepath = _validate_filepath(fp, '.pkl', overwrite)
        o = gzip_open if path_filepath.suffix == '.gz' else open
        with o(fp, 'wb') as f:
            # force overwrite as True we've already done the check above
            _export(obj, f, pickle_types, '.pkl', True)
    else:
        _export(obj, fp, pickle_types, '.pkl', overwrite)


def _normalise_extension(extension):
    # Account for the fact the user may only have passed the extension
    # without the proceeding period
    if extension[0] is not '.':
        extension = '.' + extension
    return extension.lower()


def _extension_to_export_function(extension, extensions_map):
    try:
        extension = _normalise_extension(extension)
        return extensions_map[extension.lower()]
    except KeyError:
        raise ValueError('The output file extension provided is not currently '
                         'supported.')


def _validate_filepath(fp, extension, overwrite):
    path_filepath = Path(_norm_path(fp))
    if path_filepath.exists() and not overwrite:
        raise ValueError('File already exists. Please set the overwrite '
                         'kwarg if you wish to overwrite the file.')
    if extension is not None:
        # use .suffixes[0] to handle compression suffixes correctly (see below)
        filepath_suffix = path_filepath.suffixes[0]
        # we couldn't find an exporter for all the suffixes (e.g .foo.bar)
        # maybe the file stem has '.' in it? -> try again but this time just use the
        # final suffix (.bar). (Note we first try '.foo.bar' as we want to catch
        # cases like 'pkl.gz')
        if _normalise_extension(extension) != filepath_suffix and len(path_filepath.suffixes) > 1:
            filepath_suffix = path_filepath.suffix
        if _normalise_extension(extension) != filepath_suffix:
            raise ValueError('The file path extension must match the '
                             'requested file extension.')
    return path_filepath


def _export(obj, fp, extensions_map, extension, overwrite):
    if isinstance(fp, Path):
        fp = str(fp)  # cheeky conversion to string to reuse existing code
    if isinstance(fp, basestring):
        path_filepath = _validate_filepath(fp, extension, overwrite)

        export_function = _extension_to_export_function(
            path_filepath.suffix, extensions_map)

        with path_filepath.open('wb') as file_handle:
            export_function(obj, file_handle)
    else:
        # You MUST provide an extension if a file handle is given
        if extension is None:
            raise ValueError('An export file extension must be provided if a '
                             'file-like object is passed.')
        # Apparently in Python 2.x there is no reliable way to detect something
        # that is 'file' like (file handle or a StringIO object or something
        # you can read and write to like a file). Therefore, we are going to
        # just be really Pythonic about it and just assume we were handed
        # a correctly behaving object.
        try:
            # Follow PIL like behaviour. Check the file handle extension
            # and check if matches the given extension
            _validate_filepath(fp.name, extension, overwrite)
        except AttributeError:
            pass
        export_function = _extension_to_export_function(
            _normalise_extension(extension), extensions_map)
        export_function(obj, fp)
