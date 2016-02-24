import gzip
from functools import partial
from pathlib import Path

from menpo.compatibility import basestring, str
from .extensions import landmark_types, image_types, pickle_types, video_types
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
    fp : `Path` or `file`-like object
        The Path or file-like object to save the object at/into.
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
    a `Path` or any Python type that acts like a file. If a file is provided,
    the ``extension`` kwarg **must** be provided. If no
    ``extension`` is provided and a `str` filepath is provided, then
    the export type is calculated based on the filepath extension.

    Due to the mix of string and file types, an explicit overwrite argument is
    used which is ``False`` by default.

    Parameters
    ----------
    image : :map:`Image`
        The image to export.
    fp : `Path` or `file`-like object
        The Path or file-like object to save the object at/into.
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


def export_video(images, filepath, overwrite=False, fps=30, **kwargs):
    r"""
    Exports a given list of images as a video. The ``filepath`` argument is
    a `Path` representing the path to save the video to. At this time,
    it is not possible to export videos directly to a file buffer.

    Due to the mix of string and file types, an explicit overwrite argument is
    used which is ``False`` by default.

    Note that exporting of GIF images is also supported.

    Parameters
    ----------
    images : list of :map:`Image`
        The images to export as a video.
    filepath : `Path`
        The Path to save the video at. File buffers are not supported, unlike
        other exporting formats.
    overwrite : `bool`, optional
        Whether or not to overwrite a file if it already exists.
    fps : `int`, optional
        The number of frames per second.
    **kwargs : `dict`, optional
        Extra parameters that are passed through directly to the exporter.
        Please see the documentation in the ``menpo.io.output.video`` package
        for information about the supported arguments.

    Raises
    ------
    ValueError
        File already exists and ``overwrite`` != ``True``
    ValueError
        The input is a file buffer and not a valid `Path`
    ValueError
        The provided extension does not match to an existing exporter type
        (the output type is not supported).
    """
    exporter_kwargs = {'fps': fps}
    exporter_kwargs.update(kwargs)
    path_filepath = _validate_filepath(Path(filepath), overwrite)
    extension = _parse_and_validate_extension(path_filepath, None, video_types)

    export_function = _extension_to_export_function(extension, video_types)
    export_function(images, path_filepath, **exporter_kwargs)


def export_pickle(obj, fp, overwrite=False, protocol=2):
    r"""
    Exports a given collection of Python objects with Pickle.

    The ``fp`` argument can be either a `Path` or any Python type that acts like
    a file.
    If ``fp`` is a path, it must have the suffix `.pkl` or `.pkl.gz`. If
    `.pkl`, the object will be pickled using Pickle protocol 2 without
    compression. If `.pkl.gz` the object will be pickled using Pickle protocol
    2 with gzip compression (at a fixed compression level of 3).

    Note that a special exception is made for `pathlib.Path` objects - they
    are pickled down as a `pathlib.PurePath` so that pickles can be easily
    moved between different platforms.

    Parameters
    ----------
    obj : ``object``
        The object to export.
    fp : `Path` or `file`-like object
        The string path or file-like object to save the object at/into.
    overwrite : `bool`, optional
        Whether or not to overwrite a file if it already exists.
    protocol : `int`, optional
        The Pickle protocol used to serialize the file.
        The protocols were introduced in different versions of python, thus
        it is recommended to save with the highest protocol version that
        your python distribution can support.
        The protocol refers to:

        ========= =========================================================
        Protocol                       Functionality
        ========= =========================================================
        0         Simplest protocol for text mode, backwards compatible.
        1         Protocol for binary mode, backwards compatible.
        2         Wider support for classes, compatible with python >= 2.3.
        3         Support for byte objects, compatible with python >= 3.0.
        4         Support for large objects, compatible with python >= 3.4.
        ========= =========================================================
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
    if isinstance(fp, basestring):
        fp = Path(fp)  # cheeky conversion to Path to reuse existing code
    if isinstance(fp, Path):
        # user provided a path - if it ended .gz we will compress
        path_filepath = _validate_filepath(fp, overwrite)
        extension = _parse_and_validate_extension(path_filepath, None,
                                                  pickle_types)
        o = gzip_open if extension[-3:] == '.gz' else open
        with o(str(path_filepath), 'wb') as f:
            # force overwrite as True we've already done the check above
            _export(obj, f, pickle_types, extension, True, protocol=protocol)
    else:
        _export(obj, fp, pickle_types, '.pkl', overwrite, protocol=protocol)


def _normalise_extension(extension):
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


def _extension_to_export_function(extension, extensions_map):
    r"""
    Simple function that wraps the extensions map indexing and raises
    a user friendly ``ValueError``

    Parameters
    ----------
    extension : `str`
        The string extension with period prefix e.g '.jpg'
    extensions_map : `dict` of `str` -> `callable`
        The extension map that maps extensions to export callables.

    Returns
    -------
    mapping_callable : `callable`
        The callable that performs exporting.

    Raises
    ------
    ValueError
        If ``extensions_map`` does not contain ``extension``. More friendly
        than the ``KeyError`` that would be raised.
    """
    # This just ensures that a sensible, user friendly Exception is raised.
    try:
        return extensions_map[extension]
    except KeyError:
        raise ValueError('The output file extension ({}) provided is not '
                         'currently supported.'.format(extension))


def _validate_filepath(fp, overwrite):
    r"""
    Normalise a given file path and ensure that ``overwrite == True`` if the
    file path exists. Normalisation involves things like making the given
    path absolute and expanding environment variables and user variables.

    Parameters
    ----------
    fp : `Path`
        The file path.
    overwrite : `bool`
        Whether the export method should override an existing file at the
        file path.

    Returns
    -------
    normalised_filepath : `Path`
        The normalised file path.

    Raises
    ------
    ValueError
        If ``overwrite == False`` and a file already exists at the file path.
    """
    path_filepath = _norm_path(fp)
    if path_filepath.exists() and not overwrite:
        raise ValueError('File already exists. Please set the overwrite '
                         'kwarg if you wish to overwrite the file.')
    return path_filepath


def _parse_and_validate_extension(path_filepath, extension, extensions_map):
    r"""
    If an extension is given, validate that the given file path matches
    the given extension.

    If not, parse the file path and return a correct extension. This function
    will handle cases such as file names with periods in.

    Parameters
    ----------
    path_filepath : `Path`
        The file path (normalised).
    extension : `str`
        The extension provided by the user.
    extensions_map : `dict` of `str` -> `callable`
        A dictionary mapping extensions to export callables.

    Returns
    -------
    norm_extension : `str`
        The correct extension, with leading period.

    Raises
    ------
    ValueError
        Unknown extension.
    ValueError
        File path contains extension that does not EXACTLY match the users'
        provided extension.
    """
    # If an explicit extension is passed, it must match exactly. However, file
    # names may contain periods, and therefore we need to try and parse
    # a known extension from the given file path.
    suffixes = path_filepath.suffixes
    i = 1
    while i < len(suffixes) + 1:
        try:
            suffix = ''.join(suffixes[-i:])
            _extension_to_export_function(suffix, extensions_map)
            known_extension = suffix
            break
        except ValueError:
            pass
        i += 1
    else:
        raise ValueError('Unknown file extension passed: ({})'.format(
            ''.join(suffixes)))

    if extension is not None:
        extension = _normalise_extension(extension)
        if extension != known_extension:
            raise ValueError('The file path extension must match the '
                             'requested file extension: ({}) != ({}).'.format(
                               extension, known_extension))
        known_extension = extension
    return known_extension


def _export(obj, fp, extensions_map, extension, overwrite, protocol=None):
    r"""
    The shared export function. This handles the shared logic of ensuring
    that the given ``fp`` is either a ``pathlib.Path`` or a file like
    object. All exporter methods are defined as receiving a buffer object,
    regardless of if a path is provided. If a file-like object is provided
    then the extension mut not be ``None``.

    Parameters
    ----------
    obj : `object`
        The Python object to export.
    fp : `Path` or file-like object
        The path or file buffer to write to.
    extensions_map : `dict` of `str` -> `callable`
        The dictionary mapping extensions to export callables.
    extension : `str`
        User provided extension (required if a file-like ``fp`` is passed).
    overwrite : `bool`
        If ``True``, overwrite any existing files at the given path.
    protocol : `int`, optional
        The output pickle protocol, if necessary.
    """
    if isinstance(fp, basestring):
        fp = Path(fp)  # cheeky conversion to Path to reuse existing code
    if isinstance(fp, Path):
        path_filepath = _validate_filepath(fp, overwrite)
        extension = _parse_and_validate_extension(path_filepath, extension,
                                                  extensions_map)

        export_function = _extension_to_export_function(extension,
                                                        extensions_map)

        with path_filepath.open('wb') as file_handle:
            export_function(obj, file_handle, protocol=protocol,
                            extension=extension)
    else:
        # You MUST provide an extension if a file handle is given
        if extension is None:
            raise ValueError('An export file extension must be provided if a '
                             'file-like object is passed.')
        else:
            extension = _normalise_extension(extension)

        # Apparently in Python 2.x there is no reliable way to detect something
        # that is 'file' like (file handle or a StringIO object or something
        # you can read and write to like a file). Therefore, we are going to
        # just be really Pythonic about it and just assume we were handed
        # a correctly behaving object.
        try:
            # Follow PIL like behaviour. Check the file handle extension
            # and check if matches the given extension
            filepath = Path(fp.name)
            _validate_filepath(filepath, overwrite)
            extension = _parse_and_validate_extension(filepath, extension,
                                                      extensions_map)
        except AttributeError:
            pass

        export_function = _extension_to_export_function(extension,
                                                        extensions_map)
        export_function(obj, fp, protocol=protocol, extension=extension)
