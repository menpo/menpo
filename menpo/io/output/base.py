import gzip
import warnings
from functools import partial
from pathlib import Path

from menpo.compatibility import basestring, str
from .extensions import landmark_types, image_types, pickle_types, video_types
from ..exceptions import OverwriteError
from ..utils import (_norm_path, _possible_extensions_from_filepath,
                     _normalize_extension)

# an open file handle that uses a small fast level of compression
gzip_open = partial(gzip.open, compresslevel=3)


def export_landmark_file(landmarks_object, fp, extension=None, overwrite=False):
    r"""
    Exports a given shape. The ``fp`` argument can be either or a `str` or
    any Python type that acts like a file. If a file is provided, the
    ``extension`` kwarg **must** be provided. If no ``extension`` is provided
    and a `str` filepath is provided, then the export type is calculated
    based on the filepath extension.

    Due to the mix in string and file types, an explicit overwrite argument is
    used which is ``False`` by default.

    Parameters
    ----------
    landmarks_object : dict or :map:`LandmarkManager`  or
        :map:`PointCloud` or subclass of :map:`PointCloud`
        The landmarks to export. The type of :map:`PointCloud` or
        subclass of it are supported by all exporters, while the
        rest are available only for the LJSON format.
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
    ValueError
        The provided type for landmarks_object is not supported.
    """
    extension = _normalize_extension(extension)

    try:
        landmarks_object.n_points
    except AttributeError:
        # unless this is LJSON, this is not correct.
        fp_is_path = isinstance(fp, basestring) or isinstance(fp, Path)
        if (extension is not None and extension != '.ljson') or \
                (fp_is_path and Path(fp).suffix != '.ljson'):
            m1 = ('Only the LJSON format supports multiple '
                  'keys for exporting. \nIn any other '
                  'case your input should be a PointCloud or '
                  'subclass.')
            raise ValueError(m1)
    _export(landmarks_object, fp, landmark_types, extension, overwrite)


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


def export_video(images, file_path, overwrite=False, fps=30, **kwargs):
    r"""
    Exports a given list of images as a video. Ensure that all the images
    have the same shape, otherwise you might get unexpected results from
    the ffmpeg writer. The ``file_path`` argument is a `Path` representing
    the path to save the video to. At this time, it is not possible
    to export videos directly to a file buffer.

    Due to the mix of string and file types, an explicit overwrite argument is
    used which is ``False`` by default.

    Note that exporting of GIF images is also supported.

    Parameters
    ----------
    images : list of :map:`Image`
        The images to export as a video.
    file_path : `Path`
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
        The input is a buffer and not a valid `Path`
    ValueError
        The provided extension does not match to an existing exporter type
        (the output type is not supported).
    """
    exporter_kwargs = {'fps': fps}
    exporter_kwargs.update(kwargs)

    file_path = _enforce_only_paths_supported(file_path, 'FFMPEG')
    _export_paths_only(images, file_path, video_types, None, overwrite,
                       exporter_kwargs=exporter_kwargs)


def export_pickle(obj, fp, overwrite=False, protocol=2):
    r"""
    Exports a given collection of Python objects with Pickle.

    The ``fp`` argument can be either a `Path` or any Python type that acts like
    a file.
    If ``fp`` is a path, it must have the suffix `.pkl` or `.pkl.gz`. If
    `.pkl`, the object will be pickled using the selected Pickle protocol.
    If `.pkl.gz` the object will be pickled using the selected Pickle
    protocol with gzip compression (at a fixed compression level of 3).

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
    exporter_kwargs = {'protocol': protocol}
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
            _export(obj, f, pickle_types, extension, True,
                    exporter_kwargs=exporter_kwargs)
    else:
        _export(obj, fp, pickle_types, '.pkl', overwrite,
                exporter_kwargs=exporter_kwargs)


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
    Normalize a given file path and ensure that ``overwrite == True`` if the
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
    normalized_filepath : `Path`
        The normalized file path.

    Raises
    ------
    OverwriteError
        If ``overwrite == False`` and a file already exists at the file path.
    """
    path_filepath = _norm_path(fp)
    if path_filepath.exists() and not overwrite:
        raise OverwriteError('File {} already exists. Please set the overwrite '
                             'kwarg if you wish to overwrite '
                             'the file.'.format(path_filepath.name),
                             path_filepath)
    return path_filepath


def _parse_and_validate_extension(filepath, extension, extensions_map):
    r"""
    If an extension is given, validate that the given file path matches
    the given extension.

    If not, parse the file path and return a correct extension. This function
    will handle cases such as file names with periods in.

    Parameters
    ----------
    filepath : `Path`
        The file path (normalized).
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
    possible_exts = _possible_extensions_from_filepath(filepath)

    known_extension = None
    while known_extension is None and possible_exts:
        possible_extension = possible_exts.pop(0)
        if possible_extension in extensions_map:
            known_extension = possible_extension

    if known_extension is None:
        raise ValueError('Unknown file extension passed: {}'.format(
            ''.join(filepath.suffixes)))

    if extension is not None:
        extension = _normalize_extension(extension)
        if extension != known_extension:
            raise ValueError('The file path extension must match the '
                             'requested file extension: {} != {}'.format(
                               extension, known_extension))

    return known_extension


def _enforce_only_paths_supported(file_path, exporter_name):
    r"""
    If a given exporter only supports paths rather than open file handles
    or buffers then this function can be used to enforce that. If a file
    handle is passed then an attempt is made to write to the path of the file
    handle.

    Parameters
    ----------
    file_path : `str` or `pathlib.Path` or file-like object
        The file path to write to.

    Returns
    -------
    file_path : `str`
        The path to open file handle or, if a path was passed, it is returned
        unchanged.

    Raises
    ------
    ValueError
        If given ``file_path`` is not a string, pathlib.Path or file handle.
    """
    if hasattr(file_path, 'name') and not isinstance(file_path, Path):
        file_path = file_path.name
        warnings.warn('The {} exporter only supports file paths and not '
                      'buffers or open file handles - therefore the provided '
                      'file handle will be ignored and the object will be '
                      'exported to {}.'.format(exporter_name, file_path))
    if isinstance(file_path, basestring) or isinstance(file_path, Path):
        return file_path
    else:
        raise ValueError('Cannot write to unnamed file handles or buffers.')


def _validate_and_get_export_func(file_path, extensions_map, extension,
                                  overwrite, return_extension=False):
    r"""
    Given a ``file_path``, ensure that the options chosen are valid with respect
    to overwriting and any provided extensions. If this validation is
    successful then the exporter function is returned.

    Parameters
    ----------
    file_path : `Path`
        The path to write to.
    extensions_map : `dict` of `str` -> `callable`
        The dictionary mapping extensions to export callables.
    extension : `str`
        User provided extension (required if a file-like ``fp`` is passed).
    overwrite : `bool`
        If ``True``, overwrite any existing files at the given path.
    return_extension : `bool`, optional
        If ``True``, return the correct extension as well as the export
        callable, as a tuple ``(callable, extension)``.

    Returns
    -------
    exporter_callable : `callable`
        The exporter callable.
    extension : `str`
        The correct extension for the exporter function, if
        ``return_extension==True``.
    """
    if isinstance(file_path, basestring):
        # cheeky conversion to Path to reuse existing code
        file_path = Path(file_path)

    file_path = _validate_filepath(file_path, overwrite)
    extension = _parse_and_validate_extension(file_path, extension,
                                              extensions_map)
    export_callable = _extension_to_export_function(extension, extensions_map)

    if return_extension:
        return export_callable, extension
    else:
        return export_callable


def _export_paths_only(obj, file_path, extensions_map, extension, overwrite,
                       exporter_kwargs=None):
    r"""
    A shared export function handling paths only. This handles the logic
    of ensuring that the given ``file_path`` is a ``pathlib.Path``. All exporter
    methods that are called from here are defined as receiving a
    ``pathlib.Path``.

    Parameters
    ----------
    obj : `object`
        The Python object to export.
    file_path : `Path`
        The path to write to.
    extensions_map : `dict` of `str` -> `callable`
        The dictionary mapping extensions to export callables.
    extension : `str`
        User provided extension (required if a file-like ``fp`` is passed).
    overwrite : `bool`
        If ``True``, overwrite any existing files at the given path.
    exporter_kwargs : `int`, optional
        Any kwargs to be passed through to the exporter.
    """
    if exporter_kwargs is None:
        exporter_kwargs = {}
    export_function = _validate_and_get_export_func(file_path, extensions_map,
                                                    extension, overwrite)
    export_function(obj, file_path, **exporter_kwargs)


def _export(obj, fp, extensions_map, extension, overwrite,
            exporter_kwargs=None):
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
    exporter_kwargs : `int`, optional
        Any kwargs to be passed through to the exporter.
    """
    if exporter_kwargs is None:
        exporter_kwargs = {}
    if isinstance(fp, basestring):
        fp = Path(fp)  # cheeky conversion to Path to reuse existing code
    if isinstance(fp, Path):
        export_function, extension = _validate_and_get_export_func(
            fp, extensions_map, extension, overwrite, return_extension=True)

        with fp.open('wb') as file_handle:
            export_function(obj, file_handle, extension=extension,
                            **exporter_kwargs)
    else:
        # You MUST provide an extension if a file handle is given
        if extension is None:
            raise ValueError('An export file extension must be provided if a '
                             'file-like object is passed.')
        else:
            extension = _normalize_extension(extension)

        # Apparently in Python 2.x there is no reliable way to detect something
        # that is 'file' like (file handle or a StringIO object or something
        # you can read and write to like a file). Therefore, we are going to
        # just be really Pythonic about it and just assume we were handed
        # a correctly behaving object.
        try:
            # Follow PIL like behaviour. Check the file handle extension
            # and check if matches the given extension
            export_function = _validate_and_get_export_func(
                Path(fp.name), extensions_map, extension, overwrite)
        except AttributeError:
            # Just use the extension to get the export function
            export_function = _extension_to_export_function(extension,
                                                            extensions_map)

        export_function(obj, fp, extension=extension, **exporter_kwargs)
