from pathlib import Path

from .extensions import landmark_types, image_types
from ..utils import _norm_path


def export_landmark_file(landmark_group, filepath, export_extension=None,
                         overwrite=False):
    r"""
    Exports a given landmark group. The ``filepath`` argument can be either
    or a `str` or any Python type that acts like a file. If a file is provided,
    the ``export_extension`` kwarg **must** be provided. If no
    ``export_extension`` is provided and a `str` filepath is provided, then
    the export type is calculated based on the filepath extension.

    Due to the mix in string and file types, an explicit overwrite argument is
    used which is ``False`` by default.

    Parameters
    ----------
    landmark_group : :map:`LandmarkGroup`
        The landmark group to export.
    filepath : `str` or `file`-like object
        The string path or file-like object to save the object at/into.
    export_extension : `str` or None, optional
        The export extension to use, this must match the file path if the file
        path is a string. Determines the type of exporter that is used.
    overwrite : `bool`, optional
        Whether or not to overwrite a file if it already exists.

    Raises
    ------
    ValueError
        File already exists and ``overwrite`` != ``True``
    ValueError
        ``filepath`` is a `str` and the ``export_extension`` is not ``None``
        and the two extensions do not match
    ValueError
        ``filepath`` is a `file`-like object and ``export_extension`` is
        ``None``
    ValueError
        The provided extension does not match to an existing exporter type
        (the output type is not supported).
    """
    _export(landmark_group, filepath, landmark_types, export_extension,
            overwrite)


def export_image(image, filepath, export_extension=None, overwrite=False):
    r"""
    Exports a given image. The ``filepath`` argument can be either
    or a `str` or any Python type that acts like a file. If a file is provided,
    the ``export_extension`` kwarg **must** be provided. If no
    ``export_extension`` is provided and a `str` filepath is provided, then
    the export type is calculated based on the filepath extension.

    Due to the mix in string and file types, an explicit overwrite argument is
    used which is ``False`` by default.

    Parameters
    ----------
    image : :map:`Image` or subclass
        The image to export.
    filepath : `str` or `file`-like object
        The string path or file-like object to save the object at/into.
    export_extension : `str` or None, optional
        The export extension to use, this must match the file path if the file
        path is a string. Determines the type of exporter that is used.
    overwrite : `bool`, optional
        Whether or not to overwrite a file if it already exists.

    Raises
    ------
    ValueError
        File already exists and ``overwrite`` != ``True``
    ValueError
        ``filepath`` is a `str` and the ``export_extension`` is not ``None``
        and the two extensions do not match
    ValueError
        ``filepath`` is a `file`-like object and ``export_extension`` is
        ``None``
    ValueError
        The provided extension does not match to an existing exporter type
        (the output type is not supported).
    """
    _export(image, filepath, image_types, export_extension, overwrite)


def _normalise_extension(extension):
    # Account for the fact the user may only have passed the extension
    # without the proceeding period
    if extension[0] is not '.':
        extension = '.' + extension
    return extension.lower()


def _extension_to_export_function(file_extension, extensions_map):
    try:
        file_extension = _normalise_extension(file_extension)
        return extensions_map[file_extension.lower()]
    except KeyError:
        raise ValueError('The output file extension provided is not currently '
                         'supported.')


def _validate_filepath(filepath, file_extension, overwrite):
    path_filepath = Path(_norm_path(filepath))
    if path_filepath.exists() and not overwrite:
        raise ValueError('File already exists. Please set the overwrite '
                         'kwarg if you wish to overwrite the file.')
    if file_extension is not None:
        filepath_suffix = path_filepath.suffix
        if _normalise_extension(file_extension) != filepath_suffix:
            raise ValueError('The file path extension must match the '
                             'requested file extension.')
    return path_filepath


def _export(obj, filepath, extensions_map, file_extension=None,
            overwrite=False):
    if isinstance(filepath, basestring):
        path_filepath = _validate_filepath(filepath, file_extension, overwrite)

        export_function = _extension_to_export_function(
            path_filepath.suffix, extensions_map)

        with path_filepath.open('wb') as file_handle:
            export_function(obj, file_handle)
    else:
        # You MUST provide an export function if a file handle is given
        if file_extension is None:
            raise ValueError('An export file extension must be provided if a '
                             'file-like object is passed.')
        # Apparently in Python 2.x there is no reliable way to detect something
        # that is 'file' like (file handle or a StringIO object or something
        # you can read and write to like a file). Therefore, we are going to
        # just be really Pythonic about it and just assume we were handed
        # a correctly behaving object.
        try:
            # Follow PIL like behaviour. Check the file handle extension
            # and check if matches the given file_extension
            _validate_filepath(filepath.name, file_extension, overwrite)
        except AttributeError:
            pass
        export_function = _extension_to_export_function(
            _normalise_extension(file_extension), extensions_map)
        export_function(obj, filepath)
