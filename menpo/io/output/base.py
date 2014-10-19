from pathlib import Path

from .extensions import landmark_types, image_types
from ..utils import _norm_path


def export_landmark_file(fp, landmark_group, extension=None,
                         overwrite=False):
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
    fp : `str` or `file`-like object
        The string path or file-like object to save the object at/into.
    landmark_group : :map:`LandmarkGroup`
        The landmark group to export.
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
    _export(fp, landmark_group, landmark_types, extension,
            overwrite)


def export_image(fp, image, extension=None, overwrite=False):
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
    fp : `str` or `file`-like object
        The string path or file-like object to save the object at/into.
    image : :map:`Image`
        The image to export.
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
    _export(fp, image, image_types, extension, overwrite)


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
        filepath_suffix = path_filepath.suffix
        if _normalise_extension(extension) != filepath_suffix:
            raise ValueError('The file path extension must match the '
                             'requested file extension.')
    return path_filepath


def _export(fp, obj, extensions_map, extension=None, overwrite=False):
    if isinstance(fp, basestring):
        path_filepath = _validate_filepath(fp, extension, overwrite)

        export_function = _extension_to_export_function(
            path_filepath.suffix, extensions_map)

        with path_filepath.open('wb') as file_handle:
            export_function(file_handle, obj)
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
        export_function(fp, obj)
