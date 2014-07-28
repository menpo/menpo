from pathlib import Path

from .extensions import landmark_types, image_types
from ..utils import _norm_path


def export_landmark_file(landmark_group, filepath, export_extension=None,
                         overwrite=False):
    _export(landmark_group, filepath, landmark_types, export_extension,
            overwrite)


def export_image(image, filepath, export_extension=None, overwrite=False):
    _export(image, filepath, image_types, export_extension, overwrite)


def _normalise_extension(extension):
    # Account for the fact the user may only have passed the extension
    # without the proceeding period
    if extension[0] is not '.':
        extension = '.' + extension
    return extension.lower()


def _extension_to_export_function(file_extension, extensions_map,
                                  is_path=False):
    try:
        if is_path:
            file_extension = file_extension.suffix
        file_extension = _normalise_extension(file_extension)
        return extensions_map[file_extension.lower()]
    except KeyError:
        raise ValueError('The output file extension provided is not currently '
                         'supported.')


def _export(obj, filepath, extensions_map, file_extension=None,
            overwrite=False):
    if isinstance(filepath, basestring):
        filepath = Path(_norm_path(filepath))
        if filepath.exists() and not overwrite:
            raise ValueError('File already exists. Please set the overwrite '
                             'kwarg if you wish to overwrite the file.')
        # Little trick. You either passed us the extension or we try to find
        # it from the path. We only need to find it from the path if the
        # file_extension is ``None``
        export_function = _extension_to_export_function(
            filepath, extensions_map, file_extension is None)

        with open(str(filepath), 'wb') as file_handle:
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
            file_handle_extension = Path(filepath.name).suffix
            if file_handle_extension != _normalise_extension(file_extension):
                raise ValueError('The file handle has an extension '
                                 'that does not match the export_extension.')
        except AttributeError:
            pass
        export_function = _extension_to_export_function(file_extension,
                                                        extensions_map)
        export_function(obj, filepath)
