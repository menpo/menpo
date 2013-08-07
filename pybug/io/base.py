import abc
import os
from glob import glob
import sys


def auto_import(pattern, meshes=True, images=True,
                include_texture_images=False):
    r"""
    Smart data importer. Will match all files found on the glob pattern
    passed in, build the relevant importers, and then call ``build()`` on them
    to return a list of usable objects.

    :param pattern: The glob style pattern to search for textures and meshes.

        Examples::
            Look for all meshes that have file extension ``.obj``:

            >>> auto_import('*.obj')

            Look for all files that begin with the string ``test``:

            >>> auto_import('test.*')
    :type pattern: string
    :keyword meshes: Include mesh types in results
    :type meshes: bool
    :keyword images: Include image types in results
    :type images: bool
    :keyword include_texture_images: Check if the images it has found  in
        the glob pattern are actually textures of the meshes it found.
        If this is the case, it won't import these images separately.
    :type include_texture_images: bool
    """
    mesh_objects, image_objects = [], []
    if meshes:
        mesh_paths = _glob_matching_extension(pattern, mesh_types)
        mesh_objects, mesh_importers = _multi_mesh_import(mesh_paths,
                                                          keep_importers=True)
    if images:
        image_files = _glob_matching_extension(pattern, image_types)
        if meshes and not include_texture_images:
            texture_paths = [m.texture_path for m in mesh_importers
                             if m.texture_path is not None]
            image_files = _images_unrelated_to_meshes(image_files,
                                                      texture_paths)
            image_objects = _multi_image_import(image_files)

    return mesh_objects + image_objects


def _multi_image_import(image_filepaths, keep_importers=False):
    r"""
    Creates importers for all the image filepaths passed in,
    and then calls build on them, returning a list of
    :class:`Images <pybug.image.base.Image>`.

    :param image_filepaths: List of filepaths
    :type image_filepaths: list of strings
    :keyword keep_importers: Returns the
        :class:`Importers <pybug.io.base.Importer>` as well as the images
        themselves
    :type keep_importers: bool
    :return: list of images or tuple of (images, importers)
    :rtype: [:class:`Image <pybug.image.base.Image>`] or
            ([:class:`Images <pybug.image.base.Image>`],
            [:class:`Importers <pybug.io.base.Importer>`])
    """
    return _multi_import(image_filepaths, image_types, keep_importers)


def _multi_mesh_import(mesh_filepaths, keep_importers=False):
    """
    Creates importers for all the mesh filepaths passed in,
    and then calls build on them, returning a list of TriMeshes or
    TexturedTriMeshes.
    """
    result = _multi_import(mesh_filepaths, mesh_types, keep_importers)
    # meshes come back as a nested list - unpack this for convenience
    if keep_importers:
        meshes = result[0]
    else:
        meshes = result
    meshes = [mesh for mesh_grp in meshes for mesh in mesh_grp]
    if keep_importers:
        return meshes, result[1]
    else:
        return meshes


def map_filepaths_to_importers(filepaths, extensions_map):
    """
    Expects a list of filepaths
    :param filepaths:
    :param extensions_map:
    :return:
    """
    importers = []
    for f in sorted(filepaths):
        ext = os.path.splitext(f)[1]
        importer_type = extensions_map.get(ext)
        importers.append(importer_type(f))
    return importers


def find_extensions_from_basename(filepath):
    """
    Given a filepath, find all the files that share the same name. This is
    used to find all potential matching images and landmark files for a given
    file.
    :param filepath: An absolute filepath
    :return: A list of absolute filepaths to files that share the same basename
        as filepath
    """
    basename = os.path.splitext(os.path.basename(filepath))[0] + '*'
    basepath = os.path.join(os.path.dirname(filepath), basename)
    return glob(basepath)


def filter_extensions(filepaths, extensions_map):
    """
    Given a set of filepaths, filter the files who's extensions are in the
    given map. This used to find images and landmarks from a given basename.
    :param filepaths: A list of absolute filepaths
    :param extensions_map: A mapping from extensions to importers, where the
        keys are the extensions
    :return: A list of basenames
    """
    extensions = extensions_map.keys()
    return [os.path.basename(f) for f in filepaths
            if os.path.splitext(f)[1] in extensions]


def find_alternative_files(file_type, filepath, extension_map):
    """
    Given a filepath, search for files with the same basename that match
    a given extension type, eg images
    :param file_type:
    :param filepath:
    :param extension_map:
    :return: The basename of the file that was found eg mesh.bmp
    """
    try:
        all_paths = find_extensions_from_basename(filepath)
        base_names = filter_extensions(all_paths, extension_map)
        if len(base_names) > 1:
            print "Warning: More than one {0} was found: " \
                  "{1}. Taking the first by default".format(
                  file_type, base_names)
        return base_names[0]
    except Exception as e:
        raise ImportError("Failed to find a {0} for {1} from types {2}. "
                          "Reason: {3}".format(file_type, filepath,
                                               extension_map, e.message))


def get_importer(path, extension_map):
    """
    Given the absolute path to a file, try and find an appropriate importer
    using the extension map
    :param path: Absolute path to a file
    :param extension_map: Map of extensions to importer classes
    :return: A subclass of Importer
    """
    try:
        importers = map_filepaths_to_importers([path], extension_map)
        if len(importers) > 1:
            print "Warning: More than one importer was found for " \
                  "{0}. Taking the first importer by default".format(
                  path)
        return importers[0]
    except Exception as e:
        raise ImportError("Failed to find importer for {0} "
                          "for types {1}. Reason: {2}".format(path,
                                                              extension_map,
                                                              e.message))


def _multi_import(filepaths, extensions_map, keep_importers=False):
    """
    Creates importers of type importer for all the filepaths passed in,
    and then calls build on them, returning a list of objects. Expects every
    file type in the filepaths list to have a supported importer.
    """
    object_count = len(filepaths)
    importers = map_filepaths_to_importers(filepaths, extensions_map)

    objects = []
    for i, importer in enumerate(importers):
        objects.append(importer.build())

        # Cheeky carriage return so we print on the same line
        sys.stdout.write('\rCreating importer for %s (%d of %d)'
                         % (repr(importer), i + 1, object_count))
        sys.stdout.flush()

    # New line to clear for the next print
    sys.stdout.write('\n')
    sys.stdout.flush()

    if keep_importers:
        return objects, importers
    else:
        return objects


def _glob_matching_extension(pattern, extension_map):
    """
    Filters the results from the glob pattern passed in to only those files
    that have an importer given in extension_map.

    :param pattern: A UNIX style glob pattern to match against.
    :param extension_map: Map of extensions to importer class
                          e.g {'jpg': ImageImporter}
    """
    files = glob(os.path.expanduser(pattern))
    exts = [os.path.splitext(f)[1] for f in files]
    matches = [ext in extension_map for ext in exts]

    print 'Found {0} files. ({1}/{0}) are importable'.format(
        len(exts), len(filter(lambda x: x, matches)))

    return [f for f, does_match in zip(files, matches)
            if does_match]


def _images_unrelated_to_meshes(image_paths, mesh_texture_paths):
    """
    Commonly, textures of meshes will have the same name as the mesh file
    """
    image_filenames = [os.path.splitext(f)[0] for f in image_paths]
    mesh_filenames = [os.path.splitext(f)[0] for f in mesh_texture_paths]
    images_unrelated_to_mesh = set(image_filenames) - set(mesh_filenames)
    image_name_to_path = {}
    for k, v in zip(image_filenames, image_paths):
        image_name_to_path[k] = v
    return [image_name_to_path[i] for i in images_unrelated_to_mesh]


class Importer(object):
    """
    Abstract representation of an Importer. Construction on an importer
    takes a resource path and Imports the data into the Importers internal
    representation. To get an instance of one of our datatypes you access
    the shape method.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, filepath):
        self.filepath = os.path.abspath(os.path.expanduser(filepath))
        self.filename = os.path.splitext(os.path.basename(self.filepath))[0]
        self.extension = os.path.splitext(self.filepath)[1]
        self.folder = os.path.dirname(self.filepath)

    @abc.abstractmethod
    def build(self):
        pass

from pybug.io.extensions import mesh_types, image_types