import abc
import os
import glob
import sys


def auto_import(pattern, meshes=True, images=True,
                include_texture_images=False):
    """ Smart data importer. Will match all files found on the glob pattern
    passed in, build the relevant importers, and then call build() on them to
    return a list of usable objects. To be selective, use the kwargs.

    kwargs**
    images - import images found in the pattern
    meshes - import meshes found in the pattern
    include_texture_images - by default, auto_import will check if the images
    it has found  in the glob pattern are actually textures of the meshes it
    found. If this is the case, it won't import these images separately. To
    override this behavior, set include_texture_images to True.
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
    """
    Creates importers for all the image filepaths passed in,
    and then calls build on them, returning a list of Images.
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


def _multi_import(filepaths, extensions_map, keep_importers=False):
    """
    Creates importers of type importer for all the filepaths passed in,
    and then calls build on them, returning a list of objects. Expects every
    file type in the filepaths list to have a supported importer.
    """
    object_count = len(filepaths)
    importers = []

    # Loop over the sorted filepaths (keeps logical filepath order)
    for i, (f, ext) in enumerate(sorted(filepaths)):
        importer_type = extensions_map.get(ext)
        importers.append(importer_type(f))
        # Cheeky carriage return so we print on the same line
        sys.stdout.write('\rCreating importer for %s (%d of %d)'
                         % (repr(importer_type), i + 1, object_count))
        sys.stdout.flush()

    # New line to clear for the next print
    sys.stdout.write('\n')
    sys.stdout.flush()

    objects = [i.build() for i in importers]
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
    files = glob.glob(os.path.expanduser(pattern))
    exts = [os.path.splitext(f)[1] for f in files]
    matches = [(ext, ext in extension_map) for ext in exts]

    print 'Found {0} files. ({1}/{0}) are importable'.format(
        len(exts), len(filter(lambda x: x[1] is True, matches)))

    return [(f, ext) for f, (ext, does_match) in zip(files, matches)
            if does_match]


def _images_unrelated_to_meshes(image_paths, mesh_texture_paths):
    """
    Commonly, textures of meshes will have the same name as the mesh file
    """
    image_filenames = [os.path.splitext(f)[0] for f, ext in image_paths]
    mesh_filenames = [os.path.splitext(f)[0] for f in mesh_texture_paths]
    images_unrelated_to_mesh = set(image_filenames) - set(mesh_filenames)
    image_name_to_path = {}
    for k, v in zip(image_filenames, image_paths):
        image_name_to_path[k] = v
    return [image_name_to_path[i] for i in images_unrelated_to_mesh]


class Importer:
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