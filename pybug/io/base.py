import abc
import os
from glob import glob
import sys
import collections


def auto_import(pattern, meshes=True, images=True,
                include_texture_images=False,
                max_meshes=None, max_images=None):
    r"""
    Smart data importer. Will match all files found on the glob pattern
    passed in, build the relevant importers, and then call ``build()`` on them
    to return a list of usable objects.

    It makes it's best effort to import and attach relevant related
    information such as landmarks. It searches the directory for files that
    begin with the same filename and end in a supported extension.

    Parameters
    ----------
    pattern : String
        The glob style pattern to search for textures and meshes.
    meshes: bool, optional
        If ``True``, include mesh types in results

        Default: ``True``
    images: bool, optional
        If ``True``, include image types in results

        Default: ``True``
    include_texture_images: bool, optional
        If ``True``, check if the images found in the glob pattern are actually
        textures of the meshes it found.
        If this is the case, it won't import these images separately.

        Default: ``False``
    max_meshes: positive integer, optional
        If not ``None``, only import the first max_mesh meshes found. Else,
        import all.

        Default: ``None``
    max_images: positive integer, optional
        If not ``None``, only import the first max_images found. Else,
        import all.

        Default: ``None``

    Examples
    --------
    Look for all meshes that have file extension ``.obj``:

        >>> auto_import('*.obj')

    Look for all files that begin with the string ``test``:

        >>> auto_import('test.*')

    Assuming that in the current directory that are two files, ``bunny.obj``
    and ``bunny.pts``, which represent a mesh and it's landmarks, calling

        >>> auto_import('bunny.obj')

    Will create a mesh object that **includes** the landmarks automatically.
    """
    mesh_objects, image_objects = [], []
    if meshes:
        mesh_paths = _glob_matching_extension(pattern, mesh_types)
        if max_meshes:
            mesh_paths = mesh_paths[:max_meshes]
        mesh_objects, mesh_importers = _multi_mesh_import(mesh_paths,
                                                          keep_importers=True)
    if images:
        image_files = _glob_matching_extension(pattern, all_image_types)
        if max_images:
            image_files = image_files[:max_images]
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
    :class:`pybug.image.base.Image`.

    Parameters
    ----------
    image_filepaths: list of strings
        List of filepaths
    keep_importers: bool, optional
        Returns the :class:`pybug.io.base.Importer` as well as the images
        themselves

    Returns
    -------
    images : list of :class:`pybug.image.base.Image` or tuple of ([:class:`pybug.image.base.Image`], [:class:`pybug.io.base.Importer`])
        The list of images found in the filepaths. If ``keep_importers`` is
        ``True`` then the importer for each image is returned as a tuple of s
        lists.
    """
    return _multi_import(image_filepaths, all_image_types, keep_importers)


def _multi_mesh_import(mesh_filepaths, keep_importers=False):
    r"""
    Creates importers for all the mesh filepaths passed in,
    and then calls build on them, returning a list of
    :class:`pybug.shape.mesh.base.Trimesh` or
    :class:`pybug.shape.mesh.textured.TexturedTriMesh`.

    Parameters
    ----------
    mesh_filepaths : list of strings
        List of filepaths to the meshes
    keep_importers : bool, optional
        If ``True``, return the :class:`pybug.io.base.Importer` for each mesh
        as well as the meshes.

    Returns
    -------
    meshes : list of :class:`pybug.shape.mesh.base.Trimesh` or tuple of ([:class:`pybug.shape.mesh.base.Trimesh`], [:class:`pybug.io.base.Importer`])
        The list of meshes found in the filepaths. If ``keep_importers`` is
        ``True`` then the importer for each mesh is returned as a tuple of
        lists.
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
    r"""
    Given a list of filepaths, return the appropriate importers for each path
    as mapped by the extension map.

    Parameters
    ----------
    filepaths : list of strings
        The filepaths to get importers for
    extensions_map : dictionary (String, :class:`pybug.io.base.Importer`)
        A map from extensions to importers. The importers are expected to be
        non-instantiated classes. The extensions are expected to
        contain the leading period eg. ``.obj``.

    Returns
    --------
    importers : list of :class:`pybug.io.base.Importer`
        The list of instantiated importers as found in the ``extensions_map``.

    """
    importers = []
    for f in sorted(filepaths):
        ext = os.path.splitext(f)[1]
        importer_type = extensions_map.get(ext)
        importers.append(importer_type(f))
    return importers


def find_extensions_from_basename(filepath):
    r"""
    Given a filepath, find all the files that share the same name. This is
    used to find all potential matching images and landmark files for a given
    file.

    Parameters
    ----------
    filepath : string
        An absolute filepath

    Returns
    -------
    files : list of strings
        A list of absolute filepaths to files that share the same basename
        as filepath. These files are found using ``glob``.
    """
    basename = os.path.splitext(os.path.basename(filepath))[0] + '*'
    basepath = os.path.join(os.path.dirname(filepath), basename)
    return glob(basepath)


def filter_extensions(filepaths, extensions_map):
    r"""
    Given a set of filepaths, filter the files who's extensions are in the
    given map. This is used to find images and landmarks from a given basename.

    Parameters
    ----------
    filepaths : list of strings
        A list of absolute filepaths
    extensions_map : dictionary (String, :class:`pybug.io.base.Importer`)
        A map from extensions to importers. The importers are expected to be
        non-instantiated classes. The extensions are expected to
        contain the leading period eg. ``.obj``.

    Returns
    -------
    basenames : list of strings
        A list of basenames
    """
    extensions = extensions_map.keys()
    return [os.path.basename(f) for f in filepaths
            if os.path.splitext(f)[1] in extensions]


def find_alternative_files(file_type, filepath, extensions_map):
    r"""
    Given a filepath, search for files with the same basename that match
    a given extension type, eg images. If more than one file is found, an error
    is printed and the first such basename is returned.

    Parameters
    ----------
    file_type : string
        The type of file being found. Used for the error outputs.
    filepath : string
        An absolute filepath
    extensions_map : dictionary (String, :class:`pybug.io.base.Importer`)
        A map from extensions to importers. The importers are expected to be
        non-instantiated classes. The extensions are expected to
        contain the leading period eg. ``.obj``.

    Returns
    -------
    base_name : string
        The basename of the file that was found eg ``mesh.bmp``. Only **one**
        file is ever returned. If more than one is found, the first is taken.

    Raises
    ------
    ImportError
        If no alternative file is found
    """
    try:
        all_paths = find_extensions_from_basename(filepath)
        base_names = filter_extensions(all_paths, extensions_map)
        if len(base_names) > 1:
            print "Warning: More than one {0} was found: " \
                  "{1}. Taking the first by default".format(
                  file_type, base_names)
        return base_names[0]
    except Exception as e:
        raise ImportError("Failed to find a {0} for {1} from types {2}. "
                          "Reason: {3}".format(file_type, filepath,
                                               extensions_map, e.message))


def get_importer(path, extensions_map):
    r"""
    Given the absolute path to a file, try and find an appropriate importer
    using the extension map. If more than one importer is found, an error
    is printed and the first importer is returned.

    Parameters
    ----------
    path : string
        Absolute path to a file
    extensions_map : dictionary (String, :class:`pybug.io.base.Importer`)
        A map from extensions to importers. The importers are expected to be
        non-instantiated classes. The extensions are expected to
        contain the leading period eg. ``.obj``.

    Returns
    -------
    importer : :class:`pybug.io.base.Importer`
        The first importer that was found for the given path.

    Raises
    ------
    ImportError
        If no importer is found
    """
    try:
        importers = map_filepaths_to_importers([path], extensions_map)
        if len(importers) > 1:
            print "Warning: More than one importer was found for " \
                  "{0}. Taking the first importer by default".format(
                  path)
        return importers[0]
    except Exception as e:
        raise ImportError("Failed to find importer for {0} "
                          "for types {1}. Reason: {2}".format(path,
                                                              extensions_map,
                                                              e.message))


def _multi_import(filepaths, extensions_map, keep_importers=False):
    r"""
    Creates importers for all the filepaths passed in, and then calls build on
    them, returning a list of objects. Expects every file type in the filepaths
    list to have a supported importer. The type of objects returned are
    specified implicitly by the ``extensions_map``.

    Prints out the current progress of importing to stdout.

    Parameters
    ----------
    filepaths : list of strings
        The filepaths to import
    extensions_map : dictionary (String, :class:`pybug.io.base.Importer`)
        A map from extensions to importers. The importers are expected to be
        non-instantiated classes. The extensions are expected to
        contain the leading period eg. ``.obj``.
    keep_importers : bool, optional
        If ``True``, return the :class:`pybug.io.base.Importer` for each mesh
        as well as the meshes.

    Returns
    -------
    objs : list of objects or tuple of ([object], [:class:`pybug.io.base.Importer`])
        The list of objects found in the filepaths. If ``keep_importers`` is
        ``True`` then the importer for each object is returned as a tuple of
        lists.
    """
    object_count = len(filepaths)
    importers = map_filepaths_to_importers(filepaths, extensions_map)

    objects = []
    for i, importer in enumerate(importers):
        built_objects = importer.build()
        if isinstance(built_objects, collections.Iterable):
            for x in built_objects:
                x.filepath = importer.filepath  # save the filepath
        else:
            built_objects.filepath = importer.filepath
        objects.append(built_objects)

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


def _glob_matching_extension(pattern, extensions_map):
    r"""
    Filters the results from the glob pattern passed in to only those files
    that have an importer given in ``extensions_map``.

    Prints out to stdout how many files that were found by the glob have a
    valid importer in the ``extensions_map``.

    Parameters
    ----------
    pattern : string
        A UNIX style glob pattern to match against.
    extensions_map : dictionary (String, :class:`pybug.io.base.Importer`)
        A map from extensions to importers. The importers are expected to be
        non-instantiated classes. The extensions are expected to
        contain the leading period eg. ``.obj``.

    Returns
    -------
    filepaths : list of string
        The list of filepaths that have valid extensions.
    """
    files = glob(os.path.expanduser(pattern))
    exts = [os.path.splitext(f)[1] for f in files]
    matches = [ext in extensions_map for ext in exts]

    print 'Found {0} files. ({1}/{0}) are importable'.format(
        len(exts), len(filter(lambda x: x, matches)))

    return [f for f, does_match in zip(files, matches)
            if does_match]


def _images_unrelated_to_meshes(image_paths, mesh_texture_paths):
    r"""
    Find the set of images that do not correspond to textures for the given
    meshes.

    Parameters
    ----------
    image_paths : list of strings
        List of absolute filepaths to images
    mesh_texture_paths : list of strings
        List of absolute filepaths to mesh textures

    Returns
    -------
    images : list of strings
        List of absolute filepaths to images that are unrelated to meshes.
    """
    image_filenames = [os.path.splitext(f)[0] for f in image_paths]
    mesh_filenames = [os.path.splitext(f)[0] for f in mesh_texture_paths]
    images_unrelated_to_mesh = set(image_filenames) - set(mesh_filenames)
    image_name_to_path = {}
    for k, v in zip(image_filenames, image_paths):
        image_name_to_path[k] = v
    return [image_name_to_path[i] for i in images_unrelated_to_mesh]


class Importer(object):
    r"""
    Abstract representation of an Importer. Construction of an importer simply
    sets the filepaths etc up. To actually import the object and build a valid
    representation, the ``build`` method must be called. This allows a set
    of importers to be instantiated but the heavy duty importing to happen
    separately.

    Parameters
    ----------
    filepath : string
        An absolute filepath
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, filepath):
        self.filepath = os.path.abspath(os.path.expanduser(filepath))
        self.filename = os.path.splitext(os.path.basename(self.filepath))[0]
        self.extension = os.path.splitext(self.filepath)[1]
        self.folder = os.path.dirname(self.filepath)

    @abc.abstractmethod
    def build(self):
        r"""
        Performs the heavy lifting for the importer class. This actually reads
        the file in from disk and does any necessary parsing of the data in to
        an appropriate format.

        Returns
        -------
        object : object
            An instantiated class of the expected type. For example, for an
            ``.obj`` importer, this would be a
            :class:`pybug.shape.mesh.base.Trimesh`.
        """
        pass

# Avoid circular imports
from pybug.io.extensions import mesh_types, all_image_types
