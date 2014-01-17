import abc
import os
from glob import glob
import collections
from pybug import pybug_src_dir_path


def data_dir_path():
    r"""A path to the Pybug built in ./data folder on this machine.

    Returns
    -------
    string
        The path to the local PyBug ./data folder

    """
    return os.path.join(pybug_src_dir_path(), 'data')


def data_path_to(asset_filename):
    r"""The path to a builtin asset in the ./data folder on this machine.

    Parameters:
    asset_filename : string
        The filename (with extension) of a file builtin to PyBug. The full
        set of allowed names is given by ls_ls_builtin_assets()
    Returns
    -------
    string
        The path to a given asset in the ./data folder

    Raises
    ------
    ValueError
        If the asset_filename doesn't exist in the ./data folder.

    """
    asset_path = os.path.join(data_dir_path(), asset_filename)
    if not os.path.isfile(asset_path):
        raise ValueError("{} is not a builtin asset: {}".format(
            asset_filename, ls_builtin_assets()))
    return asset_path


def import_auto(pattern, max_meshes=None, max_images=None):
    r"""
    Smart data importer generator. Matches all files found on the glob pattern
    passed in, builds the relevant importers, and calls ``build()`` on
    them to return instantiated objects.

    Makes it's best effort to import and attach relevant related
    information such as landmarks. It searches the directory for files that
    begin with the same filename and end in a supported extension.

    Note that this is a generator function. This allows for pre-processing
    of data to take place as data is imported (e.g. cropping images to
    landmarks as they are imported for memory efficiency).

    Parameters
    ----------
    pattern : String
        The glob style pattern to search for textures and meshes.
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
    Import all meshes that have file extension ``.obj``:

        >>> meshes = list(import_auto('*.obj'))

    (note the cast to a list as auto_import is a generator and we want to
    exhaust it's values)

    Look for all files that begin with the string ``test``:

        >>> test_images = list(import_auto('test.*'))

    Assuming that in the current directory that are two files, ``bunny.obj``
    and ``bunny.pts``, which represent a mesh and it's landmarks, calling

        >>> bunny = list(import_auto('bunny.obj'))

    Will create a mesh object that **includes** the landmarks automatically.

    Import a crop of the top 100 square pixels of all images from a huge
    collection

        >>> images = []
        >>> for im in import_auto('./massive_image_db/*'):
        >>>    images.append(im.crop((0, 0), (100, 100)))

    """
    texture_paths = []

    # MESHES
    #  find all meshes that we can import
    mesh_paths = _glob_matching_extension(pattern, mesh_types)
    if max_meshes:
        mesh_paths = mesh_paths[:max_meshes]
    for mesh, mesh_i in _multi_import_generator(mesh_paths, mesh_types,
                                                keep_importers=True):
        # need to keep track of texture images to not double import
        if mesh_i.texture_path is not None:
            texture_paths.append(mesh_i.texture_path)
        yield mesh

    # IMAGES
    # find all images that we can import
    image_files = _glob_matching_extension(pattern, all_image_types)
    image_files = _images_unrelated_to_meshes(image_files,
                                              texture_paths)
    if max_images:
        image_files = image_files[:max_images]
    for image in _multi_import_generator(image_files, all_image_types):
        yield image


def import_image(filepath):
    return _import(filepath, all_image_types)


def import_mesh(filepath):
    return _import(filepath, mesh_types)


def import_images(pattern, max_images=None):
    for asset in _import_glob_generator(pattern, all_image_types,
                                        max_assets=max_images):
        yield asset


def import_meshes(pattern, max_meshes=None):
    for asset in _import_glob_generator(pattern, mesh_types,
                                        max_assets=max_meshes):
        yield asset


def import_builtin_asset(asset_name):
    asset_path = data_path_to(asset_name)
    return _import(asset_path, all_mesh_and_image_types)


def ls_builtin_assets():
    return os.listdir(data_dir_path())


def _import_glob_generator(pattern, extension_map, max_assets=None):
    filepaths = _glob_matching_extension(pattern, extension_map)
    if max_assets:
        filepaths = filepaths[:max_assets]
    for asset in _multi_import_generator(filepaths, extension_map):
        yield asset


def _import(filepath, extensions_map, keep_importer=False):
    r"""
    Creates an importer for the filepath passed in, and then calls build on
    it, returning a list of assets or a single asset, depending on the
    file type.

    The type of assets returned are specified by the ``extensions_map``.

    Parameters
    ----------
    filepath : string
        The filepath to import
    extensions_map : dictionary (String, :class:`pybug.io.base.Importer`)
        A map from extensions to importers. The importers are expected to be
        non-instantiated classes. The extensions are expected to
        contain the leading period eg. ``.obj``.
    keep_importers : bool, optional
        If ``True``, return the :class:`pybug.io.base.Importer` for each mesh
        as well as the meshes.

    Returns
    -------
    assets : list of assets or tuple of (assets, [:class:`pybug.io.base
    .Importer`])
        The asset or list of assets found in the filepath. If
        `keep_importers` is `True` then the importer is returned.
    """
    filepath = _norm_path(filepath)
    if not os.path.isfile(filepath):
        raise ValueError("{} is not a file".format(filepath))
    # below could raise ValueError as well...
    importer = map_filepath_to_importer(filepath, extensions_map)
    built_objects = importer.build()
    if isinstance(built_objects, collections.Iterable):
        for x in built_objects:
            x.filepath = importer.filepath  # save the filepath
    else:
        built_objects.filepath = importer.filepath
    if keep_importer:
        return built_objects, importer
    else:
        return built_objects


def _multi_import_generator(filepaths, extensions_map, keep_importers=False):
    r"""
    Generator yielding assets from the filepaths provided.

    Note that if a single file yields multiple assets, each is yielded in
    turn (this function will never yield an iterable of assets in one go).
    Assets are yielded in alphabetical order from the filepaths provided.

    Parameters
    ----------
    filepaths : list of strings
        The filepaths to import. Assets are imported in alphabetical order
    extensions_map : dictionary (String, :class:`pybug.io.base.Importer`)
        A map from extensions to importers. The importers are expected to be
        non-instantiated classes. The extensions are expected to
        contain the leading period eg. ``.obj``.
    keep_importers : bool, optional
        If ``True``, return the :class:`pybug.io.base.Importer` for each mesh
        as well as the meshes.

    Yields
    ------
    asset :
        An asset found at one of the filepaths.
    importer: :class:`pybug.io.base.Importer`
        Only if ``keep_importers`` is ``True``. The importer used for the
        yielded asset.
    """
    importer = None
    for f in sorted(filepaths):
        imported = _import(f, extensions_map, keep_importer=keep_importers)
        if keep_importers:
            assets, importer = imported
        else:
            assets = imported
        # could be that there are many assets returned from one file.
        if isinstance(assets, collections.Iterable):
            # there are multiple assets, and one importer.
            # -> yield each asset in turn with the shared importer (if
            # requested)
            for asset in assets:
                if keep_importers:
                    yield asset, importer
                else:
                    yield asset
        else:
            # assets is a single item. Rather than checking (again! for
            # importers, just yield the imported tuple
            yield imported


def _glob_matching_extension(pattern, extensions_map):
    r"""
    Filters the results from the glob pattern passed in to only those files
    that have an importer given in ``extensions_map``.

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
    pattern = _norm_path(pattern)
    files = glob(pattern)
    exts = [os.path.splitext(f)[1] for f in files]
    matches = [ext in extensions_map for ext in exts]
    return [f for f, does_match in zip(files, matches)
            if does_match]


def map_filepath_to_importer(filepath, extensions_map):
    r"""
    Given a filepath, return the appropriate importer as mapped by the
    extension map.

    Parameters
    ----------
    filepath : string
        The filepath to get importers for
    extensions_map : dictionary (String, :class:`pybug.io.base.Importer`)
        A map from extensions to importers. The importers are expected to be
        a subclass of :class:`Importer`. The extensions are expected to
        contain the leading period eg. ``.obj``.

    Returns
    --------
    importer: :class:`pybug.io.base.Importer` instance
        Importer as found in the ``extensions_map`` instantiated for the
        filepath provided.

    """
    ext = os.path.splitext(filepath)[1]
    importer_type = extensions_map.get(ext)
    if importer_type is None:
        raise ValueError("{} does not have a suitable importer.".format(ext))
    return importer_type(filepath)


def find_extensions_from_basename(filepath):
    r"""
    Given a filepath, find all the files that share the same name.

    Can be used to find all potential matching images and landmark files for a
    given mesh for instance.

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


def _norm_path(filepath):
    r"""
    Uses all the tricks in the book to expand a path out to an absolute one.
    """
    return os.path.abspath(os.path.normpath(
        os.path.expandvars(os.path.expanduser(filepath))))

# Avoid circular imports
from pybug.io.extensions import (mesh_types, all_image_types,
                                 all_mesh_and_image_types)
