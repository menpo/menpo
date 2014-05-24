import abc
from copy import deepcopy
import os
from glob import glob
from menpo import menpo_src_dir_path
from menpo.visualize import progress_bar_str, print_dynamic


def data_dir_path():
    r"""A path to the Menpo built in ./data folder on this machine.

    Returns
    -------
    string
        The path to the local Menpo ./data folder

    """
    return os.path.join(menpo_src_dir_path(), 'data')


def data_path_to(asset_filename):
    r"""The path to a builtin asset in the ./data folder on this machine.

    Parameters
    ----------
    asset_filename : string
        The filename (with extension) of a file builtin to Menpo. The full
        set of allowed names is given by :func:`ls_builtin_assets()`

    Returns
    -------
    data_path : string
        The path to a given asset in the ./data folder

    Raises
    ------
    ValueError
        If the asset_filename doesn't exist in the `data` folder.

    """
    asset_path = os.path.join(data_dir_path(), asset_filename)
    if not os.path.isfile(asset_path):
        raise ValueError("{} is not a builtin asset: {}".format(
            asset_filename, ls_builtin_assets()))
    return asset_path


def import_auto(pattern, max_meshes=None, max_images=None):
    r"""Smart mixed asset import generator.

    Makes it's best effort to import and attach relevant related
    information such as landmarks. It searches the directory for files that
    begin with the same filename and end in a supported extension.
    Furthermore, this method attempts to handle mixed assets (e.g. textured
    meshes in the same folder as images) without 'double importing' anything.

    Note that this is a generator function. This allows for pre-processing
    of data to take place as data is imported (e.g. cropping images to
    landmarks as they are imported for memory efficiency).


    Parameters
    ----------
    pattern : String
        The glob path pattern to search for textures and meshes.
    max_meshes: positive integer, optional
        If not `None`, only import the first max_mesh meshes found. Else,
        import all.

        Default: `None`
    max_images: positive integer, optional
        If not `None`, only import the first max_images found. Else,
        import all.

        Default: `None`

    Yields
    ------
    asset
        Assets found to match the glob pattern provided.

    Examples
    --------
    Import all meshes that have file extension `.obj`:

        >>> meshes = list(import_auto('*.obj'))

    (note the cast to a list as auto_import is a generator and we want to
    exhaust it's values)

    Look for all files that begin with the string `test`:

        >>> test_images = list(import_auto('test.*'))

    Assuming that in the current directory that are two files, `bunny.obj`
    and `bunny.pts`, which represent a mesh and it's landmarks, calling

        >>> bunny = list(import_auto('bunny.obj'))

    Will create a mesh object that **includes** the landmarks automatically.

    """
    texture_paths = []

    # MESHES
    #  find all meshes that we can import
    mesh_files = mesh_paths(pattern)
    if max_meshes:
        mesh_files = mesh_files[:max_meshes]
    for mesh, mesh_i in _multi_import_generator(mesh_files, mesh_types,
                                                keep_importers=True):
        # need to keep track of texture images to not double import
        if mesh_i.texture_path is not None:
            texture_paths.append(mesh_i.texture_path)
        yield mesh

    # IMAGES
    # find all images that we can import
    image_files = image_paths(pattern)
    image_files = _images_unrelated_to_meshes(image_files,
                                              texture_paths)
    if max_images:
        image_files = image_files[:max_images]
    for image in _multi_import_generator(image_files, all_image_types):
        yield image


def import_image(filepath, landmark_resolver=None):
    r"""Single image (and associated landmarks) importer.

    Iff an image file is found at `filepath`, returns a :class:`menpo.image
    .MaskedImage` representing it. Landmark files sharing the same filename
    will be imported and attached too. If the image defines a mask,
    this mask will be imported.

    This method is valid for both traditional images and spatial image types.

    Parameters
    ----------
    filepath : String
        A relative or absolute filepath to an image file.
    landmark_resolver: function, optional
        If not None, this function will be used to find landmarks for the
        image. The function should take one argument (the image itself) and
        return a dictionary of the form {'group_name': 'landmark_filepath'}


    Returns
    -------
    :class:`menpo.image.Image`
        An instantiated image class built from the image file.

    """
    return _import(filepath, all_image_types, has_landmarks=True,
                   landmark_resolver=landmark_resolver)


def import_mesh(filepath, landmark_resolver=None, texture=True):
    r"""Single mesh (and associated landmarks and texture) importer.

    Iff an mesh file is found at `filepath`, returns a :class:`menpo.shape
    .TriMesh` representing it. Landmark files sharing the same filename
    will be imported and attached too. If texture coordinates and a suitable
    texture are found the object returned will be a :class:`menpo.shape
    .TexturedTriMesh`.

    Parameters
    ----------
    filepath : String
        A relative or absolute filepath to an image file.
    landmark_resolver: function, optional
        If not None, this function will be used to find landmarks for the
        mesh. The function should take one argument (the mesh itself) and
        provide a string or list of strings detailing the landmarks to be
        imported.

    texture: Boolean, optional
        If False, don't search for textures.

        Default: True
    Returns
    -------
    :class:`menpo.shape.TriMesh`
        An instantiated trimesh (or textured trimesh) file object

    """
    kwargs = {'texture': texture}
    return _import(filepath, mesh_types, has_landmarks=True,
                   landmark_resolver=landmark_resolver, importer_kwargs=kwargs)


def import_landmark_file(filepath):
    r"""Single landmark group importer.

    Iff an landmark file is found at `filepath`, returns a :class:`menpo
    .landmarks.LandmarkGroup` representing it.

    Parameters
    ----------
    filepath : String
        A relative or absolute filepath to an landmark file.

    Returns
    -------
    :class:`menpo.shape.LandmarkGroup`
        The LandmarkGroup that the file format represents.

    """
    return _import(filepath, all_landmark_types, has_landmarks=False)


def import_images(pattern, max_images=None, landmark_resolver=None,
                  verbose=False):
    r"""Multiple image import generator.

    Makes it's best effort to import and attach relevant related
    information such as landmarks. It searches the directory for files that
    begin with the same filename and end in a supported extension.

    Note that this is a generator function. This allows for pre-processing
    of data to take place as data is imported (e.g. cropping images to
    landmarks as they are imported for memory efficiency).


    Parameters
    ----------
    pattern : `str`
        The glob path pattern to search for images.

    max_images : positive `int`, optional
        If not ``None``, only import the first ``max_images`` found. Else,
        import all.

    landmark_resolver : `function`, optional
        If not ``None``, this function will be used to find landmarks for each
        image. The function should take one argument (an image itself) and
        return a dictionary of the form ``{'group_name': 'landmark_filepath'}``

    verbose : `bool`, optional
        If ``True`` progress of the importing will be dynamically reported.

    Yields
    ------
    :map:`MaskedImage`
        Images found to match the glob pattern provided.

    Raises
    ------
    ValueError
        If no images are found at the provided glob.

    Examples
    --------
    Import crops of the top 100 square pixels from a huge collection of images

        >>> images = []
        >>> for im in import_images('./massive_image_db/*'):
        >>>    im.crop_inplace((0, 0), (100, 100))  # crop to a sensible size as we go
        >>>    images.append(im)
    """
    for asset in _import_glob_generator(pattern, all_image_types,
                                        max_assets=max_images,
                                        has_landmarks=True,
                                        landmark_resolver=landmark_resolver,
                                        verbose=verbose):
        yield asset


def import_meshes(pattern, max_meshes=None, landmark_resolver=None,
                  textures=True, verbose=False):
    r"""Multiple mesh import generator.

    Makes it's best effort to import and attach relevant related
    information such as landmarks. It searches the directory for files that
    begin with the same filename and end in a supported extension.

    If texture coordinates and a suitable texture are found the object
    returned will be a :map:`TexturedTriMesh`.

    Note that this is a generator function. This allows for pre-processing
    of data to take place as data is imported (e.g. cleaning meshes
    as they are imported for memory efficiency).

    Parameters
    ----------
    pattern : `str`
        The glob path pattern to search for textures and meshes.

    max_meshes : positive `int`, optional
        If not ``None``, only import the first ``max_meshes`` meshes found.
        Else, import all.

    landmark_resolver : `function`, optional
        If not ``None``, this function will be used to find landmarks for each
        mesh. The function should take one argument (a mesh itself) and
        return a dictionary of the form ``{'group_name': 'landmark_filepath'}``

    texture : `bool`, optional
        If ``False``, don't search for textures.

    verbose : `bool`, optional
        If ``True`` progress of the importing will be dynamically reported.

    Yields
    ------
    :map:`TriMesh` or :map:`TexturedTriMesh`
        Meshes found to match the glob pattern provided.

    Raises
    ------
    ValueError
        If no meshes are found at the provided glob.

    """
    kwargs = {'texture': textures}
    for asset in _import_glob_generator(pattern, mesh_types,
                                        max_assets=max_meshes,
                                        has_landmarks=True,
                                        landmark_resolver=landmark_resolver,
                                        importer_kwargs=kwargs,
                                        verbose=verbose):
        yield asset


def import_landmark_files(pattern, max_landmarks=None, verbose=False):
    r"""Multiple landmark file import generator.

    Note that this is a generator function.

    Parameters
    ----------
    pattern : `str`
        The glob path pattern to search for images.

    max_landmark_files : positive `int`, optional
        If not ``None``, only import the first ``max_landmark_files`` found.
        Else, import all.

    verbose : `bool`, optional
        If ``True`` progress of the importing will be dynamically reported.

    Yields
    ------
    :map:`LandmarkGroup`
        Landmark found to match the glob pattern provided.

    Raises
    ------
    ValueError
        If no landmarks are found at the provided glob.

    """
    for asset in _import_glob_generator(pattern, all_landmark_types,
                                        max_assets=max_landmarks,
                                        has_landmarks=False,
                                        verbose=verbose):
        yield asset


def import_builtin_asset(asset_name):
    r"""Single builtin asset (mesh or image) importer.

    Imports the relevant builtin asset from the ./data directory that
    ships with Menpo.

    Parameters
    ----------
    asset_name : String
        The filename of a builtin asset (see :func:`ls_ls_builtin_assets()`
        for allowed values)

    Returns
    -------
    asset
        An instantiated asset (mesh, trimesh, or image)

    """
    asset_path = data_path_to(asset_name)
    return _import(asset_path, all_mesh_and_image_types, has_landmarks=True)


def ls_builtin_assets():
    r"""List all the builtin asset examples provided in Menpo.

    Returns
    -------
    list of strings
        Filenames of all assets in the data directory shipped with Menpo

    """
    return os.listdir(data_dir_path())


def mesh_paths(pattern):
    r"""
    Return mesh filepaths that Menpo can import that match the glob pattern.
    """
    return _glob_matching_extension(pattern, mesh_types)


def image_paths(pattern):
    r"""
    Return image filepaths that Menpo can import that match the glob pattern.
    """
    return _glob_matching_extension(pattern, all_image_types)


def _import_glob_generator(pattern, extension_map, max_assets=None,
                           has_landmarks=False, landmark_resolver=None,
                           importer_kwargs=None, verbose=False):
    filepaths = _glob_matching_extension(pattern, extension_map)
    if max_assets:
        filepaths = filepaths[:max_assets]
    n_files = len(filepaths)
    if n_files == 0:
        raise ValueError('The glob {} yields no assets'.format(pattern))
    for i, asset in enumerate(_multi_import_generator(filepaths, extension_map,
                                         has_landmarks=has_landmarks,
                                         landmark_resolver=landmark_resolver,
                                         importer_kwargs=importer_kwargs)):
        if verbose:
            print_dynamic('- Loading {} assets: {}'.format(
                n_files, progress_bar_str(float(i + 1) / n_files,
                                          show_bar=True)))
        yield asset


def _import(filepath, extensions_map, keep_importer=False,
            has_landmarks=True, landmark_resolver=None,
            asset=None, importer_kwargs=None):
    r"""
    Creates an importer for the filepath passed in, and then calls build on
    it, returning a list of assets or a single asset, depending on the
    file type.

    The type of assets returned are specified by the `extensions_map`.

    Parameters
    ----------
    filepath : string
        The filepath to import
    extensions_map : dictionary (String, :class:`menpo.io.base.Importer`)
        A map from extensions to importers. The importers are expected to be
        non-instantiated classes. The extensions are expected to
        contain the leading period eg. `.obj`.
    keep_importer : bool, optional
        If `True`, return the :class:`menpo.io.base.Importer` for each mesh
        as well as the meshes.
    has_landmarks : bool, optional
        If `True`, an attempt will be made to find relevant landmarks.
    landmark_resolver: function, optional
        If not None, this function will be used to find landmarks for each
        asset. The function should take one argument (the asset itself) and
        return a dictionary of the form {'group_name': 'landmark_filepath'}
    asset: object, optional
        If not None, the asset will be passed to the importer's build method
        as the asset kwarg
    importer_kwargs: dict, optional:
        kwargs that will be supplied to the importer if not None

    Returns
    -------
    assets : list of assets or tuple of (assets, [:class:`menpo.io.base
    .Importer`])
        The asset or list of assets found in the filepath. If
        `keep_importers` is `True` then the importer is returned.
    """
    filepath = _norm_path(filepath)
    if not os.path.isfile(filepath):
        raise ValueError("{} is not a file".format(filepath))
    # below could raise ValueError as well...
    importer = map_filepath_to_importer(filepath, extensions_map,
                                        importer_kwargs=importer_kwargs)
    if asset is not None:
        built_objects = importer.build(asset=asset)
    else:
        built_objects = importer.build()
    # landmarks are iterable so check for list precisely
    ioinfo = importer.build_ioinfo()
    # enforce a list to make processing consistent
    if not isinstance(built_objects, list):
        built_objects = [built_objects]

    # attach ioinfo
    for x in built_objects:
        x.ioinfo = deepcopy(ioinfo)

    # handle landmarks
    if has_landmarks:
        if landmark_resolver is None:
            # user isn't customising how landmarks are found.
            lm_pattern = os.path.join(ioinfo.dir, ioinfo.filename + '.*')
            # find all the landmarks we can
            lms_paths = _glob_matching_extension(lm_pattern, all_landmark_types)
            for lm_path in lms_paths:
                # manually trigger _import (so we can set the asset!)
                lms = _import(lm_path, all_landmark_types, keep_importer=False,
                              has_landmarks=False, asset=asset)
                for x in built_objects:
                    try:
                        x.landmarks[lms.group_label] = deepcopy(lms)
                    except ValueError:
                        pass
        else:
            for x in built_objects:
                lm_paths = landmark_resolver(x)  # use the users fcn to find
                # paths
                if lm_paths is None:
                    continue
                for group_name, lm_path in lm_paths.iteritems():
                    lms = import_landmark_file(lm_path)
                    x.landmarks[group_name] = lms

    # undo list-if-cation (if we added it!)
    if len(built_objects) == 1:
        built_objects = built_objects[0]

    if keep_importer:
        return built_objects, importer
    else:
        return built_objects


def _multi_import_generator(filepaths, extensions_map, keep_importers=False,
                            has_landmarks=False, landmark_resolver=None,
                            importer_kwargs=None):
    r"""
    Generator yielding assets from the filepaths provided.

    Note that if a single file yields multiple assets, each is yielded in
    turn (this function will never yield an iterable of assets in one go).
    Assets are yielded in alphabetical order from the filepaths provided.

    Parameters
    ----------
    filepaths : list of strings
        The filepaths to import. Assets are imported in alphabetical order
    extensions_map : dictionary (String, :class:`menpo.io.base.Importer`)
        A map from extensions to importers. The importers are expected to be
        non-instantiated classes. The extensions are expected to
        contain the leading period eg. `.obj`.
    keep_importers : bool, optional
        If `True`, return the :class:`menpo.io.base.Importer` for each mesh
        as well as the meshes.
    has_landmarks : bool, optional
        If `True`, an attempt will be made to find relevant landmarks.
    landmark_resolver: function, optional
        If not None, this function will be used to find landmarks for each
        asset. The function should take one argument (the asset itself) and
        return a dictionary of the form {'group_name': 'landmark_filepath'}
    importer_kwargs: dict, optional
        kwargs to be supplied to the importer if not None

    Yields
    ------
    asset :
        An asset found at one of the filepaths.
    importer: :class:`menpo.io.base.Importer`
        Only if `keep_importers` is `True`. The importer used for the
        yielded asset.
    """
    importer = None
    for f in sorted(filepaths):
        imported = _import(f, extensions_map, keep_importer=keep_importers,
                           has_landmarks=has_landmarks,
                           landmark_resolver=landmark_resolver,
                           importer_kwargs=importer_kwargs)
        if keep_importers:
            assets, importer = imported
        else:
            assets = imported
        # could be that there are many assets returned from one file.
        # landmarks are iterable so check for list precisely
        if isinstance(assets, list):
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
    that have an importer given in `extensions_map`.

    Parameters
    ----------
    pattern : string
        A UNIX style glob pattern to match against.
    extensions_map : dictionary (String, :class:`menpo.io.base.Importer`)
        A map from extensions to importers. The importers are expected to be
        non-instantiated classes. The extensions are expected to
        contain the leading period eg. `.obj`.

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


def map_filepath_to_importer(filepath, extensions_map, importer_kwargs=None):
    r"""
    Given a filepath, return the appropriate importer as mapped by the
    extension map.

    Parameters
    ----------
    filepath : string
        The filepath to get importers for
    extensions_map : dictionary (String, :class:`menpo.io.base.Importer`)
        A map from extensions to importers. The importers are expected to be
        a subclass of :class:`Importer`. The extensions are expected to
        contain the leading period eg. `.obj`.
    importer_kwargs: dictionary, optional
        kwargs that will be supplied to the importer if not None.

    Returns
    --------
    importer: :class:`menpo.io.base.Importer` instance
        Importer as found in the `extensions_map` instantiated for the
        filepath provided.

    """
    ext = os.path.splitext(filepath)[1]
    importer_type = extensions_map.get(ext)
    if importer_type is None:
        raise ValueError("{} does not have a suitable importer.".format(ext))
    if importer_kwargs is not None:
        return importer_type(filepath, **importer_kwargs)
    else:
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
        as filepath. These files are found using `glob`.

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
    extensions_map : dictionary (String, :class:`menpo.io.base.Importer`)
        A map from extensions to importers. The importers are expected to be
        non-instantiated classes. The extensions are expected to
        contain the leading period eg. `.obj`.

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
    extensions_map : dictionary (String, :class:`menpo.io.base.Importer`)
        A map from extensions to importers. The importers are expected to be
        non-instantiated classes. The extensions are expected to
        contain the leading period eg. `.obj`.

    Returns
    -------
    base_name : string
        The basename of the file that was found eg `mesh.bmp`. Only **one**
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
                                               extensions_map, e))


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
    representation, the `build` method must be called. This allows a set
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
        object : object or list
            An instantiated class of the expected type. For example, for an
            `.obj` importer, this would be a
            :class:`menpo.shape.mesh.base.Trimesh`. If multiple objects need
            to be returned from one importer, a list must be returned.
        """
        pass

    def build_ioinfo(self):
        return IOInfo(self.filepath)


def _norm_path(filepath):
    r"""
    Uses all the tricks in the book to expand a path out to an absolute one.
    """
    return os.path.abspath(os.path.normpath(
        os.path.expandvars(os.path.expanduser(filepath))))

# Avoid circular imports
from menpo.io.extensions import (mesh_types, all_image_types,
                                 all_mesh_and_image_types,
                                 all_landmark_types)


class IOInfo(object):
    r"""
    Simple state object for recording IO information.
    """

    def __init__(self, filepath):
        self.filepath = os.path.abspath(os.path.expanduser(filepath))
        self.filename = os.path.splitext(os.path.basename(self.filepath))[0]
        self.extension = os.path.splitext(self.filepath)[1]
        self.dir = os.path.dirname(self.filepath)

    def __str__(self):
        return 'filename: {}\nextension: {}\ndir: {}\nfilepath: {}'.format(
            self.filename, self.extension, self.dir, self.filepath
        )
