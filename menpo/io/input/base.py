import abc
import os
from glob import glob

from pathlib import Path
import hdf5able

from ..utils import _norm_path
from menpo import menpo_src_dir_path
from menpo.visualize import progress_bar_str, print_dynamic


def load(path):
    r"""
    Load a given HDF5 file of serialized Menpo objects or base types.

    Parameters
    ----------
    path : `str`
        A path to a HDF5 file that conforms to the hdf5able specification.

    Returns
    -------
    hdf5able :
        Any collection of HDF5able objects.

    """
    return hdf5able.load(_norm_path(path))


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
    asset_filename : `str`
        The filename (with extension) of a file builtin to Menpo. The full
        set of allowed names is given by :func:`ls_builtin_assets()`

    Returns
    -------
    data_path : `str`
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


def same_name(asset):
    r"""
    Menpo's default landmark resolver. Returns all landmarks found to have
    the same stem as the asset.
    """
    # pattern finding all landmarks with the same stem
    pattern = os.path.join(asset.ioinfo.dir, asset.ioinfo.filename + '.*')
    # find all the landmarks we can with this name. Key is ext (without '.')
    return {os.path.splitext(p)[-1][1:].upper(): p
            for p in landmark_file_paths(pattern)}


def import_image(filepath, landmark_resolver=same_name):
    r"""Single image (and associated landmarks) importer.

    Iff an image file is found at `filepath`, returns a :class:`menpo.image
    .MaskedImage` representing it. Landmark files sharing the same filename
    will be imported and attached too. If the image defines a mask,
    this mask will be imported.

    This method is valid for both traditional images and spatial image types.

    Parameters
    ----------
    filepath : `str`
        A relative or absolute filepath to an image file.

    landmark_resolver : `function`, optional
        This function will be used to find landmarks for the
        image. The function should take one argument (the image itself) and
        return a dictionary of the form ``{'group_name': 'landmark_filepath'}``
        Default finds landmarks with the same name as the image file.

    Returns
    -------
    :map:`Image`
        An instantiated :map:`Image` or subclass thereof

    """
    return _import(filepath, all_image_types, has_landmarks=True,
                   landmark_resolver=landmark_resolver)


def import_mesh(filepath, landmark_resolver=same_name, texture=True):
    r"""Single mesh (and associated landmarks and texture) importer.

    Iff an mesh file is found at `filepath`, returns a :class:`menpo.shape
    .TriMesh` representing it. Landmark files sharing the same filename
    will be imported and attached too. If texture coordinates and a suitable
    texture are found the object returned will be a :class:`menpo.shape
    .TexturedTriMesh`.

    Parameters
    ----------
    filepath : `str`
        A relative or absolute filepath to an image file.

    landmark_resolver : `function`, optional
        This function will be used to find landmarks for the
        mesh. The function should take one argument (the mesh itself) and
        return a dictionary of the form ``{'group_name': 'landmark_filepath'}``
        Default finds landmarks with the same name as the mesh file.

    texture : `bool`, optional
        If False, don't search for textures.

        Default: ``True``

    Returns
    -------
    :map:`TriMesh`
        An instantiated :map:`TriMesh` (or subclass thereof)

    """
    kwargs = {'texture': texture}
    return _import(filepath, mesh_types, has_landmarks=True,
                   landmark_resolver=landmark_resolver, importer_kwargs=kwargs)


def import_landmark_file(filepath, asset=None):
    r"""Single landmark group importer.

    Iff an landmark file is found at `filepath`, returns a :class:`menpo
    .landmarks.LandmarkGroup` representing it.

    Parameters
    ----------
    filepath : `str`
        A relative or absolute filepath to an landmark file.

    Returns
    -------
    :map:`LandmarkGroup`
        The :map:`LandmarkGroup` that the file format represents.

    """
    return _import(filepath, all_landmark_types, has_landmarks=False,
                   asset=asset)


def import_images(pattern, max_images=None, landmark_resolver=same_name,
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
        This function will be used to find landmarks for the
        image. The function should take one argument (the image itself) and
        return a dictionary of the form ``{'group_name': 'landmark_filepath'}``
        Default finds landmarks with the same name as the image file.

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


def import_meshes(pattern, max_meshes=None, landmark_resolver=same_name,
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
        This function will be used to find landmarks for the
        mesh. The function should take one argument (the mesh itself) and
        return a dictionary of the form ``{'group_name': 'landmark_filepath'}``
        Default finds landmarks with the same name as the mesh file.

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


def _import_builtin_asset(asset_name):
    r"""Single builtin asset (mesh or image) importer.

    Imports the relevant builtin asset from the ./data directory that
    ships with Menpo.

    Parameters
    ----------
    asset_name : `str`
        The filename of a builtin asset (see :map:`ls_builtin_assets`
        for allowed values)

    Returns
    -------
    asset
        An instantiated :map:`Image` or :map:`TriMesh` asset.

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


def import_builtin(x):

    def execute():
        return _import_builtin_asset(x)

    return execute


class BuiltinAssets(object):

    def __call__(self, asset_name):
        return _import_builtin_asset(asset_name)

import_builtin_asset = BuiltinAssets()

for asset in ls_builtin_assets():
    setattr(import_builtin_asset, asset.replace('.', '_'), import_builtin(asset))


def mesh_paths(pattern):
    r"""
    Return mesh filepaths that Menpo can import that match the glob pattern.
    """
    return glob_with_suffix(pattern, mesh_types)


def image_paths(pattern):
    r"""
    Return image filepaths that Menpo can import that match the glob pattern.
    """
    return glob_with_suffix(pattern, all_image_types)


def landmark_file_paths(pattern):
    r"""
    Return landmark file filepaths that Menpo can import that match the glob
    pattern.
    """
    return glob_with_suffix(pattern, all_landmark_types)


def _import_glob_generator(pattern, extension_map, max_assets=None,
                           has_landmarks=False, landmark_resolver=same_name,
                           importer_kwargs=None, verbose=False):
    filepaths = list(glob_with_suffix(pattern, extension_map))
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
            has_landmarks=True, landmark_resolver=same_name,
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
        x.ioinfo = ioinfo

    # handle landmarks
    if has_landmarks:
        for x in built_objects:
            lm_paths = landmark_resolver(x)  # use the users fcn to find
            # paths
            if lm_paths is None:
                continue
            for group_name, lm_path in lm_paths.iteritems():
                lms = import_landmark_file(lm_path, asset=x)
                if x.n_dims == lms.n_dims:
                    x.landmarks[group_name] = lms

    # undo list-ification (if we added it!)
    if len(built_objects) == 1:
        built_objects = built_objects[0]

    if keep_importer:
        return built_objects, importer
    else:
        return built_objects


def _multi_import_generator(filepaths, extensions_map, keep_importers=False,
                            has_landmarks=False, landmark_resolver=same_name,
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


def _pathlib_glob_for_pattern(pattern):
    r"""Generator for glob matching a string path pattern

    Splits the provided ``pattern`` into a root path for pathlib and a
    subsequent glob pattern to be applied.

    Parameters
    ----------
    pattern : `str`
        Path including glob patterns. If no glob patterns are present and the
        pattern is a dir, a '**/*' pattern will be automatically added.

    Yields
    ------
    Path : A path to a file matching the provided pattern.

    Raises
    ------
    ValueError
        If the pattern doesn't contain a '*' wildcard and is not a directory
    """
    pattern = _norm_path(pattern)
    gsplit = pattern.split('*', 1)
    if len(gsplit) == 1:
        # no glob provided. Is the provided pattern a dir?
        if Path(pattern).is_dir():
            preglob = pattern
            pattern = '*'
        else:
            raise ValueError('{} is an invalid glob and '
                             'not a dir'.format(pattern))
    else:
        preglob = gsplit[0]
        pattern = '*' + gsplit[1]
    if not os.path.isdir(preglob):
        # the glob pattern is in the middle of a path segment. pair back
        # to the nearest dir and add the reminder to the pattern
        preglob, pattern_prefix = os.path.split(preglob)
        pattern  = pattern_prefix + pattern
    p = Path(preglob)
    return p.glob(str(pattern))


def glob_with_suffix(pattern, extensions_map):
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

    Yields
    ------
    filepaths : list of string
        The list of filepaths that have valid extensions.
    """
    for path in _pathlib_glob_for_pattern(pattern):
        if path.suffix in extensions_map:
            yield str(path)


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
    basename = os.path.splitext(os.path.basename(filepath))[0] + '.*'
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
            print("Warning: More than one {0} was found: "
                  "{1}. Taking the first by default".format(
                  file_type, base_names))
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


# Avoid circular imports
from menpo.io.input.extensions import (mesh_types, all_image_types,
                                       all_mesh_and_image_types,
                                       all_landmark_types)


class IOInfo(hdf5able.HDF5able):
    r"""
    Simple state object for recording IO information.
    """
    def __init__(self, filepath):
        self._path = Path(os.path.abspath(os.path.expanduser(filepath)))

    @property
    def path(self):
        return self._path

    @property
    def filepath(self):
        return str(self._path)

    @property
    def filename(self):
        return self._path.stem

    @property
    def dir(self):
        return str(self._path.parent)

    @property
    def extension(self):
        return ''.join(self._path.suffixes)

    def __str__(self):
        return 'filename: {}\nextension: {}\ndir: {}\nfilepath: {}'.format(
            self.filename, self.extension, self.dir, self.filepath
        )
