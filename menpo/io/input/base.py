import abc
import os
from pathlib import Path

from ..utils import _norm_path
from menpo.base import menpo_src_dir_path
from menpo.visualize import progress_bar_str, print_dynamic


def data_dir_path():
    r"""A path to the Menpo built in ./data folder on this machine.

    Returns
    -------
    ``pathlib.Path``
        The path to the local Menpo ./data folder

    """
    return menpo_src_dir_path() / 'data'


def data_path_to(asset_filename):
    r"""
    The path to a builtin asset in the ./data folder on this machine.

    Parameters
    ----------
    asset_filename : `str`
        The filename (with extension) of a file builtin to Menpo. The full
        set of allowed names is given by :func:`ls_builtin_assets()`

    Returns
    -------
    data_path : `pathlib.Path`
        The path to a given asset in the ./data folder

    Raises
    ------
    ValueError
        If the asset_filename doesn't exist in the `data` folder.

    """
    asset_path = data_dir_path() / asset_filename
    if not asset_path.is_file():
        raise ValueError("{} is not a builtin asset: {}".format(
            asset_filename, ls_builtin_assets()))
    return asset_path


def same_name(asset):
    r"""
    Menpo's default landmark resolver. Returns all landmarks found to have
    the same stem as the asset.
    """
    # pattern finding all landmarks with the same stem
    pattern = asset.path.with_suffix('.*')
    # find all the landmarks we can with this name. Key is ext (without '.')
    return {p.suffix[1:].upper(): p for p in landmark_file_paths(pattern)}


def import_image(filepath, landmark_resolver=same_name, normalise=True):
    r"""Single image (and associated landmarks) importer.

    If an image file is found at `filepath`, returns an :map:`Image` or
    subclass representing it. By default, landmark files sharing the same
    filename stem will be imported and attached with a group name based on the
    extension of the landmark file, although this behavior can be customised
    (see `landmark_resolver`). If the image defines a mask, this mask will be
    imported.

    Parameters
    ----------
    filepath : `pathlib.Path` or `str`
        A relative or absolute filepath to an image file.
    landmark_resolver : `function`, optional
        This function will be used to find landmarks for the
        image. The function should take one argument (the image itself) and
        return a dictionary of the form ``{'group_name': 'landmark_filepath'}``
        Default finds landmarks with the same name as the image file.
    normalise : `bool`, optional
        If ``True``, normalise the image pixels between 0 and 1 and convert
        to floating point. If false, the native datatype of the image will be
        maintained (commonly `uint8`). Note that in general Menpo assumes
        :map:`Image` instances contain floating point data - if you disable
        this flag you will have to manually convert the images you import to
        floating point before doing most Menpo operations. This however can be
        useful to save on memory usage if you only wish to view or crop images.

    Returns
    -------
    images : :map:`Image` or list of
        An instantiated :map:`Image` or subclass thereof or a list of images.
    """
    kwargs = {'normalise': normalise}
    return _import(filepath, image_types,
                   landmark_ext_map=image_landmark_types,
                   landmark_resolver=landmark_resolver,
                   importer_kwargs=kwargs)


def import_landmark_file(filepath, asset=None):
    r"""Single landmark group importer.

    If a landmark file is found at ``filepath``, returns a
    :map:`LandmarkGroup` representing it.

    Parameters
    ----------
    filepath : `pathlib.Path` or `str`
        A relative or absolute filepath to an landmark file.

    Returns
    -------
    landmark_group : :map:`LandmarkGroup`
        The :map:`LandmarkGroup` that the file format represents.
    """
    return _import(filepath, image_landmark_types, asset=asset)


def import_pickle(filepath):
    r"""Import a pickle file of arbitrary Python objects.

    Menpo unambiguously uses ``.pkl`` as it's choice of extension for Pickle
    files. Menpo also supports automatic importing and exporting of gzip
    compressed pickle files - just choose a ``filepath`` ending ``pkl.gz`` and
    gzip compression will automatically be applied. Compression can massively
    reduce the filesize of a pickle file at the cost of longer import and
    export times.

    Parameters
    ----------
    filepath : `pathlib.Path` or `str`
        A relative or absolute filepath to a ``.pkl`` or ``.pkl.gz`` file.

    Returns
    -------
    object : `object`
        Whatever Python objects are present in the Pickle file
    """
    return _import(filepath, pickle_types)


def import_images(pattern, max_images=None, landmark_resolver=same_name,
                  normalise=True, verbose=False):
    r"""Multiple image (and associated landmarks) importer.

    For each image found yields an :map:`Image` or
    subclass representing it. By default, landmark files sharing the same
    filename stem will be imported and attached with a group name based on the
    extension of the landmark file, although this behavior can be customised
    (see `landmark_resolver`). If the image defines a mask, this mask will be
    imported.

    Note that this is a generator function. This allows for pre-processing
    of data to take place as data is imported (e.g. cropping images to
    landmarks as they are imported for memory efficiency).

    Parameters
    ----------
    pattern : `str`
        A glob path pattern to search for images. Every image found to match
        the glob will be imported one by one. See :map:`image_paths` for more
        details of what images will be found.
    max_images : positive `int`, optional
        If not ``None``, only import the first ``max_images`` found. Else,
        import all.
    landmark_resolver : `function`, optional
        This function will be used to find landmarks for the
        image. The function should take one argument (the image itself) and
        return a dictionary of the form ``{'group_name': 'landmark_filepath'}``
        Default finds landmarks with the same name as the image file.
    normalise : `bool`, optional
        If ``True``, normalise the image pixels between 0 and 1 and convert
        to floating point. If false, the native datatype of the image will be
        maintained (commonly `uint8`). Note that in general Menpo assumes
        :map:`Image` instances contain floating point data - if you disable
        this flag you will have to manually convert the images you import to
        floating point before doing most Menpo operations. This however can be
        useful to save on memory usage if you only wish to view or crop images.
    verbose : `bool`, optional
        If ``True`` progress of the importing will be dynamically reported with
        a progress bar.

    Returns
    -------
    generator : `generator` yielding :map:`Image` or list of
        Generator yielding :map:`Image` instances found to match the glob
        pattern provided.

    Raises
    ------
    ValueError
        If no images are found at the provided glob.

    Examples
    --------
    Import images at 20% scale from a huge collection:

    >>> images = []
    >>> for img in menpo.io.import_images('./massive_image_db/*'):
    >>>    # rescale to a sensible size as we go
    >>>    images.append(img.rescale(0.2))
    """
    kwargs = {'normalise': normalise}
    for asset in _import_glob_generator(pattern, image_types,
                                        max_assets=max_images,
                                        landmark_resolver=landmark_resolver,
                                        landmark_ext_map=image_landmark_types,
                                        verbose=verbose,
                                        importer_kwargs=kwargs):
        yield asset


def import_landmark_files(pattern, max_landmarks=None, verbose=False):
    r"""Multiple landmark file import generator.

    Note that this is a generator function.

    Parameters
    ----------
    pattern : `str`
        A glob path pattern to search for landmark files. Every
        landmark file found to match the glob will be imported one by one.
        See :map:`landmark_file_paths` for more details of what landmark files
        will be found.

    max_landmark_files : positive `int`, optional
        If not ``None``, only import the first ``max_landmark_files`` found.
        Else, import all.

    verbose : `bool`, optional
        If ``True`` progress of the importing will be dynamically reported.

    Returns
    -------
    generator : `generator` yielding :map:`LandmarkGroup`
        Generator yielding :map:`LandmarkGroup` instances found to match the
        glob pattern provided.

    Raises
    ------
    ValueError
        If no landmarks are found at the provided glob.
    """
    for asset in _import_glob_generator(pattern, image_landmark_types,
                                        max_assets=max_landmarks,
                                        verbose=verbose):
        yield asset


def import_pickles(pattern, max_pickles=None, verbose=False):
    r"""Multiple pickle file import generator.

    Note that this is a generator function.

    Menpo unambiguously uses ``.pkl`` as it's choice of extension for pickle
    files. Menpo also supports automatic importing of gzip compressed pickle
    files - matching files with extension ``pkl.gz`` will be automatically
    un-gzipped and imported.

    Parameters
    ----------
    pattern : `str`
        The glob path pattern to search for pickles. Every pickle file found
        to match the glob will be imported one by one.

    max_pickles : positive `int`, optional
        If not ``None``, only import the first ``max_pickles`` found.
        Else, import all.

    verbose : `bool`, optional
        If ``True`` progress of the importing will be dynamically reported.

    Returns
    -------
    generator : generator yielding `object`
        Generator yielding whatever Python object is present in the pickle
        files that match the glob pattern provided.

    Raises
    ------
    ValueError
        If no pickles are found at the provided glob.

    """
    for asset in _import_glob_generator(pattern, pickle_types,
                                        max_assets=max_pickles,
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
    return _import(asset_path, image_types,
                   landmark_ext_map=image_landmark_types)


def ls_builtin_assets():
    r"""List all the builtin asset examples provided in Menpo.

    Returns
    -------
    list of strings
        Filenames of all assets in the data directory shipped with Menpo

    """
    return [p.name for p in data_dir_path().glob('*')]


def import_builtin(x):

    def execute():
        return _import_builtin_asset(x)

    return execute


class BuiltinAssets(object):

    def __call__(self, asset_name):
        return _import_builtin_asset(asset_name)

import_builtin_asset = BuiltinAssets()

for asset in ls_builtin_assets():
    setattr(import_builtin_asset, asset.replace('.', '_'),
            import_builtin(asset))


def image_paths(pattern):
    r"""
    Return image filepaths that Menpo can import that match the glob pattern.
    """
    return glob_with_suffix(pattern, image_types)


def landmark_file_paths(pattern):
    r"""
    Return landmark file filepaths that Menpo can import that match the glob
    pattern.
    """
    return glob_with_suffix(pattern, image_landmark_types)


def _import_glob_generator(pattern, extension_map, max_assets=None,
                           landmark_resolver=same_name,
                           landmark_ext_map=None, importer_kwargs=None,
                           verbose=False):
    filepaths = list(glob_with_suffix(pattern, extension_map))
    if max_assets:
        filepaths = filepaths[:max_assets]
    n_files = len(filepaths)
    if n_files == 0:
        raise ValueError('The glob {} yields no assets'.format(pattern))
    for i, asset in enumerate(_multi_import_generator(filepaths, extension_map,
                              landmark_resolver=landmark_resolver,
                              landmark_ext_map=landmark_ext_map,
                              importer_kwargs=importer_kwargs)):
        if verbose:
            print_dynamic('- Loading {} assets: {}'.format(
                n_files, progress_bar_str(float(i + 1) / n_files,
                                          show_bar=True)))
        yield asset


def _import(filepath, extensions_map, keep_importer=False,
            landmark_resolver=same_name,
            landmark_ext_map=None, asset=None, importer_kwargs=None):
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
    landmark_ext_map : dictionary (str, :map:`Importer`), optional
        If not None an attempt will be made to import annotations with
        extensions defined in this mapping. If None, no attempt will be
        made to import annotations.
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
    path = Path(_norm_path(filepath))
    if not path.is_file():
        raise ValueError("{} is not a file".format(path))
    # below could raise ValueError as well...
    importer = importer_for_filepath(path, extensions_map,
                                     importer_kwargs=importer_kwargs)
    if asset is not None:
        built_objects = importer.build(asset=asset)
    else:
        built_objects = importer.build()
    # landmarks are iterable so check for list precisely
    # enforce a list to make processing consistent
    if not isinstance(built_objects, list):
        built_objects = [built_objects]

    # attach path if there is no x.path already.
    for x in built_objects:
        if not hasattr(x, 'path'):
            try:
                x.path = path
            except AttributeError:
                pass  # that's fine! Probably a dict/list from PickleImporter.
    # handle landmarks
    if landmark_ext_map is not None:
        for x in built_objects:
            lm_paths = landmark_resolver(x)  # use the users fcn to find
            # paths
            if lm_paths is None:
                continue
            for group_name, lm_path in lm_paths.iteritems():
                lms = _import(lm_path, landmark_ext_map, asset=x)
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
                            landmark_resolver=same_name,
                            landmark_ext_map=None, importer_kwargs=None):
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
    landmark_ext_map : dictionary (str, :map:`Importer`), optional
        If not None an attempt will be made to import annotations with
        extensions defined in this mapping. If None, no attempt will be
        made to import annotations.
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
                           landmark_resolver=landmark_resolver,
                           landmark_ext_map=landmark_ext_map,
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
        pattern = pattern_prefix + pattern
    p = Path(preglob)
    return sorted(p.glob(str(pattern)))


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
        # we want to extract '.pkl.gz' as an extension - for this we need to
        # use suffixes and join.
        # .suffix only takes
        if ''.join(path.suffixes) in extensions_map:
            yield path


def importer_for_filepath(filepath, extensions_map, importer_kwargs=None):
    r"""
    Given a filepath, return the appropriate importer as mapped by the
    extension map.

    Parameters
    ----------
    filepath : `pathlib.Path`
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
    suffix = ''.join(filepath.suffixes)
    importer_type = extensions_map.get(suffix)
    # we couldn't find an importer for all the suffixes (e.g .foo.bar)
    # maybe the file stem has '.' in it? -> try again but this time just use the
    # final suffix (.bar). (Note we first try '.foo.bar' as we want to catch
    # cases like 'pkl.gz')
    if importer_type is None and len(filepath.suffixes) > 1:
        suffix = filepath.suffix
        importer_type = extensions_map.get(suffix)
    if importer_type is None:
        raise ValueError("{} does not have a "
                         "suitable importer.".format(suffix))
    if importer_kwargs is not None:
        return importer_type(str(filepath), **importer_kwargs)
    else:
        return importer_type(str(filepath))


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


# Avoid circular imports
from menpo.io.input.extensions import (image_landmark_types, image_types,
                                       pickle_types)
