import warnings
from functools import partial
import os
from pathlib import Path
import random

from menpo.base import (menpo_src_dir_path, LazyList, partial_doc,
                        MenpoDeprecationWarning)
from menpo.compatibility import basestring
from menpo.visualize import print_progress

from ..utils import (_norm_path, _possible_extensions_from_filepath,
                     _normalize_extension)
from .extensions import (image_landmark_types, image_types, pickle_types,
                         ffmpeg_video_types)


# TODO: Remove once deprecated
def _parse_deprecated_normalise(normalise, normalize):
    if normalise is not None and normalize is not None:
        raise ValueError('normalise is now deprecated, do not set both '
                         'normalize and normalise.')
    elif normalise is not None:
        warnings.warn('normalise is no longer supported and will be removed in '
                      'a future version of Menpo. Use normalize instead.',
                      MenpoDeprecationWarning)
        normalize = normalise
    elif normalize is None:
        normalize = True
    return normalize


def _data_dir_path(base_path):
    r"""A path to the built in ./data folder on this machine.

    Returns
    -------
    path : ``pathlib.Path``
        The path to the local ./data folder
    """
    return base_path() / 'data'


def _data_path_to(data_dir_path, builtin_assets, asset_filename):
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
            asset_filename, builtin_assets()))
    return asset_path


def _import_builtin_asset(data_path_to, object_types, landmark_types,
                          asset_name, **kwargs):
    r"""Single builtin asset (landmark or image) importer.

    Imports the relevant builtin asset from the ``./data`` directory that
    ships with the project.

    Parameters
    ----------
    asset_name : `str`
        The filename of a builtin asset (see :map:`ls_builtin_assets`
        for allowed values)

    Returns
    -------
    asset :
        An instantiated :map:`Image` or :map:`LandmarkGroup` asset.
    """
    if kwargs != {}:
        normalize = _parse_deprecated_normalise(kwargs.get('normalise'),
                                                kwargs.get('normalize'))
        kwargs['normalize'] = normalize
        if 'normalise' in kwargs:
            del kwargs['normalise']

    asset_path = data_path_to(asset_name)
    # Import could be either an image or a set of landmarks, so we try
    # importing them both separately.
    try:
        return _import(asset_path, object_types,
                       landmark_ext_map=landmark_types,
                       landmark_attach_func=_import_object_attach_landmarks,
                       importer_kwargs=kwargs)
    except ValueError:
        return _import(asset_path, landmark_types,
                       importer_kwargs=kwargs)


def _ls_builtin_assets(data_dir_path):
    r"""List all the builtin asset examples provided.

    Returns
    -------
    file_paths : list of `str`
        Filenames of all assets in the data directory shipped with the
        project.
    """
    return [p.name for p in data_dir_path().glob('*') if not p.is_dir()]


def _register_importer(ext_map, extension, callable):
    r"""
    Register a new importer for the given extension.

    Parameters
    ----------
    ext_map : `{'str' -> 'callable'}` dict
        Extensions map to callable.
    extension : `str`
        File extension to support. May be multi-part e.g. '.tar.gz'
    callable : `callable`
        The callable to invoke if a file with the provided extension is
        discovered during importing. Should take a single argument (the
        filepath) and any number of kwargs.
    """
    if not isinstance(extension, basestring):
        raise ValueError('Only string type keys are supported.')
    if extension in ext_map:
        warnings.warn("Replacing an existing importer for the '{}' "
                      "extension.".format(extension))
    ext_map[_normalize_extension(extension)] = callable


register_image_importer = partial_doc(_register_importer, image_types)

register_video_importer = partial_doc(_register_importer, ffmpeg_video_types)

register_landmark_importer = partial_doc(_register_importer,
                                         image_landmark_types)

register_pickle_importer = partial_doc(_register_importer, pickle_types)


menpo_data_dir_path = partial_doc(_data_dir_path, menpo_src_dir_path)

menpo_ls_builtin_assets = partial_doc(_ls_builtin_assets, menpo_data_dir_path)

menpo_data_path_to = partial_doc(_data_path_to, menpo_data_dir_path,
                                 menpo_ls_builtin_assets)

_menpo_import_builtin_asset = partial_doc(_import_builtin_asset,
                                          menpo_data_path_to,
                                          image_types, image_landmark_types)


def image_paths(pattern):
    r"""
    Return image filepaths that Menpo can import that match the glob pattern.
    """
    return glob_with_suffix(pattern, image_types)


def video_paths(pattern):
    r"""
    Return video filepaths that Menpo can import that match the glob pattern.
    """
    return glob_with_suffix(pattern, ffmpeg_video_types)


def landmark_file_paths(pattern):
    r"""
    Return landmark file filepaths that Menpo can import that match the glob
    pattern.
    """
    return glob_with_suffix(pattern, image_landmark_types)


def pickle_paths(pattern):
    r"""
    Return pickle filepaths that Menpo can import that match the glob
    pattern.
    """
    return glob_with_suffix(pattern, pickle_types)


def same_name(path, paths_callable=landmark_file_paths):
    r"""
    Default image landmark resolver. Returns all landmarks found to have
    the same stem as the asset.
    """
    # pattern finding all landmarks with the same stem
    pattern = path.with_suffix('.*')
    # find all the assets we can with this name. Key is extension.
    return {p.suffix[1:].upper(): p for p in paths_callable(pattern)}


def same_name_video(path, frame_number,
                    paths_callable=landmark_file_paths):
    r"""
    Default video landmark resolver. Returns all landmarks found to have
    the same stem as the asset.
    """
    # pattern finding all landmarks with the same stem
    pattern = path.with_name('{}_{}.*'.format(path.stem, frame_number))
    # find all the assets we can with this name. Key is extension
    return {p.suffix[1:].upper(): p for p in paths_callable(pattern)}


def import_image(filepath, landmark_resolver=same_name, normalize=None,
                 normalise=None):
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
        image. The function should take one argument (the path to the image) and
        return a dictionary of the form ``{'group_name': 'landmark_filepath'}``
        Default finds landmarks with the same name as the image file.
    normalize : `bool`, optional
        If ``True``, normalize the image pixels between 0 and 1 and convert
        to floating point. If false, the native datatype of the image will be
        maintained (commonly `uint8`). Note that in general Menpo assumes
        :map:`Image` instances contain floating point data - if you disable
        this flag you will have to manually convert the images you import to
        floating point before doing most Menpo operations. This however can be
        useful to save on memory usage if you only wish to view or crop images.
    normalise: `bool`, optional
        Deprecated version of normalize. Please use the normalize arg.

    Returns
    -------
    images : :map:`Image` or list of
        An instantiated :map:`Image` or subclass thereof or a list of images.
    """
    normalize = _parse_deprecated_normalise(normalise, normalize)
    kwargs = {'normalize': normalize}
    return _import(filepath, image_types,
                   landmark_ext_map=image_landmark_types,
                   landmark_resolver=landmark_resolver,
                   landmark_attach_func=_import_object_attach_landmarks,
                   importer_kwargs=kwargs)


def import_video(filepath, landmark_resolver=same_name_video, normalize=None,
                 normalise=None, importer_method='ffmpeg', exact_frame_count=True):
    r"""Single video (and associated landmarks) importer.

    If a video file is found at `filepath`, returns an :map:`LazyList` wrapping
    all the frames of the video. By default, landmark files sharing the same
    filename stem will be imported and attached with a group name based on the
    extension of the landmark file appended with the frame number, although this
    behavior can be customised (see `landmark_resolver`).

    .. warning::

        This method currently uses ffmpeg to perform the importing. In order
        to recover accurate frame counts from videos it is necessary to use
        ffprobe to count the frames. This involves reading the entire
        video in to memory which may cause a delay in loading despite the lazy
        nature of the video loading within Menpo. 
        If ffprobe cannot be found, and `exact_frame_count` is ``False``,
        Menpo falls back to ffmpeg itself which is not accurate and the user
        should proceed at their own risk.

    Parameters
    ----------
    filepath : `pathlib.Path` or `str`
        A relative or absolute filepath to a video file.
    landmark_resolver : `function`, optional
        This function will be used to find landmarks for the
        video. The function should take two arguments (the path to the video and
        the frame number) and return a dictionary of the form ``{'group_name':
        'landmark_filepath'}`` Default finds landmarks with the same name as the
        video file, appended with '_{frame_number}'.
    normalize : `bool`, optional
        If ``True``, normalize the frame pixels between 0 and 1 and convert
        to floating point. If ``False``, the native datatype of the image will
        be maintained (commonly `uint8`). Note that in general Menpo assumes
        :map:`Image` instances contain floating point data - if you disable this
        flag you will have to manually convert the farmes you import to floating
        point before doing most Menpo operations. This however can be useful to
        save on memory usage if you only wish to view or crop the frames.
    normalise : `bool`, optional
        Deprecated version of normalize. Please use the normalize arg.
    importer_method : {'ffmpeg'}, optional
        A string representing the type of importer to use, by default ffmpeg
        is used.
    exact_frame_count: `bool`, optional
        If True, the import fails if ffmprobe is not available
        (reading from ffmpeg's output returns inexact frame count)

    Returns
    -------
    frames : :map:`LazyList`
        An lazy list of :map:`Image` or subclass thereof which wraps the frames
        of the video. This list can be treated as a normal list, but the frame
        is only read when the video is indexed or iterated.

    Examples
    --------
    >>> video = menpo.io.import_video('video.avi')
    >>> # Lazily load the 100th frame without reading the entire video
    >>> frame100 = video[100]
    """
    normalize = _parse_deprecated_normalise(normalise, normalize)

    kwargs = {'normalize': normalize, 'exact_frame_count':exact_frame_count}

    video_importer_methods = {'ffmpeg': ffmpeg_video_types}
    if importer_method not in video_importer_methods:
        raise ValueError('Unsupported importer method requested. Valid values '
                         'are: {}'.format(video_importer_methods.keys()))

    return _import(filepath, video_importer_methods[importer_method],
                   landmark_ext_map=image_landmark_types,
                   landmark_resolver=landmark_resolver,
                   landmark_attach_func=_import_lazylist_attach_landmarks,
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


def import_images(pattern, max_images=None, shuffle=False,
                  landmark_resolver=same_name, normalize=None,
                  normalise=None, as_generator=False, verbose=False):
    r"""Multiple image (and associated landmarks) importer.

    For each image found creates an importer than returns a :map:`Image` or
    subclass representing it. By default, landmark files sharing the same
    filename stem will be imported and attached with a group name based on the
    extension of the landmark file, although this behavior can be customised
    (see `landmark_resolver`). If the image defines a mask, this mask will be
    imported.

    Note that this is a function returns a :map:`LazyList`. Therefore, the
    function will return immediately and indexing into the returned list
    will load an image at run time. If all images should be loaded, then simply
    wrap the returned :map:`LazyList` in a Python `list`.

    Parameters
    ----------
    pattern : `str`
        A glob path pattern to search for images. Every image found to match
        the glob will be imported one by one. See :map:`image_paths` for more
        details of what images will be found.
    max_images : positive `int`, optional
        If not ``None``, only import the first ``max_images`` found. Else,
        import all.
    shuffle : `bool`, optional
        If ``True``, the order of the returned images will be randomised. If
        ``False``, the order of the returned images will be alphanumerically
        ordered.
    landmark_resolver : `function`, optional
        This function will be used to find landmarks for the
        image. The function should take one argument (the image itself) and
        return a dictionary of the form ``{'group_name': 'landmark_filepath'}``
        Default finds landmarks with the same name as the image file.
    normalize : `bool`, optional
        If ``True``, normalize the image pixels between 0 and 1 and convert
        to floating point. If false, the native datatype of the image will be
        maintained (commonly `uint8`). Note that in general Menpo assumes
        :map:`Image` instances contain floating point data - if you disable
        this flag you will have to manually convert the images you import to
        floating point before doing most Menpo operations. This however can be
        useful to save on memory usage if you only wish to view or crop images.
    normalise : `bool`, optional
        Deprecated version of normalize. Please use the normalize arg.
    as_generator : `bool`, optional
        If ``True``, the function returns a generator and assets will be yielded
        one after another when the generator is iterated over.
    verbose : `bool`, optional
        If ``True`` progress of the importing will be dynamically reported with
        a progress bar.

    Returns
    -------
    lazy_list : :map:`LazyList` or generator of :map:`Image`
        A :map:`LazyList` or generator yielding :map:`Image` instances found
        to match the glob pattern provided.

    Raises
    ------
    ValueError
        If no images are found at the provided glob.

    Examples
    --------
    Import images at 20% scale from a huge collection:

    >>> rescale_20p = lambda x: x.rescale(0.2)
    >>> images =  menpo.io.import_images('./massive_image_db/*')  # Returns immediately
    >>> images = images.map(rescale_20p)  # Returns immediately
    >>> images[0]  # Get the first image, resize, lazily loaded
    """
    normalize = _parse_deprecated_normalise(normalise, normalize)

    kwargs = {'normalize': normalize}
    return _import_glob_lazy_list(
        pattern, image_types,
        max_assets=max_images, shuffle=shuffle,
        landmark_resolver=landmark_resolver,
        landmark_ext_map=image_landmark_types,
        landmark_attach_func=_import_object_attach_landmarks,
        as_generator=as_generator,
        verbose=verbose,
        importer_kwargs=kwargs
    )


def import_videos(pattern, max_videos=None, shuffle=False,
                  landmark_resolver=same_name_video, normalize=None,
                  normalise=None, importer_method='ffmpeg',
                  exact_frame_count=True, as_generator=False, verbose=False):
    r"""Multiple video (and associated landmarks) importer.

    For each video found yields a :map:`LazyList`. By default, landmark files
    sharing the same filename stem will be imported and attached with a group
    name based on the extension of the landmark file appended with the frame
    number, although this behavior can be customised (see `landmark_resolver`).

    Note that this is a function returns a :map:`LazyList`. Therefore, the
    function will return immediately and indexing into the returned list
    will load an image at run time. If all images should be loaded, then simply
    wrap the returned :map:`LazyList` in a Python `list`.

    .. warning::

        This method currently uses ffmpeg to perform the importing. In order
        to recover accurate frame counts from videos it is necessary to use
        ffprobe to count the frames. This involves reading the entire
        video in to memory which may cause a delay in loading despite the lazy
        nature of the video loading within Menpo. 
        If ffprobe cannot be found, and `exact_frame_count` is ``False``,
        Menpo falls back to ffmpeg itself which is not accurate and the user
        should proceed at their own risk.

    Parameters
    ----------
    pattern : `str`
        A glob path pattern to search for videos. Every video found to match
        the glob will be imported one by one. See :map:`video_paths` for more
        details of what videos will be found.
    max_videos : positive `int`, optional
        If not ``None``, only import the first ``max_videos`` found. Else,
        import all.
    shuffle : `bool`, optional
        If ``True``, the order of the returned videos will be randomised. If
        ``False``, the order of the returned videos will be alphanumerically
        ordered.
    landmark_resolver : `function`, optional
        This function will be used to find landmarks for the
        video. The function should take two arguments (the path to the video and
        the frame number) and return a dictionary of the form ``{'group_name':
        'landmark_filepath'}`` Default finds landmarks with the same name as the
        video file, appended with '_{frame_number}'.
    normalize : `bool`, optional
        If ``True``, normalize the frame pixels between 0 and 1 and convert
        to floating point. If ``False``, the native datatype of the image will
        be maintained (commonly `uint8`). Note that in general Menpo assumes
        :map:`Image` instances contain floating point data - if you disable this
        flag you will have to manually convert the frames you import to floating
        point before doing most Menpo operations. This however can be useful to
        save on memory usage if you only wish to view or crop the frames.
    normalise : `bool`, optional
        Deprecated version of normalize. Please use the normalize arg.
    importer_method : {'ffmpeg'}, optional
        A string representing the type of importer to use, by default ffmpeg
        is used.
    as_generator : `bool`, optional
        If ``True``, the function returns a generator and assets will be yielded
        one after another when the generator is iterated over.
    exact_frame_count: `bool`, optional
        If True, the import fails if ffmprobe is not available
        (reading from ffmpeg's output returns inexact frame count)
    verbose : `bool`, optional
        If ``True`` progress of the importing will be dynamically reported with
        a progress bar.

    Returns
    -------
    lazy_list : :map:`LazyList` or generator of :map:`LazyList`
        A :map:`LazyList` or generator yielding :map:`LazyList` instances that
        wrap the video object.

    Raises
    ------
    ValueError
        If no videos are found at the provided glob.

    Examples
    --------
    Import videos at and rescale every frame of each video:

    >>> videos = []
    >>> for video in menpo.io.import_videos('./set_of_videos/*'):
    >>>    frames = []
    >>>    for frame in video:
    >>>        # rescale to a sensible size as we go
    >>>        frames.append(frame.rescale(0.2))
    >>>    videos.append(frames)
    """
    normalize = _parse_deprecated_normalise(normalise, normalize)

    kwargs = {'normalize': normalize, 'exact_frame_count':exact_frame_count}
    video_importer_methods = {'ffmpeg': ffmpeg_video_types}
    if importer_method not in video_importer_methods:
        raise ValueError('Unsupported importer method requested. Valid values '
                         'are: {}'.format(video_importer_methods.keys()))

    return _import_glob_lazy_list(
        pattern, video_importer_methods[importer_method],
        max_assets=max_videos, shuffle=shuffle,
        landmark_resolver=landmark_resolver,
        landmark_ext_map=image_landmark_types,
        landmark_attach_func=_import_lazylist_attach_landmarks,
        as_generator=as_generator,
        verbose=verbose,
        importer_kwargs=kwargs
    )


def import_landmark_files(pattern, max_landmarks=None, shuffle=False,
                          as_generator=False, verbose=False):
    r"""Import Multiple landmark files.

    For each landmark file found returns an importer than
    returns a :map:`LandmarkGroup`.

    Note that this is a function returns a :map:`LazyList`. Therefore, the
    function will return immediately and indexing into the returned list
    will load the landmarks at run time. If all landmarks should be loaded, then
    simply wrap the returned :map:`LazyList` in a Python `list`.

    Parameters
    ----------
    pattern : `str`
        A glob path pattern to search for landmark files. Every
        landmark file found to match the glob will be imported one by one.
        See :map:`landmark_file_paths` for more details of what landmark files
        will be found.
    max_landmarks : positive `int`, optional
        If not ``None``, only import the first ``max_landmark_files`` found.
        Else, import all.
    shuffle : `bool`, optional
        If ``True``, the order of the returned landmark files will be
        randomised. If ``False``, the order of the returned landmark files will
        be  alphanumerically  ordered.
    as_generator : `bool`, optional
        If ``True``, the function returns a generator and assets will be yielded
        one after another when the generator is iterated over.
    verbose : `bool`, optional
        If ``True`` progress of the importing will be dynamically reported.

    Returns
    -------
    lazy_list : :map:`LazyList` or generator of :map:`LandmarkGroup`
        A :map:`LazyList` or generator yielding :map:`LandmarkGroup` instances
        found to match the glob pattern provided.

    Raises
    ------
    ValueError
        If no landmarks are found at the provided glob.
    """
    return _import_glob_lazy_list(pattern, image_landmark_types,
                                  max_assets=max_landmarks, shuffle=shuffle,
                                  as_generator=as_generator, verbose=verbose)


def import_pickles(pattern, max_pickles=None, shuffle=False, as_generator=False,
                   verbose=False):
    r"""Import multiple pickle files.

    Menpo unambiguously uses ``.pkl`` as it's choice of extension for pickle
    files. Menpo also supports automatic importing of gzip compressed pickle
    files - matching files with extension ``pkl.gz`` will be automatically
    un-gzipped and imported.

    Note that this is a function returns a :map:`LazyList`. Therefore, the
    function will return immediately and indexing into the returned list
    will load the landmarks at run time. If all pickles should be loaded, then
    simply wrap the returned :map:`LazyList` in a Python `list`.

    Parameters
    ----------
    pattern : `str`
        The glob path pattern to search for pickles. Every pickle file found
        to match the glob will be imported one by one.
    max_pickles : positive `int`, optional
        If not ``None``, only import the first ``max_pickles`` found.
        Else, import all.
    shuffle : `bool`, optional
        If ``True``, the order of the returned pickles will be randomised. If
        ``False``, the order of the returned pickles will be alphanumerically
        ordered.
    as_generator : `bool`, optional
        If ``True``, the function returns a generator and assets will be yielded
        one after another when the generator is iterated over.
    verbose : `bool`, optional
        If ``True`` progress of the importing will be dynamically reported.

    Returns
    -------
    lazy_list : :map:`LazyList` or generator of Python objects
        A :map:`LazyList` or generator yielding Python objects inside the
        pickle files found to match the glob pattern provided.

    Raises
    ------
    ValueError
        If no pickles are found at the provided glob.
    """
    return _import_glob_lazy_list(pattern, pickle_types,
                                  max_assets=max_pickles, shuffle=shuffle,
                                  as_generator=as_generator, verbose=verbose)


def _import_glob_lazy_list(pattern, extension_map, max_assets=None,
                           landmark_resolver=same_name, shuffle=False,
                           as_generator=False, landmark_ext_map=None,
                           landmark_attach_func=None, importer_kwargs=None,
                           verbose=False):
    filepaths = list(glob_with_suffix(pattern, extension_map,
                                      sort=(not shuffle)))
    if shuffle:
        random.shuffle(filepaths)
    if (max_assets is not None) and max_assets <= 0:
        raise ValueError('Max elements should be positive'
                         ' ({} provided)'.format(max_assets))
    elif max_assets:
        filepaths = filepaths[:max_assets]

    n_files = len(filepaths)
    if n_files == 0:
        raise ValueError('The glob {} yields no assets'.format(pattern))

    lazy_list = LazyList([partial(_import, f, extension_map,
                                  landmark_resolver=landmark_resolver,
                                  landmark_ext_map=landmark_ext_map,
                                  landmark_attach_func=landmark_attach_func,
                                  importer_kwargs=importer_kwargs)
                          for f in filepaths])

    if verbose and as_generator:
        # wrap the generator with the progress reporter
        lazy_list = print_progress(lazy_list, prefix='Importing assets',
                                   n_items=n_files)
    elif verbose:
        print('Found {} assets, index the returned LazyList to import.'.format(
            n_files))

    if as_generator:
        return (a for a in lazy_list)
    else:
        return lazy_list


def _import_object_attach_landmarks(built_objects, landmark_resolver,
                                    landmark_ext_map=None):
    # handle landmarks
    if landmark_ext_map is not None:
        for x in built_objects:
            lm_paths = landmark_resolver(x.path)  # use the users fcn to find
            # paths
            if lm_paths is None:
                continue
            for group_name, lm_path in lm_paths.items():
                lms = _import(lm_path, landmark_ext_map, asset=x)
                if x.n_dims == lms.n_dims:
                    x.landmarks[group_name] = lms


def _import_lazylist_attach_landmarks(built_objects, landmark_resolver,
                                      landmark_ext_map=None):
    # handle landmarks
    if landmark_ext_map is not None:
        for k, x in enumerate(built_objects):
            # Use the users function to find landmarks - builds a list
            # of functions that we will map against the frames in order to
            # attach a landmark per frame.
            lm_resolvers = [partial(landmark_resolver, x.path, i)
                            for i in range(len(x))]

            def wrap_landmarks(lm_resolver, obj):
                lm_paths = lm_resolver()
                for group_name, lm_path in lm_paths.items():
                    lms = _import(lm_path, landmark_ext_map, asset=obj)
                    if obj.n_dims == lms.n_dims:
                        obj.landmarks[group_name] = lms
                return obj

            # Provide the lm_resolver for each wrap_landmarks function and then
            # lazily map against the underlying importers.
            new_ll = x.map([partial(wrap_landmarks, lmr)
                            for lmr in lm_resolvers])
            built_objects[k] = new_ll


def _import(filepath, extensions_map, landmark_resolver=same_name,
            landmark_ext_map=None, landmark_attach_func=None,
            asset=None, importer_kwargs=None):
    r"""
    Finds an importer for the filepath passed in and then calls it with the
    filepath and optionally an asset, returning either a list of assets or a
    single asset, depending on the file type.

    The type of assets returned are specified by the `extensions_map`.

    Parameters
    ----------
    filepath : `Path` or `str`
        The filepath to import.
    extensions_map : `dict` (String, :class:`menpo.io.base.Importer`)
        A map from extensions to importers. The importers are expected to be
        non-instantiated classes. The extensions are expected to
        contain the leading period eg. `.obj`.
    landmark_ext_map : `dict` (str, :map:`Importer`), optional
        If not None an attempt will be made to import annotations with
        extensions defined in this mapping. If None, no attempt will be
        made to import annotations.
    landmark_resolver : `callable`, optional
        If not None, this function will be used to find landmarks for each
        asset. The function should take one argument (the asset itself) and
        return a dictionary of the form {'group_name': 'landmark_filepath'}
    asset : `object`, optional
        Passed through to the importer callable.
    importer_kwargs : `dict`, optional
        kwargs that will be supplied to the importer if not None

    Returns
    -------
    assets : asset or list of assets
        The loaded asset or list of assets.
    """
    path = _norm_path(filepath)
    if not path.is_file():
        raise ValueError("{} is not a file".format(path))

    # below could raise ValueError as well...
    importer_callable = importer_for_filepath(path, extensions_map)
    if importer_kwargs is None:
        importer_kwargs = {}
    built_objects = importer_callable(path, asset=asset, **importer_kwargs)

    # landmarks are iterable so check for list precisely
    if not isinstance(built_objects, list):
        built_objects = [built_objects]

    # attach path if there is no x.path already.
    for x in built_objects:
        if not hasattr(x, 'path'):
            try:
                x.path = path
            except AttributeError:
                pass  # that's fine! Probably a dict/list from PickleImporter.

    if landmark_attach_func is not None:
        landmark_attach_func(built_objects, landmark_resolver,
                             landmark_ext_map=landmark_ext_map)

    if len(built_objects) == 1:
        built_objects = built_objects[0]

    return built_objects


def _pathlib_glob_for_pattern(pattern, sort=True):
    r"""Generator for glob matching a string path pattern

    Splits the provided ``pattern`` into a root path for pathlib and a
    subsequent glob pattern to be applied.

    Parameters
    ----------
    pattern : `str`
        Path including glob patterns. If no glob patterns are present and the
        pattern is a dir, a '**/*' pattern will be automatically added.
    sort : `bool`, optional
        If True, the returned paths will be sorted. If False, no guarantees are
        made about the ordering of the results.

    Yields
    ------
    Path : A path to a file matching the provided pattern.

    Raises
    ------
    ValueError
        If the pattern doesn't contain a '*' wildcard and is not a directory
    """
    pattern = _norm_path(pattern)
    pattern_str = str(pattern)
    gsplit = pattern_str.split('*', 1)
    if len(gsplit) == 1:
        # no glob provided. Is the provided pattern a dir?
        if Path(pattern).is_dir():
            preglob = pattern_str
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
    paths = p.glob(str(pattern))
    if sort:
        paths = sorted(paths)
    return paths


def glob_with_suffix(pattern, extensions_map, sort=True):
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
    sort : `bool`, optional
        If True, the returned paths will be sorted. If False, no guarantees are
        made about the ordering of the results.

    Yields
    ------
    filepaths : list of string
        The list of filepaths that have valid extensions.
    """
    for path in _pathlib_glob_for_pattern(pattern, sort=sort):
        possible_exts = _possible_extensions_from_filepath(path)
        if any([ext in extensions_map for ext in possible_exts]):
            yield path


def importer_for_filepath(filepath, extensions_map):
    r"""
    Given a filepath, return the appropriate importer as mapped by the
    extension map.

    Parameters
    ----------
    filepath : `pathlib.Path`
        The filepath to get importers for.
    extensions_map : dictionary (String, :class:`menpo.io.base.Importer`)
        A map from extensions to importers. The importers are expected to be
        a subclass of :class:`Importer`. The extensions are expected to
        contain the leading period eg. `.obj`.

    Returns
    --------
    importer: :class:`menpo.io.base.Importer` instance
        Importer as found in the `extensions_map` instantiated for the
        filepath provided.
    """
    possible_exts = _possible_extensions_from_filepath(filepath)

    # we couldn't find an importer for all the suffixes (e.g .foo.bar)
    # maybe the file stem has '.' in it? -> try again but this time just use the
    # final suffix (.bar). (Note we first try '.foo.bar' as we want to catch
    # cases like '.pkl.gz')
    importer_callable = None
    while importer_callable is None and possible_exts:
        importer_callable = extensions_map.get(possible_exts.pop(0))

    if importer_callable is None:
        raise ValueError("{} does not have a "
                         "suitable importer.".format(filepath.name))
    return importer_callable


# Create special callable that can both be called with a builtin asset name
# and has dynamic methods attached that list the available builtin assets
class BuiltinAssets(object):

    def __init__(self, import_builtin_callable):
        self.import_builtin_asset = import_builtin_callable

    def __call__(self, asset_name, **kwargs):
        return self.import_builtin_asset(asset_name, **kwargs)

import_builtin_asset = BuiltinAssets(_menpo_import_builtin_asset)

for asset in menpo_ls_builtin_assets():
    setattr(import_builtin_asset, asset.replace('.', '_'),
            partial(_menpo_import_builtin_asset, asset))
