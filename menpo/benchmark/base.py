import os

import menpo.io as pio
from menpo.visualize.text_utils import print_dynamic, progress_bar_str
from menpo.fitmultilevel.aam import AAMBuilder, LucasKanadeAAMFitter
from menpo.fit.fittingresult import FittingResultList
from menpo.landmark import labeller


def aam_benchmark(training_db_path, training_db_ext, fitting_db_path,
                  fitting_db_ext, db_loading_options=None,
                  training_options=None, fitting_options=None,
                  initialization_options=None, verbose=False):
    r"""
    Trains an AAM model and fits it to a database.

    Parameters
    ----------
    training_db_path: str
        The path of the training database images.
    training_db_ext: str
        The extension (file format) of the image files. (e.g. '.png' or 'png')
    fitting_db_path: str
        The path of the fitting database images.
    fitting_db_ext: str
        The extension (file format) of the image files. (e.g. '.png' or 'png')
    db_loading_options: dictionary, optional
        A dictionary with the parameters that will be passed in the
        _load_database_images() method.
        If None, the default options will be used.
        This is an example of the dictionary with the default options:
            db_loading_options = {'crop_proportion': 0.1,
                                  'convert_to_grey': True
                                  }
        For an explanation of the options, please refer to the
        _load_database_images() documentation.

        Default: None
    training_options: dictionary, optional
        A dictionary with the parameters that will be passed in the AAMBuilder
        (:class:menpo.fitmultilevel.aam.AAMBuilder).
        If None, the default options will be used.
        This is an example of the dictionary with the default options:
            training_options = {'group': 'PTS',
                                'feature_type': 'igo',
                                'transform': PiecewiseAffine,
                                'trilist': None,
                                'normalization_diagonal': None,
                                'n_levels': 3,
                                'downscale': 2,
                                'scaled_shape_models': True,
                                'max_shape_components': None,
                                'max_appearance_components': None,
                                'boundary': 3,
                                'interpolator': 'scipy'
                                }
        For an explanation of the options, please refer to the AAMBuilder
        documentation.

        Default: None
    fitting_options: dictionary, optional
        A dictionary with the parameters that will be passed in the
        LucasKanadeAAMFitter (:class:menpo.fitmultilevel.aam.base).
        If None, the default options will be used.
        This is an example of the dictionary with the default options:
            fitting_options = {'algorithm': AlternatingInverseCompositional,
                               'md_transform': OrthoMDTransform,
                               'global_transform': AlignmentSimilarity,
                               'n_shape': None,
                               'n_appearance': None,
                               'max_iters': 50,
                               'error_type': 'me_norm'
                               }
        For an explanation of the options, please refer to the
        LucasKanadeAAMFitter documentation.

        Default: None
    initialization_options: dictionary, optional
        A dictionary with parameters that define the initialization scheme to
        be used during fitting. Currently the only supported initialization is
        perturbation on the ground truth shape with noise of specified std.
        If None, the default options will be used.
        This is an example of the dictionary with the default options:
            initialization_options = {'noise_std': 0.04,
                                      'rotation': False
                                      }
        For an explanation of the options, please refer to the perturb_shape()
        method documentation of :class:menpo.fitmultilevel.MultilevelFitter.
    verbose: boolean, optional
        If True, it prints information regarding the AAM training and fitting.

        Default: False

    Returns
    -------
    fitting_results: :class:menpo.fit.fittingresult.FittingResultList object
        A list with the FittingResult object per image.
    """
    # train aam
    aam = aam_build_benchmark(training_db_path, training_db_ext,
                              db_loading_options=db_loading_options,
                              training_options=training_options,
                              verbose=verbose)
    # fit aam
    fitting_results = aam_fit_benchmark(
        fitting_db_path, fitting_db_ext, aam,
        db_loading_options=db_loading_options, fitting_options=fitting_options,
        initialization_options=initialization_options, verbose=verbose)
    return fitting_results


def aam_fit_benchmark(fitting_db_path, fitting_db_ext, aam,
                      db_loading_options=None, fitting_options=None,
                      initialization_options=None, verbose=False):
    r"""
    Fits a trained AAM model to a database.

    Parameters
    ----------
    fitting_db_path: str
        The path of the fitting database images.
    fitting_db_ext: str
        The extension (file format) of the image files. (e.g. '.png' or 'png')
    aam: :class:menpo.fitmultilevel.aam.AAM object
        The trained AAM object. It can be generate from the
        aam_build_benchmark() method.
    db_loading_options: dictionary, optional
        A dictionary with the parameters that will be passed in the
        _load_database_images() method.
        If None, the default options will be used.
        This is an example of the dictionary with the default options:
            db_loading_options = {'crop_proportion': 0.1,
                                  'convert_to_grey': True
                                  }
        For an explanation of the options, please refer to the
        _load_database_images() documentation.

        Default: None
    fitting_options: dictionary, optional
        A dictionary with the parameters that will be passed in the
        LucasKanadeAAMFitter (:class:menpo.fitmultilevel.aam.base).
        If None, the default options will be used.
        This is an example of the dictionary with the default options:
            fitting_options = {'algorithm': AlternatingInverseCompositional,
                               'md_transform': OrthoMDTransform,
                               'global_transform': AlignmentSimilarity,
                               'n_shape': None,
                               'n_appearance': None,
                               'max_iters': 50,
                               'error_type': 'me_norm'
                               }
        For an explanation of the options, please refer to the
        LucasKanadeAAMFitter documentation.

        Default: None
    initialization_options: dictionary, optional
        A dictionary with parameters that define the initialization scheme to
        be used during fitting. Currently the only supported initialization is
        perturbation on the ground truth shape with noise of specified std.
        If None, the default options will be used.
        This is an example of the dictionary with the default options:
            initialization_options = {'noise_std': 0.04,
                                      'rotation': False
                                      }
        For an explanation of the options, please refer to the perturb_shape()
        method documentation of :class:menpo.fitmultilevel.MultilevelFitter.
    verbose: boolean, optional
        If True, it prints information regarding the AAM fitting including
        progress bar, current image error and percentage of images with errors
        less or equal than a value.

        Default: False

    Returns
    -------
    fitting_results: :class:menpo.fit.fittingresult.FittingResultList object
        A list with the FittingResult object per image.
    """
    if verbose:
        print('AAM Fitting:')
        perc1 = 0.
        perc2 = 0.

    # parse options
    if db_loading_options is None:
        db_loading_options = {}
    if fitting_options is None:
        fitting_options = {}
    if initialization_options is None:
        initialization_options = {}

    # extract some options
    group = fitting_options.pop('gt_group', 'PTS')
    max_iters = fitting_options.pop('max_iters', 50)
    error_type = fitting_options.pop('error_type', 'me_norm')

    # create fitter
    fitter = LucasKanadeAAMFitter(aam, **fitting_options)

    # load fitting images
    db_loading_options['verbose'] = verbose
    fitting_images = _load_database_images(fitting_db_path, fitting_db_ext,
                                           **db_loading_options)
    n_images = len(fitting_images)

    # fit images
    fitting_results = []
    for j, i in enumerate(fitting_images):
        # perturb shape
        gt_s = i.landmarks[group].lms
        s = fitter.perturb_shape(gt_s, **initialization_options)

        # fit
        fr = fitter.fit(i, s, gt_shape=gt_s, max_iters=max_iters,
                        error_type=error_type, verbose=False)
        fitting_results.append(fr)

        # print
        if verbose:
            if error_type is 'me_norm':
                if fr.final_error <= 0.03:
                    perc1 += 1.
                if fr.final_error <= 0.04:
                    perc2 += 1.
            print_dynamic('- {0} - [<=0.03: {1:.1f}%, <=0.04: {2:.1f}%] - '
                          'Image {3}/{4} (error: {5:.3f} --> {6:.3f})'.format(
                progress_bar_str(float(j + 1) / n_images, show_bar=False),
                perc1 * 100. / n_images, perc2 * 100. / n_images, j + 1,
                n_images, fr.initial_error, fr.final_error))
    if verbose:
        print_dynamic('- Fitting completed: [<=0.03: {0:.1f}%, <=0.04: '
                      '{1:.1f}%]\n'.format(perc1 * 100. / n_images,
                                           perc2 * 100. / n_images))

    # fit images
    fitting_results = FittingResultList(fitting_results)

    return fitting_results


def aam_build_benchmark(training_db_path, training_db_ext,
                        db_loading_options=None, training_options=None,
                        verbose=False):
    r"""
    Builds an AAM model.

    Parameters
    ----------
    training_db_path: str
        The path of the training database images.
    training_db_ext: str
        The extension (file format) of the image files. (e.g. '.png' or 'png')
    db_loading_options: dictionary, optional
        A dictionary with the parameters that will be passed in the
        _load_database_images() method.
        If None, the default options will be used.
        This is an example of the dictionary with the default options:
            db_loading_options = {'crop_proportion': 0.1,
                                  'convert_to_grey': True
                                  }
        For an explanation of the options, please refer to the
        _load_database_images() documentation.

        Default: None
    training_options: dictionary, optional
        A dictionary with the parameters that will be passed in the AAMBuilder
        (:class:menpo.fitmultilevel.aam.AAMBuilder).
        If None, the default options will be used.
        This is an example of the dictionary with the default options:
            training_options = {'group': 'PTS',
                                'feature_type': 'igo',
                                'transform': PiecewiseAffine,
                                'trilist': None,
                                'normalization_diagonal': None,
                                'n_levels': 3,
                                'downscale': 2,
                                'scaled_shape_models': True,
                                'max_shape_components': None,
                                'max_appearance_components': None,
                                'boundary': 3,
                                'interpolator': 'scipy'
                                }
        For an explanation of the options, please refer to the AAMBuilder
        documentation.

        Default: None
    verbose: boolean, optional
        If True, it prints information regarding the AAM training.

        Default: False

    Returns
    -------
    aam: :class:menpo.fitmultilevel.aam.AAM object
        The trained AAM model.
    """
    if verbose:
        print('AAM Training:')

    # parse options
    if db_loading_options is None:
        db_loading_options = {}
    if training_options is None:
        training_options = {}

    # load training images
    db_loading_options['verbose'] = verbose
    training_images = _load_database_images(training_db_path, training_db_ext,
                                            **db_loading_options)

    # group option
    group = training_options.pop('group', None)

    # trilist option
    trilist = training_options.pop('trilist', None)
    if trilist is not None:
        labeller(training_images[0], 'PTS', trilist)
        training_options['trilist'] = \
            training_images[0].landmarks[trilist.__name__].lms.trilist

    # build aam
    aam = AAMBuilder(**training_options).build(training_images, group=group,
                                               verbose=verbose)

    return aam


def _load_database_images(database_path, files_extension, crop_proportion=0.1,
                          convert_to_grey=True, verbose=False):
    r"""
    Loads the database images, crops them and converts them.

    Parameters
    ----------
    database_path: str
        The path of the database images.
    files_extension: str
        The extension (file format) of the image files. (e.g. '.png' or 'png')
    crop_proportion: float, optional
        Additional padding to be added all around the landmarks bounds when the
        images are cropped. It is defined as a proportion of the landmarks'
        range.

        Default: 0.1
    convert_to_grey: boolean, optional
        If True, the RGB images will be converted to greyscale.

        Default: True
    verbose: boolean, optional
        If True, it prints a progress percentage bar.

        Default: False

    Returns
    -------
    training_images: list of :class:MaskedImage objects
        A list of the loaded images.

    Raises
    ------
    ValueError
        Invalid path given
    ValueError
        No {files_extension} files in given path
    """
    # check given path
    database_path = os.path.abspath(os.path.expanduser(database_path))
    if os.path.isdir(database_path) is not True:
        raise ValueError('Invalid path given')

    # check given extension
    if files_extension[0] is not '.' and len(files_extension) == 3:
        files_extension = '.{}'.format(files_extension)

    # create final path
    final_path = os.path.abspath(os.path.expanduser(os.path.join(
        database_path, '*{}'.format(files_extension))))

    # find number of files
    n_files = len(pio.image_paths(final_path))
    if n_files < 1:
        raise ValueError('No {} files in given path'.format(files_extension))

    # load images
    training_images = []
    for c, i in enumerate(pio.import_images(final_path)):
        # print progress bar
        if verbose:
            print_dynamic('- Loading database with {} images: {}'.format(
                n_files, progress_bar_str(float(c + 1) / n_files,
                                          show_bar=True)))

        # crop image
        i.crop_to_landmarks_proportion(crop_proportion)

        # convert it to greyscale if needed
        if convert_to_grey is True and i.n_channels == 3:
            i = i.as_greyscale(mode='luminosity')

        # append it to the list
        training_images.append(i)
    if verbose:
        print_dynamic('- Loading database with {} images: Done\n'.format(
            n_files))
    return training_images