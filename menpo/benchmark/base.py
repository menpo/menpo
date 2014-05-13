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
    if verbose:
        print('AAM Fitting:')

    # parse options
    if db_loading_options is None:
        db_loading_options = {}
    if fitting_options is None:
        fitting_options = {}
    if initialization_options is None:
        initialization_options = {}

    # group option
    group = fitting_options.pop('gt_group', 'PTS')

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
        fr = fitter.fit(i, s, gt_shape=gt_s, verbose=False)
        fitting_results.append(fr)

        # print
        if verbose:
            print_dynamic('- {0} - Image {1}/{2} ({3:.4f} --> {4:.4f})'.format(
                progress_bar_str(float(j + 1) / n_images, show_bar=False),
                j + 1, n_images, fr.initial_error, fr.final_error))
    if verbose:
        print_dynamic('- Fitting completed\n')

    # fit images
    fitting_results = FittingResultList(fitting_results)

    return fitting_results


def aam_build_benchmark(training_db_path, training_db_ext,
                        db_loading_options=None, training_options=None,
                        verbose=False):
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
    # make sure given path is correct
    database_path = os.path.abspath(os.path.expanduser(database_path))
    if os.path.isdir(database_path) is not True:
        raise ValueError('Invalid path given')

    # make sure given extension is correct
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

        # convert it to grayscale if needed
        if convert_to_grey is True and i.n_channels == 3:
            i = i.as_greyscale(mode='luminosity')

        # append it to the list
        training_images.append(i)
    if verbose:
        print_dynamic('- Loading database with {} images: Done\n'.format(
            n_files))

    return training_images