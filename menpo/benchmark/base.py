import os
import numpy as np

import menpo.io as mio
from menpo.visualize.text_utils import print_dynamic, progress_bar_str
from menpo.fitmultilevel.aam import AAMBuilder, LucasKanadeAAMFitter
from menpo.fitmultilevel.clm import CLMBuilder, GradientDescentCLMFitter
from menpo.fitmultilevel.sdm import SDMTrainer, SDMFitter
from menpo.fit.fittingresult import FittingResultList
from menpo.landmark import labeller
from menpo.visualize.base import GraphPlotter


def aam_fit_benchmark(fitting_images, aam, fitting_options=None,
                      perturb_options=None, verbose=False):
    r"""
    Fits a trained AAM model to a database.

    Parameters
    ----------
    fitting_images: list of :class:MaskedImage objects
        A list of the fitting images.
    aam: :class:menpo.fitmultilevel.aam.AAM object
        The trained AAM object. It can be generated from the
        aam_build_benchmark() method.
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
    bounding_boxes: list of (2, 2) ndarray, optional
        If provided, fits will be initialized from a bounding box. If
        None, perturbation of ground truth will be used instead.
        can be provided). Interpreted as [[min_x, min_y], [max_x, max_y]].
    perturb_options: dictionary, optional
        A dictionary with parameters that control the perturbation on the
        ground truth shape with noise of specified std. Note that if
        bounding_box is provided perturb_options is ignored and not used.
        If None, the default options will be used.
        This is an example of the dictionary with the default options:
            initialization_options = {'noise_std': 0.04,
                                      'rotation': False
                                      }
        For an explanation of the options, please refer to the perturb_shape()
        method documentation of :map:`MultilevelFitter`.
    verbose: bool, optional
        If True, it prints information regarding the AAM fitting including
        progress bar, current image error and percentage of images with errors
        less or equal than a value.

        Default: False

    Returns
    -------
    fitting_results: :map:`FittingResultList`
        A list with the :map:`FittingResult` object per image.
    """
    if verbose:
        print('AAM Fitting:')
        perc1 = 0.
        perc2 = 0.

    # parse options
    if fitting_options is None:
        fitting_options = {}
    if perturb_options is None:
        perturb_options = {}

    # extract some options
    group = fitting_options.pop('gt_group', 'PTS')
    max_iters = fitting_options.pop('max_iters', 50)
    error_type = fitting_options.pop('error_type', 'me_norm')

    # create fitter
    fitter = LucasKanadeAAMFitter(aam, **fitting_options)

    # fit images
    n_images = len(fitting_images)
    fitting_results = []
    for j, i in enumerate(fitting_images):
        # perturb shape
        gt_s = i.landmarks[group].lms
        if 'bbox' in i.landmarks:
            # shape from bounding box
            s = fitter.obtain_shape_from_bb(i.landmarks['bbox'].lms.points)
        else:
            # shape from perturbation
            s = fitter.perturb_shape(gt_s, **perturb_options)
        # fit
        fr = fitter.fit(i, s, gt_shape=gt_s, max_iters=max_iters,
                        error_type=error_type, verbose=False)
        fitting_results.append(fr)

        # print
        if verbose:
            if error_type == 'me_norm':
                if fr.final_error <= 0.03:
                    perc1 += 1.
                if fr.final_error <= 0.04:
                    perc2 += 1.
            elif error_type == 'rmse':
                if fr.final_error <= 0.05:
                    perc1 += 1.
                if fr.final_error <= 0.06:
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
    fitting_results_list = FittingResultList(fitting_results)

    return fitting_results_list


def aam_build_benchmark(training_images, training_options=None, verbose=False):
    r"""
    Builds an AAM model.

    Parameters
    ----------
    training_images: list of :class:MaskedImage objects
        A list of the training images.
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
                                'pyramid_on_features': True,
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
    if training_options is None:
        training_options = {}

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


def clm_fit_benchmark(fitting_images, clm, fitting_options=None,
                      perturb_options=None, verbose=False):
    r"""
    Fits a trained CLM model to a database.

    Parameters
    ----------
    fitting_images: list of :class:MaskedImage objects
        A list of the fitting images.
    clm: :class:menpo.fitmultilevel.clm.CLM object
        The trained CLM object. It can be generated from the
        clm_build_benchmark() method.
    fitting_options: dictionary, optional
        A dictionary with the parameters that will be passed in the
        GradientDescentCLMFitter (:class:menpo.fitmultilevel.clm.base).
        If None, the default options will be used.
        This is an example of the dictionary with the default options:
            fitting_options = {'algorithm': RegularizedLandmarkMeanShift,
                               'pdm_transform': OrthoPDM,
                               'global_transform': AlignmentSimilarity,
                               'n_shape': None,
                               'max_iters': 50,
                               'error_type': 'me_norm'
                               }
        For an explanation of the options, please refer to the
        GradientDescentCLMFitter documentation.

        Default: None
    bounding_boxes: list of (2, 2) ndarray, optional
        If provided, fits will be initialized from a bounding box. If
        None, perturbation of ground truth will be used instead.
        can be provided). Interpreted as [[min_x, min_y], [max_x, max_y]].
    perturb_options: dictionary, optional
        A dictionary with parameters that control the perturbation on the
        ground truth shape with noise of specified std. Note that if
        bounding_box is provided perturb_options is ignored and not used.
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
        print('CLM Fitting:')
        perc1 = 0.
        perc2 = 0.

    # parse options
    if fitting_options is None:
        fitting_options = {}

    # extract some options
    group = fitting_options.pop('gt_group', 'PTS')
    max_iters = fitting_options.pop('max_iters', 50)
    error_type = fitting_options.pop('error_type', 'me_norm')

    # create fitter
    fitter = GradientDescentCLMFitter(clm, **fitting_options)

    # fit images
    n_images = len(fitting_images)
    fitting_results = []
    for j, i in enumerate(fitting_images):
        # perturb shape
        gt_s = i.landmarks[group].lms
        if 'bbox' in i.landmarks:
            # shape from bounding box
            s = fitter.obtain_shape_from_bb(i.landmarks['bbox'].lms.points)
        else:
            # shape from perturbation
            s = fitter.perturb_shape(gt_s, **perturb_options)
        # fit
        fr = fitter.fit(i, s, gt_shape=gt_s, max_iters=max_iters,
                        error_type=error_type, verbose=False)
        fitting_results.append(fr)

        # print
        if verbose:
            if error_type == 'me_norm':
                if fr.final_error <= 0.03:
                    perc1 += 1.
                if fr.final_error <= 0.04:
                    perc2 += 1.
            elif error_type == 'rmse':
                if fr.final_error <= 0.05:
                    perc1 += 1.
                if fr.final_error <= 0.06:
                    perc2 += 1.
            print_dynamic('- {0} - [<=0.03: {1:.1f}%, <=0.04: {2:.1f}%] - '
                          'Image {3}/{4} (error: {5:.3f} --> {6:.3f})'.format(
                          progress_bar_str(float(j + 1) / n_images,
                                           show_bar=False),
                          perc1 * 100. / n_images, perc2 * 100. / n_images,
                          j + 1, n_images, fr.initial_error, fr.final_error))
    if verbose:
        print_dynamic('- Fitting completed: [<=0.03: {0:.1f}%, <=0.04: '
                      '{1:.1f}%]\n'.format(perc1 * 100. / n_images,
                                           perc2 * 100. / n_images))

    # fit images
    fitting_results_list = FittingResultList(fitting_results)

    return fitting_results_list


def clm_build_benchmark(training_images, training_options=None, verbose=False):
    r"""
    Builds an CLM model.

    Parameters
    ----------
    training_images: list of :class:MaskedImage objects
        A list of the training images.
    training_options: dictionary, optional
        A dictionary with the parameters that will be passed in the CLMBuilder
        (:class:menpo.fitmultilevel.clm.CLMBuilder).
        If None, the default options will be used.
        This is an example of the dictionary with the default options:
            training_options = {'group': 'PTS',
                                'classifier_type': linear_svm_lr,
                                'patch_shape': (5, 5),
                                'feature_type': sparse_hog,
                                'normalization_diagonal': None,
                                'n_levels': 3,
                                'downscale': 1.1,
                                'scaled_shape_models': True,
                                'pyramid_on_features': True,
                                'max_shape_components': None,
                                'boundary': 3,
                                'interpolator': 'scipy'
                                }
        For an explanation of the options, please refer to the CLMBuilder
        documentation.

        Default: None
    verbose: boolean, optional
        If True, it prints information regarding the CLM training.

        Default: False

    Returns
    -------
    clm: :class:menpo.fitmultilevel.clm.CLM object
        The trained CLM model.
    """
    if verbose:
        print('CLM Training:')

    # parse options
    if training_options is None:
        training_options = {}

    # group option
    group = training_options.pop('group', None)

    # build aam
    aam = CLMBuilder(**training_options).build(training_images, group=group,
                                               verbose=verbose)

    return aam


def sdm_fit_benchmark(fitting_images, fitter, perturb_options=None,
                      fitting_options=None, verbose=False):
    r"""
    Fits a trained SDM to a database.

    Parameters
    ----------
    fitting_images: list of :class:MaskedImage objects
        A list of the fitting images.
    fitter: :map:`SDMFitter`
        The trained AAM object. It can be generated from the
        aam_build_benchmark() method.
    fitting_options: dictionary, optional
        A dictionary with the parameters that will be passed in the
        LucasKanadeAAMFitter (:class:menpo.fitmultilevel.sdm.base).
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
    bounding_boxes: list of (2, 2) ndarray, optional
        If provided, fits will be initialized from a bounding box. If
        None, perturbation of ground truth will be used instead.
        can be provided). Interpreted as [[min_x, min_y], [max_x, max_y]].
    perturb_options: dictionary, optional
        A dictionary with parameters that control the perturbation on the
        ground truth shape with noise of specified std. Note that if
        bounding_box is provided perturb_options is ignored and not used.
        If None, the default options will be used.
        This is an example of the dictionary with the default options:
            initialization_options = {'noise_std': 0.04,
                                      'rotation': False
                                      }
        For an explanation of the options, please refer to the perturb_shape()
        method documentation of :map:`MultilevelFitter`.
    verbose: bool, optional
        If True, it prints information regarding the AAM fitting including
        progress bar, current image error and percentage of images with errors
        less or equal than a value.

        Default: False

    Returns
    -------
    fitting_results: :map:`FittingResultList`
        A list with the :map:`FittingResult` object per image.
    """
    if verbose:
        print('SDM Fitting:')
        perc1 = 0.
        perc2 = 0.

    # parse options
    if fitting_options is None:
        fitting_options = {}
    if perturb_options is None:
        perturb_options = {}

    # extract some options
    group = fitting_options.pop('gt_group', 'PTS')
    error_type = fitting_options.pop('error_type', 'me_norm')

    # fit images
    n_images = len(fitting_images)
    fitting_results = []
    for j, i in enumerate(fitting_images):
        # perturb shape
        gt_s = i.landmarks[group].lms
        if 'bbox' in i.landmarks:
            # shape from bounding box
            s = fitter.obtain_shape_from_bb(i.landmarks['bbox'].lms.points)
        else:
            # shape from perturbation
            s = fitter.perturb_shape(gt_s, **perturb_options)
        # fit
        fr = fitter.fit(i, s, gt_shape=gt_s, error_type=error_type,
                        verbose=False)
        fitting_results.append(fr)

        # print
        if verbose:
            if error_type == 'me_norm':
                if fr.final_error <= 0.03:
                    perc1 += 1.
                if fr.final_error <= 0.04:
                    perc2 += 1.
            elif error_type == 'rmse':
                if fr.final_error <= 0.05:
                    perc1 += 1.
                if fr.final_error <= 0.06:
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
    fitting_results_list = FittingResultList(fitting_results)

    return fitting_results_list


def sdm_build_benchmark(training_images, training_options=None, verbose=False):
    r"""
    Builds an SDM model.

    Parameters
    ----------
    training_images: list of :class:MaskedImage objects
        A list of the training images.
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
                                'pyramid_on_features': True,
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
        print('SDM Training:')

    # parse options
    if training_options is None:
        training_options = {}

    # group option
    group = training_options.pop('group', None)

    # build sdm
    sdm = SDMTrainer(**training_options).train(training_images, group=group,
                                               verbose=verbose)
    return sdm


def load_database(database_path, bounding_boxes=None,
                  db_loading_options=None, verbose=False):
    r"""
    Loads the database images, crops them and converts them.

    Parameters
    ----------
    database_path: str
        The path of the database images.
    db_loading_options: dictionary, optional
        A dictionary with options related to image loading.
        If None, the default options will be used.
        This is an example of the dictionary with the default options:
            training_options = {'crop_proportion': 0.1,
                                'convert_to_grey': True,
                                }

        crop_proportion (float) defines the additional padding to be added all
        around the landmarks bounds when the images are cropped. It is defined
        as a proportion of the landmarks' range.

        convert_to_grey (boolean)defines whether the images will be converted
        to greyscale.

        Default: None
    verbose: boolean, optional
        If True, it prints a progress percentage bar.

        Default: False

    Returns
    -------
    images: list of :class:MaskedImage objects
        A list of the loaded images.

    Raises
    ------
    ValueError
        Invalid path given
    ValueError
        No {files_extension} files in given path
    """
    # check input options
    if db_loading_options is None:
        db_loading_options = {}

    # check given path
    database_path = os.path.abspath(os.path.expanduser(database_path))
    if os.path.isdir(database_path) is not True:
        raise ValueError('Invalid path given')

    # create final path
    final_path = os.path.join(database_path, '*')

    # get options
    crop_proportion = db_loading_options.pop('crop_proportion', 0.5)
    convert_to_grey = db_loading_options.pop('convert_to_grey', True)

    # load images
    images = []
    for i in mio.import_images(final_path, verbose=verbose):
        # If we have bounding boxes then we need to make sure we crop to them!
        # If we don't crop to the bounding box then we might crop out part of
        # the image the bounding box belongs to.
        landmark_group_label = None
        if bounding_boxes is not None:
            fname = i.ioinfo.filename + i.ioinfo.extension
            landmark_group_label = 'bbox'
            i.landmarks[landmark_group_label] = bounding_boxes[fname].detector

        # crop image
        i.crop_to_landmarks_proportion_inplace(crop_proportion,
                                               group=landmark_group_label)

        # convert it to greyscale if needed
        if convert_to_grey and i.n_channels == 3:
            i = i.as_greyscale(mode='luminosity')

        # append it to the list
        images.append(i)
    if verbose:
        print("\nAssets loaded.")
    return images


def convert_fitting_results_to_ced(fitting_results, max_error_bin=0.05,
                                   bins_error_step=0.005):
    r"""
    Method that given a fitting_result object, it converts it to the
    cumulative error distribution values that can be used for plotting.

    Parameters
    ----------
    fitting_results: :class:menpo.fit.fittingresult.FittingResultList object
        A list with the FittingResult object per image.
    max_error_bin: float, Optional
        The maximum error of the distribution.

        Default: 0.05
    bins_error_step: float, Optional
        The sampling step of the distribution values.

        Default: 0.005

    Returns
    -------
    final_error_dist: list
        Cumulative distribution values of the final errors.
    initial_error_dist: list
        Cumulative distribution values of the initial errors.
    """
    error_bins = np.arange(0., max_error_bin + bins_error_step,
                           bins_error_step)
    final_error_dist = np.array(
        [float(np.sum(fitting_results.final_error <= k)) /
         len(fitting_results.final_error) for k in error_bins])
    initial_error_dist = np.array(
        [float(np.sum(fitting_results.initial_error <= k)) /
         len(fitting_results.final_error) for k in error_bins])
    return final_error_dist, initial_error_dist, error_bins


def plot_fitting_curves(x_axis, ceds, title, figure_id=None, new_figure=False,
                        x_label='Point-to-Point Normalized RMS Error',
                        y_limit=1, x_limit=0.05, legend=None, **kwargs):
    r"""
    Method that plots Cumulative Error Distributions in a single figure.

    Parameters
    ----------
    x_axis: ndarray
        The horizontal axis values (errors).
    ceds: list of ndarrays
        The vertical axis values (percentages).
    title: string
        The plot title.
    figure_id, Optional
        A figure handle.

        Default: None
    new_figure: boolean, Optional
        If True, a new figure window will be created.

        Default: False
    y_limit: float, Optional
        The maximum value of the vertical axis.

        Default: 1
    x_limit: float, Optional
        The maximum value of the vertical axis.

        Default: 0.05
    x_label: string
        The label of the horizontal axis.

        Default: 'Point-to-Point Normalized RMS Error'
    legend: list of strings or None
        The legend of the plot. If None, the legend will include an incremental
        number per curve.

        Default: None

    Returns
    -------
    final_error_dist: list
        Cumulative distribution values of the final errors.
    initial_error_dist: list
        Cumulative distribution values of the initial errors.
    """
    if legend is None:
        legend = [str(i + 1) for i in range(len(ceds))]
    y_label = 'Proportion of images'
    axis_limits = [0, x_limit, 0, y_limit]
    return GraphPlotter(figure_id, new_figure, x_axis, ceds, title=title,
                        legend=legend, x_label=x_label, y_label=y_label,
                        axis_limits=axis_limits).render(**kwargs)
