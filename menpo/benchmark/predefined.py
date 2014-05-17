from menpo.landmark import *
from menpo.fit.lucaskanade.appearance import *
from menpo.transform import PiecewiseAffine, ThinPlateSplines
from menpo.transform.modeldriven import GlobalMDTransform, OrthoMDTransform
from menpo.transform import AlignmentSimilarity
from menpo.fitmultilevel.featurefunctions import sparse_hog
from menpo.model.pdm import PDM, OrthoPDM
from menpo.fit.gradientdescent import RegularizedLandmarkMeanShift
from menpo.fitmultilevel.clm.classifierfunctions import linear_svm_lr

from .base import (load_database, aam_build_benchmark, aam_fit_benchmark,
                   clm_build_benchmark, clm_fit_benchmark,
                   convert_fitting_results_to_ced, plot_fitting_curves)


def aam_fastest_alternating(training_db_path, training_db_ext, fitting_db_path,
                            fitting_db_ext, feature_type='igo', noise_std=0.04,
                            verbose=False, plot=False):
    # check feature
    if not isinstance(feature_type, str):
        if not hasattr(feature_type, '__call__'):
            if feature_type is not None:
                raise ValueError("feature_type must be a string or "
                                 "function/closure or None")

    # predefined options
    db_loading_options = {'crop_proportion': 0.1,
                          'convert_to_grey': True
                          }
    training_options = {'group': 'PTS',
                        'feature_type': 'igo',
                        'transform': PiecewiseAffine,
                        'trilist': ibug_68_trimesh,
                        'normalization_diagonal': None,
                        'n_levels': 3,
                        'downscale': 2,
                        'scaled_shape_models': True,
                        'pyramid_on_features': True,
                        'max_shape_components': 25,
                        'max_appearance_components': 250,
                        'boundary': 3,
                        'interpolator': 'scipy'
                        }
    fitting_options = {'algorithm': AlternatingInverseCompositional,
                       'md_transform': OrthoMDTransform,
                       'global_transform': AlignmentSimilarity,
                       'n_shape': [3, 6, 12],
                       'n_appearance': 50,
                       'max_iters': 50,
                       'error_type': 'me_norm'
                       }
    initialization_options = {'noise_std': 0.04,
                              'rotation': False}

    # set passed parameters
    training_options['feature_type'] = feature_type
    initialization_options['noise_std'] = noise_std

    # run experiment
    training_images = load_database(training_db_path, training_db_ext,
                                    db_loading_options=db_loading_options,
                                    verbose=verbose)
    aam = aam_build_benchmark(training_images,
                              training_options=training_options,
                              verbose=verbose)
    fitting_images = load_database(fitting_db_path, fitting_db_ext,
                                   db_loading_options=db_loading_options,
                                   verbose=verbose)
    fitting_results = aam_fit_benchmark(fitting_images, aam,
                                        initialization_options=
                                        initialization_options,
                                        fitting_options=fitting_options,
                                        verbose=verbose)

    # convert results
    max_error_bin = 0.05
    bins_error_step = 0.005
    final_error_curve, initial_error_curve, error_bins = \
        convert_fitting_results_to_ced(fitting_results,
                                       max_error_bin=max_error_bin,
                                       bins_error_step=bins_error_step)

    # plot results
    if plot:
        title = "AAMs using {} and Alternating IC".format(
            training_options['feature_type'])
        y_axis = [final_error_curve, initial_error_curve]
        legend = ['Fitting', 'Initialization']
        plot_fitting_curves(error_bins, y_axis, title, new_figure=True,
                            x_limit=max_error_bin, legend=legend,
                            color_list=['r', 'b'], marker_list=['o', 'x'])
    return fitting_results, final_error_curve, initial_error_curve, error_bins


def aam_best_performance_alternating(training_db_path, training_db_ext,
                                     fitting_db_path, fitting_db_ext,
                                     feature_type='igo', noise_std=0.04,
                                     verbose=False, plot=False):
    # check feature
    if not isinstance(feature_type, str):
        if not hasattr(feature_type, '__call__'):
            if feature_type is not None:
                raise ValueError("feature_type must be a string or "
                                 "function/closure or None")

    # predefined options
    db_loading_options = {'crop_proportion': 0.1,
                          'convert_to_grey': True
                          }
    training_options = {'group': 'PTS',
                        'feature_type': 'igo',
                        'transform': PiecewiseAffine,
                        'trilist': ibug_68_trimesh,
                        'normalization_diagonal': None,
                        'n_levels': 3,
                        'downscale': 2,
                        'scaled_shape_models': False,
                        'pyramid_on_features': True,
                        'max_shape_components': 25,
                        'max_appearance_components': 250,
                        'boundary': 3,
                        'interpolator': 'scipy'
                        }
    fitting_options = {'algorithm': AlternatingInverseCompositional,
                       'md_transform': OrthoMDTransform,
                       'global_transform': AlignmentSimilarity,
                       'n_shape': [3, 6, 12],
                       'n_appearance': 50,
                       'max_iters': 50,
                       'error_type': 'me_norm'
                       }
    initialization_options = {'noise_std': 0.04,
                              'rotation': False}

    # set passed parameters
    training_options['feature_type'] = feature_type
    initialization_options['noise_std'] = noise_std

    # run experiment
    training_images = load_database(training_db_path, training_db_ext,
                                    db_loading_options=db_loading_options,
                                    verbose=verbose)
    aam = aam_build_benchmark(training_images,
                              training_options=training_options,
                              verbose=verbose)
    fitting_images = load_database(fitting_db_path, fitting_db_ext,
                                   db_loading_options=db_loading_options,
                                   verbose=verbose)
    fitting_results = aam_fit_benchmark(fitting_images, aam,
                                        initialization_options=
                                        initialization_options,
                                        fitting_options=fitting_options,
                                        verbose=verbose)

    # convert results
    max_error_bin = 0.05
    bins_error_step = 0.005
    final_error_curve, initial_error_curve, error_bins = \
        convert_fitting_results_to_ced(fitting_results,
                                       max_error_bin=max_error_bin,
                                       bins_error_step=bins_error_step)

    # plot results
    if plot:
        title = "AAMs using {} and Alternating IC".format(
            training_options['feature_type'])
        y_axis = [final_error_curve, initial_error_curve]
        legend = ['Fitting', 'Initialization']
        plot_fitting_curves(error_bins, y_axis, title, new_figure=True,
                            x_limit=max_error_bin, legend=legend,
                            color_list=['r', 'b'], marker_list=['o', 'x'])
    return fitting_results, final_error_curve, initial_error_curve, error_bins


def clm_basic(training_db_path, training_db_ext, fitting_db_path,
              fitting_db_ext, feature_type=sparse_hog,
              classifier_type=linear_svm_lr, noise_std=0.04, verbose=False,
              plot=False):
    # check feature
    if not isinstance(feature_type, str):
        if not hasattr(feature_type, '__call__'):
            if feature_type is not None:
                raise ValueError("feature_type must be a string or "
                                 "function/closure or None")

    # predefined options
    db_loading_options = {'crop_proportion': 0.4,
                          'convert_to_grey': True
                          }
    training_options = {'group': 'PTS',
                        'classifier_type': linear_svm_lr,
                        'patch_shape': (5, 5),
                        'feature_type': sparse_hog,
                        'normalization_diagonal': None,
                        'n_levels': 3,
                        'downscale': 1.1,
                        'scaled_shape_models': False,
                        'pyramid_on_features': False,
                        'max_shape_components': None,
                        'boundary': 3,
                        'interpolator': 'scipy'
                        }
    fitting_options = {'algorithm': RegularizedLandmarkMeanShift,
                       'pdm_transform': OrthoPDM,
                       'global_transform': AlignmentSimilarity,
                       'n_shape': [6, 6, 6],
                       'max_iters': 50,
                       'error_type': 'me_norm'
                       }
    initialization_options = {'noise_std': 0.01,
                              'rotation': False}

    # set passed parameters
    training_options['feature_type'] = feature_type
    training_options['classifier_type'] = classifier_type
    initialization_options['noise_std'] = noise_std

    # run experiment
    training_images = load_database(training_db_path, training_db_ext,
                                    db_loading_options=db_loading_options,
                                    verbose=verbose)
    clm = clm_build_benchmark(training_images,
                              training_options=training_options,
                              verbose=verbose)
    fitting_images = load_database(fitting_db_path, fitting_db_ext,
                                   db_loading_options=db_loading_options,
                                   verbose=verbose)
    fitting_results = clm_fit_benchmark(fitting_images, clm,
                                        initialization_options=
                                        initialization_options,
                                        fitting_options=fitting_options,
                                        verbose=verbose)

    # convert results
    max_error_bin = 0.05
    bins_error_step = 0.005
    final_error_curve, initial_error_curve, error_bins = \
        convert_fitting_results_to_ced(fitting_results,
                                       max_error_bin=max_error_bin,
                                       bins_error_step=bins_error_step)

    # plot results
    if plot:
        title = "CLMs with {} and {} classifier using RLMS".format(
            training_options['feature_type'],
            training_options['classifier_type'])
        y_axis = [final_error_curve, initial_error_curve]
        legend = ['Fitting', 'Initialization']
        plot_fitting_curves(error_bins, y_axis, title, new_figure=True,
                            x_limit=max_error_bin, legend=legend,
                            color_list=['r', 'b'], marker_list=['o', 'x'])
    return fitting_results, final_error_curve, initial_error_curve, error_bins


def clm_temp(training_db_path, training_db_ext, fitting_db_path,
             fitting_db_ext, feature_type=sparse_hog, patch_size=(5, 5),
             classifier_type=linear_svm_lr, scaled_shape_models=True,
             pyramid_on_features=True, n_shape=None,
             noise_std=0.04, verbose=False,
             plot=False):
    # check feature
    if not isinstance(feature_type, str):
        if not hasattr(feature_type, '__call__'):
            if feature_type is not None:
                raise ValueError("feature_type must be a string or "
                                 "function/closure or None")

    # predefined options
    db_loading_options = {'crop_proportion': 0.4,
                          'convert_to_grey': True
                          }
    training_options = {'group': 'PTS',
                        'classifier_type': linear_svm_lr,
                        'patch_shape': (5, 5),
                        'feature_type': sparse_hog,
                        'normalization_diagonal': None,
                        'n_levels': 3,
                        'downscale': 1.1,
                        'scaled_shape_models': False,
                        'pyramid_on_features': False,
                        'max_shape_components': None,
                        'boundary': 3,
                        'interpolator': 'scipy'
                        }
    fitting_options = {'algorithm': RegularizedLandmarkMeanShift,
                       'pdm_transform': OrthoPDM,
                       'global_transform': AlignmentSimilarity,
                       'n_shape': [6, 6, 6],
                       'max_iters': 50,
                       'error_type': 'me_norm'
                       }
    initialization_options = {'noise_std': 0.01,
                              'rotation': False}

    # set passed parameters
    training_options['feature_type'] = feature_type
    training_options['patch_size'] = patch_size
    training_options['scaled_shape_models'] = scaled_shape_models
    training_options['pyramid_on_features'] = pyramid_on_features
    fitting_options['n_shape'] = n_shape
    training_options['classifier_type'] = classifier_type
    initialization_options['noise_std'] = noise_std

    # run experiment
    training_images = load_database(training_db_path, training_db_ext,
                                    db_loading_options=db_loading_options,
                                    verbose=verbose)
    clm = clm_build_benchmark(training_images,
                              training_options=training_options,
                              verbose=verbose)
    fitting_images = load_database(fitting_db_path, fitting_db_ext,
                                   db_loading_options=db_loading_options,
                                   verbose=verbose)
    fitting_results = clm_fit_benchmark(fitting_images, clm,
                                        initialization_options=
                                        initialization_options,
                                        fitting_options=fitting_options,
                                        verbose=verbose)

    # convert results
    max_error_bin = 0.05
    bins_error_step = 0.005
    final_error_curve, initial_error_curve, error_bins = \
        convert_fitting_results_to_ced(fitting_results,
                                       max_error_bin=max_error_bin,
                                       bins_error_step=bins_error_step)

    # plot results
    if plot:
        title = "CLMs with {} and {} classifier using RLMS".format(
            training_options['feature_type'],
            training_options['classifier_type'])
        y_axis = [final_error_curve, initial_error_curve]
        legend = ['Fitting', 'Initialization']
        plot_fitting_curves(error_bins, y_axis, title, new_figure=True,
                            x_limit=max_error_bin, legend=legend,
                            color_list=['r', 'b'], marker_list=['o', 'x'])
    return fitting_results, final_error_curve, initial_error_curve, error_bins
