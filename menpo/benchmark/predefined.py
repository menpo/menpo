from menpo.landmark import *
from menpo.fit.lucaskanade.appearance import *
from menpo.transform import PiecewiseAffine, ThinPlateSplines
from menpo.transform.modeldriven import GlobalMDTransform, OrthoMDTransform
from menpo.transform import AlignmentSimilarity
from menpo.fitmultilevel.featurefunctions import sparse_hog
from menpo.model.modelinstance import PDM, OrthoPDM
from menpo.fit.gradientdescent import RegularizedLandmarkMeanShift
from menpo.fitmultilevel.clm.classifierfunctions import linear_svm_lr
from .io import import_bounding_boxes

from .base import (load_database, aam_build_benchmark, aam_fit_benchmark,
                   clm_build_benchmark, clm_fit_benchmark,
                   sdm_build_benchmark, sdm_fit_benchmark,
                   convert_fitting_results_to_ced, plot_fitting_curves)


def aam_fastest_alternating_noise(training_db_path, fitting_db_path,
                                  feature_type='igo', noise_std=0.04,
                                  verbose=False, plot=False):
    # check feature
    if not isinstance(feature_type, str):
        if not hasattr(feature_type, '__call__'):
            if feature_type is not None:
                raise ValueError("feature_type must be a string or "
                                 "function/closure or None")

    # predefined options
    db_loading_options = {'crop_proportion': 0.2,
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
    perturb_options = {'noise_std': 0.04,
                       'rotation': False}

    # set passed parameters
    training_options['feature_type'] = feature_type
    perturb_options['noise_std'] = noise_std

    # run experiment
    training_images = load_database(training_db_path,
                                    db_loading_options=db_loading_options,
                                    verbose=verbose)
    aam = aam_build_benchmark(training_images,
                              training_options=training_options,
                              verbose=verbose)
    fitting_images = load_database(fitting_db_path,
                                   db_loading_options=db_loading_options,
                                   verbose=verbose)
    fitting_results = aam_fit_benchmark(fitting_images, aam,
                                        perturb_options=perturb_options,
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


def aam_fastest_alternating_bbox(training_db_path, fitting_db_path,
                                 fitting_bboxes_path, feature_type='igo',
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
                        'pyramid_on_features': False,
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

    # set passed parameters
    training_options['feature_type'] = feature_type

    # run experiment
    training_images = load_database(training_db_path,
                                    db_loading_options=db_loading_options,
                                    verbose=verbose)
    aam = aam_build_benchmark(training_images,
                              training_options=training_options,
                              verbose=verbose)

    # import bounding boxes
    bboxes_list = import_bounding_boxes(fitting_bboxes_path)

    # for all fittings, we crop to 0.5
    db_loading_options['crop_proportion'] = 0.5
    fitting_images = load_database(fitting_db_path,
                                   db_loading_options=db_loading_options,
                                   bounding_boxes=bboxes_list,
                                   verbose=verbose)

    fitting_results = aam_fit_benchmark(fitting_images, aam,
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


def aam_best_performance_alternating_noise(training_db_path, fitting_db_path,
                                           feature_type='igo', noise_std=0.04,
                                           verbose=False, plot=False):
    # check feature
    if not isinstance(feature_type, str):
        if not hasattr(feature_type, '__call__'):
            if feature_type is not None:
                raise ValueError("feature_type must be a string or "
                                 "function/closure or None")

    # predefined options
    db_loading_options = {'crop_proportion': 0.2,
                          'convert_to_grey': True
                          }
    training_options = {'group': 'PTS',
                        'feature_type': 'igo',
                        'transform': PiecewiseAffine,
                        'trilist': ibug_68_trimesh,
                        'normalization_diagonal': None,
                        'n_levels': 3,
                        'downscale': 1.2,
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
    perturb_options = {'noise_std': 0.04,
                       'rotation': False}

    # set passed parameters
    training_options['feature_type'] = feature_type
    perturb_options['noise_std'] = noise_std

    # run experiment
    training_images = load_database(training_db_path,
                                    db_loading_options=db_loading_options,
                                    verbose=verbose)
    aam = aam_build_benchmark(training_images,
                              training_options=training_options,
                              verbose=verbose)
    fitting_images = load_database(fitting_db_path,
                                   db_loading_options=db_loading_options,
                                   verbose=verbose)
    fitting_results = aam_fit_benchmark(fitting_images, aam,
                                        perturb_options=perturb_options,
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


def aam_best_performance_alternating_bbox(training_db_path, fitting_db_path,
                                          fitting_bboxes_path,
                                          feature_type='igo', verbose=False,
                                          plot=False):
    # check feature
    if not isinstance(feature_type, str):
        if not hasattr(feature_type, '__call__'):
            if feature_type is not None:
                raise ValueError("feature_type must be a string or "
                                 "function/closure or None")

    # predefined options
    db_loading_options = {'crop_proportion': 0.5,
                          'convert_to_grey': True
    }
    training_options = {'group': 'PTS',
                        'feature_type': 'igo',
                        'transform': PiecewiseAffine,
                        'trilist': ibug_68_trimesh,
                        'normalization_diagonal': 200,
                        'n_levels': 3,
                        'downscale': 2,
                        'scaled_shape_models': True,
                        'pyramid_on_features': True,
                        'max_shape_components': 25,
                        'max_appearance_components': 100,
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

    # set passed parameters
    training_options['feature_type'] = feature_type

    # run experiment
    training_images = load_database(training_db_path,
                                    db_loading_options=db_loading_options,
                                    verbose=verbose)
    aam = aam_build_benchmark(training_images,
                              training_options=training_options,
                              verbose=verbose)

    # import bounding boxes
    bboxes_list = import_bounding_boxes(fitting_bboxes_path)

    # for all fittings, we crop to 0.5
    db_loading_options['crop_proportion'] = 0.5
    fitting_images = load_database(fitting_db_path,
                                   db_loading_options=db_loading_options,
                                   bounding_boxes=bboxes_list,
                                   verbose=verbose)

    fitting_results = aam_fit_benchmark(fitting_images, aam,
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


def clm_basic_noise(training_db_path,  fitting_db_path,
                    feature_type=sparse_hog, classifier_type=linear_svm_lr,
                    noise_std=0.04, verbose=False, plot=False):
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
                        'scaled_shape_models': True,
                        'pyramid_on_features': False,
                        'max_shape_components': None,
                        'boundary': 3,
                        'interpolator': 'scipy'
                        }
    fitting_options = {'algorithm': RegularizedLandmarkMeanShift,
                       'pdm_transform': OrthoPDM,
                       'global_transform': AlignmentSimilarity,
                       'n_shape': [3, 6, 12],
                       'max_iters': 50,
                       'error_type': 'me_norm'
                       }
    perturb_options = {'noise_std': 0.01,
                       'rotation': False}

    # set passed parameters
    training_options['feature_type'] = feature_type
    training_options['classifier_type'] = classifier_type
    perturb_options['noise_std'] = noise_std

    # run experiment
    training_images = load_database(training_db_path,
                                    db_loading_options=db_loading_options,
                                    verbose=verbose)
    clm = clm_build_benchmark(training_images,
                              training_options=training_options,
                              verbose=verbose)
    fitting_images = load_database(fitting_db_path,
                                   db_loading_options=db_loading_options,
                                   verbose=verbose)
    fitting_results = clm_fit_benchmark(fitting_images, clm,
                                        perturb_options=perturb_options,
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


def clm_basic_bbox(training_db_path,  fitting_db_path, fitting_bboxes_path,
                   feature_type=sparse_hog, classifier_type=linear_svm_lr,
                   verbose=False, plot=False):
    # check feature
    if not isinstance(feature_type, str):
        if not hasattr(feature_type, '__call__'):
            if feature_type is not None:
                raise ValueError("feature_type must be a string or "
                                 "function/closure or None")

    # predefined options
    db_loading_options = {'crop_proportion': 0.5,
                          'convert_to_grey': True
    }
    training_options = {'group': 'PTS',
                        'classifier_type': linear_svm_lr,
                        'patch_shape': (5, 5),
                        'feature_type': sparse_hog,
                        'normalization_diagonal': None,
                        'n_levels': 3,
                        'downscale': 1.1,
                        'scaled_shape_models': True,
                        'pyramid_on_features': False,
                        'max_shape_components': None,
                        'boundary': 3,
                        'interpolator': 'scipy'
    }
    fitting_options = {'algorithm': RegularizedLandmarkMeanShift,
                       'pdm_transform': OrthoPDM,
                       'global_transform': AlignmentSimilarity,
                       'n_shape': [3, 6, 12],
                       'max_iters': 50,
                       'error_type': 'me_norm'
    }

    # set passed parameters
    training_options['feature_type'] = feature_type
    training_options['classifier_type'] = classifier_type

    # run experiment
    training_images = load_database(training_db_path,
                                    db_loading_options=db_loading_options,
                                    verbose=verbose)

    clm = clm_build_benchmark(training_images,
                              training_options=training_options,
                              verbose=verbose)

    # import bounding boxes
    bboxes_list = import_bounding_boxes(fitting_bboxes_path)

    # for all fittings, we crop to 0.5
    db_loading_options['crop_proportion'] = 0.5
    fitting_images = load_database(fitting_db_path,
                                   db_loading_options=db_loading_options,
                                   bounding_boxes=bboxes_list,
                                   verbose=verbose)

    fitting_results = clm_fit_benchmark(fitting_images, clm,
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


def sdm_fastest_bbox(training_db_path, fitting_db_path,
                                 fitting_bboxes_path, feature_type=None,
                                 verbose=False, plot=False):
    # check feature
    if not isinstance(feature_type, str):
        if not hasattr(feature_type, '__call__'):
            if feature_type is not None:
                raise ValueError("feature_type must be a string or "
                                 "function/closure or None")

    # predefined options
    db_loading_options = {'crop_proportion': 0.8,
                          'convert_to_grey': True
    }
    training_options = {'group': 'PTS',
                        'normalization_diagonal': 200,
                        'n_levels': 4,
                        'downscale': 1.01,
                        'noise_std': 0.08,
                        'patch_shape': (16, 16),
                        'n_perturbations': 15,
                        'pyramid_on_features': False,
                        'interpolator': 'scipy'
    }
    fitting_options = {
                       'error_type': 'me_norm'
    }

    # run experiment
    training_images = load_database(training_db_path,
                                    db_loading_options=db_loading_options,
                                    verbose=verbose)
    sdm = sdm_build_benchmark(training_images,
                              training_options=training_options,
                              verbose=verbose)

    # import bounding boxes
    bboxes_list = import_bounding_boxes(fitting_bboxes_path)
    # predefined options

    # for all fittings, we crop to 0.5
    db_loading_options['crop_proportion'] = 0.5
    fitting_images = load_database(fitting_db_path,
                                   db_loading_options=db_loading_options,
                                   bounding_boxes=bboxes_list,
                                   verbose=verbose)

    fitting_results = sdm_fit_benchmark(fitting_images, sdm,
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
        title = "SDMs using default (sparse hogs)".format(
            training_options['feature_type'])
        y_axis = [final_error_curve, initial_error_curve]
        legend = ['Fitting', 'Initialization']
        plot_fitting_curves(error_bins, y_axis, title, new_figure=True,
                            x_limit=max_error_bin, legend=legend,
                            color_list=['r', 'b'], marker_list=['o', 'x'])
    return fitting_results, final_error_curve, initial_error_curve, error_bins


def aam_params_combinations_noise(training_db_path, fitting_db_path,
                                  n_experiments=1, feature_type=None,
                                  scaled_shape_models=None,
                                  pyramid_on_features=None, n_shape=None,
                                  n_appearance=None, noise_std=None,
                                  rotation=None, verbose=False, plot=False):

    # parse input
    if feature_type is None:
        feature_type = ['igo'] * n_experiments
    elif len(feature_type) is not n_experiments:
        raise ValueError("feature_type has wrong length")
    if scaled_shape_models is None:
        scaled_shape_models = [True] * n_experiments
    elif len(scaled_shape_models) is not n_experiments:
        raise ValueError("scaled_shape_models has wrong length")
    if pyramid_on_features is None:
        pyramid_on_features = [True] * n_experiments
    elif len(pyramid_on_features) is not n_experiments:
        raise ValueError("pyramid_on_features has wrong length")
    if n_shape is None:
        n_shape = [[3, 6, 12]] * n_experiments
    elif len(n_shape) is not n_experiments:
        raise ValueError("n_shape has wrong length")
    if n_appearance is None:
        n_appearance = [50] * n_experiments
    elif len(n_appearance) is not n_experiments:
        raise ValueError("n_appearance has wrong length")
    if noise_std is None:
        noise_std = [0.04] * n_experiments
    elif len(noise_std) is not n_experiments:
        raise ValueError("noise_std has wrong length")
    if rotation is None:
        rotation = [False] * n_experiments
    elif len(rotation) is not n_experiments:
        raise ValueError("rotation has wrong length")

    # load images
    db_loading_options = {'crop_proportion': 0.1,
                          'convert_to_grey': True
                          }
    training_images = load_database(training_db_path,
                                    db_loading_options=db_loading_options,
                                    verbose=verbose)
    fitting_images = load_database(fitting_db_path,
                                   db_loading_options=db_loading_options,
                                   verbose=verbose)

    # run experiments
    max_error_bin = 0.05
    bins_error_step = 0.005
    curves_to_plot = []
    all_fitting_results = []
    for i in range(n_experiments):
        if verbose:
            print("\nEXPERIMENT {}/{}:".format(i + 1, n_experiments))
            print("- feature_type: {}\n- scaled_shape_models: {}\n"
                  "- pyramid_on_features: {}\n- n_shape: {}\n"
                  "- n_appearance: {}\n- noise_std: {}\n"
                  "- rotation: {}".format(
                  feature_type[i], scaled_shape_models[i],
                  pyramid_on_features[i], n_shape[i], n_appearance[i],
                  noise_std[i], rotation[i]))

        # predefined option dictionaries
        training_options = {'group': 'PTS',
                            'feature_type': 'igo',
                            'transform': PiecewiseAffine,
                            'trilist': ibug_68_trimesh,
                            'normalization_diagonal': None,
                            'n_levels': 3,
                            'downscale': 1.1,
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
        pertrub_options = {'noise_std': 0.04,
                           'rotation': False}

        # training
        training_options['feature_type'] = feature_type[i]
        training_options['scaled_shape_models'] = scaled_shape_models[i]
        training_options['pyramid_on_features'] = pyramid_on_features[i]
        aam = aam_build_benchmark(training_images,
                                  training_options=training_options,
                                  verbose=verbose)

        # fitting
        fitting_options['n_shape'] = n_shape[i]
        fitting_options['n_appearance'] = n_appearance[i]
        pertrub_options['noise_std'] = noise_std[i]
        pertrub_options['rotation'] = rotation[i]
        fitting_results = aam_fit_benchmark(fitting_images, aam,
                                            perturb_options=pertrub_options,
                                            fitting_options=fitting_options,
                                            verbose=verbose)
        all_fitting_results.append(fitting_results)

        # convert results
        final_error_curve, initial_error_curve, error_bins = \
            convert_fitting_results_to_ced(fitting_results,
                                           max_error_bin=max_error_bin,
                                           bins_error_step=bins_error_step)
        curves_to_plot.append(final_error_curve)
        if i == n_experiments - 1:
            curves_to_plot.append(initial_error_curve)

    # plot results
    if plot:
        title = "AAMs using Alternating IC"
        color_list = ['r', 'b', 'g', 'y', 'c'] * n_experiments
        marker_list = ['o', 'x', 'v', 'd'] * n_experiments
        plot_fitting_curves(error_bins, curves_to_plot, title, new_figure=True,
                            x_limit=max_error_bin,  color_list=color_list,
                            marker_list=marker_list)
    return all_fitting_results


def clm_params_combinations_noise(training_db_path, fitting_db_path,
                                  n_experiments=1, classifier_type=None,
                                  patch_shape=None, feature_type=None,
                                  scaled_shape_models=None,
                                  pyramid_on_features=None, n_shape=None,
                                  noise_std=None, rotation=None, verbose=False,
                                  plot=False):

    # parse input
    if classifier_type is None:
        classifier_type = [linear_svm_lr] * n_experiments
    elif len(classifier_type) is not n_experiments:
        raise ValueError("classifier_type has wrong length")
    if patch_shape is None:
        patch_shape = [(5, 5)] * n_experiments
    elif len(patch_shape) is not n_experiments:
        raise ValueError("patch_shape has wrong length")
    if feature_type is None:
        feature_type = ['igo'] * n_experiments
    elif len(feature_type) is not n_experiments:
        raise ValueError("feature_type has wrong length")
    if scaled_shape_models is None:
        scaled_shape_models = [True] * n_experiments
    elif len(scaled_shape_models) is not n_experiments:
        raise ValueError("scaled_shape_models has wrong length")
    if pyramid_on_features is None:
        pyramid_on_features = [True] * n_experiments
    elif len(pyramid_on_features) is not n_experiments:
        raise ValueError("pyramid_on_features has wrong length")
    if n_shape is None:
        n_shape = [[3, 6, 12]] * n_experiments
    elif len(n_shape) is not n_experiments:
        raise ValueError("n_shape has wrong length")
    if noise_std is None:
        noise_std = [0.04] * n_experiments
    elif len(noise_std) is not n_experiments:
        raise ValueError("noise_std has wrong length")
    if rotation is None:
        rotation = [False] * n_experiments
    elif len(rotation) is not n_experiments:
        raise ValueError("rotation has wrong length")

    # load images
    db_loading_options = {'crop_proportion': 0.4,
                          'convert_to_grey': True
                          }
    training_images = load_database(training_db_path,
                                    db_loading_options=db_loading_options,
                                    verbose=verbose)
    fitting_images = load_database(fitting_db_path,
                                   db_loading_options=db_loading_options,
                                   verbose=verbose)

    # run experiments
    max_error_bin = 0.05
    bins_error_step = 0.005
    curves_to_plot = []
    all_fitting_results = []
    for i in range(n_experiments):
        if verbose:
            print("\nEXPERIMENT {}/{}:".format(i + 1, n_experiments))
            print("- classifier_type: {}\n- patch_shape: {}\n"
                  "- feature_type: {}\n- scaled_shape_models: {}\n"
                  "- pyramid_on_features: {}\n- n_shape: {}\n"
                  "- noise_std: {}\n- rotation: {}".format(
                  classifier_type[i], patch_shape[i], feature_type[i],
                  scaled_shape_models[i], pyramid_on_features[i], n_shape[i],
                  noise_std[i], rotation[i]))

        # predefined option dictionaries
        training_options = {'group': 'PTS',
                            'classifier_type': linear_svm_lr,
                            'patch_shape': (5, 5),
                            'feature_type': sparse_hog,
                            'normalization_diagonal': None,
                            'n_levels': 3,
                            'downscale': 1.1,
                            'scaled_shape_models': False,
                            'pyramid_on_features': True,
                            'max_shape_components': None,
                            'boundary': 3,
                            'interpolator': 'scipy'
                            }
        fitting_options = {'algorithm': RegularizedLandmarkMeanShift,
                           'pdm_transform': OrthoPDM,
                           'global_transform': AlignmentSimilarity,
                           'n_shape': [3, 6, 12],
                           'max_iters': 50,
                           'error_type': 'me_norm'
                           }
        perturb_options = {'noise_std': 0.01,
                           'rotation': False}

        # training
        training_options['classifier_type'] = classifier_type[i]
        training_options['patch_shape'] = patch_shape[i]
        training_options['feature_type'] = feature_type[i]
        training_options['scaled_shape_models'] = scaled_shape_models[i]
        training_options['pyramid_on_features'] = pyramid_on_features[i]
        clm = clm_build_benchmark(training_images,
                                  training_options=training_options,
                                  verbose=verbose)

        # fitting
        fitting_options['n_shape'] = n_shape[i]
        perturb_options['noise_std'] = noise_std[i]
        perturb_options['rotation'] = rotation[i]
        fitting_results = clm_fit_benchmark(fitting_images, clm,
                                            perturb_options=perturb_options,
                                            fitting_options=fitting_options,
                                            verbose=verbose)
        all_fitting_results.append(fitting_results)

        # convert results
        final_error_curve, initial_error_curve, error_bins = \
            convert_fitting_results_to_ced(fitting_results,
                                           max_error_bin=max_error_bin,
                                           bins_error_step=bins_error_step)
        curves_to_plot.append(final_error_curve)
        if i == n_experiments - 1:
            curves_to_plot.append(initial_error_curve)

    # plot results
    if plot:
        title = "CLMs using RLMS"
        color_list = ['r', 'b', 'g', 'y', 'c'] * n_experiments
        marker_list = ['o', 'x', 'v', 'd'] * n_experiments
        plot_fitting_curves(error_bins, curves_to_plot, title, new_figure=True,
                            x_limit=max_error_bin,  color_list=color_list,
                            marker_list=marker_list)
    return all_fitting_results
