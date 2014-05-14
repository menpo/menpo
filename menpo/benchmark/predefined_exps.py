from menpo.landmark import *
from menpo.fit.lucaskanade.appearance import *
from menpo.transform import PiecewiseAffine, ThinPlateSplines
from menpo.transform.modeldriven import GlobalMDTransform, OrthoMDTransform
from menpo.transform import AlignmentSimilarity

from .base import aam_benchmark, aam_fit_benchmark, aam_build_benchmark


def aam_experiments_all(training_db_path, training_db_ext, fitting_db_path,
                        fitting_db_ext, verbose=False):
    # options that will be combined
    scaled_shape_models_list = [True, False]
    n_shape_list = [[3, 6, 12], [5, 10, 15]]
    n_appearance_list = [50, 100]
    noise_std_list = [0.01, 0.03]

    # number of experiments
    n_experiments = 16

    # loading predefined options
    db_loading_options = {'crop_proportion': 0.1,
                          'convert_to_grey': True
                          }

    # training predefined options
    training_options = {'group': 'PTS',
                        'feature_type': 'igo',
                        'transform': PiecewiseAffine,
                        'trilist': None,
                        'normalization_diagonal': None,
                        'n_levels': 3,
                        'downscale': 2,
                        'scaled_shape_models': True,
                        'max_shape_components': 25,
                        'max_appearance_components': 250,
                        'boundary': 3,
                        'interpolator': 'scipy'
                        }

    # fitting predefined options
    fitting_options = {'algorithm': AlternatingInverseCompositional,
                       'md_transform': OrthoMDTransform,
                       'global_transform': AlignmentSimilarity,
                       'n_shape': [3, 6, 12],
                       'n_appearance': 100,
                       'max_iters': 50,
                       'error_type': 'me_norm'
                       }

    # initialization predefined options
    initialization_options = {'noise_std': 0.04,
                              'rotation': False
                              }

    # train the two aam models
    if verbose:
        print('\nTraining AAM with Smoothing pyramid:')
    training_options['scaled_shape_models'] = True
    aam_true = aam_build_benchmark(training_db_path, training_db_ext,
                                   db_loading_options=db_loading_options,
                                   training_options=training_options,
                                   verbose=verbose)
    if verbose:
        print('\nTraining AAM with Gaussian pyramid:')
    training_options['scaled_shape_models'] = False
    aam_false = aam_build_benchmark(training_db_path, training_db_ext,
                                    db_loading_options=db_loading_options,
                                    training_options=training_options,
                                    verbose=verbose)

    # fittings
    exp_counter = 0
    fitting_results_all = []
    parameters_all = []
    for noise_std in noise_std_list:
        for n_shape in n_shape_list:
            for n_appearance in n_appearance_list:
                for scaled_shape_models in scaled_shape_models_list:
                    exp_counter += 1
                    if verbose:
                        print('\nEXPERIMENT {0}/{1}\n----------------\n'
                              'Parameters:\n'
                              '- noise_std: {2:.2f}\n'
                              '- n_shape: {3}\n'
                              '- n_appearance: {4}\n'
                              '- scaled_shape_models: {5}\n'.format(
                              exp_counter, n_experiments, noise_std, n_shape,
                              n_appearance, scaled_shape_models))

                    # set parameters
                    fitting_options['n_shape'] = n_shape
                    fitting_options['n_appearance'] = n_appearance
                    initialization_options['noise_std'] = noise_std

                    # save parameters
                    parameters_all.append({'n_shape': n_shape,
                                           'n_appearance': n_appearance,
                                           'noise_std': noise_std,
                                           'scaled_shape_models':
                                               scaled_shape_models})

                    # fit
                    if scaled_shape_models is True:
                        fitting_results_all.append(aam_fit_benchmark(
                            fitting_db_path, fitting_db_ext, aam_true,
                            db_loading_options=db_loading_options,
                            fitting_options=fitting_options,
                            initialization_options=initialization_options,
                            verbose=verbose))
                    else:
                        fitting_results_all.append(aam_fit_benchmark(
                            fitting_db_path, fitting_db_ext, aam_false,
                            db_loading_options=db_loading_options,
                            fitting_options=fitting_options,
                            initialization_options=initialization_options,
                            verbose=verbose))

    return fitting_results_all, parameters_all


def aam_igo_smoothing_pyramid(training_db_path, training_db_ext,
                              fitting_db_path, fitting_db_ext, verbose=False):
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
                        'max_shape_components': 25,
                        'max_appearance_components': 250,
                        'boundary': 3,
                        'interpolator': 'scipy'
                        }
    fitting_options = {'algorithm': AlternatingInverseCompositional,
                       'md_transform': OrthoMDTransform,
                       'global_transform': AlignmentSimilarity,
                       'n_shape': [3, 6, 12],
                       'n_appearance': 100,
                       'max_iters': 50,
                       'error_type': 'me_norm'
                       }
    initialization_options = {'noise_std': 0.04,
                              'rotation': False
                              }
    # run experiment
    fitting_results = aam_benchmark(
        training_db_path, training_db_ext, fitting_db_path, fitting_db_ext,
        db_loading_options=db_loading_options,
        training_options=training_options, fitting_options=fitting_options,
        initialization_options=initialization_options, verbose=verbose)

    # plot results
    fitting_results.plot_cumulative_error_dist(color_list=['r', 'b'],
                                               marker_list=['o', 'x'])
    return fitting_results
