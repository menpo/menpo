from menpo.landmark import *
from menpo.fit.lucaskanade.appearance import *
from menpo.transform import PiecewiseAffine, ThinPlateSplines
from menpo.transform.modeldriven import GlobalMDTransform, OrthoMDTransform
from menpo.transform import AlignmentSimilarity

from .base import aam_benchmark


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
