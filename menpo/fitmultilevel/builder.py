from __future__ import division
import abc
import numpy as np

from menpo.transform import Scale, Translation, GeneralizedProcrustesAnalysis
from menpo.model.pca import PCAModel
from menpo.visualize import print_dynamic, progress_bar_str

from .functions import mean_pointcloud


#TODO: Document me
class DeformableModelBuilder(object):
    r"""
    """
    __metaclass__ = abc.ABCMeta

    @classmethod
    def check_n_levels(cls, n_levels):
        if n_levels < 1:
            raise ValueError("n_levels must be > 0")

    @classmethod
    def check_downscale(cls, downscale):
        if downscale < 1:
            raise ValueError("downscale must be >= 1")

    @classmethod
    def check_normalization_diagonal(cls, normalization_diagonal):
        if normalization_diagonal is not None and normalization_diagonal < 20:
            raise ValueError("normalization_diagonal must be >= 20")

    @classmethod
    def check_boundary(cls, boundary):
        if boundary < 0:
            raise ValueError("boundary must be >= 0")

    @classmethod
    def check_max_components(cls, max_components, n_levels, var_name):
        str_error = ("{} must be None or an int > 0 or a 0 <= float <= 1 or "
                     "a list of those containing 1 or {} elements").format(
                         var_name, n_levels)
        if not isinstance(max_components, list):
            max_components_list = [max_components] * n_levels
        elif len(max_components) is 1:
            max_components_list = [max_components[0]] * n_levels
        elif len(max_components) is n_levels:
            max_components_list = max_components
        else:
            raise ValueError(str_error)
        for comp in max_components_list:
            if comp is not None:
                if not isinstance(comp, int):
                    if not isinstance(comp, float):
                        raise ValueError(str_error)
        return max_components_list

    @classmethod
    def check_feature_type(cls, feature_type, n_levels):
        feature_type_str_error = ("feature_type must be a str or a "
                                  "function/closure or a list of "
                                  "those containing 1 or {} "
                                  "elements").format(n_levels)
        if not isinstance(feature_type, list):
            feature_type_list = [feature_type] * n_levels
        elif len(feature_type) is 1:
            feature_type_list = [feature_type[0]] * n_levels
        elif len(feature_type) is n_levels:
            feature_type_list = feature_type
        else:
            raise ValueError(feature_type_str_error)
        for ft in feature_type_list:
            if ft is not None:
                if not isinstance(ft, str):
                    if not hasattr(ft, '__call__'):
                        raise ValueError(feature_type_str_error)
        return feature_type_list

    @abc.abstractmethod
    def build(self, images, group=None, label='all'):
        r"""
        """
        pass

    @classmethod
    def _preprocessing(cls, images, group, label, normalization_diagonal,
                       interpolator, scaled_shape_models, n_levels, downscale,
                       verbose=False):
        r"""
        """
        if verbose:
            intro_str = '- Preprocessing: '

        # the mean shape of the images' landmarks is the reference_shape
        if verbose:
            print_dynamic('{}Computing reference shape'.format(intro_str))
        shapes = [i.landmarks[group][label].lms for i in images]
        reference_shape = mean_pointcloud(shapes)

        # fix the reference_shape's diagonal length if asked
        if normalization_diagonal:
            x, y = reference_shape.range()
            scale = normalization_diagonal / np.sqrt(x**2 + y**2)
            Scale(scale, reference_shape.n_dims).apply_inplace(reference_shape)

        # normalize the scaling of all shapes wrt the reference_shape
        normalized_images = []
        for c, i in enumerate(images):
            if verbose:
                print_dynamic('{}Normalizing object size - {}'.format(
                    intro_str, progress_bar_str(float(c + 1) / len(images),
                                                show_bar=False)))
            normalized_images.append(i.rescale_to_reference_shape(
                reference_shape, group=group, label=label,
                interpolator=interpolator))

        # generate pyramid
        if verbose:
            print_dynamic('{}Generating multilevel scale space'.format(
                intro_str))
        if scaled_shape_models:
            generator = [i.smoothing_pyramid(n_levels=n_levels,
                                             downscale=downscale)
                         for i in normalized_images]
        else:
            generator = [i.gaussian_pyramid(n_levels=n_levels,
                                            downscale=downscale)
                         for i in normalized_images]

        if verbose:
            print_dynamic('{}Done\n'.format(intro_str))

        return reference_shape, generator

    #TODO: this seems useful on its own, maybe it shouldn't be underscored...
    @classmethod
    def _build_shape_model(cls, shapes, max_components):
        r"""
        """
        # centralize shapes
        centered_shapes = [Translation(-s.centre).apply(s) for s in shapes]
        # align centralized shape using Procrustes Analysis
        gpa = GeneralizedProcrustesAnalysis(centered_shapes)
        aligned_shapes = [s.aligned_source for s in gpa.transforms]

        # build shape model
        shape_model = PCAModel(aligned_shapes)
        if max_components is not None:
            # trim shape model if required
            shape_model.trim_components(max_components)

        return shape_model
