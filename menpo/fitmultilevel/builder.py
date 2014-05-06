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

    @abc.abstractmethod
    def build(self, images, group=None, label='all'):
        r"""
        """
        pass

    @classmethod
    def _preprocessing(cls, images, group, label, normalization_diagonal,
                       interpolator, scaled_levels, n_levels, downscale,
                       verbose=False):
        r"""
        """
        # the mean shape of the images' landmarks is the reference shape
        if verbose:
            print('  - Computing reference shape')
        shapes = [i.landmarks[group][label].lms for i in images]
        reference_shape = mean_pointcloud(shapes)

        # normalize the scaling of all shapes wrt the reference shape
        if normalization_diagonal:
            x, y = reference_shape.range()
            scale = normalization_diagonal / np.sqrt(x**2 + y**2)
            Scale(scale, reference_shape.n_dims).apply_inplace(
                reference_shape)
        normalized_images = []
        for c, i in enumerate(images):
            if verbose:
                print_str = '  - Normalizing object size: ' + progress_bar_str(
                    float(c + 1) / len(images), show_bar=False)
                print_dynamic(print_str, new_line=(c == len(images) - 1))
            normalized_images.append(i.rescale_to_reference_shape(
                reference_shape, group=group, label=label,
                interpolator=interpolator))

        # generate pyramid
        if verbose:
            print('  - Generating multilevel scale space')
        if scaled_levels:
            generator = [i.gaussian_pyramid(n_levels=n_levels,
                                            downscale=downscale)
                         for i in normalized_images]
        else:
            generator = [i.smoothing_pyramid(n_levels=n_levels,
                                             downscale=downscale)
                         for i in normalized_images]

        return reference_shape, generator

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
