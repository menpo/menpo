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
