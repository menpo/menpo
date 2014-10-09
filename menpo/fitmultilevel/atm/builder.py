from __future__ import division
import numpy as np

from menpo.fitmultilevel.aam.builder import build_reference_frame
from menpo.shape import TriMesh
from menpo.image import MaskedImage
from menpo.transform import Translation, Scale
from menpo.transform.piecewiseaffine import PiecewiseAffine
from menpo.fitmultilevel.base import create_pyramid
from menpo.transform.thinplatesplines import ThinPlateSplines
from menpo.model import PCAModel
from menpo.fitmultilevel.builder import (DeformableModelBuilder,
                                         compute_reference_shape,
                                         build_shape_model)
from menpo.fitmultilevel import checks
from menpo.visualize import print_dynamic, progress_bar_str
from menpo.feature import igo


class ATMBuilder(DeformableModelBuilder):
    r"""
    Class that builds Multilevel Active Template Models.

    Parameters
    ----------
    features : `callable` or ``[callable]``, optional
        If list of length ``n_levels``, feature extraction is performed at
        each level after downscaling of the image.
        The first element of the list specifies the features to be extracted at
        the lowest pyramidal level and so on.

        If ``callable`` the specified feature will be applied to the original
        image and pyramid generation will be performed on top of the feature
        image. Also see the `pyramid_on_features` property.

        Note that from our experience, this approach of extracting features
        once and then creating a pyramid on top tends to lead to better
        performing AAMs.

    transform : :map:`PureAlignmentTransform`, optional
        The :map:`PureAlignmentTransform` that will be
        used to warp the images.

    trilist : ``(t, 3)`` `ndarray`, optional
        Triangle list that will be used to build the reference frame. If
        ``None``, defaults to performing Delaunay triangulation on the points.

    normalization_diagonal : `int` >= ``20``, optional
        During building an AAM, all images are rescaled to ensure that the
        scale of their landmarks matches the scale of the mean shape.

        If `int`, it ensures that the mean shape is scaled so that the diagonal
        of the bounding box containing it matches the normalization_diagonal
        value.

        If ``None``, the mean shape is not rescaled.

        Note that, because the reference frame is computed from the mean
        landmarks, this kwarg also specifies the diagonal length of the
        reference frame (provided that features computation does not change
        the image size).

    n_levels : `int` > 0, optional
        The number of multi-resolution pyramidal levels to be used.

    downscale : `float` >= ``1``, optional
        The downscale factor that will be used to create the different
        pyramidal levels. The scale factor will be::

            (downscale ** k) for k in range(``n_levels``)

    scaled_shape_models : `boolean`, optional
        If ``True``, the reference frames will be the mean shapes of
        each pyramid level, so the shape models will be scaled.

        If ``False``, the reference frames of all levels will be the mean shape
        of the highest level, so the shape models will not be scaled; they will
        have the same size.

        Note that from our experience, if ``scaled_shape_models`` is ``False``,
        AAMs tend to have slightly better performance.

    max_shape_components : ``None`` or `int` > 0 or ``0`` <= `float` <= ``1`` or list of those, optional
        If list of length ``n_levels``, then a number of shape components is
        defined per level. The first element of the list specifies the number
        of components of the lowest pyramidal level and so on.

        If not a list or a list with length ``1``, then the specified number of
        shape components will be used for all levels.

        Per level:
            If `int`, it specifies the exact number of components to be
            retained.

            If `float`, it specifies the percentage of variance to be retained.

            If ``None``, all the available components are kept
            (100% of variance).

    boundary : `int` >= ``0``, optional
        The number of pixels to be left as a safe margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).

    Returns
    -------
    atm : :map:`ATMBuilder`
        The ATM Builder object

    Raises
    -------
    ValueError
        ``n_levels`` must be `int` > ``0``
    ValueError
        ``downscale`` must be >= ``1``
    ValueError
        ``normalization_diagonal`` must be >= ``20``
    ValueError
        ``max_shape_components`` must be ``None`` or an `int` > 0 or
        a ``0`` <= `float` <= ``1`` or a list of those containing 1 or
        ``n_levels`` elements
    ValueError
        ``features`` must be a `function` or a list of those
        containing ``1`` or ``n_levels`` elements
    """
    def __init__(self, features=igo, transform=PiecewiseAffine,
                 trilist=None, normalization_diagonal=None, n_levels=3,
                 downscale=2, scaled_shape_models=True,
                 max_shape_components=None, boundary=3):
        # check parameters
        checks.check_n_levels(n_levels)
        checks.check_downscale(downscale)
        checks.check_normalization_diagonal(normalization_diagonal)
        checks.check_boundary(boundary)
        max_shape_components = checks.check_max_components(
            max_shape_components, n_levels, 'max_shape_components')
        features = checks.check_features(features, n_levels)
        # store parameters
        self.features = features
        self.transform = transform
        self.trilist = trilist
        self.normalization_diagonal = normalization_diagonal
        self.n_levels = n_levels
        self.downscale = downscale
        self.scaled_shape_models = scaled_shape_models
        self.max_shape_components = max_shape_components
        self.boundary = boundary

    def build(self, shapes, template, group=None, label=None, verbose=False):
        r"""
        Builds a Multilevel Active Template Model given a list of shapes and a
        template image.

        Parameters
        ----------
        shapes : list of :map:`PointCloud`
            The set of shapes from which to build the shape model of the ATM.

        template : :map:`Image` or subclass
            The image to be used as template.

        group : `string`, optional
            The key of the landmark set of the template that should be used. If
            ``None``, and if there is only one set of landmarks, this set will
            be used.

        label : `string`, optional
            The label of the landmark manager of the template that you wish to
            use. If ``None`` is passed, the convex hull of all landmarks is
            used.

        verbose : `boolean`, optional
            Flag that controls information and progress printing.

        Returns
        -------
        atm : :map:`ATM`
            The ATM object. Shape and appearance models are stored from lowest
            to highest level.
        """
        # compute reference_shape
        self.reference_shape = compute_reference_shape(
            shapes, self.normalization_diagonal, verbose=verbose)

        # normalize the template size using the reference_shape scaling
        if verbose:
            print_dynamic('- Normalizing template size')
        normalized_template = template.rescale_to_reference_shape(
            self.reference_shape, group=group, label=label)

        # create pyramid for template image
        generator = create_pyramid([normalized_template], self.n_levels,
                                   self.downscale, self.features)

        # build the model at each pyramid level
        if verbose:
            if self.n_levels > 1:
                print_dynamic('- Building model for each of the {} pyramid '
                              'levels\n'.format(self.n_levels))
            else:
                print_dynamic('- Building model\n')

        shape_models = []
        warped_templates = []
        # for each pyramid level (high --> low)
        for j in range(self.n_levels):
            # since models are built from highest to lowest level, the
            # parameters in form of list need to use a reversed index
            rj = self.n_levels - j - 1

            if verbose:
                level_str = '  - '
                if self.n_levels > 1:
                    level_str = '  - Level {}: '.format(j + 1)

            # rescale shapes if required
            if j > 0 and self.scaled_shape_models:
                scale_transform = Scale(scale_factor=1.0 / self.downscale,
                                        n_dims=2)
                shapes = [scale_transform.apply(s) for s in shapes]

            # train shape model and find reference frame
            if verbose:
                print_dynamic('{}Building shape model'.format(level_str))
            shape_model = build_shape_model(shapes,
                                            self.max_shape_components[rj])
            reference_frame = self._build_reference_frame(shape_model.mean)

            # add shape model to the list
            shape_models.append(shape_model)

            # get template's feature image of current level
            if verbose:
                print_dynamic('{}Warping template'.format(level_str))
            feature_template = next(generator)

            # compute transform
            transform = self.transform(reference_frame.landmarks['source'].lms,
                                       feature_template.landmarks[group][label])

            # warp template to reference frame
            warped_templates.append(
                feature_template.warp_to_mask(reference_frame.mask, transform))

            # attach reference_frame to template's source shape
            warped_templates[j].landmarks['source'] = \
                reference_frame.landmarks['source']

            if verbose:
                print_dynamic('{}Done\n'.format(level_str))

        # reverse the list of shape and appearance models so that they are
        # ordered from lower to higher resolution
        shape_models.reverse()
        warped_templates.reverse()
        n_training_shapes = len(shapes)

        return self._build_atm(shape_models, warped_templates,
                               n_training_shapes)

    def _build_reference_frame(self, mean_shape):
        r"""
        Generates the reference frame given a mean shape.

        Parameters
        ----------
        mean_shape : :map:`PointCloud`
            The mean shape to use.

        Returns
        -------
        reference_frame : :map:`MaskedImage`
            The reference frame.
        """
        return build_reference_frame(mean_shape, boundary=self.boundary,
                                     trilist=self.trilist)

    def _build_atm(self, shape_models, warped_templates, n_training_shapes):
        r"""
        Returns an ATM object.

        Parameters
        ----------
        shape_models : `list` of :map:`PCAModel`
            The trained multilevel shape models.

        warped_templates : `list` of :map:`MaskedImage`
            The warped multilevel templates.

        n_training_shapes : `int`
            The number of training shapes.

        Returns
        -------
        atm : :map:`ATM`
            The trained ATM object.
        """
        from .base import ATM
        return ATM(shape_models, warped_templates, n_training_shapes,
                   self.transform, self.features, self.reference_shape,
                   self.downscale, self.scaled_shape_models)
