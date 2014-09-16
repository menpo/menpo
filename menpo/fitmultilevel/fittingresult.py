from __future__ import division

from itertools import chain
from hdf5able import HDF5able

from menpo.transform import Scale
from menpo.fit.fittingresult import FittingResult

from .base import name_of_callable


def _rescale_shapes_to_reference(fitting_results, n_levels, downscale,
                                 affine_correction):
    n = n_levels - 1
    shapes = []
    for j, f in enumerate(fitting_results):
        transform = Scale(downscale ** (n - j), f.final_shape.n_dims)
        for t in f.shapes:
            t = transform.apply(t)
            shapes.append(affine_correction.apply(t))
    return shapes


class MultilevelFittingResult(FittingResult):
    r"""
    Class that holds the state of a :map:`MultilevelFitter` object before,
    during and after it has fitted a particular image.

    Parameters
    -----------
    image : :map:`Image` or subclass
        The fitted image.
    multilevel_fitter : :map:`MultilevelFitter`
        The multilevel fitter object used to fit the image.
    fitting_results : `list` of :map:`FittingResult`
        The list of fitting results.
    affine_correction : :map:`Affine`
        The affine transform between the initial shape of the highest
        pyramidal level and the initial shape of the original image
    gt_shape : class:`PointCloud`, optional
        The ground truth shape associated to the image.
    """
    def __init__(self, image, multiple_fitter, fitting_results,
                 affine_correction, gt_shape=None):
        super(MultilevelFittingResult, self).__init__(image, gt_shape=gt_shape)
        self.fitter = multiple_fitter
        self.fitting_results = fitting_results
        self._affine_correction = affine_correction

    @property
    def n_levels(self):
        r"""
        The number of levels of the fitter object.

        :type: `int`
        """
        return self.fitter.n_levels

    @property
    def downscale(self):
        r"""
        The downscale factor used by the multiple fitter.

        :type: `float`
        """
        return self.fitter.downscale

    @property
    def n_iters(self):
        r"""
        The total number of iterations used to fitter the image.

        :type: `int`
        """
        n_iters = 0
        for f in self.fitting_results:
            n_iters += f.n_iters
        return n_iters

    @property
    def shapes(self):
        return _rescale_shapes_to_reference(self.fitting_results, self.n_levels,
                                            self.downscale,
                                            self._affine_correction)

    @property
    def final_shape(self):
        r"""
        The final fitted shape.

        :type: :map:`PointCloud`
        """
        return self._affine_correction.apply(
            self.fitting_results[-1].final_shape)

    @property
    def initial_shape(self):
        r"""
        The initial shape from which the fitting started.

        :type: :map:`PointCloud`
        """
        n = self.n_levels - 1
        initial_shape = self.fitting_results[0].initial_shape
        Scale(self.downscale ** n, initial_shape.n_dims).apply_inplace(
            initial_shape)

        return self._affine_correction.apply(initial_shape)

    @FittingResult.gt_shape.setter
    def gt_shape(self, value):
        r"""
        Setter for the ground truth shape associated to the image.

        type: :map:`PointCloud`
        """
        self._gt_shape = value

    def __str__(self):
        if self.fitter.pyramid_on_features:
            feat_str = name_of_callable(self.fitter.features)
        else:
            feat_str = []
            for j in range(self.n_levels):
                if isinstance(self.fitter.features[j], str):
                    feat_str.append(self.fitter.features[j])
                elif self.fitter.features[j] is None:
                    feat_str.append("none")
                else:
                    feat_str.append(name_of_callable(self.fitter.features[j]))
        out = "Fitting Result\n" \
              " - Initial error: {0:.4f}\n" \
              " - Final error: {1:.4f}\n" \
              " - {2} method with {3} pyramid levels, {4} iterations " \
              "and using {5} features.".format(
              self.initial_error(), self.final_error(), self.fitter.algorithm,
              self.n_levels, self.n_iters, feat_str)
        return out

    def as_serializable(self):
        r""""
        Returns a serializable version of the fitting result. This is a much
        lighter weight object than the initial fitting result. For example,
        it won't contain the original fitting object.

        Returns
        -------
        serializable_fitting_result : :map:`SerializableFittingResult`
            The lightweight serializable version of this fitting result.
        """
        gt_shape = self.gt_shape.copy() if self.gt_shape else None
        fr_copies = [fr.as_serializable() for fr in self.fitting_results]

        return SerializableMultilevelFittingResult(
            self.image.copy(), fr_copies,
            gt_shape, self.n_levels, self.downscale, self.n_iters,
            self._affine_correction.copy())


class AAMMultilevelFittingResult(MultilevelFittingResult):
    r"""
    Class that holds the state of a :map:`AAMFitter` object before,
    during and after it has fitted a particular image.
    """
    @property
    def costs(self):
        r"""
        Returns a list containing the cost at each fitting iteration.

        :type: `list` of `float`
        """
        raise ValueError('costs not implemented yet.')

    @property
    def final_cost(self):
        r"""
        Returns the final fitting cost.

        :type: `float`
        """
        raise ValueError('costs not implemented yet.')

    @property
    def initial_cost(self):
        r"""
        Returns the initial fitting cost.

        :type: `float`
        """
        raise ValueError('costs not implemented yet.')

    @property
    def warped_images(self):
        r"""
        The list containing the warped images obtained at each fitting
        iteration.

        :type: `list` of :map:`Image` or subclass
        """
        mask = self.fitting_results[-1].fitter.template.mask
        transform = self.fitting_results[-1].fitter.transform
        warped_images = []
        for s in self.shapes():
            transform.set_target(s)
            image = self.image.warp_to_mask(mask, transform)
            warped_images.append(image)

        return warped_images

    @property
    def appearance_reconstructions(self):
        r"""
        The list containing the appearance reconstruction obtained at
        each fitting iteration.

        :type: `list` of :map:`Image` or subclass
        """
        return list(chain(
            *[f.appearance_reconstructions for f in self.fitting_results]))

    @property
    def error_images(self):
        r"""
        The list containing the error images obtained at each fitting
        iteration.

        :type: `list` of :map:`Image` or subclass
        """
        return list(chain(
            *[f.error_images for f in self.fitting_results]))

    @property
    def aam_reconstructions(self):
        r"""
        The list containing the aam reconstruction (i.e. the appearance
        reconstruction warped on the shape instance reconstruction) obtained at
        each fitting iteration.

        Note that this reconstruction is only tested to work for the
        :map:`OrthoMDTransform`

        :type: list` of :map:`Image` or subclass
        """
        aam_reconstructions = []
        for level, f in enumerate(self.fitting_results):
            if f.weights:
                for shape_w, aw in zip(f.parameters, f.weights):
                    shape_w = shape_w[4:]
                    sm_level = self.fitter.aam.shape_models[level]
                    am_level = self.fitter.aam.appearance_models[level]
                    swt = shape_w / sm_level.eigenvalues[:len(shape_w)] ** 0.5
                    awt = aw / am_level.eigenvalues[:len(aw)] ** 0.5
                    aam_reconstructions.append(self.fitter.aam.instance(
                        shape_weights=swt, appearance_weights=awt, level=level))
            else:
                for shape_w in f.parameters:
                    shape_w = shape_w[4:]
                    sm_level = self.fitter.aam.shape_models[level]
                    swt = shape_w / sm_level.eigenvalues[:len(shape_w)] ** 0.5
                    aam_reconstructions.append(self.fitter.aam.instance(
                        shape_weights=swt, appearance_weights=None,
                        level=level))
        return aam_reconstructions


class SerializableMultilevelFittingResult(HDF5able, FittingResult):
    r"""
    Designed to allow the fitting results to be easily serializable. In
    comparison to the other fitting result objects, the serializable fitting
    results contain a much stricter set of data. For example, the major data
    components of a serializable fitting result are the fitted shapes, the
    parameters and the fitted image.

    Parameters
    -----------
    image : :map:`Image`
        The fitted image.
    shapes : `list` of :map:`PointCloud`
        The list of fitted shapes per iteration of the fitting procedure.
    gt_shape : :map:`PointCloud`
        The ground truth shape associated to the image.
    n_levels : `int`
        Number of levels within the multilevel fitter.
    downscale : `int`
        Scale of downscaling applied to the image.
    n_iters : `int`
        Number of iterations the fitter performed.
    """
    def __init__(self, image, fitting_results, gt_shape, n_levels,
                 downscale, n_iters, affine_correction):
        FittingResult.__init__(self, image, gt_shape=gt_shape)
        self.fitting_results = fitting_results
        self.n_levels = n_levels
        self._n_iters = n_iters
        self.downscale = downscale
        self.affine_correction = affine_correction

    @property
    def n_iters(self):
        return self.n_iters

    @property
    def final_shape(self):
        return self.shapes[-1]

    @property
    def initial_shape(self):
        return self.shapes[0]

    @property
    def shapes(self):
        return _rescale_shapes_to_reference(self.fitting_results, self.n_levels,
                                            self.downscale,
                                            self.affine_correction)
