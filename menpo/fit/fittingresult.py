from __future__ import division
import abc
from hdf5able import HDF5able

from menpo.shape.pointcloud import PointCloud
from menpo.image import Image
from menpo.fitmultilevel.functions import compute_error
from menpo.visualize.base import Viewable, FittingViewer


class FittingResult(Viewable):
    r"""
    Object that holds the state of a single fitting object, during and after it
    has fitted a particular image.

    Parameters
    -----------
    image : :map:`Image` or subclass
        The fitted image.
    gt_shape : :map:`PointCloud`
        The ground truth shape associated to the image.
    """

    def __init__(self, image, gt_shape=None):
        self.image = image
        self._gt_shape = gt_shape

    @property
    def n_iters(self):
        return len(self.shapes) - 1

    @abc.abstractproperty
    def shapes(self):
        r"""
        A list containing the shapes obtained at each fitting iteration.

        :type: `list` of :map:`PointCloud`
        """

    @abc.abstractproperty
    def final_shape(self):
        r"""
        Returns the final fitted shape.
        """

    @abc.abstractproperty
    def initial_shape(self):
        r"""
        Returns the initial shape from which the fitting started.
        """

    @property
    def gt_shape(self):
        r"""
        Returns the original ground truth shape associated to the image.
        """
        return self._gt_shape

    @property
    def fitted_image(self):
        r"""
        Returns a copy of the fitted image with the following landmark
        groups attached to it:
            - ``initial``, containing the initial fitted shape .
            - ``final``, containing the final shape.
            - ``ground``, containing the ground truth shape. Only returned if
            the ground truth shape was provided.

        :type: :map:`Image`
        """
        image = Image(self.image.pixels)

        image.landmarks['initial'] = self.initial_shape
        image.landmarks['final'] = self.final_shape
        if self.gt_shape is not None:
            image.landmarks['ground'] = self.gt_shape
        return image

    @property
    def iter_image(self):
        r"""
        Returns a copy of the fitted image with as many landmark groups as
        iteration run by fitting procedure:
            - ``iter_0``, containing the initial shape.
            - ``iter_1``, containing the the fitted shape at the first
            iteration.
            - ``...``
            - ``iter_n``, containing the final fitted shape.

        :type: :map:`Image`
        """
        image = Image(self.image.pixels)
        for j, s in enumerate(self.shapes):
            key = 'iter_{}'.format(j)
            image.landmarks[key] = s
        return image

    def errors(self, error_type='me_norm'):
        r"""
        Returns a list containing the error at each fitting iteration.

        Parameters
        -----------
        error_type : `str` ``{'me_norm', 'me', 'rmse'}``, optional
            Specifies the way in which the error between the fitted and
            ground truth shapes is to be computed.

        Returns
        -------
        errors : `list` of `float`
            The errors at each iteration of the fitting process.
        """
        if self.gt_shape is not None:
            return [compute_error(t, self.gt_shape, error_type)
                    for t in self.shapes]
        else:
            raise ValueError('Ground truth has not been set, errors cannot '
                             'be computed')

    def final_error(self, error_type='me_norm'):
        r"""
        Returns the final fitting error.

        Parameters
        -----------
        error_type : `str` ``{'me_norm', 'me', 'rmse'}``, optional
            Specifies the way in which the error between the fitted and
            ground truth shapes is to be computed.

        Returns
        -------
        final_error : `float`
            The final error at the end of the fitting procedure.
        """
        if self.gt_shape is not None:
            return compute_error(self.final_shape, self.gt_shape, error_type)
        else:
            raise ValueError('Ground truth has not been set, final error '
                             'cannot be computed')

    def initial_error(self, error_type='me_norm'):
        r"""
        Returns the initial fitting error.

        Parameters
        -----------
        error_type : `str` ``{'me_norm', 'me', 'rmse'}``, optional
            Specifies the way in which the error between the fitted and
            ground truth shapes is to be computed.

        Returns
        -------
        initial_error : `float`
            The initial error at the start of the fitting procedure.
        """
        if self.gt_shape is not None:
            return compute_error(self.initial_shape, self.gt_shape, error_type)
        else:
            raise ValueError('Ground truth has not been set, final error '
                             'cannot be computed')

    def _view(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        Displays the whole fitting procedure.
        """
        pixels = self.image.pixels
        targets = [s.points for s in self.shapes]
        return FittingViewer(figure_id, new_figure, self.image.n_dims, pixels,
                             targets).render(**kwargs)

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
        if self.parameters is not None:
            parameters = [p.copy() for p in self.parameters]
        else:
            parameters = []
        gt_shape = self.gt_shape.copy() if self.gt_shape else None
        return SerializableFittingResult(self.image.copy(),
                                         parameters,
                                         [s.copy() for s in self.shapes],
                                         gt_shape)


class NonParametricFittingResult(FittingResult):
    r"""
    Object that holds the state of a Non Parametric :map:`Fitter` object
    before, during and after it has fitted a particular image.

    Parameters
    -----------
    image : :map:`Image`
        The fitted image.
    fitter : :map:`Fitter`
        The Fitter object used to fitter the image.
    shapes : `list` of :map:`PointCloud`
        The list of fitted shapes per iteration of the fitting procedure.
    gt_shape : :map:`PointCloud`
        The ground truth shape associated to the image.
    """

    def __init__(self, image, fitter, parameters=None, gt_shape=None):
        super(NonParametricFittingResult, self).__init__(image,
                                                         gt_shape=gt_shape)
        self.fitter = fitter
        # The parameters are the shapes for Non-Parametric algorithms
        self.parameters = parameters

    @property
    def shapes(self):
        return self.parameters

    @property
    def final_shape(self):
        return self.parameters[-1].copy()

    @property
    def initial_shape(self):
        return self.parameters[0].copy()

    @FittingResult.gt_shape.setter
    def gt_shape(self, value):
        r"""
        Setter for the ground truth shape associated to the image.
        """
        if isinstance(value, PointCloud):
            self._gt_shape = value
        else:
            raise ValueError("Accepted values for gt_shape setter are "
                             "PointClouds.")


class SemiParametricFittingResult(FittingResult):
    r"""
    Object that holds the state of a Semi Parametric :map:`Fitter` object
    before, during and after it has fitted a particular image.

    Parameters
    -----------
    image : :map:`Image`
        The fitted image.
    fitter : :map:`Fitter`
        The Fitter object used to fitter the image.
    parameters : `list` of `ndarray`
        The list of optimal transform parameters per iteration of the fitting
        procedure.
    gt_shape : :map:`PointCloud`
        The ground truth shape associated to the image.
    """

    def __init__(self, image, fitter, parameters=None, gt_shape=None):
        FittingResult.__init__(self, image, gt_shape=gt_shape)
        self.fitter = fitter
        self.parameters = parameters

    @property
    def transforms(self):
        r"""
        Generates a list containing the transforms obtained at each fitting
        iteration.
        """
        return [self.fitter.transform.from_vector(p) for p in self.parameters]

    @property
    def final_transform(self):
        r"""
        Returns the final transform.
        """
        return self.fitter.transform.from_vector(self.parameters[-1])

    @property
    def initial_transform(self):
        r"""
        Returns the initial transform from which the fitting started.
        """
        return self.fitter.transform.from_vector(self.parameters[0])

    @property
    def shapes(self):
        return [self.fitter.transform.from_vector(p).target
                for p in self.parameters]

    @property
    def final_shape(self):
        return self.final_transform.target

    @property
    def initial_shape(self):
        return self.initial_transform.target

    @FittingResult.gt_shape.setter
    def gt_shape(self, value):
        r"""
        Setter for the ground truth shape associated to the image.
        """
        if type(value) is PointCloud:
            self._gt_shape = value
        elif type(value) is list and value[0] is float:
            transform = self.fitter.transform.from_vector(value)
            self._gt_shape = transform.target
        else:
            raise ValueError("Accepted values for gt_shape setter are "
                             "PointClouds or float lists "
                             "specifying transform shapes.")


class ParametricFittingResult(SemiParametricFittingResult):
    r"""
    Object that holds the state of a Fully Parametric :map:`Fitter` object
    before, during and after it has fitted a particular image.

    Parameters
    -----------
    image : :map:`Image`
        The fitted image.
    fitter : :map:`Fitter`
        The Fitter object used to fitter the image.
    parameters : `list` of `ndarray`
        The list of optimal transform parameters per iteration of the fitting
        procedure.
    weights : `list` of `ndarray`
        The list of optimal appearance parameters per iteration of the fitting
        procedure.
    gt_shape : :map:`PointCloud`
        The ground truth shape associated to the image.
    """
    def __init__(self, image, fitter, parameters=None, weights=None,
                 gt_shape=None):
        SemiParametricFittingResult.__init__(self, image, fitter, parameters,
                                             gt_shape=gt_shape)
        self.weights = weights

    @property
    def warped_images(self):
        r"""
        The list containing the warped images obtained at each fitting
        iteration.

        :type: `list` of :map:`Image` or subclass
        """
        mask = self.fitter.template.mask
        transform = self.fitter.transform
        return [self.image.warp_to_mask(mask, transform.from_vector(p))
                for p in self.parameters]

    @property
    def appearance_reconstructions(self):
        r"""
        The list containing the appearance reconstruction obtained at
        each fitting iteration.

        :type: list` of :map:`Image` or subclass
        """
        if self.weights:
            return [self.fitter.appearance_model.instance(w)
                    for w in self.weights]
        else:
            return [self.fitter.template for _ in self.shapes()]

    @property
    def error_images(self):
        r"""
        The list containing the error images obtained at
        each fitting iteration.

        :type: list` of :map:`Image` or subclass
        """
        template = self.fitter.template
        warped_images = self.warped_images
        appearances = self.appearance_reconstructions

        error_images = []
        for a, i in zip(appearances, warped_images):
            error = a.as_vector() - i.as_vector()
            error_image = template.from_vector(error)
            error_images.append(error_image)

        return error_images


class SerializableFittingResult(HDF5able, FittingResult):
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
    parameters : `list` of `ndarray`
        The list of optimal transform parameters per iteration of the fitting
        procedure.
    shapes : `list` of :map:`PointCloud`
        The list of fitted shapes per iteration of the fitting procedure.
    gt_shape : :map:`PointCloud`
        The ground truth shape associated to the image.
    """
    def __init__(self, image, parameters, shapes, gt_shape):
        FittingResult.__init__(self, image, gt_shape=gt_shape)

        self.parameters = parameters
        self._shapes = shapes

    @property
    def shapes(self):
        return self._shapes

    @property
    def initial_shape(self):
        return self._shapes[0]

    @property
    def final_shape(self):
        return self._shapes[-1]
