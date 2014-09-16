from nose.plugins.attrib import attr
import numpy as np
from numpy.testing import assert_approx_equal
from nose.tools import raises
from menpo.fit.fittingresult import (FittingResult, NonParametricFittingResult,
                                     SemiParametricFittingResult,
                                     ParametricFittingResult)
from menpo.shape import PointCloud
from menpo.testing import is_same_array
from menpo.image import MaskedImage
from mock import MagicMock


class MockedFittingResult(FittingResult):

    def __init__(self, gt_shape=None):
        FittingResult.__init__(self, MaskedImage.blank((10, 10)),
                               gt_shape=gt_shape)
    @property
    def n_iters(self):
        return 1

    @property
    def shapes(self):
        return [PointCloud(np.ones([3, 2]))]

    @property
    def final_shape(self):
        return PointCloud(np.ones([3, 2]))

    @property
    def initial_shape(self):
        return PointCloud(np.ones([3, 2]))


@attr('fuzzy')
def test_fittingresult_errors_me_norm():
    pcloud = PointCloud(np.array([[1., 2], [3, 4], [5, 6]]))
    fr = MockedFittingResult(gt_shape=pcloud)

    assert_approx_equal(fr.errors()[0], 0.9173896)


@raises(ValueError)
def test_fittingresult_errors_no_gt():
    fr = MockedFittingResult()
    fr.errors()


def test_fittingresult_gt_shape():
    pcloud = PointCloud(np.ones([3, 2]))
    fr = MockedFittingResult(gt_shape=pcloud)
    assert (is_same_array(fr.gt_shape.points, pcloud.points))


@attr('fuzzy')
def test_fittingresult_final_error_me_norm():
    pcloud = PointCloud(np.array([[1., 2], [3, 4], [5, 6]]))
    fr = MockedFittingResult(gt_shape=pcloud)

    assert_approx_equal(fr.final_error(), 0.9173896)


@raises(ValueError)
def test_fittingresult_final_error_no_gt():
    fr = MockedFittingResult()
    fr.final_error()


@attr('fuzzy')
def test_fittingresult_initial_error_me_norm():
    pcloud = PointCloud(np.array([[1., 2], [3, 4], [5, 6]]))
    fr = MockedFittingResult(gt_shape=pcloud)

    assert_approx_equal(fr.initial_error(), 0.9173896)


@raises(ValueError)
def test_fittingresult_initial_error_no_gt():
    fr = MockedFittingResult()
    fr.initial_error()


def test_nonpara_fittingresult_as_serialized():
    image = MagicMock()
    fitter = MagicMock()
    parameters = [MagicMock()]
    gt_shape = MagicMock()
    fr = NonParametricFittingResult(image, fitter, parameters=parameters,
                                    gt_shape=gt_shape)
    s_fr = fr.as_serializable()

    image.copy.assert_called_once()
    parameters[0].copy.assert_called_once()
    gt_shape.copy.assert_called_once()


def test_nonpara_fittingresult_as_serialized():
    image = MagicMock()
    fitter = MagicMock()
    parameters = [MagicMock()]
    gt_shape = MagicMock()
    fr = NonParametricFittingResult(image, fitter, parameters=parameters,
                                    gt_shape=gt_shape)
    s_fr = fr.as_serializable()

    image.copy.assert_called_once()
    parameters[0].copy.assert_called_once()
    gt_shape.copy.assert_called_once()