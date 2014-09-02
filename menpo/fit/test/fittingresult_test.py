from nose.plugins.attrib import attr
import numpy as np
from numpy.testing import assert_approx_equal
from nose.tools import raises
from menpo.fit.fittingresult import FittingResult
from menpo.shape import PointCloud
from menpo.testing import is_same_array
from menpo.image import MaskedImage


class MockedFittingResult(FittingResult):

    def __init__(self, image, fitter, **kwargs):
        FittingResult.__init__(self, MaskedImage.blank((10, 10)), fitter,
                               **kwargs)
    @property
    def n_iters(self):
        return 1

    def shapes(self, as_points=False):
        if as_points:
            return [np.ones([3, 2])]
        else:
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
    fr = MockedFittingResult(None, None, gt_shape=pcloud)

    assert_approx_equal(fr.errors()[0], 0.9173896)


@raises(ValueError)
def test_fittingresult_errors_no_gt():
    fr = MockedFittingResult(None, None)
    fr.errors()


def test_fittingresult_gt_shape():
    pcloud = PointCloud(np.ones([3, 2]))
    fr = MockedFittingResult(None, None, gt_shape=pcloud)
    assert (is_same_array(fr.gt_shape.points, pcloud.points))


@attr('fuzzy')
def test_fittingresult_final_error_me_norm():
    pcloud = PointCloud(np.array([[1., 2], [3, 4], [5, 6]]))
    fr = MockedFittingResult(None, None, gt_shape=pcloud)

    assert_approx_equal(fr.final_error(), 0.9173896)


@raises(ValueError)
def test_fittingresult_final_error_no_gt():
    fr = MockedFittingResult(None, None)
    fr.final_error()


@attr('fuzzy')
def test_fittingresult_initial_error_me_norm():
    pcloud = PointCloud(np.array([[1., 2], [3, 4], [5, 6]]))
    fr = MockedFittingResult(None, None, gt_shape=pcloud)

    assert_approx_equal(fr.initial_error(), 0.9173896)


@raises(ValueError)
def test_fittingresult_initial_error_no_gt():
    fr = MockedFittingResult(None, None)
    fr.initial_error()
