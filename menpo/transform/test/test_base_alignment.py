from mock import Mock
from pytest import raises
import numpy as np
from numpy.testing import assert_equal

from menpo.transform.base import Alignment


# Mock a pointcloud
source_array = np.random.random([5, 2])
target_array = np.random.random([5, 2])
target_mismatch_array = np.random.random([8, 2])
target_array_3d = np.random.random([5, 3])
source = Mock(points=source_array, n_dims=source_array.shape[1],
              n_points=source_array.shape[0])
target = Mock(points=target_array, n_dims=target_array.shape[1],
              n_points=target_array.shape[0])
target_mismatch = Mock(points=target_mismatch_array,
                       n_dims=target_mismatch_array.shape[1],
                       n_points=target_mismatch_array.shape[0])
target_3d = Mock(points=target_array_3d, n_dims=target_array_3d.shape[1],
                 n_points=target_array_3d.shape[0])


class MockedAlignment(Alignment):
    def _sync_state_from_target(self):
        pass

    def apply(self, x):
        return x


def test_alignment_init():
    MockedAlignment(source, target)


def test_alignment_init_n_dims_mismatch():
    with raises(ValueError):
        MockedAlignment(source, target_3d)


def test_alignment_init_n_dims_mismatch():
    with raises(ValueError):
        MockedAlignment(source, target_3d)


def test_alignment_init_n_points_mismatch():
    with raises(ValueError):
        MockedAlignment(source, target_mismatch)


def test_alignment_source():
    al = MockedAlignment(source, target)
    assert (al.source is source)


def test_alignment_target():
    al = MockedAlignment(source, target)
    assert (al.target is target)


def test_alignment_aligned_source():
    al = MockedAlignment(source, target)
    assert (al.aligned_source() is source)


def test_alignment_alignment_error():
    al = MockedAlignment(source, target)
    error = al.alignment_error()
    assert_equal(error, np.linalg.norm(al.source.points - al.target.points))


def test_alignment_view_non_2d():
    al = MockedAlignment(target_3d, target_3d)
    with raises(NotImplementedError):
        al.view()
