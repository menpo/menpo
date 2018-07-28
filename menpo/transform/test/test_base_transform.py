import numpy as np
from numpy.testing import assert_allclose
from pytest import raises
from mock import Mock

from menpo.transform import Transform


x = np.zeros([5, 5])


class MockTransform(Transform):
    def _apply(self, x, **kwargs):
        return np.array(x)


def test_transform_n_dims_return_None():
    tr = MockTransform()
    n_dims = tr.n_dims
    assert (n_dims is None)


def test_transform_n_dims_output_return_None():
    tr = MockTransform()
    n_dims = tr.n_dims_output
    assert (n_dims is None)


def test_transform_apply_x_not_transformable():
    tr = MockTransform()
    new_x = tr.apply(x)

    assert (new_x is not x)
    assert_allclose(new_x, x)


def test_transform_apply_x_transformable():
    mocked = Mock()
    mocked._transform.return_value = mocked
    tr = MockTransform()
    transformed_mock = tr.apply(mocked)

    assert (transformed_mock is mocked)
    assert mocked._transform.called


def test_transform_apply_inplace_x_transformable():
    mocked = Mock()
    mocked._transform_inplace.return_value = mocked
    tr = MockTransform()
    no_return = tr._apply_inplace(mocked)

    assert (no_return is None)
    assert mocked._transform_inplace.called


def test_transform_apply_inplace_x_not_transformable():
    tr = MockTransform()
    with raises(ValueError):
        tr._apply_inplace(x)


def test_transform_compose_before():
    mocked = Mock()
    mocked._apply.return_value = x
    tr = MockTransform()
    chain = tr.compose_before(mocked)
    # Check transform chain
    assert (len(chain.transforms) == 2)
    assert (chain.transforms[0] is tr)
    assert (chain.transforms[1] is mocked)


def test_transform_compose_after():
    mocked = Mock()
    mocked._apply.return_value = x
    tr = MockTransform()
    chain = tr.compose_after(mocked)
    # Check transform chain
    assert (len(chain.transforms) == 2)
    assert (chain.transforms[0] is mocked)
    assert (chain.transforms[1] is tr)
