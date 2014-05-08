import numpy as np
from mock import Mock

from menpo.transform import Transform


x = np.zeros([5, 5])


class MockTransform(Transform):
    def _apply(self, x, **kwargs):
        return x


def transform_n_dims_return_None_test():
    tr = MockTransform()
    n_dims = tr.n_dims
    assert (n_dims is None)


def transform_n_dims_output_return_None_test():
    tr = MockTransform()
    n_dims = tr.n_dims_output
    assert (n_dims is None)


def transform_apply_x_transformable_test():
    mocked = Mock(**{'_transform.return_value': x})
    tr = MockTransform()
    new_x = tr.apply(mocked)

    assert (new_x is x)
    assert mocked._transform.called


def transform_compose_before_test():
    mocked = Mock(**{'_apply.return_value': x})
    tr = MockTransform()
    chain = tr.compose_before(mocked)
    # Check transform chain
    assert (len(chain.transforms) == 2)
    assert (chain.transforms[0] is tr)
    assert (chain.transforms[1] is mocked)
    # Apply transform chain
    result = chain.apply(x)
    assert (result is x)
    assert mocked._apply.called


def transform_compose_after_test():
    mocked = Mock(**{'_apply.return_value': x})
    tr = MockTransform()
    chain = tr.compose_after(mocked)
    # Check transform chain
    assert (len(chain.transforms) == 2)
    assert (chain.transforms[0] is mocked)
    assert (chain.transforms[1] is tr)
    # Apply transform chain
    result = chain.apply(x)
    assert (result is x)
    assert mocked._apply.called
