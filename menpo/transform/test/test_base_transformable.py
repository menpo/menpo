from mock import Mock

from menpo.transform.base import Transformable


class MockTransformable(Transformable):
    def _transform_inplace(self, transform):
        transform()


def test_transformable_transform():
    mocked = Mock(return_value=1)
    tr = MockTransformable()
    new_tr = tr._transform(mocked)
    assert mocked.called
    assert (new_tr is not tr)


def test_transformable_transform_inplace():
    mocked = Mock(return_value=1)
    tr = MockTransformable()
    no_return = tr._transform_inplace(mocked)
    assert mocked.called
    assert (no_return is None)
