from mock import Mock
from nose.tools import raises

from menpo.transform import TransformChain

from menpo.transform.base import ComposableTransform


class OtherMockedComposable(ComposableTransform):
    pass


class MockedComposable(ComposableTransform):
    def __init__(self):
        self.transforms = [self]

    @property
    def composes_inplace_with(self):
        return OtherMockedComposable

    def _compose_before_inplace(self, transform):
        self.transforms.append(transform)

    def _compose_after_inplace(self, transform):
        self.transforms.insert(0, transform)

    def _apply(self, x, **kwargs):
        return x


def composable_compose_before_not_composes_with_test():
    mocked = Mock()
    co = MockedComposable()
    new_co = co.compose_before(mocked)
    assert (new_co is not co)
    assert (isinstance(new_co, TransformChain))


def composable_compose_before_composes_with_test():
    mocked = Mock(spec=OtherMockedComposable)
    co = MockedComposable()
    new_co = co.compose_before(mocked)
    assert (new_co is not co)
    assert (new_co.transforms[1] is mocked)


def composable_compose_before_inplace_composes_with_test():
    mocked = Mock(spec=OtherMockedComposable)
    co = MockedComposable()
    ref = co
    no_return = co.compose_before_inplace(mocked)
    assert (no_return is None)
    assert (ref is co)
    assert (co.transforms[1] is mocked)


@raises(ValueError)
def composable_compose_before_inplace_not_composes_with_test():
    mocked = Mock()
    co = MockedComposable()
    co.compose_before_inplace(mocked)


def composable_compose_after_not_composes_with_test():
    mocked = Mock()
    co = MockedComposable()
    new_co = co.compose_after(mocked)
    assert (new_co is not co)
    assert (isinstance(new_co, TransformChain))


def composable_compose_after_composes_with_test():
    mocked = Mock(spec=OtherMockedComposable)
    co = MockedComposable()
    new_co = co.compose_after(mocked)
    assert (new_co is not co)
    assert (new_co.transforms[0] is mocked)


def composable_compose_after_inplace_composes_with_test():
    mocked = Mock(spec=OtherMockedComposable)
    co = MockedComposable()
    ref = co
    no_return = co.compose_after_inplace(mocked)
    assert (no_return is None)
    assert (ref is co)
    assert (co.transforms[0] is mocked)


@raises(ValueError)
def composable_compose_after_inplace_not_composes_with_test():
    mocked = Mock()
    co = MockedComposable()
    co.compose_after_inplace(mocked)
