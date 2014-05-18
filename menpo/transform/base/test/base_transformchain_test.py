from mock import Mock

from menpo.transform import TransformChain, Transform


def transformchain_apply_test():
    mocked_transform = Mock()
    mocked_transform._apply.return_value = 3
    mocked_transform2 = Mock()
    mocked_transform2._apply.return_value = 4
    transforms = [mocked_transform, mocked_transform2]
    tr = TransformChain(transforms)
    result = tr.apply(1)
    assert (result == 4)


def transformchain_composes_inplace_with_test():
    tr = TransformChain([])
    assert (tr.composes_inplace_with == Transform)


def transformchain_compose_before_composes_with_test():
    tr = TransformChain([])
    new_tr = tr.compose_before(Mock(spec=Transform))
    assert (new_tr is not tr)
    assert (len(new_tr.transforms) is 1)


def transformchain_compose_before_inplace_order_test():
    m1 = Mock(spec=Transform)
    m2 = Mock(spec=Transform)
    tr = TransformChain([m1])
    tr.compose_before_inplace(m2)
    assert (tr.transforms[1] is m2)


def transformchain_compose_after_inplace_order_test():
    m1 = Mock(spec=Transform)
    m2 = Mock(spec=Transform)
    tr = TransformChain([m1])
    tr.compose_after_inplace(m2)
    assert (tr.transforms[0] is m2)


def transformchain_compose_before_inplace_composes_with_test():
    tr = TransformChain([])
    ref = tr
    no_return = tr.compose_after_inplace(Mock(spec=Transform))
    assert (no_return is None)
    assert (ref is tr)
    assert (len(tr.transforms) is 1)


def transformchain_compose_after_composes_with_test():
    tr = TransformChain([])
    new_tr = tr.compose_after(Mock(spec=Transform))
    assert (new_tr is not tr)
    assert (len(new_tr.transforms) is 1)


def transformchain_compose_after_inplace_composes_with_test():
    tr = TransformChain([])
    ref = tr
    no_return = tr.compose_after_inplace(Mock(spec=Transform))
    assert (no_return is None)
    assert (ref is tr)
    assert (len(tr.transforms) is 1)
