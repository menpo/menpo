try:
    import collections.abc as collections_abc
except ImportError:
    import collections as collections_abc
import numpy as np
from mock import Mock
from pytest import raises

from menpo.base import LazyList


def test_lazylist_get():
    mock_func = Mock()
    mock_func.return_value = 1
    ll = LazyList([mock_func] * 10)
    assert len(ll) == 10
    assert ll[0] == 1


def test_lazylist_copy_lazy():
    mock_func = Mock()
    mock_func.return_value = 1
    ll = LazyList([mock_func] * 10)
    copied_ll = ll.copy()
    assert len(copied_ll) == 10
    assert id(ll._callables) != id(copied_ll._callables)
    mock_func.assert_not_called()


def test_lazylist_copy_duck_typed():
    mock_func = Mock()
    mock_func.return_value = 1
    ll = LazyList([mock_func] * 10)
    ll.fps = 50
    copied_ll = ll.copy()
    assert len(copied_ll) == 10
    assert id(ll._callables) != id(copied_ll._callables)
    mock_func.assert_not_called()
    assert copied_ll.fps == 50


def test_lazylist_multiple_calls():
    mock_func = Mock()
    mock_func.return_value = 1
    ll = LazyList([mock_func] * 10)
    ll[0]
    ll[0]
    assert mock_func.call_count == 2


def test_lazylist_multi_map():
    two_func = lambda: 2
    double_func = [lambda x: x * 2] * 2
    ll = LazyList([two_func] * 2)
    ll_mapped = ll.map(double_func)
    assert len(ll_mapped) == 2
    assert id(ll) != id(ll_mapped)
    assert all(x == 4 for x in ll_mapped)


def test_lazylist_multi_map_unequal_lengths():
    two_func = lambda: 2
    double_func = [lambda x: x * 2] * 2
    ll = LazyList([two_func])
    with raises(ValueError):
        ll.map(double_func)


def test_lazylist_multi_map_iterable_and_callable():
    class double_func(collections_abc.Iterable):
        def __call__(self, x, **kwargs):
            return x * 2

        def __iter__(self):
            yield 1

    f = double_func()
    two_func = lambda: 2
    ll = LazyList([two_func])
    with raises(ValueError):
        ll.map(f)


def test_lazylist_repeat():
    ll = LazyList.init_from_iterable([0, 1])
    ll_repeated = ll.repeat(2)
    assert len(ll_repeated) == 4
    assert all([a == b for a, b in zip([0, 0, 1, 1], ll_repeated)])


def test_lazylist_map():
    two_func = lambda: 2
    double_func = lambda x: x * 2
    ll = LazyList([two_func])
    ll_mapped = ll.map(double_func)
    assert id(ll) != id(ll_mapped)
    assert ll_mapped[0] == 4


def test_lazylist_double_map():
    two_func = lambda: 2
    double_func = lambda x: x * 2
    ll = LazyList([two_func])
    ll_mapped = ll.map(double_func)
    ll_mapped = ll_mapped.map(double_func)
    assert id(ll) != id(ll_mapped)
    assert ll_mapped[0] == 8


def test_lazylist_map_no_call():
    mock_func = Mock()
    double_func = lambda x: x * 2
    ll = LazyList([mock_func])
    ll_mapped = ll.map(double_func)
    assert id(ll) != id(ll_mapped)
    mock_func.assert_not_called()


def test_lazylist_init_from_iterable_identity():
    ll = LazyList.init_from_iterable([0, 1])
    assert ll[0] == 0
    assert ll[1] == 1


def test_lazylist_init_from_iterable_with_f():
    double_func = lambda x: x * 2
    ll = LazyList.init_from_iterable([0, 1], f=double_func)
    assert ll[0] == 0
    assert ll[1] == 2


def test_lazylist_init_from_index_callable():
    identity_func = lambda x: x
    ll = LazyList.init_from_index_callable(identity_func, 5)
    assert ll[0] == 0
    assert ll[-1] == 4


def test_lazylist_immutable():
    ll = LazyList([])
    with raises(TypeError):
        ll[0] = 1


def test_lazylist_add_lazylist():
    a = Mock()
    b = Mock()
    ll1 = LazyList([a])
    ll2 = LazyList([b])
    new_ll = ll1 + ll2
    assert len(new_ll) == 2
    assert new_ll._callables[0] is a
    assert new_ll._callables[1] is b


def test_lazylist_add_list():
    a = Mock()
    b = Mock()
    ll1 = LazyList([a])
    l2 = [b]
    new_ll = ll1 + l2
    assert len(new_ll) == 2
    assert new_ll._callables[0] is a
    assert new_ll._callables[1] is not b
    assert new_ll[1] is b


def test_lazylist_add_non_iterable_non_lazy_list_rases_value_error():
    with raises(ValueError):
        LazyList([1]) + None


def test_lazylist_slice_with_ndarray():
    index = np.array([1, 0, 3], dtype=np.int)
    l = LazyList.init_from_iterable(['a', 'b', 'c', 'd', 'e'])
    l_indexed = l[index]
    assert list(l_indexed) == ['b', 'a', 'd']
