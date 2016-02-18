from mock import Mock
from nose.tools import raises

from menpo.base import LazyList


def test_lazylist_get():
    mock_func = Mock()
    mock_func.return_value = 1
    ll = LazyList([mock_func] * 10)
    assert len(ll) == 10
    assert ll[0] == 1


def test_lazylist_multiple_calls():
    mock_func = Mock()
    mock_func.return_value = 1
    ll = LazyList([mock_func] * 10)
    ll[0]
    ll[0]
    assert mock_func.call_count == 2


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


def test_lazylist_init_from_index_callable():
    identity_func = lambda x: x
    ll = LazyList.init_from_index_callable(identity_func, 5)
    assert ll[0] == 0
    assert ll[-1] == 4


@raises(TypeError)
def test_lazylist_immutable():
    ll = LazyList([])
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
