from menpo.feature import sparse_hog, igo
from menpo.fitmultilevel.base import (is_pyramid_on_features, name_of_callable)


class Foo():
    def __call__(self):
        pass


def test_is_pyramid_on_features_true():
    assert is_pyramid_on_features(igo)


def test_is_pyramid_on_features_false():
    assert not is_pyramid_on_features([igo, sparse_hog])


def test_name_of_callable_partial():
    assert name_of_callable(sparse_hog) == 'sparse_hog'


def test_name_of_callable_function():
    assert name_of_callable(igo) == 'igo'


def test_name_of_callable_object_with_call():
    assert name_of_callable(Foo()) == 'Foo'
