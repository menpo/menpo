import numpy as np
from numpy.testing import assert_allclose
from pytest import raises

from menpo.base import LazyList
from menpo.math import as_matrix
from menpo.model import PCAModel, PCAVectorModel
from menpo.shape import PointCloud


# ---- building from a data matrix, or from a list/LazyList of vectors
def _shapes(n=15, seed=7):
    rng = np.random.RandomState(seed)
    return [PointCloud(rng.rand(5, 2)) for _ in range(n)]


def test_pca_model_init_from_data_matrix_is_the_same_model_as_from_samples():
    shapes = _shapes()
    expected = PCAModel(shapes)
    data, template = as_matrix(shapes, return_template=True)
    model = PCAModel.init_from_data_matrix(data, template)
    assert model.n_samples == expected.n_samples == 15
    assert_allclose(model.mean().as_vector(), expected.mean().as_vector())
    assert_allclose(model.eigenvalues, expected.eigenvalues)
    assert_allclose(np.abs(model.components), np.abs(expected.components), atol=1e-8)
    assert (
        isinstance(model.mean(), PointCloud) and model.template_instance.n_points == 5
    )


def test_pca_model_init_from_data_matrix_options():
    shapes = _shapes()
    data, template = as_matrix(shapes, return_template=True)
    untouched = data.copy()
    model = PCAModel.init_from_data_matrix(
        data, template, max_n_components=3, inplace=False
    )
    assert model.n_components == 3
    assert_allclose(data, untouched)  # inplace=False leaves the matrix alone
    centred = PCAModel.init_from_data_matrix(
        data - data.mean(axis=0), template, centre=False
    )
    assert_allclose(centred.mean().as_vector(), 0.0)


def test_pca_model_init_from_data_matrix_checks_its_arguments():
    shapes = _shapes()
    data, template = as_matrix(shapes, return_template=True)
    with raises(TypeError):
        PCAModel.init_from_data_matrix(data, template.as_vector())  # an ndarray
    with raises(ValueError):
        PCAModel.init_from_data_matrix(data[:, :-1], template)  # wrong n_features
    with raises(ValueError):
        PCAModel.init_from_data_matrix(data[0], template)  # not 2D


def test_pca_vector_model_from_a_list_of_vectors_is_the_same_as_from_a_matrix():
    data = np.random.RandomState(8).rand(10, 6)
    from_matrix = PCAVectorModel(data.copy())
    from_list = PCAVectorModel(list(data))
    assert from_list.n_samples == 10
    assert_allclose(from_list.mean(), from_matrix.mean())
    assert_allclose(from_list.eigenvalues, from_matrix.eigenvalues)


def test_pca_vector_model_from_a_lazy_list_loads_each_sample_once():
    data = np.random.RandomState(9).rand(10, 6)
    loaded = []

    def loader(i):
        def load():
            loaded.append(i)
            return data[i].copy()

        return load

    model = PCAVectorModel(LazyList([loader(i) for i in range(10)]))
    assert sorted(loaded) == list(range(10))  # no sample was loaded twice
    assert_allclose(model.mean(), data.mean(axis=0))


def test_pca_vector_model_n_samples_takes_only_the_first_ones():
    data = np.random.RandomState(10).rand(10, 6)
    loaded = []
    lazy = LazyList(
        [(lambda i=i: loaded.append(i) or data[i].copy()) for i in range(10)]
    )
    model = PCAVectorModel(lazy, n_samples=4)
    assert model.n_samples == 4 and sorted(loaded) == [0, 1, 2, 3]
    assert_allclose(model.mean(), data[:4].mean(axis=0))
    assert_allclose(PCAVectorModel(data.copy(), n_samples=4).mean(), model.mean())


def test_pca_vector_model_more_samples_than_there_are_is_an_error():
    data = np.random.RandomState(11).rand(4, 6)
    with raises(ValueError):
        PCAVectorModel(list(data), n_samples=5)
    with raises(ValueError):
        PCAVectorModel(data, n_samples=5)


def test_pca_vector_model_verbose_shows_progress_for_a_list(capsys):
    PCAVectorModel(list(np.random.RandomState(12).rand(6, 4)), verbose=True)
    assert "Building data matrix" in capsys.readouterr().out
