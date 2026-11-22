import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_equal,
)
from pytest import raises

from menpo.math import as_matrix
from menpo.model import LinearVectorModel, PCAModel, PCAVectorModel
from menpo.shape import PointCloud


def test_linear_model_creation():
    data = np.zeros((3, 120))
    LinearVectorModel(data)


def test_linear_model_basics():
    data = np.random.random((3, 120))
    linear_model = LinearVectorModel(data)
    assert linear_model.n_components == 3
    assert linear_model.n_components == 3
    assert linear_model.n_features == 120


def test_linear_model_project_vector():
    data = np.zeros((3, 120))
    data[0, 0] = 1
    data[1, 1] = 1
    data[2, 2] = 1
    linear_model = LinearVectorModel(data)
    sample = np.random.random(120)
    weights = linear_model.project(sample)
    assert_allclose(weights, sample[:3])


def test_linear_model_component():
    data = np.random.random((3, 120))
    linear_model = LinearVectorModel(data)
    assert_equal(linear_model.component(2), data[2])


def test_linear_model_instance_vector():
    data = np.zeros((3, 120))
    data[0, 0] = 1
    data[1, 1] = 1
    data[2, 2] = 1
    linear_model = LinearVectorModel(data)
    weights = np.array([0.263, 7, 41.2])
    projected = linear_model.instance(weights)
    # only the first 3 features are non zero...
    assert_allclose(projected[:3], weights)
    # rest should be nil
    assert_allclose(projected[3:], 0)


def test_pca_trim_components():
    samples = [np.random.randn(10) for _ in range(10)]
    model = PCAVectorModel(samples)
    # integer
    small = model.trim_components(5)
    assert_equal(small.n_components, 5)
    # the model it came from is as it was
    assert_equal(model.n_components, 9)


def test_pca_trim_components_too_many():
    samples = [np.random.randn(10) for _ in range(10)]
    model = PCAVectorModel(samples)
    # too many components: all of them
    assert_equal(model.trim_components(100).n_components, 9)
    # a smaller number of components, then too many again
    small = model.trim_components(5)
    assert_equal(small.n_components, 5)
    assert_equal(small.trim_components(100).n_components, 5)


def test_pca_trim_components_negative():
    samples = [np.random.randn(10) for _ in range(10)]
    model = PCAVectorModel(samples)
    with raises(ValueError):
        model.trim_components(-5)


def test_pca_trim():
    samples = [PointCloud(np.random.randn(10)) for _ in range(10)]
    model = PCAModel(samples)
    # trim components
    model.trim_components_inplace(5)
    assert_equal(model.n_components, 5)


def test_pca_trim_variance_limit():
    samples = [np.random.randn(10) for _ in range(10)]
    model = PCAVectorModel(samples)
    with raises(ValueError):
        # impossible to keep more than 1.0 ratio variance
        model.trim_components_inplace(2.5)


def test_pca_trim_negative_integers():
    samples = [PointCloud(np.random.randn(10)) for _ in range(10)]
    model = PCAModel(samples)
    with raises(ValueError):
        # no negative number of components
        model.trim_components_inplace(-2)


def test_pca_trim_negative_float():
    samples = [PointCloud(np.random.randn(10)) for _ in range(10)]
    model = PCAModel(samples)
    with raises(ValueError):
        # no negative number of components
        model.trim_components_inplace(-2)


def test_pca_variance():
    samples = [PointCloud(np.random.randn(10)) for _ in range(10)]
    model = PCAModel(samples)
    # kept variance must be equal to total variance
    assert_equal(model.variance(), model.original_variance())
    # kept variance ratio must be 1.0
    assert_equal(model.variance_ratio(), 1.0)
    # noise variance must be 0.0
    assert_equal(model.noise_variance(), 0.0)
    # noise variance ratio must be also 0.0
    assert_equal(model.noise_variance_ratio(), 0.0)


def test_pca_inverse_noise_variance():
    samples = [PointCloud(np.random.randn(10)) for _ in range(10)]
    model = PCAModel(samples)
    with raises(ValueError):
        # inverse noise_variance it's not computable
        model.inverse_noise_variance()


def test_pca_variance_after_trim_components():
    samples = [np.random.randn(10) for _ in range(10)]
    model = PCAVectorModel(samples)
    # use fewer components
    model = model.trim_components(5)
    # kept variance must be smaller than total variance
    assert model.variance() < model.original_variance()
    # kept variance ratio must be smaller than 1.0
    assert model.variance_ratio() < 1.0
    # noise variance must be bigger than 0.0
    assert model.noise_variance() > 0.0
    # noise variance ratio must also be bigger than 0.0
    assert model.noise_variance_ratio() > 0.0
    # inverse noise variance is computable
    assert model.inverse_noise_variance() == 1 / model.noise_variance()


def test_pca_variance_after_trim():
    samples = [np.random.randn(10) for _ in range(10)]
    model = PCAVectorModel(samples)
    # set number of active components
    model.trim_components_inplace(5)
    # kept variance must be smaller than total variance
    assert model.variance() < model.original_variance()
    # kept variance ratio must be smaller than 1.0
    assert model.variance_ratio() < 1.0
    # noise variance must be bigger than 0.0
    assert model.noise_variance() > 0.0
    # noise variance ratio must also be bigger than 0.0
    assert model.noise_variance_ratio() > 0.0
    # inverse noise variance is computable
    assert model.inverse_noise_variance() == 1 / model.noise_variance()


def test_pca_orthogonalize_against():
    pca_samples = np.random.randn(10, 10)
    pca_model = PCAVectorModel(pca_samples)
    lm_samples = np.asarray([np.random.randn(10) for _ in range(4)])
    lm_model = LinearVectorModel(np.asarray(lm_samples))
    # orthogonalize
    pca_model.orthonormalize_against_inplace(lm_model)
    # number of active components must remain the same
    assert_equal(pca_model.n_components, 6)


def test_pca_increment_centred():
    pca_samples = [PointCloud(np.random.randn(10, 2)) for _ in range(10)]
    ipca_model = PCAModel(pca_samples[:3])
    ipca_model.increment_inplace(pca_samples[3:6])
    ipca_model.increment_inplace(pca_samples[6:])

    bpca_model = PCAModel(pca_samples)

    assert_almost_equal(np.abs(ipca_model.components), np.abs(bpca_model.components))
    assert_almost_equal(ipca_model.eigenvalues, bpca_model.eigenvalues)
    assert_almost_equal(ipca_model.mean().as_vector(), bpca_model.mean().as_vector())


def test_pca_increment_noncentred():
    pca_samples = [np.random.randn(10) for _ in range(10)]
    ipca_model = PCAVectorModel(pca_samples[:3], centre=False)
    ipca_model.increment_inplace(pca_samples[3:6])
    ipca_model.increment_inplace(pca_samples[6:])

    bpca_model = PCAVectorModel(pca_samples, centre=False)

    assert_almost_equal(np.abs(ipca_model.components), np.abs(bpca_model.components))
    assert_almost_equal(ipca_model.eigenvalues, bpca_model.eigenvalues)
    assert_almost_equal(ipca_model.mean(), bpca_model.mean())


def test_pca_vector_init_from_covariance():
    n_samples = 30
    n_features = 10
    centre_values = [True, False]
    for centre in centre_values:
        # generate samples matrix and mean vector
        samples = np.random.randn(n_samples, n_features)
        mean = np.mean(samples, axis=0)
        # compute covariance matrix
        if centre:
            X = samples - mean
            C = np.dot(X.T, X) / (n_samples - 1)
        else:
            C = np.dot(samples.T, samples) / (n_samples - 1)
        # create the 2 pca models
        pca1 = PCAVectorModel.init_from_covariance_matrix(
            C, mean, centred=centre, n_samples=n_samples
        )
        pca2 = PCAVectorModel(samples, centre=centre, inplace=False)
        # compare them
        assert_array_almost_equal(pca1.mean(), pca2.mean())
        assert_array_almost_equal(
            pca1.component(0, with_mean=False), pca2.component(0, with_mean=False)
        )
        assert_array_almost_equal(pca1.component(7), pca2.component(7))
        assert_array_almost_equal(pca1.components, pca2.components)
        assert_array_almost_equal(pca1.eigenvalues, pca2.eigenvalues)
        assert_array_almost_equal(
            pca1.eigenvalues_cumulative_ratio(), pca2.eigenvalues_cumulative_ratio()
        )
        assert_array_almost_equal(pca1.eigenvalues_ratio(), pca2.eigenvalues_ratio())
        weights = np.random.randn(pca1.n_components - 4)
        assert_array_almost_equal(pca1.instance(weights), pca2.instance(weights))
        assert pca1.n_components == pca2.n_components
        assert pca1.n_components == pca2.n_components
        assert pca1.n_features == pca2.n_features
        assert pca1.n_samples == pca2.n_samples
        assert pca1.noise_variance() == pca2.noise_variance()
        assert pca1.noise_variance_ratio() == pca2.noise_variance_ratio()
        assert_allclose(pca1.variance(), pca2.variance())
        assert pca1.variance_ratio() == pca2.variance_ratio()
        assert_array_almost_equal(
            pca1.whitened_components(), pca2.whitened_components()
        )


def test_pca_init_from_covariance():
    n_samples = 30
    n_features = 10
    n_dims = 2
    centre_values = [True, False]
    for centre in centre_values:
        # generate samples list and convert it to nd.array
        samples = [
            PointCloud(np.random.randn(n_features, n_dims)) for _ in range(n_samples)
        ]
        data, template = as_matrix(samples, return_template=True)
        # compute covariance matrix and mean
        if centre:
            mean_vector = np.mean(data, axis=0)
            mean = template.from_vector(mean_vector)
            X = data - mean_vector
            C = np.dot(X.T, X) / (n_samples - 1)
        else:
            mean = samples[0]
            C = np.dot(data.T, data) / (n_samples - 1)
        # create the 2 pca models
        pca1 = PCAModel.init_from_covariance_matrix(
            C, mean, centred=centre, n_samples=n_samples
        )
        pca2 = PCAModel(samples, centre=centre)
        # compare them
        assert_array_almost_equal(
            pca1.component_vector(0, with_mean=False),
            pca2.component_vector(0, with_mean=False),
        )
        assert_array_almost_equal(
            pca1.component(7).as_vector(), pca2.component(7).as_vector()
        )
        assert_array_almost_equal(pca1.components, pca2.components)
        assert_array_almost_equal(pca1.eigenvalues, pca2.eigenvalues)
        assert_array_almost_equal(
            pca1.eigenvalues_cumulative_ratio(), pca2.eigenvalues_cumulative_ratio()
        )
        assert_array_almost_equal(pca1.eigenvalues_ratio(), pca2.eigenvalues_ratio())
        weights = np.random.randn(pca1.n_components)
        assert_array_almost_equal(
            pca1.instance(weights).as_vector(), pca2.instance(weights).as_vector()
        )
        weights2 = np.random.randn(pca1.n_components - 4)
        assert_array_almost_equal(
            pca1.instance_vector(weights2), pca2.instance_vector(weights2)
        )
        assert_array_almost_equal(pca1.mean().as_vector(), pca2.mean().as_vector())
        assert_array_almost_equal(pca1.mean_vector, pca2.mean_vector)
        assert pca1.n_components == pca2.n_components
        assert pca1.n_components == pca2.n_components
        assert pca1.n_features == pca2.n_features
        assert pca1.n_samples == pca2.n_samples
        assert pca1.noise_variance() == pca2.noise_variance()
        assert pca1.noise_variance_ratio() == pca2.noise_variance_ratio()
        assert_almost_equal(pca1.variance(), pca2.variance())
        assert_almost_equal(pca1.variance_ratio(), pca2.variance_ratio())
        assert_array_almost_equal(
            pca1.whitened_components(), pca2.whitened_components()
        )


def test_pca_project():
    pca_samples = [PointCloud(np.random.randn(10, 2)) for _ in range(10)]
    pca_model = PCAModel(pca_samples)
    projected = pca_model.project(pca_samples[0])
    assert projected.shape[0] == 9


# ---- imputation: the missing features are taken to be the mean (zero without one)
def _pca_vector_model(n_samples=20, n_features=12, seed=0):
    rng = np.random.RandomState(seed)
    return PCAVectorModel(rng.rand(n_samples, n_features))


def _mask(n_features, missing):
    mask = np.ones(n_features, dtype=bool)
    mask[list(missing)] = False
    return mask


def test_linear_model_impute_fills_missing_with_zero_then_reconstructs():
    rng = np.random.RandomState(1)
    model = LinearVectorModel(np.linalg.qr(rng.rand(12, 4))[0].T)  # 4 orthonormal rows
    mask = _mask(12, [1, 5, 6])
    full = rng.rand(12)
    filled = np.where(mask, full, 0.0)
    assert_allclose(model.impute(full[mask], mask), model.reconstruct(filled))


def test_pca_vector_model_impute_fills_missing_with_the_mean_then_reconstructs():
    model = _pca_vector_model()
    mask = _mask(12, [0, 3, 4, 11])
    full = np.random.RandomState(2).rand(12)
    filled = np.where(mask, full, model.mean())
    assert_allclose(model.impute(full[mask], mask), model.reconstruct(filled))


def test_impute_with_nothing_missing_is_reconstruct():
    model = _pca_vector_model()
    full = np.random.RandomState(3).rand(12)
    mask = np.ones(12, dtype=bool)
    assert_allclose(model.impute(full, mask), model.reconstruct(full))


def test_impute_vectors_matches_impute_one_vector_at_a_time():
    model = _pca_vector_model()
    mask = _mask(12, [2, 7])
    vectors = np.random.RandomState(4).rand(5, mask.sum())
    imputed = model.impute_vectors(vectors, mask)
    assert imputed.shape == (5, 12)
    for v, row in zip(vectors, imputed):
        assert_allclose(model.impute(v, mask), row)


def test_impute_returns_a_vector_in_the_span_of_the_active_components():
    model = _pca_vector_model().trim_components(3)
    mask = _mask(12, [0, 1])
    imputed = model.impute(np.random.RandomState(5).rand(10), mask)
    weights = model.project(imputed)
    assert weights.shape == (3,)
    assert_allclose(model.instance(weights), imputed)


def test_impute_recovers_a_missing_feature_that_the_model_ties_to_the_others():
    # every sample has x[1] == 2 * x[0]: PCA finds the single direction (1, 2) / sqrt(5)
    t = np.linspace(-1, 1, 30)
    model = PCAVectorModel(np.c_[t, 2 * t])
    assert model.n_components == 1
    imputed = model.impute(np.array([0.5]), np.array([True, False]))
    # the missing feature is the mean (0) in the projection, so the weight is 0.5 / sqrt(5)
    # onto the direction (1, 2) / sqrt(5): the imputed vector is (0.1, 0.2) away from a
    # perfect (0.5, 1.0) but points along the correct direction
    assert_allclose(imputed[1] / imputed[0], 2.0)


def test_impute_checks_the_mask_and_the_vectors():
    model = _pca_vector_model()
    mask = _mask(12, [0])
    with raises(ValueError):
        model.impute(np.zeros(11), np.ones(12, dtype=int))  # not a boolean mask
    with raises(ValueError):
        model.impute(np.zeros(11), mask[:-1])  # wrong length
    with raises(ValueError):
        model.impute(np.zeros(10), mask)  # 10 features, mask has 11 True
    with raises(ValueError):
        model.impute_vectors(np.zeros((2, 3, 11)), mask)  # not 2D


def test_pca_model_impute_a_partial_instance():
    rng = np.random.RandomState(6)
    shapes = [PointCloud(rng.rand(4, 2)) for _ in range(15)]
    model = PCAModel(shapes)
    # the last point is missing: features are (x0, y0, x1, y1, ...)
    mask = np.array([True] * 6 + [False] * 2)
    partial = PointCloud(rng.rand(3, 2))
    imputed = model.impute(partial, mask)
    assert isinstance(imputed, PointCloud) and imputed.n_points == 4
    filled = np.concatenate([partial.as_vector(), model.mean().as_vector()[~mask]])
    assert_allclose(imputed.as_vector(), model.reconstruct_vector(filled))
