import numpy as np
from nose.tools import raises
from numpy.testing import (assert_allclose, assert_equal, assert_almost_equal,
                           assert_array_almost_equal)
from menpo.shape import PointCloud
from menpo.model import LinearVectorModel, PCAModel, PCAVectorModel
from menpo.math import as_matrix


def test_linear_model_creation():
    data = np.zeros((3, 120))
    LinearVectorModel(data)


def test_linear_model_basics():
    data = np.random.random((3, 120))
    linear_model = LinearVectorModel(data)
    assert(linear_model.n_components == 3)
    assert(linear_model.n_components == 3)
    assert(linear_model.n_features == 120)


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


def test_pca_n_active_components():
    samples = [np.random.randn(10) for _ in range(10)]
    model = PCAVectorModel(samples)
    # integer
    model.n_active_components = 5
    assert_equal(model.n_active_components, 5)


def test_pca_n_active_components_too_many():
    samples = [np.random.randn(10) for _ in range(10)]
    model = PCAVectorModel(samples)
    # too many components
    model.n_active_components = 100
    assert_equal(model.n_active_components, 9)
    # reset too smaller number of components
    model.n_active_components = 5
    assert_equal(model.n_active_components, 5)
    # reset to too many components
    model.n_active_components = 100
    assert_equal(model.n_active_components, 9)


@raises(ValueError)
def test_pca_n_active_components_negative():
    samples = [np.random.randn(10) for _ in range(10)]
    model = PCAVectorModel(samples)
    # not sufficient components
    model.n_active_components = -5


def test_pca_trim():
    samples = [PointCloud(np.random.randn(10)) for _ in range(10)]
    model = PCAModel(samples)
    # trim components
    model.trim_components(5)
    # number of active components should be the same as number of components
    assert_equal(model.n_active_components, model.n_components)


@raises(ValueError)
def test_pca_trim_variance_limit():
    samples = [np.random.randn(10) for _ in range(10)]
    model = PCAVectorModel(samples)
    # impossible to keep more than 1.0 ratio variance
    model.trim_components(2.5)

@raises(ValueError)
def test_pca_trim_variance_limit():
    samples = [np.random.randn(10) for _ in range(10)]
    model = PCAVectorModel(samples)
    # impossible to keep more than 1.0 ratio variance
    model.trim_components(2.5)


@raises(ValueError)
def test_pca_trim_negative_integers():
    samples = [PointCloud(np.random.randn(10)) for _ in range(10)]
    model = PCAModel(samples)
    # no negative number of components
    model.trim_components(-2)


@raises(ValueError)
def test_pca_trim_negative_float():
    samples = [PointCloud(np.random.randn(10)) for _ in range(10)]
    model = PCAModel(samples)
    # no negative number of components
    model.trim_components(-2)


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


@raises(ValueError)
def test_pca_inverse_noise_variance():
    samples = [PointCloud(np.random.randn(10)) for _ in range(10)]
    model = PCAModel(samples)
    # inverse noise_variance it's not computable
    model.inverse_noise_variance()


def test_pca_variance_after_change_n_active_components():
    samples = [np.random.randn(10) for _ in range(10)]
    model = PCAVectorModel(samples)
    # set number of active components
    model.n_active_components = 5
    # kept variance must be smaller than total variance
    assert(model.variance() < model.original_variance())
    # kept variance ratio must be smaller than 1.0
    assert(model.variance_ratio() < 1.0)
    # noise variance must be bigger than 0.0
    assert(model.noise_variance() > 0.0)
    # noise variance ratio must also be bigger than 0.0
    assert(model.noise_variance_ratio() > 0.0)
    # inverse noise variance is computable
    assert(model.inverse_noise_variance() == 1/model.noise_variance())


def test_pca_variance_after_trim():
    samples = [np.random.randn(10) for _ in range(10)]
    model = PCAVectorModel(samples)
    # set number of active components
    model.trim_components(5)
    # kept variance must be smaller than total variance
    assert(model.variance() < model.original_variance())
    # kept variance ratio must be smaller than 1.0
    assert(model.variance_ratio() < 1.0)
    # noise variance must be bigger than 0.0
    assert(model.noise_variance() > 0.0)
    # noise variance ratio must also be bigger than 0.0
    assert(model.noise_variance_ratio() > 0.0)
    # inverse noise variance is computable
    assert(model.inverse_noise_variance() == 1 / model.noise_variance())


def test_pca_orthogonalize_against():
    pca_samples = np.random.randn(10, 10)
    pca_model = PCAVectorModel(pca_samples)
    lm_samples = np.asarray([np.random.randn(10) for _ in range(4)])
    lm_model = LinearVectorModel(np.asarray(lm_samples))
    # orthogonalize
    pca_model.orthonormalize_against_inplace(lm_model)
    # number of active components must remain the same
    assert_equal(pca_model.n_active_components, 6)


def test_pca_orthogonalize_against_with_less_active_components():
    pca_samples = np.random.randn(10, 10)
    pca_model = PCAVectorModel(pca_samples)
    lm_samples = np.asarray([np.random.randn(10) for _ in range(4)])
    lm_model = LinearVectorModel(np.asarray(lm_samples))
    # set number of active components
    pca_model.n_active_components = 5
    # orthogonalize
    pca_model.orthonormalize_against_inplace(lm_model)
    # number of active components must remain the same
    assert_equal(pca_model.n_active_components, 5)


def test_pca_increment_centred():
    pca_samples = [PointCloud(np.random.randn(10, 2)) for _ in range(10)]
    ipca_model = PCAModel(pca_samples[:3])
    ipca_model.increment(pca_samples[3:6])
    ipca_model.increment(pca_samples[6:])

    bpca_model = PCAModel(pca_samples)

    assert_almost_equal(np.abs(ipca_model.components),
                        np.abs(bpca_model.components))
    assert_almost_equal(ipca_model.eigenvalues, bpca_model.eigenvalues)
    assert_almost_equal(ipca_model.mean().as_vector(),
                        bpca_model.mean().as_vector())


def test_pca_increment_noncentred():
    pca_samples = [np.random.randn(10) for _ in range(10)]
    ipca_model = PCAVectorModel(pca_samples[:3], centre=False)
    ipca_model.increment(pca_samples[3:6])
    ipca_model.increment(pca_samples[6:])

    bpca_model = PCAVectorModel(pca_samples, centre=False)

    assert_almost_equal(np.abs(ipca_model.components),
                        np.abs(bpca_model.components))
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
        pca1 = PCAVectorModel.init_from_covariance_matrix(C, mean, centred=centre,
                                                          n_samples=n_samples)
        pca2 = PCAVectorModel(samples, centre=centre, inplace=False)
        # compare them
        assert_array_almost_equal(pca1.mean(), pca2.mean())
        assert_array_almost_equal(pca1.component(0, with_mean=False),
                                  pca2.component(0, with_mean=False))
        assert_array_almost_equal(pca1.component(7), pca2.component(7))
        assert_array_almost_equal(pca1.components, pca2.components)
        assert_array_almost_equal(pca1.eigenvalues, pca2.eigenvalues)
        assert_array_almost_equal(pca1.eigenvalues_cumulative_ratio(),
                                  pca2.eigenvalues_cumulative_ratio())
        assert_array_almost_equal(pca1.eigenvalues_ratio(),
                                  pca2.eigenvalues_ratio())
        weights = np.random.randn(pca1.n_active_components - 4)
        assert_array_almost_equal(pca1.instance(weights),
                                  pca2.instance(weights))
        assert(pca1.n_active_components == pca2.n_active_components)
        assert(pca1.n_components == pca2.n_components)
        assert(pca1.n_features == pca2.n_features)
        assert(pca1.n_samples == pca2.n_samples)
        assert(pca1.noise_variance() == pca2.noise_variance())
        assert(pca1.noise_variance_ratio() == pca2.noise_variance_ratio())
        assert_allclose(pca1.variance(), pca2.variance())
        assert(pca1.variance_ratio() == pca2.variance_ratio())
        assert_array_almost_equal(pca1.whitened_components(),
                                  pca2.whitened_components())


def test_pca_init_from_covariance():
    n_samples = 30
    n_features = 10
    n_dims = 2
    centre_values = [True, False]
    for centre in centre_values:
        # generate samples list and convert it to nd.array
        samples = [PointCloud(np.random.randn(n_features, n_dims))
                   for _ in range(n_samples)]
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
        pca1 = PCAModel.init_from_covariance_matrix(C, mean,
                                                    centred=centre,
                                                    n_samples=n_samples)
        pca2 = PCAModel(samples, centre=centre)
        # compare them
        assert_array_almost_equal(pca1.component_vector(0, with_mean=False),
                                  pca2.component_vector(0, with_mean=False))
        assert_array_almost_equal(pca1.component(7).as_vector(),
                                  pca2.component(7).as_vector())
        assert_array_almost_equal(pca1.components, pca2.components)
        assert_array_almost_equal(pca1.eigenvalues, pca2.eigenvalues)
        assert_array_almost_equal(pca1.eigenvalues_cumulative_ratio(),
                                  pca2.eigenvalues_cumulative_ratio())
        assert_array_almost_equal(pca1.eigenvalues_ratio(),
                                  pca2.eigenvalues_ratio())
        weights = np.random.randn(pca1.n_active_components)
        assert_array_almost_equal(pca1.instance(weights).as_vector(),
                                  pca2.instance(weights).as_vector())
        weights2 = np.random.randn(pca1.n_active_components - 4)
        assert_array_almost_equal(pca1.instance_vector(weights2),
                                  pca2.instance_vector(weights2))
        assert_array_almost_equal(pca1.mean().as_vector(),
                                  pca2.mean().as_vector())
        assert_array_almost_equal(pca1.mean_vector,
                                  pca2.mean_vector)
        assert(pca1.n_active_components == pca2.n_active_components)
        assert(pca1.n_components == pca2.n_components)
        assert(pca1.n_features == pca2.n_features)
        assert(pca1.n_samples == pca2.n_samples)
        assert(pca1.noise_variance() == pca2.noise_variance())
        assert(pca1.noise_variance_ratio() == pca2.noise_variance_ratio())
        assert_almost_equal(pca1.variance(), pca2.variance())
        assert_almost_equal(pca1.variance_ratio(), pca2.variance_ratio())
        assert_array_almost_equal(pca1.whitened_components(),
                                  pca2.whitened_components())


def test_pca_project():
    pca_samples = [PointCloud(np.random.randn(10, 2)) for _ in range(10)]
    pca_model = PCAModel(pca_samples)
    projected = pca_model.project(pca_samples[0])
    assert projected.shape[0] == 9
