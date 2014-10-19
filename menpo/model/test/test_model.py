import numpy as np
from nose.tools import raises
from numpy.testing import assert_allclose, assert_equal
from menpo.shape import PointCloud
from menpo.model import LinearModel, MeanLinearModel, PCAModel
from menpo.model import InstanceLinearModel, MeanInstanceLinearModel


def test_linear_model_creation():
    data = np.zeros((3, 120))
    LinearModel(data)


def test_linear_model_basics():
    data = np.random.random((3, 120))
    linear_model = LinearModel(data)
    assert(linear_model.n_components == 3)
    assert(linear_model.n_components == 3)
    assert(linear_model.n_features == 120)


def test_linear_model_project_vector():
    data = np.zeros((3, 120))
    data[0, 0] = 1
    data[1, 1] = 1
    data[2, 2] = 1
    linear_model = LinearModel(data)
    sample = np.random.random(120)
    weights = linear_model.project_vector(sample)
    assert_allclose(weights, sample[:3])


def test_linear_model_component():
    data = np.random.random((3, 120))
    linear_model = LinearModel(data)
    assert_equal(linear_model.component_vector(2), data[2])


def test_linear_model_instance_vector():
    data = np.zeros((3, 120))
    data[0, 0] = 1
    data[1, 1] = 1
    data[2, 2] = 1
    linear_model = LinearModel(data)
    weights = np.array([0.263, 7, 41.2])
    projected = linear_model.instance_vector(weights)
    # only the first 3 features are non zero...
    assert_allclose(projected[:3], weights)
    # rest should be nil
    assert_allclose(projected[3:], 0)


def test_pca_n_active_components():
    samples = [PointCloud(np.random.randn(10)) for _ in range(10)]
    model = PCAModel(samples)
    # integer
    model.n_active_components = 5
    assert_equal(model.n_active_components, 5)


def test_pca_n_active_components_too_many():
    samples = [PointCloud(np.random.randn(10)) for _ in range(10)]
    model = PCAModel(samples)
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
    samples = [PointCloud(np.random.randn(10)) for _ in range(10)]
    model = PCAModel(samples)
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
    samples = [PointCloud(np.random.randn(10)) for _ in range(10)]
    model = PCAModel(samples)
    # impossible to keep more than 1.0 ratio variance
    model.trim_components(2.5)

@raises(ValueError)
def test_pca_trim_variance_limit():
    samples = [PointCloud(np.random.randn(10)) for _ in range(10)]
    model = PCAModel(samples)
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
    samples = [PointCloud(np.random.randn(10)) for _ in range(10)]
    model = PCAModel(samples)
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
    samples = [PointCloud(np.random.randn(10)) for _ in range(10)]
    model = PCAModel(samples)
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
    pca_samples = [PointCloud(np.random.randn(10)) for _ in range(10)]
    pca_model = PCAModel(pca_samples)
    lm_samples = np.asarray([np.random.randn(10) for _ in range(4)])
    lm_model = LinearModel(np.asarray(lm_samples))
    # orthogonalize
    pca_model.orthonormalize_against_inplace(lm_model)
    # number of active components must remain the same
    assert_equal(pca_model.n_active_components, 6)


def test_pca_orthogonalize_against_with_less_active_components():
    pca_samples = [PointCloud(np.random.randn(10)) for _ in range(10)]
    pca_model = PCAModel(pca_samples)
    lm_samples = np.asarray([np.random.randn(10) for _ in range(4)])
    lm_model = LinearModel(np.asarray(lm_samples))
    # set number of active components
    pca_model.n_active_components = 5
    # orthogonalize
    pca_model.orthonormalize_against_inplace(lm_model)
    # number of active components must remain the same
    assert_equal(pca_model.n_active_components, 5)

