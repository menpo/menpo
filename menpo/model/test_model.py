import numpy as np
from numpy.testing import assert_allclose, assert_equal
from menpo.model import LinearModel, MeanLinearModel
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
