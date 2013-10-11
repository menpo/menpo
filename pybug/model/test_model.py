import numpy as np
from numpy.testing import assert_allclose
from pybug.model import LinearModel, MeanLinearModel
from pybug.model import InstanceLinearModel, MeanInstanceLinearModel


def test_linear_model_creation():
    data = np.zeros((120, 3))
    LinearModel(data)


def test_linear_model_project_vector():
    data = np.zeros((120, 3))
    data[0, 0] = 1
    data[1, 1] = 1
    data[2, 2] = 1
    linear_model = LinearModel(data)
    sample = np.random.random(120)
    weights = linear_model.project_vector(sample)
    assert_allclose(weights, sample[:3])


def test_linear_model_instance_vector():
    data = np.zeros((120, 3))
    data[0, 0] = 1
    data[1, 1] = 1
    data[2, 2] = 1
    linear_model = LinearModel(data)
    weights = np.array([0.263, 7, 41.2])
    projected = linear_model.instance_vector(weights)
    assert_allclose(projected, weights)
