import numpy as np
from pybug.model.linear import LinearModel, MeanLinearModel, \
    MeanInstanceLinearModel, InstanceBackedModel

def test_linear_model_creation():
    data = np.zeros((120, 3))
    instnc