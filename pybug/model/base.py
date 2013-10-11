import numpy as np
from pybug.model.instancebacked import InstanceBackedModel
from pybug.model.linear import LinearModel, MeanLinearModel


class InstanceLinearModel(LinearModel, InstanceBackedModel):

    def __int__(self, components, template_instance):
        LinearModel.__init__(self, components)
        InstanceBackedModel.__init__(self, template_instance)


class MeanInstanceLinearModel(MeanLinearModel, InstanceBackedModel):

    def __init__(self, components, mean_vector, template_instance):
        MeanLinearModel.__init__(self, components, mean_vector)
        InstanceBackedModel.__init__(self, template_instance)

    @property
    def mean(self):
        return self.template_instance.from_vector(self.mean_vector)


#noinspection PyPep8Naming
def Similarity2dInstanceModel(shape):
    shape_vector = shape.as_vector()
    components = np.zeros(4, shape_vector.shape[0])
    components[0, :] = shape_vector
    aux = shape.points[:, [1, 0]]
    aux[:, 0] = -aux[:, 0]
    components[1, :] = aux.flatten()
    components[2, ::2] = 1
    components[3, 1::2] = 1
    return MeanInstanceLinearModel(components, shape.as_vector(), shape)
