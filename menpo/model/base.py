import numpy as np
from menpo.model.instancebacked import InstanceBackedModel
from menpo.model.linear import LinearModel, MeanLinearModel


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

    def component(self, index, with_mean=True, scale=1.0):
        r"""
        Return a particular component of the linear model.

        Parameters
        ----------
        index : int
            The component that is to be returned

        with_mean: boolean (optional)
            If True, the component will be blended with the mean vector
            before being returned. If not, the component is returned on it's
            own.

            Default: True
        scale : float
            A scale factor that should be applied to the component. Only
            valid in the case where with_mean is True. See
            :meth:`component_vector` for how this scale factor is interpreted.

        :type: `type(self.template_instance)`
        """
        return self.template_instance.from_vector(self.component_vector(
            index, with_mean=with_mean, scale=scale))


#noinspection PyPep8Naming
def Similarity2dInstanceModel(shape):
    r"""
    A MeanInstanceLinearModel that encodes all possible 2D similarity
    transforms of a 2D shape (of n_points).

        Parameters
        ----------
        shape : 2D :class:`menpo.shape.Shape`

        Returns
        -------
        model : `menpo.model.linear.MeanInstanceLinearModel`
            Model with four components, linear combinations of which
            represent the original shape under a similarity transform. The
            model is exhaustive (that is, all possible similarity transforms
            can be expressed in the model).

    """
    shape_vector = shape.as_vector()
    components = np.zeros((4, shape_vector.shape[0]))
    components[0, :] = shape_vector  # Comp. 1 - just the shape
    rotated_ccw = shape.points[:, ::-1].copy()  # flip x,y -> y,x
    rotated_ccw[:, 0] = -rotated_ccw[:, 0]  # negate (old) y
    components[1, :] = rotated_ccw.flatten()  # C2 - the shape rotated 90 degs
    components[2, ::2] = 1  # Tx
    components[3, 1::2] = 1  # Ty
    return MeanInstanceLinearModel(components, shape_vector, shape)
