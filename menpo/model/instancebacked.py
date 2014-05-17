
class InstanceBackedModel(object):
    r"""
    Mixin for models constructed from a set of Vectorizable objects.
    Allows for models where visualizing the meaning of a set of components
    is trivial. Requires that the following attributes to be present:

    n_components
    components

    and the following methods implemented:

    component_vector(index)
    instance_vectors(weightings)
    project_vector(vector)
    reconstruct_vectors(vectors, n_components)
    project_out_vector(vector)

    The constructor takes an instance of :map:`Vectorizable`. This is used for
    all conversions to and from numpy vectors and instances.
    """

    def __init__(self, template_instance):
        self.template_instance = template_instance

    def component(self, index):
        r"""
        Return a particular component of the linear model.

        Parameters
        ----------
        index : int
            The component that is to be returned

        :type: `type(self.template_instance)`
        """
        return self.template_instance.from_vector(self.component_vector(index))

    def instance(self, weights):
        """
        Creates a new instance of the model using the first `len(weights)`
        components.

        Parameters
        ----------
        weights : (n_weights,) ndarray or list
            `weights[i]` is the linear contribution of the i'th component
            to the instance vector.

        Raises
        ------
        ValueError: If n_weights > n_components

        Returns
        -------
        instance : `type(self.template_instance)`
            An instance of the model.
        """
        return self.template_instance.from_vector(
            self.instance_vector(weights))

    def project(self, instance):
        """
        Projects the `instance` onto the model, retrieving the optimal
        linear weightings.

        Parameters
        -----------
        novel_instance : :class:`menpo.base.Vectorizable`
            A novel instance.

        Returns
        -------
        projected : (n_components,)
            A vector of optimal linear weightings
        """
        return self.project_vector(instance.as_vector())

    def reconstruct(self, instance):
        """
        Projects a `instance` onto the linear space and rebuilds from the
        weights found.

        Syntactic sugar for:

            >>> instance(project(instance))

        but faster, as it avoids the conversion that takes place each time.

        Parameters
        ----------
        instance : :class:`menpo.base.Vectorizable`
            A novel instance of Vectorizable

        Returns
        -------
        reconstructed : `self.instance_class`
            The reconstructed object.
        """
        reconstruction_vector = self.reconstruct_vector(instance.as_vector())
        return instance.from_vector(reconstruction_vector)

    def project_out(self, instance):
        """
        Returns a version of `instance` where all the basis of the model
        have been projected out.

        Parameters
        ----------
        instance : :class:`menpo.base.Vectorizable`
            A novel instance.

        Returns
        -------
        projected_out : `self.instance_class`
            A copy of `instance`, with all basis of the model projected out.
        """
        vector_instance = self.project_out_vector(instance.as_vector())
        return instance.from_vector(vector_instance)
