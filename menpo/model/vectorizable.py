
class VectorizableBackedModel(object):
    r"""
    Mixin for models constructed from a set of :map:`Vectorizable` objects.
    Supports models for which visualizing the meaning of a set of components
    is trivial.

    Requires that the following methods are implemented:

    1. `component_vector(index)`
    2. `instance_vector(weights)`
    3. `project_vector(vector)`
    4. `reconstruct_vector(vectors)`
    5. `project_out_vector(vector)`

    The constructor takes an instance of :map:`Vectorizable`. This is used for
    all conversions to and from numpy vectors and instances.

    Parameters
    ----------
    template_instance : :map:`Vectorizable`
        The template instance.
    """

    def __init__(self, template_instance):
        self.template_instance = template_instance

    def component_vector(self, index):
        r"""
        A particular component of the model, in vectorized form.

        Parameters
        ----------
        index : `int`
            The component that is to be returned.

        Returns
        -------
        component_vector : `ndarray`
            The component vector.
        """
        raise NotImplementedError()

    def component(self, index):
        r"""
        A particular component of the model.

        Parameters
        ----------
        index : `int`
            The component that is to be returned.

        Returns
        -------
        component : `type(self.template_instance)`
            The component instance.
        """
        return self.template_instance.from_vector(self.component_vector(index))

    def instance_vector(self, weights):
        """
        Creates a new vector instance of the model using the first ``len(weights)``
        components.

        Parameters
        ----------
        weights : ``(n_weights,)`` `ndarray` or `list`
            ``weights[i]`` is the linear contribution of the i'th component
            to the instance vector.

        Raises
        ------
        ValueError
            If n_weights > n_components

        Returns
        -------
        instance_vector : `ndarray`
            An instance of the model, in vectorized form.
        """
        raise NotImplementedError()

    def instance(self, weights):
        """
        Creates a new instance of the model using the first ``len(weights)``
        components.

        Parameters
        ----------
        weights : ``(n_weights,)`` `ndarray` or `list`
            ``weights[i]`` is the linear contribution of the i'th component
            to the instance vector.

        Raises
        ------
        ValueError
            If n_weights > n_components

        Returns
        -------
        instance : `type(self.template_instance)`
            An instance of the model.
        """
        return self.template_instance.from_vector(
            self.instance_vector(weights))

    def project_vector(self, instance_vector):
        """
        Projects the `instance_vector` onto the model, retrieving the optimal
        linear weightings.

        Parameters
        ----------
        instance_vector : `ndarray`
            A novel instance vector.

        Returns
        -------
        projected_vector : ``(n_components,)`` `ndarray`
            A vector of optimal linear weightings.
        """
        raise NotImplementedError()

    def project(self, instance):
        """
        Projects the `instance` onto the model, retrieving the optimal
        linear weightings.

        Parameters
        ----------
        instance : :map:`Vectorizable`
            A novel instance.

        Returns
        -------
        projected : ``(n_components,)`` `ndarray`
            A vector of optimal linear weightings.
        """
        return self.project_vector(instance.as_vector())

    def reconstruct_vector(self, instance_vector):
        """
        Projects an `instance_vector` onto the linear space and rebuilds from the
        weights found.

        Syntactic sugar for: ::

            instance_vector(project_vector(instance_vector))

        but faster, as it avoids the conversion that takes place each time.

        Parameters
        ----------
        instance_vector : `ndarray`
            A novel instance vector.

        Returns
        -------
        reconstructed_vector : `ndarray`
            The reconstructed vector.
        """
        raise NotImplementedError()

    def reconstruct(self, instance):
        """
        Projects a `instance` onto the linear space and rebuilds from the
        weights found.

        Syntactic sugar for: ::

            instance(project(instance))

        but faster, as it avoids the conversion that takes place each time.

        Parameters
        ----------
        instance : :class:`Vectorizable`
            A novel instance of :class:`Vectorizable`.

        Returns
        -------
        reconstructed : `self.instance_class`
            The reconstructed object.
        """
        reconstruction_vector = self.reconstruct_vector(instance.as_vector())
        return instance.from_vector(reconstruction_vector)

    def project_out_vector(self, instance_vector):
        """
        Returns a version of `instance_vector` where all the basis of the model
        have been projected out.

        Parameters
        ----------
        instance_vector : `ndarray`
            A novel instance vector.

        Returns
        -------
        projected_out_vector : `ndarray`
            A copy of `instance_vector`, with all bases of the model projected out.
        """
        raise NotImplementedError()

    def project_out(self, instance):
        """
        Returns a version of `instance` where all the basis of the model
        have been projected out.

        Parameters
        ----------
        instance : :class:`Vectorizable`
            A novel instance of :class:`Vectorizable`.

        Returns
        -------
        projected_out : `self.instance_class`
            A copy of `instance`, with all basis of the model projected out.
        """
        vector_instance = self.project_out_vector(instance.as_vector())
        return instance.from_vector(vector_instance)
