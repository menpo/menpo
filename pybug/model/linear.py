import abc
import numpy as np
from scipy.linalg.blas import dgemm
from pybug.decomposition import PCA as PybugPCA
from pybug.model.base import StatisticalModel


#noinspection PyNoneFunctionAssignment
class LinearModel(object):
    r"""
    Abstract base class representing a linear model.
    """

    @abc.abstractproperty
    def n_components(self):
        r"""
        The number of components on the linear model

        type: int
        """
        pass

    @property
    def n_features(self):
        r"""
        The number of components on the linear model

        type: int
        """
        return self._component(0).size


    @abc.abstractproperty
    def components(self):
        r"""
        The components of the linear model.

        type: (n_features, n_components) ndarray
        """
        pass

    @components.setter
    @abc.abstractmethod
    def components(self, components):
        r"""
        Update the components of this linear model
        """
        pass

    @abc.abstractproperty
    def template_instance(self):
        r"""
        An instantiated vectorizable class. This is used to rebuild objects
        from the statistical model.

        type: :class:`pybug.base.Vectorizable`
        """
        pass

    def component(self, index):
        r"""
        Return a particular component of the linear model.

        :type: ``type(self.template_instance)``
        """
        return self.template_instance.from_vector(self._component(index))

    @abc.abstractmethod
    def _component(self, index):
        """
        A particular component of the model, in vectorized form.

        :type: (n_features,) ndarray
        """
        pass

    def instance(self, weights):
        """
        Creates a new instance of the model using the first ``len(weights)``
        components.

        Parameters
        ----------
        weights : (<=n_components,) ndarray
            The weights that should be used by the model to create an
            instance of itself.

        Returns
        -------
        instance : ``type(self.template_instance)``
            An instance of the model. Created via a linear combination of the
            model vectors and the ``weights``.
        """
        return self.template_instance.from_vector(self._instance(weights))

    @abc.abstractmethod
    def _instance(self, weights):
        """
        Creates a new instance of the model using the first len(weights)
        components.

        Parameters
        ----------
        weights : (<=n_components,) ndarray
            The weights that should be used by the model to create an
            instance of itself.

        Returns
        -------
        instance : (n_features,) ndarray
            The instance vector.
        """
        pass

    def project(self, instance):
        """
        Projects the ``instance`` onto the model, retrieving the optimal
        linear weightings.

        Parameters
        -----------
        novel_instance : :class:`pybug.base.Vectorizable`
            A novel instance.

        Returns
        -------
        projected : (n_components,)
            A vector of optimal linear weightings
        """
        return self._project(instance.as_vector()).flatten()

    @abc.abstractmethod
    def _project(self, vector_instance):
        """
        Projects the ``vector_instance`` onto the model, retrieving the optimal
        linear reconstruction weights

        Parameters
        -----------
        vector_instance : (n_features,) ndarray
            A vectorized novel instance.

        Returns
        -------
        projected : (n_components,)
            A vector of optimal linear weights
        """
        pass

    def reconstruct(self, instance, n_components=None):
        """
        Projects a ``instance`` onto the linear space and rebuilds from the
        weights found.

        Syntactic sugar for:

            >>> pca.instance(pca.project(novel_instance)[:n_components])

        but faster, as it avoids the conversion that takes place each time.

        Parameters
        ----------
        instance : :class:`pybug.base.Vectorizable`
            A novel instance of Vectorizable
        n_components : int, optional
            The number of components to use in the reconstruction.

            Default: ``weights.shape[0]``

        Returns
        -------
        reconstructed : ``self.instance_class``
            The reconstructed object.
        """
        vec_reconstruction = self._reconstruct(instance.as_vector(),
                                               n_components)
        return instance.from_vector(vec_reconstruction)

    def _reconstruct(self, vector_instance, n_components=None):
        """
        Project a flattened ``novel_instance`` onto the linear space and
        rebuild from the weights found.

        Syntactic sugar for:

            >>> pca._instance(pca._project(vector_instance)[:n_components])

        Parameters
        ----------
        vector_instance : (n_features, ) ndarray
            A vectorized novel instance to project
        n_components : int, optional
            The number of components to use in the reconstruction

            Default: ``weights.shape[0]``

        Returns
        -------
        reconstructed : (n_features,) ndarray
            The reconstructed vector.
        """
        weights = self._project(vector_instance)
        if n_components is not None:
            weights = weights[..., :n_components]
        return self._instance(weights)

    def project_out(self, instance):
        """
        Returns a version of ``instance`` where all the basis of the model
        have been projected out.

        Parameters
        ----------
        instance : :class:`pybug.base.Vectorizable`
            A novel instance.

        Returns
        -------
        projected_out : ``self.instance_class``
            A copy of ``instance``, with all basis of the model projected out.
        """
        vec_instance = self._project_out(instance.as_vector())
        return instance.from_vector(vec_instance)

    def _project_out(self, vector_instance):
        """
        Returns a version of ``vector_instance`` where all the basis of the
        model have been projected out.

        Parameters
        ----------
        vector_instance : (n_features,) ndarray
            A novel vector.

        Returns
        -------
        projected_out : (n_features,) ndarray
            A copy of ``vector_instance`` with all basis of the model
            projected out.
        """
        weights = dgemm(alpha=1.0, a=vector_instance.T, b=self.components.T,
                        trans_a=True)
        return (vector_instance -
                dgemm(alpha=1.0, a=weights.T, b=self.components.T,
                      trans_a=True, trans_b=True))

    @property
    def jacobian(self):
        """
        Returns the Jacobian of the PCA model. In this case, simply the
        components of the model reshaped to have the standard Jacobian shape:

            n_points    x  n_params      x  n_dims
            n_features  x  n_components  x  n_dims

        Returns
        -------
        jacobian : (n_features, n_components, n_dims) ndarray
            The Jacobian of the model in the standard Jacobian shape.
        """
        jacobian = self._jacobian.reshape(self.n_components, -1,
                                          self.template_instance.n_dims)
        return jacobian.swapaxes(0, 1)

    @abc.abstractproperty
    def _jacobian(self):
        """
        Returns the Jacobian of the PCA model, i.e. the components of the
        model.

        Returns
        -------
        jacobian : (n_features x n_dims, n_components) ndarray
            The Jacobian of the model in matrix form.
        """
        pass

    def orthonormalize_against_inplace(self, linear_model):
        """
        """
        Q = (np.linalg.qr(np.hstack((linear_model.components.T,
                                     self.components.T)))[0]).T

        linear_model.components = Q[:linear_model.n_components, :]
        self.components = Q[linear_model.n_components:, :]


#TODO: give a description of what it means to be a PCA model
class PCAModel(LinearModel, StatisticalModel):
    """
    A Linear model based around PCA. Automatically mean centres the input
    data.

    Parameters
    ----------
    samples: list of :class:`pybug.base.Vectorizable`
        List of samples to build the model from.

    PCA: specific class implementing PCA, optional

        Default: `PybugPCA`

        .. note::

            This will currently break if `sklearn.decomposition.pca.PCA` is
            set to be the specific implementation of PCA. This is because
            this implementation does not support the concept of
            `noise_variance`. Support for this concept is expected on their
            next upcoming release.
    """

    def __init__(self, samples, PCA=PybugPCA, **kwargs):
        StatisticalModel.__init__(self, samples)
        data = np.zeros((self.n_samples, self.n_features))
        for i, sample in enumerate(self.samples):
            data[i] = sample.as_vector()

        self._pca = PCA(**kwargs)
        self._pca.fit(data)
        self.inv_noise_variance = 1 / self.noise_variance
        self.whitened_components = \
            (self.explained_variance ** (-1 / 2))[..., None] * self.components

    @property
    def explained_variance(self):
        """
        Total variance explained by each of the components.

        :type: (``n_components``,) ndarray
        """
        return self._pca.explained_variance_

    @property
    def explained_variance_ratio(self):
        """
        Percentage of variance explained by each of the components.

        :type: (``n_components``,) ndarray
        """
        return self._pca.explained_variance_ratio_

    @property
    def noise_variance(self):
        return self._pca.noise_variance_

    @property
    def mean(self):
        """
        The mean of the sample vectors.

        :type: ``self.instance_class``
        """
        return self.template_instance.from_vector(self._mean)

    @property
    def _mean(self):
        """
        The mean vector of the samples.

        :type: (n_features,) ndarray
        """
        return self._pca.mean_

    @property
    def n_components(self):
        """
        The number of kept principal components.

        :type: int
        """
        return self._pca.n_components_

    @property
    def components(self):
        """
        The principal components.

        :type: (``n_components``, ``n_features``) ndarray
        """
        return self._pca.components_

    @components.setter
    def components(self, components):
        r"""
        Setting the weights value automatically triggers a recalculation of
        the target, and an update of the transform
        """
        if self.components.shape == components.shape:
            self._pca.components_ = components
        else:
            raise ValueError('the new components must be of the same shape '
                             'as the original components')

    @property
    def _jacobian(self):
        return self.components

    def _component(self, index):
        """
        A particular principal component.

        :type: (n_features,) ndarray
        """
        return self._pca.components_[index, :]

    def _instance(self, weights):
        if weights.shape[-1] > self.n_components:
            raise Exception(
                "Number of weightings cannot be greater than {}".format(
                    self.n_components))
        elif weights.shape[-1] < self.n_components:
            if len(weights.shape) == 1:
                full_weights = np.zeros(self.n_components)
            else:
                full_weights = np.zeros((weights.shape[0], self.n_components))
            full_weights[..., :weights.shape[-1]] = weights
            weights = full_weights
        return self._pca.inverse_transform(weights)

    def _project(self, vector_instance):
        return self._pca.transform(vector_instance)

    def to_subspace(self, instance):
        """
        Returns a version of ``instance`` where all the basis of the model
        have been projected out and which has been scaled by the inverse of
        the ``noise_variance``

        Parameters
        ----------
        instance : :class:`pybug.base.Vectorizable`
            A novel instance.

        Returns
        -------
        scaled_projected_out : ``self.instance_class``
            A copy of ``instance``, with all basis of the model projected out
            and scaled by the inverse of the ``noise_variance``.
        """
        vec_instance = self._to_subspace(instance.as_vector())
        return instance.from_vector(vec_instance)

    def _to_subspace(self, vector_instance):
        """
        Returns a version of ``instance`` where all the basis of the model
        have been projected out and which has been scaled by the inverse of
        the ``noise_variance``.

        Parameters
        ----------
        vector_instance : (n_features,) ndarray
            A novel vector.

        Returns
        -------
        scaled_projected_out: (n_features,) ndarray
            A copy of ``vector_instance`` with all basis of the model projected
            out and scaled by the inverse of the ``noise_variance``.
        """
        return self.inv_noise_variance * self._project_out(vector_instance)

    def within_subspace(self, instance):
        """
        Returns a sheared (non-orthogonal) reconstruction of ``instance``.

        Parameters
        ----------
        instance : :class:`pybug.base.Vectorizable`
            A novel instance.

        Returns
        -------
        sheared_reconstruction : ``self.instance_class``
            A sheared (non-orthogonal) reconstruction of ``instance``.
        """
        vector_instance = self._within_subspace(instance.as_vector())
        return instance.from_vector(vector_instance)

    def _within_subspace(self, vector_instance):
        """
        Returns a sheared (non-orthogonal) reconstruction of
        ``vector_instance``.

        Parameters
        ----------
        vector_instance : (n_features,) ndarray
            A novel vector.

        Returns
        -------
        sheared_reconstruction : (n_features,) ndarray
            A sheared (non-orthogonal) reconstruction of ``vector_instance``
        """
        weights = dgemm(alpha=1.0, a=vector_instance.T,
                        b=self.whitened_components.T, trans_a=True)
        return dgemm(alpha=1.0, a=weights.T, b=self.whitened_components.T,
                     trans_a=True, trans_b=True)


#TODO: give a description of what it means to be a Similarity Model
#TODO: note this is 2D only
class SimilarityModel(LinearModel):

    def __init__(self, mean):
        self.mean = mean
        components = np.zeros((4, self.mean.as_vector().shape[0]))
        components[0, :] = self.mean.as_vector()
        aux = self.mean.points[:, [1, 0]]
        aux[:, 0] = -aux[:, 0]
        components[1, :] = aux.flatten()
        components[2, ::2] = 1
        components[3, 1::2] = 1
        self._components = components

    @property
    def template_instance(self):
        return self.mean

    @property
    def n_components(self):
        """
        The number of kept principal components.

        :type: int
        """
        return self.components.shape[0]

    @property
    def components(self):
        """
        The principal components.

        :type: (``n_components``, ``n_features``) ndarray
        """
        return self._components

    @components.setter
    def components(self, components):
        r"""
        """
        if self.components.shape == components.shape:
            self._components = components
        else:
            raise ValueError('the new components must be of the same shape '
                             'as the original components')

    @property
    def _jacobian(self):
        return self._components

    def _instance(self, weights):
        return self.mean.as_vector() + np.dot(weights, self.components)

    def _project(self, vector_instance):
        return np.dot(self.components, (vector_instance - self.mean.as_vector()))
