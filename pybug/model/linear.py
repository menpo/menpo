import abc
import numpy as np
from scipy.linalg.blas import dgemm
from pybug.decomposition import PCA as PybugPCA
from pybug.model.base import StatisticalModel


# TODO: better document what a linear model does.
class LinearModel(StatisticalModel):
    r"""
    Abstract base class representing a linear model.
    """

    __metaclass__ = abc.ABCMeta

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
        instance : ``self.sample_data_class``
            An instance of the model. Created via a linear combination of the
            model vectors and the ``weights``.
        """
        return self.template_sample.from_vector(self._instance(weights))

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
    def _project(self, vec_instance):
        """
        Projects the ``vec_instance`` onto the model, retrieving the optimal
         linear reconstruction weights

        Parameters
        -----------
        vec_instance : (n_features,) ndarray
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
        reconstructed : ``self.sample_data_class``
            The reconstructed object.
        """
        vec_reconstruction = self._reconstruct(instance.as_vector(),
                                               n_components)
        return instance.from_vector(vec_reconstruction)

    def _reconstruct(self, vec_instance, n_components=None):
        """
        Project a flattened ``novel_instance`` onto the linear space and
        rebuild from the weights found.

        Syntactic sugar for:

            >>> pca._instance(pca._project(novel_vectorized_instance)[:n_components])

        Parameters
        ----------
        vec_instance : (n_features, ) ndarray
            A vectorized novel instance to project
        n_components : int, optional
            The number of components to use in the reconstruction

            Default: ``weights.shape[0]``

        Returns
        -------
        reconstructed : (n_features,) ndarray
            The reconstructed vector.
        """
        weights = self._project(vec_instance)
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
        projected_out : ``self.sample_data_class``
            A copy of ``instance``, with all basis of the model projected out.
        """
        vec_instance = self._project_out(instance.as_vector())
        return instance.from_vector(vec_instance)

    @abc.abstractmethod
    def _project_out(self, novel_vectorized_instance):
        """
        Returns a version of ``instance`` where all the basis of the model
        have been projected out.

        Parameters
        ----------
        vec_instance : (n_features,) ndarray
            A novel vector.

        Returns
        -------
        projected_out : (n_features,) ndarray
            A copy of ``vec_instance`` with all basis of the model projected
            out.
        """
        pass

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
        scaled_projected_out : ``self.sample_data_class``
            A copy of ``instance``, with all basis of the model projected out
            and scaled by the inverse of the ``noise_variance``.
        """
        vec_instance = self._to_subspace(instance.as_vector())
        return instance.from_vector(vec_instance)

    @abc.abstractmethod
    def _to_subspace(self, vec_instance):
        """
        Returns a version of ``instance`` where all the basis of the model
        have been projected out and which has been scaled by the inverse of
        the ``noise_variance``.

        Parameters
        ----------
        vec_instance : (n_features,) ndarray
            A novel vector.

        Returns
        -------
        scaled_projected_out: (n_features,) ndarray
            A copy of ``vec_instance`` with all basis of the model projected
            out and scaled by the inverse of the ``noise_variance``.
        """
        pass

    def within_subspace(self, instance):
        """
        Returns a sheared (non-orthogonal) reconstruction of ``vec_instance``.

        Parameters
        ----------
        instance : :class:`pybug.base.Vectorizable`
            A novel instance.

        Returns
        -------
        sheared_reconstruction : ``self.sample_data_class``
            A sheared (non-orthogonal) reconstruction of ``instance``.
        """
        vec_instance = self._within_subspace(instance.as_vector())
        return instance.from_vector(vec_instance)

    @abc.abstractmethod
    def _within_subspace(self, vec_instance):
        """
        Returns a sheared (non-orthogonal) reconstruction of ``vec_instance``.

        Parameters
        ----------
        vec_instance : (n_features,) ndarray
            A novel vector.

        Returns
        -------
        sheared_reconstruction : (n_features,) ndarray
            A sheared (non-orthogonal) reconstruction of ``vec_instance``
        """
        pass

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
                                          self.template_sample.n_dims)
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


#TODO: give a description of what it means to be a PCA model
class PCAModel(LinearModel):
    """
    A Linear model based around PCA. Automatically mean centres the input
    data.

    Parameters
    ----------
    samples: list of :class:`pybug.base.Vectorizable`
        List of samples to build the model from.
    n_components: int, optional
        The number of components to internally keep.

        .. note::

            The number of components utilized in the model can be
            curtailed on invocation of methods like reconstruct and instance -
            setting a low number of components here permanently removes other
            components, and should only be used as a memory and performance
            saving measure.
    PCA: specific class implementing PCA, optional

        Default: `PybugPCA`

        .. note::

            This will currently break if `sklearn.decomposition.pca.PCA` is
            set to be the specific implementation of PCA. This is because
            this implementation does not support the concept of
            `noise_variance`. Support for this concept is expected on their
            next upcoming release.


    """

    def __init__(self, samples, n_components=None, PCA=PybugPCA):
        self.samples = samples
        self.n_samples = len(samples)
        self.n_features = len(samples[0].as_vector())
        self.n_components = n_components
        if self.n_components is None:
            # -1 to prevent us from getting noise in the final component
            self.n_components = min(self.n_samples, self.n_features) - 1

        # create and populate the data matrix
        data = np.zeros((self.n_samples, self.n_features))
        for i, sample in enumerate(self.samples):
            data[i] = sample.as_vector()

        # build PCA object.
        self._pca = PCA(n_components=self.n_components)
        # compute PCA
        self._pca.fit(data)

        # store inverse noise variance
        self.inv_noise_variance = 1 / self.noise_variance
        # pre-compute whiten components: U * L^{-1/2}
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

        :type: ``self.sample_data_class``
        """
        return self.template_sample.from_vector(self._mean)

    @property
    def _mean(self):
        """
        The mean vector of the samples.

        :type: (N,) ndarray
        """
        return self._pca.mean_

    @property
    def components(self):
        """
        The principal components.

        :type: (``n_components``, ``n_features``) ndarray
        """
        return self._pca.components_

    @property
    def _jacobian(self):
        return self.components

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

    def _project(self, vec_instance):
        return self._pca.transform(vec_instance)

    def _project_out(self, vec_instance):
        weights = dgemm(alpha=1.0, a=vec_instance.T, b=self.components.T,
                        trans_a=True)
        return (vec_instance -
                dgemm(alpha=1.0, a=weights.T, b=self.components.T,
                      trans_a=True, trans_b=True))

    def _to_subspace(self, vec_instance):
        return self.inv_noise_variance * self._project_out(vec_instance)

    def _within_subspace(self, vec_instance):
        weights = dgemm(alpha=1.0, a=vec_instance.T,
                        b=self.whitened_components.T, trans_a=True)
        return dgemm(alpha=1.0, a=weights.T, b=self.whitened_components.T,
                     trans_a=True, trans_b=True)
