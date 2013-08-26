import abc
import numpy as np
from sklearn.decomposition import PCA as SklearnPCA
from pybug.model.base import StatisticalModel


class LinearModel(StatisticalModel):

    @abc.abstractproperty
    def jacobian(self):
        return self._jacobian

    def instance(self, weightings):
        """
        Creates a new instance of the model using the first len(weightings)
        components.
        :param weightings: A 1D ndarray prescribing the weightings that
        should be used in the model
        :return: An instance of self.sample_data_class
        """
        return self.template_sample.from_vector(self._instance(weightings))

    @abc.abstractmethod
    def _instance(self, weightings):
        """
         Creates a new instance of the model using the first len(weightings)
        components.
        :param weightings: A 1D ndarray prescribing the weightings that
        should be used in the model
        :return: The resulting vector
        """
        pass

    def project(self, novel_instance):
        """
        Projects the novel_instance onto the model, retrieving the optimal
        linear weightings
        :param novel_instance: A novel instance of Vectorizable
        :return: A vector of optimal linear weightings
        """
        return self._project(novel_instance.as_vector())


    @abc.abstractmethod
    def _project(self, novel_vectorized_instance):
        """
        Projects the novel_vectorized_instance onto the model, retrieving the
        optimal linear reconstruction weights
        :param novel_vectorized_instance: A vectorized novel instance to
        project
        :return: A vector of optimal linear weightings
        """
        pass

    def reconstruct(self, novel_instance, n_components=None):
        """
        Project a novel_instance onto the linear space and rebuild from the
        weightings found. Syntactic sugar for:
        >>> pca.instance(pca.project(novel_instance)[:n_components])
        but faster, as it avoids the conversion that takes place each time
        :param novel_instance: A novel instance of Vectorizable
        :param n_components: The number of components to use in the
        reconstruction
        :return: An instance of self.sample_data_class
        """
        vectorized_reconstruction = self._reconstruct(
            novel_instance.as_vector(), n_components)
        return novel_instance.from_vector(vectorized_reconstruction)

    def _reconstruct(self, novel_vectorized_instance, n_components=None):
        """
        Project a flattened novel_instance onto the linear space and rebuild
        from the weightings found. Syntactic sugar for:
        >>> pca._instance(pca._project(novel_vectorized_instance)[:n_components])
        :param novel_vectorized_instance: A vectorized novel instance to
        project
        :param n_components: The number of components to use in the
        reconstruction
        :return: A vectorized reconstruction
        """
        weightings = self._project(novel_vectorized_instance)
        if n_components is not None:
            weightings = weightings[:n_components]
        return self._instance(weightings)

    def project_out(self, novel_instance):
        """
        Returns a version of novel_instance where all the information in
        the first n_components of the model has been projected out.
        :param novel_instance: A novel instance of Vectorizable
        :return: A copy of novel instance, with all features of the
        model projected out
        """
        vectorized_instance = self._project_out(novel_instance.as_vector())
        return novel_instance.from_vector(vectorized_instance.flatten())

    @abc.abstractmethod
    def _project_out(self, novel_vectorized_instance):
        """
        Returns a version of novel_instance where all the information in
        the first n_components of the model has been projected out.
        :param novel_vectorized_instance: A vectorized novel instance of the
        model
        :return: The resulting vectorized instance where the features of the
        model from the first n_components have been removed.
        """
        pass

    def distance_to_subspace(self, novel_instance):
        """
        Returns a version of novel_instance where all the information in
        the first n_components of the model has been projected out.
        :param novel_instance: A novel instance of Vectorizable
        :return: A copy of novel instance, with all features of the
        model projected out
        """
        vectorized_instance = self._distance_to_subspace(novel_instance.as_vector())
        return novel_instance.from_vector(vectorized_instance.flatten())

    @abc.abstractmethod
    def _distance_to_subspace(self, novel_vectorized_instance):
        """
        Returns a version of novel_instance where all the information in
        the first n_components of the model has been projected out.
        :param novel_vectorized_instance: A vectorized novel instance of the
        model
        :return: The resulting vectorized instance where the features of the
        model from the first n_components have been removed.
        """
        pass

    def distance_within_subspace(self, novel_instance):
        """
        Returns a version of novel_instance where all the information in
        the first n_components of the model has been projected out.
        :param novel_instance: A novel instance of Vectorizable
        :return: A copy of novel instance, with all features of the
        model projected out
        """
        vectorized_instance = self._distance_within_subspace(novel_instance.as_vector())
        return novel_instance.from_vector(vectorized_instance.flatten())

    @abc.abstractmethod
    def _distance_within_subspace(self, novel_vectorized_instance):
        """
        Returns a version of novel_instance where all the information in
        the first n_components of the model has been projected out.
        :param novel_vectorized_instance: A vectorized novel instance of the
        model
        :return: The resulting vectorized instance where the features of the
        model from the first n_components have been removed.
        """
        pass


class PCAModel(LinearModel):
    """
    A Linear model based around PCA. Automatically mean centres the input
    data.
    """

    def __init__(self, samples, n_components=None):
        """
        :param samples: A list of Vectorizable objects to build the model from.
        :param n_components: The number of components to internally keep.
        Note that the number of components utilized in the model can be
        curtailed on invocation of methods like reconstruct and instance -
        setting a low number of components here permanently removes other
        components, and should only be used as a memory and performance
        saving measure.
        """
        self.samples = samples
        self.n_samples = len(samples)
        self.n_features = len(samples[0].as_vector())
        self.n_components = n_components
        if self.n_components is None:
            # -1 to prevent us from getting noise in the final component
            self.n_components = min(self.n_samples, self.n_features) - 1
        # flatten one sample to find the n_features we need

        # create and populate the data matrix
        print "Building the data matrix..."
        data = np.zeros((self.n_samples, self.n_features))
        for i, sample in enumerate(self.samples):
            data[i] = sample.as_vector()

        # build the SKlearn PCA passing in the number of components.
        self._pca = SklearnPCA(n_components=self.n_components)
        print "Calculating Principal Components..."
        self._pca.fit(data)

    @property
    def explained_variance(self):
        return self._pca.explained_variance_

    @property
    def explained_variance_ratio(self):
        return self._pca.explained_variance_ratio_

    @property
    def mean(self):
        return self.template_sample.from_vector(self._mean)

    @property
    def _mean(self):
        return self._pca.mean_

    @property
    def components(self):
        """
        The Principal components themselves.
        :return: A (n_components x n_features) ndarray
        """
        return self._pca.components_

    @property
    def jacobian(self):
        return self.components

    def _instance(self, weightings):
        if weightings.shape[0] > self.n_components:
            raise Exception(
                "Number of weightings cannot be greater than {}".format(
                    self.n_components))
        elif weightings.shape[0] < self.n_components:
            full_weightings = np.zeros(self.n_components)
            full_weightings[:weightings.shape[0]] = weightings
            weightings = full_weightings
        return self._pca.inverse_transform(
            weightings.reshape((1, -1))).flatten()

    def _project(self, novel_vectorized_instance):
        return self._pca.transform(
            novel_vectorized_instance.reshape((1, -1))).flatten()

    def _project_out(self, novel_vectorized_instance):
        weights = np.dot(self.components, novel_vectorized_instance)
        # TODO:
        return (novel_vectorized_instance -
                np.dot(self.components.T, weights))

    def _distance_to_subspace(self, novel_vectorized_instance):
        return (self.inv_avg_variance) * obj._project_out(novel_vectorized_instance)

    def _distance_from_subspace(self, novel_vectorized_instance):
        weights = np.dot(self.components, novel_vectorized_instance)
        L = 1 / np.diag(self.explained_variance)
        return np.dot(self.components, np.dot(L, weights))
