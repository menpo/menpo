from __future__ import division, print_function
import numpy as np
from menpo.transform.affine import Scale
from menpo.image import Image
from menpo.fitmultilevel.builder import DeformableModelBuilder
from menpo.fitmultilevel.functions import build_sampling_grid
from menpo.fitmultilevel.featurefunctions import compute_features, sparse_hog
from menpo.fitmultilevel.clm.classifierfunctions import classifier, linear_svm


# TODO: document me
class CLMBuilder(DeformableModelBuilder):
    r"""
    """
    def __init__(self, classifier_type=linear_svm, patch_shape=(5, 5),
                 feature_type=sparse_hog, diagonal_range=None, n_levels=3,
                 downscale=1.1, scaled_levels=True, max_shape_components=None,
                 boundary=3, interpolator='scipy'):
        self.classifier_type = classifier_type
        self.patch_shape = patch_shape
        self.feature_type = feature_type
        self.diagonal_range = diagonal_range
        self.n_levels = n_levels
        self.downscale = downscale
        self.scaled_levels = scaled_levels
        self.max_shape_components = max_shape_components
        self.boundary = boundary
        self.interpolator = interpolator

    def build(self, images, group=None, label='all'):
        r"""
        Builds a Multilevel Constrained Local Model from a list of
        landmarked images.

        Parameters
        ----------
        images: list of :class:`menpo.image.Image`
            The set of landmarked images from which to build the AAM.

        group : string, Optional
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.

            Default: None

        label: string, Optional
            The label of of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

            Default: 'all'

        Returns
        -------
        aam : :class:`menpo.fitmultiple.clm.builder.CLM`
            The CLM object
        """
        print('- Preprocessing')
        self.reference_shape, generator = self._preprocessing(
            images, group, label, self.diagonal_range, self.interpolator,
            self.scaled_levels, self.n_levels, self.downscale)

        print('- Building model pyramids')
        shape_models = []
        classifiers = []
        # for each level
        for j in np.arange(self.n_levels):
            print(' - Level {}'.format(j))

            print('  - Computing feature space')
            images = [compute_features(g.next(), self.feature_type)
                      for g in generator]
            # extract potentially rescaled shapes
            shapes = [i.landmarks[group][label].lms for i in images]

            if j == 0 or self.scaled_levels:
                print('  - Building shape model')
                shape_model = self._build_shape_model(
                    shapes, self.max_shape_components)

            # add shape model to the list
            shape_models.append(shape_model)

            print('  - Building classifiers')
            sampling_grid = build_sampling_grid(self.patch_shape)
            n_points = shapes[0].n_points

            level_classifiers = []
            for k in range(n_points):

                print(' - {} % '.format(round(100*(k+1)/n_points)), end='\r')
                positive_labels = []
                negative_labels = []
                positive_samples = []
                negative_samples = []

                for i, s in zip(images, shapes):

                    max_x = i.shape[0] - 1
                    max_y = i.shape[1] - 1

                    point = (np.round(s.points[k, :])).astype(int)
                    patch_grid = sampling_grid + point[None, None, ...]
                    positive, negative = get_pos_neg_grid_positions(
                        patch_grid, positive_grid_size=(1, 1))

                    x = positive[:, 0]
                    y = positive[:, 1]
                    x[x > max_x] = max_x
                    y[y > max_y] = max_y
                    x[x < 0] = 0
                    y[y < 0] = 0

                    positive_sample = i.pixels[positive[:, 0],
                                               positive[:, 1], :]
                    positive_samples.append(positive_sample)
                    positive_labels.append(np.ones(positive_sample.shape[0]))

                    x = negative[:, 0]
                    y = negative[:, 1]
                    x[x > max_x] = max_x
                    y[y > max_y] = max_y
                    x[x < 0] = 0
                    y[y < 0] = 0

                    negative_sample = i.pixels[x, y, :]
                    negative_samples.append(negative_sample)
                    negative_labels.append(-np.ones(negative_sample.shape[0]))

                positive_samples = np.asanyarray(positive_samples)
                positive_samples = np.reshape(positive_samples,
                                              (-1, positive_samples.shape[-1]))
                positive_labels = np.asanyarray(positive_labels).flatten()

                negative_samples = np.asanyarray(negative_samples)
                negative_samples = np.reshape(negative_samples,
                                              (-1, negative_samples.shape[-1]))
                negative_labels = np.asanyarray(negative_labels).flatten()

                X = np.vstack((positive_samples, negative_samples))
                t = np.hstack((positive_labels, negative_labels))

                clf = classifier(X, t, self.classifier_type)
                level_classifiers.append(clf)

            # add level classifiers to the list
            classifiers.append(level_classifiers)

        # reverse the list of shape and appearance models so that they are
        # ordered from lower to higher resolution
        shape_models.reverse()
        classifiers.reverse()

        return CLM(shape_models, classifiers, self.patch_shape,
                   self.feature_type, self.reference_shape, self.downscale,
                   self.scaled_levels, self.interpolator)


#TODO: Document me
class CLM(object):
    r"""
    Constrained Local Model class.

    Parameters
    -----------
    shape_models: :class:`menpo.model.PCA` list
        A list containing the shape models of the CLM.

    classifiers: classifier_closure list of lists
        A list containing the list of classifier_closures per each pyramidal
        level of the CLM.

    patch_shape: tuple of ints
        The shape of the patches used to train the classifiers.

    transform: :class:`menpo.transform.PureAlignmentTransform`
        The transform used to warp the images from which the AAM was
        constructed.

    feature_type: str or function
        The image feature that was be used to build the appearance_models. Will
        subsequently be used by fitter objects using this class to fitter to
        novel images.

        If None, the appearance model was built immediately from the image
        representation, i.e. intensity.

        If string, the appearance model was built using one of Menpo's default
        built-in feature representations - those
        accessible at image.features.some_feature(). Note that this case can
        only be used with default feature weights - for custom feature
        weights, use the functional form of this argument instead.

        If function, the user can directly provide the feature that was
        calculated on the images. This class will simply invoke this
        function, passing in as the sole argument the image to be fitted,
        and expect as a return type an Image representing the feature
        calculation ready for further fitting. See the examples for
        details.

    reference_shape: PointCloud
        The reference shape that was used to resize all training images to a
        consistent object size.

    downscale: float
        The constant downscale factor used to create the different levels of
        the AAM. For example, a factor of 2 would imply that the second level
        of the AAM pyramid is half the width and half the height of the first.
        The third would be 1/2 * 1/2 = 1/4 the width and 1/4 the height of
        the original.

    scaled_levels: boolean
        Boolean value specifying whether the AAM levels are scaled or not.

    interpolator: string
        The interpolator that was used to build the AAM.

        Default: 'scipy'
    """
    def __init__(self, shape_models, classifiers, patch_shape, feature_type,
                 reference_shape, downscale, scaled_levels, interpolator):
        self.shape_models = shape_models
        self.classifiers = classifiers
        self.patch_shape = patch_shape
        self.feature_type = feature_type
        self.reference_shape = reference_shape
        self.downscale = downscale
        self.scaled_levels = scaled_levels
        self.interpolator = interpolator

    @property
    def n_levels(self):
        """
        The number of multi-resolution pyramidal levels of the CLM.

        :type: int
        """
        return len(self.shape_models)

    @property
    def n_classifiers_per_level(self):
        """
        The number of classifiers per pyramidal level of the CLM.

        :type: int
        """
        return [len(clf) for clf in self.classifiers]

    def instance(self, shape_weights=None, level=-1):
        r"""
        Generates a novel CLM instance given a set of shape weights. If no
        weights are provided, the mean CLM instance is returned.

        Parameters
        -----------
        shape_weights: (n_weights,) ndarray or float list
            Weights of the shape model that will be used to create
            a novel shape instance. If None, the mean shape
            (shape_weights = [0, 0, ..., 0]) is used.

            Default: None

        level: int, optional
            The pyramidal level to be used.

            Default: -1

        Returns
        -------
        image: :class:`menpo.shape.PointCloud`
            The novel CLM instance.
        """
        sm = self.shape_models[level]
        # TODO: this bit of logic should to be transferred down to PCAModel
        if shape_weights is None:
            shape_weights = [0]
        n_shape_weights = len(shape_weights)
        shape_weights *= sm.eigenvalues[:n_shape_weights] ** 0.5
        shape_instance = sm.instance(shape_weights)
        return shape_instance

    def random_instance(self, level=-1):
        r"""
        Generates a novel random CLM instance.

        Parameters
        -----------
        level: int, optional
            The pyramidal level to be used.

            Default: -1

        Returns
        -------
        image: :class:`menpo.shape.PointCloud`
            The novel CLM instance.
        """
        sm = self.shape_models[level]
        # TODO: this bit of logic should to be transferred down to PCAModel
        shape_weights = (np.random.randn(sm.n_active_components) *
                         sm.eigenvalues[:sm.n_active_components]**0.5)
        shape_instance = sm.instance(shape_weights)
        return shape_instance

    def response_image(self, image, group=None, label='all', level=-1):
        r"""
        Generates a response image result of applying the classifiers of a
        particular pyramidal level of the CLM to an image.

        Parameters
        -----------
        image: :class:`menpo.image.base.Image`
            The image.

        group : string, Optional
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.

            Default: None

        label: string, Optional
            The label of of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

            Default: 'all'

        level: int, optional
            The pyramidal level to be used.

            Default: -1

        Returns
        -------
        image: :class:`menpo.image.base.Image`
            The response image.
        """
        image = image.rescale_to_reference_shape(self.reference_shape,
                                                 group=group, label=label)

        pyramid = image.gaussian_pyramid(n_levels=self.n_levels,
                                         downscale=self.downscale)
        images = [compute_features(i, self.feature_type)
                  for i in pyramid]
        images.reverse()

        image = images[level]
        image_pixels = np.reshape(image.pixels, (-1, image.n_channels))
        response_data = np.zeros((image.shape[0], image.shape[1],
                                  self.n_classifiers_per_level[level]))
        # Compute responses
        for j, clf in enumerate(self.classifiers[level]):
            response_data[:, :, j] = np.reshape(clf(image_pixels),
                                                image.shape)

        return Image(image_data=response_data)


def get_pos_neg_grid_positions(sampling_grid, positive_grid_size=(1, 1)):
    r"""
    Divides a sampling grid in positive and negative pixel positions. By
    default only the center of the grid is considered to be positive.
    """
    positive_grid_size = np.array(positive_grid_size)
    mask = np.zeros(sampling_grid.shape[:-1], dtype=np.bool)
    center = np.round(np.array(mask.shape) / 2).astype(int)
    positive_grid_size -= [1, 1]
    start = center - positive_grid_size
    end = center + positive_grid_size + 1
    mask[start[0]:end[0], start[1]:end[1]] = True
    positive = sampling_grid[mask]
    negative = sampling_grid[~mask]
    return positive, negative