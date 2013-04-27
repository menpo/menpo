from .base import Transformation
import numpy as np


class AffineTransformation(Transformation):
    """
    The base class for all n-dimensional affine transformations. Provides
    methods to break the transform down into it's constituent
    scale/rotation/translation, to view the homogeneous matrix equivalent,
    and to chain this transformation with other affine transformations
    """

    def __init__(self, homogeneous_matrix):
        """

        :param homogeneous_matrix: (n_dim + 1, n_dim + 1) matrix of the
        format [ rotatationscale translation; 0 1]
        """
        shape = homogeneous_matrix.shape
        if len(shape) != 2 and shape[0] != shape[1]:
            raise Exception("You need to provide a square homogeneous matrix.")
        self.n_dim = shape[0] - 1
        self.homogeneous_matrix = homogeneous_matrix

    def chain(self, affinetransform):
        """
        Chains this affine transformation with another one,
        producing a new affine transformation
        :param affinetransform: Transform to be applied FOLLOWING self
        """
        # note we dot this way as we have our data in the transposed
        # representation to normal
        return AffineTransformation(np.dot(affinetransform.homogeneous_matrix,
                                           self.homogeneous_matrix))

    def apply(self, x):
        """
        Applies this transformation to a new set of vectors
        :param x: A (n_dim, n_points) ndarray to apply this transformation to.

        :return: The transformed version of x
        """
        return np.dot(x, self.linear_transformation) + self.translation

    @property
    def linear_transformation(self):
        """
        Returns just the linear transformation component of this affine
        transformation.
        """
        return self.homogeneous_matrix[:-1, :-1]

    @property
    def translation(self):
        """
        Returns just the transformation aspect of this affine transformation.
        """
        return self.homogeneous_matrix[-1, :-1]


class Translation(AffineTransformation):
    """
    An n_dim translation transformation.
    """

    def __init__(self, transformation):
        """
        translation : a 1-d ndarray of length n_dim (i.e.
        if you want to make a 3d translation you must specify the
        translation in each dimension explicitly).
        """
        homogeneous_matrix = np.eye(transformation.size + 1)
        homogeneous_matrix[-1, :-1] = transformation
        super(Translation, self).__init__(homogeneous_matrix)


class Rotation(AffineTransformation):
    """
    An n_dim rotation transformation.
    """

    def __init__(self, rotation):
        """ The rotation must be a 2-d square ndarray of shape (n_dim, n_dim)
        By default
        """
        homogeneous_matrix = np.eye(rotation.shape[0] + 1)
        homogeneous_matrix[:-1, :-1] = rotation
        super(Rotation, self).__init__(homogeneous_matrix)

    # TODO - be able to calculate rotations based on angles around axes
    # here is a start, and see
    # http://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    # for details.
    # @classmethod
    # def aboutaxis(cls, axis, angle):
    #     """
    #     Returns a rotation object defined by angles about the axes
    #     """
    #     n_dim = axis.size
    #     rotation = np.eye(n_dim) * np.cos(angle) + np.sin(angle) *
    #
    #     return cls()


class Scale(AffineTransformation):
    """
    An n_dim rotation transformation.
    """

    def __init__(self, scale):
        """ The scale must be a 2-d square ndarray of shape (n_dim, n_dim)
        By default
        :param scale:
        """
        homogeneous_matrix = np.eye(rotation.shape[0] + 1)
        homogeneous_matrix[:-1, :-1] = rotation
        super(Rotation, self).__init__(homogeneous_matrix)
