from .base import Transform
import numpy as np


class AffineTransform(Transform):
    """
    The base class for all n-dimensional affine transformations. Provides
    methods to break the transform down into it's constituent
    scale/rotation/translation, to view the homogeneous matrix equivalent,
    and to chain this transform with other affine transformations
    """

    def __init__(self, homogeneous_matrix):
        """

        :param homogeneous_matrix: (n_dim + 1, n_dim + 1) matrix of the
        format [ rotatationscale translation; 0 1]
        """
        shape = homogeneous_matrix.shape
        print homogeneous_matrix
        if len(shape) != 2 and shape[0] != shape[1]:
            raise Exception("You need to provide a square homogeneous matrix.")
        self.n_dim = shape[0] - 1
        self.homogeneous_matrix = homogeneous_matrix

    def chain(self, affine_transform):
        """
        Chains this affine transform with another one,
        producing a new affine transform
        :param affine_transform: Transform to be applied FOLLOWING self
        """
        # note we dot this way as we have our data in the transposed
        # representation to normal
        return AffineTransform(np.dot(affine_transform.homogeneous_matrix,
                                           self.homogeneous_matrix))

    def _apply(self, x):
        """
        Applies this transform to a new set of vectors
        :param x: A (n_dim, n_points) ndarray to apply this transform to.

        :return: The transformed version of x
        """
        return np.dot(x, self.linear_transform.T) + self.translation

    @property
    def linear_transform(self):
        """
        Returns just the linear transform component of this affine
        transform.
        """
        return self.homogeneous_matrix[:-1, :-1]

    @property
    def translation(self):
        """
        Returns just the transform aspect of this affine transform.
        """
        return self.homogeneous_matrix[:-1, -1]


class Translation(AffineTransform):
    """
    An n_dim translation transform.
    """

    def __init__(self, transformation):
        """
        translation : a 1-d ndarray of length n_dim (i.e.
        if you want to make a 3d translation you must specify the
        translation in each dimension explicitly).
        """
        homogeneous_matrix = np.eye(transformation.size + 1)
        homogeneous_matrix[:-1, -1] = transformation
        super(Translation, self).__init__(homogeneous_matrix)


class Rotation(AffineTransform):
    """
    An n_dim rotation transform.
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


class Scale(AffineTransform):
    """
    An n_dim rotation transform.
    """

    def __init__(self, scale):
        """ The scale must be a 2-d square ndarray of shape (n_dim, n_dim)
        By default
        :param scale:
        """
        homogeneous_matrix = np.eye(scale.size + 1)
        np.fill_diagonal(homogeneous_matrix, scale)
        homogeneous_matrix[-1, -1] = 1
        super(Scale, self).__init__(homogeneous_matrix)
