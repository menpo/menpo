import abc
import copy
from .base import Transform
from pybug.exceptions import DimensionalityError
import numpy as np


class AffineTransform(Transform):
    """
    The base class for all n-dimensional affine transformations. Provides
    methods to break the transform down into it's constituent
    scale/rotation/translation, to view the homogeneous matrix equivalent,
    and to chain this transform with other affine transformations.
    """

    def __init__(self, homogeneous_matrix):
        """
        :param homogeneous_matrix: (n_dim + 1, n_dim + 1) matrix of the
        format [ rotatationscale translation; 0 1]
        """
        #TODO Check am I a valid Affine transform
        shape = homogeneous_matrix.shape
        if len(shape) != 2 and shape[0] != shape[1]:
            raise Exception("You need to provide a square homogeneous matrix.")
        self.n_dim = shape[0] - 1
        # this restriction is because we have to be able to decompose
        # transforms to find their meaning, and I haven't explored 4D+
        # rotations (everything else is obvious). If there is need we can
        # relax it in the future
        if self.n_dim not in [2, 3]:
            raise DimensionalityError("Affine Transforms can only be 2D or "
                                      "3D")
        self.homogeneous_matrix = homogeneous_matrix

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
        Returns just the n-dim translation component of this affine transform.
        """
        return self.homogeneous_matrix[:-1, -1]

    def __eq__(self, other):
        return np.allclose(self.homogeneous_matrix, other.homogeneous_matrix)

    def __str__(self):
        rep = repr(self) + '\n'
        rep += str(self.homogeneous_matrix) + '\n'
        rep += self._transform_str()
        return rep

    def _transform_str(self):
        """
        A string representation explaining what this affine transform does.
        Has to be implemented by base classes.
        """
        list_str = [t._transform_str() for t in self.decompose()]
        return reduce(lambda x, y: x + '\n' + y, list_str)

    def _apply(self, x):
        """
        Applies this transform to a new set of vectors
        :param x: A (n_dim, n_points) ndarray to apply this transform to.

        :return: The transformed version of x
        """
        return np.dot(x, self.linear_transform.T) + self.translation

    def chain(self, affine_transform):
        """
        Chains this affine transform with another one,
        producing a new affine transform
        :param affine_transform: Transform to be applied FOLLOWING self
        :return: the resulting affine transform
        """
        # note we dot this way as we have our data in the transposed
        # representation to normal
        return AffineTransform(np.dot(affine_transform.homogeneous_matrix,
                                      self.homogeneous_matrix))

    def decompose(self):
        """
        Uses an SVD to decompose this transform into discrete Affine
        Transforms.

        :return transforms: A list of a DiscreteAffineTransforms that are
        equivalent to this affine transform, s.t.
        reduce(lambda x,y: x.chain(y), self.decompose()) == self
        """
        U, S, V = np.linalg.svd(self.linear_transform)
        rotation_2 = Rotation(U)
        rotation_1 = Rotation(V)
        scale = Scale(S)
        translation = Translation(self.translation)
        return [rotation_1, scale, rotation_2, translation]


class SimilarityTransform(AffineTransform):
    """
    Specialist version of an AffineTransform that is guaranteed to be a
    Similarity transform.
    """

    def __init__(self, homogeneous_matrix):
        #TODO check that I am a similarity transform
        super(SimilarityTransform, self).__init__(homogeneous_matrix)

    def chain(self, transform):
        """
        Chains this similarity transform with another one. If the second
        transform is also a Similarity transform, the result will be a
        SimilarityTransform. If not, the result will be an AffineTransform.
        :param transform: Transform to be applied FOLLOWING self
        :return: the resulting transform
        """
        if isinstance(transform, SimilarityTransform):
            return SimilarityTransform(np.dot(transform.homogeneous_matrix,
                                              self.homogeneous_matrix))
        else:
            return super(SimilarityTransform, self).chain(transform)


class DiscreteAffineTransform(object):
    """
    A discrete Affine transform operation (such as a Scale,
    Translation or Rotation). Has to be able to invert itself. Make sure you
    inherit from DiscreteAffineTransform first, for optimal decompose()
    behavior
    """

    __metaclass__ = abc.ABCMeta

    def decompose(self):
        """ A DiscreteAffineTransform is already maximally decomposed -
        return a copy of self in a list
        """
        return [copy.deepcopy(self)]

    @abc.abstractproperty
    def inverse(self):
        pass


def Rotation(rotation_matrix):
    """
    Factory function for producing Rotation transforms.
    :param rotation_matrix: A square legal 2D or 3D rotation matrix
    :return: A Rotation2D or Rotation3D object.
    """
    if rotation_matrix.shape[0] == 2:
        return Rotation2D(rotation_matrix)
    elif rotation_matrix.shape[0] == 3:
        return Rotation3D(rotation_matrix)
    else:
        raise DimensionalityError("Can only construct 2D or 3D Rotations")
    # TODO build rotations about axis, euler angles etc
    # see
    # http://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    # for details.


class AbstractRotation(DiscreteAffineTransform, SimilarityTransform):
    """
    Abstract n_dim rotation transform.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, rotation_matrix):
        """ The rotation_matrix must be a 2-d square ndarray of shape
        (n_dim, n_dim)
        """
        #TODO check that I am a valid rotation
        homogeneous_matrix = np.eye(rotation_matrix.shape[0] + 1)
        homogeneous_matrix[:-1, :-1] = rotation_matrix
        super(AbstractRotation, self).__init__(homogeneous_matrix)

    @property
    def rotation_matrix(self):
        """Returns the rotation matrix
        """
        return self.linear_transform

    @property
    def inverse(self):
        return Rotation(np.linalg.inv(self.rotation_matrix))

    def _transform_str(self):
        axis, rad_angle_of_rotation = self.axis_and_angle_of_rotation()
        if axis is None:
            return "NO OP"
        angle_of_rot = (rad_angle_of_rotation * 180.0) / np.pi
        message = 'CCW Rotation of %d degrees about %s' % (angle_of_rot, axis)
        return message

    @abc.abstractmethod
    def axis_and_angle_of_rotation(self):
        pass


class Rotation2D(AbstractRotation):

    def __init__(self, rotation_matrix):
        super(Rotation2D, self).__init__(rotation_matrix)
        if self.n_dim != 2:
            raise DimensionalityError("Rotation2D has to be built from a 2D"
                                      " rotation matrix")

    def axis_and_angle_of_rotation(self):
        """
        Decomposes this 2D rotation's rotation matrix into a angular rotation
        The rotation is considered in a right handed sense. The axis is, by
        definition, [0, 0, 1].

        :return: (axis, angle_of_rotation)
        axis: A unit vector, the axis about which the rotation takes place
        angle_of_rotation: The angle in radians of the rotation about the
        axis.
        The angle is signed in a right handed sense.
        """
        axis = np.array([0, 0, 1])
        test_vector = np.array([1, 0])
        transformed_vector = np.dot(self.rotation_matrix,
                                    test_vector)
        angle_of_rotation = np.arccos(
            np.dot(transformed_vector, test_vector))
        return axis, angle_of_rotation


class Rotation3D(AbstractRotation):

    def __init__(self, rotation_matrix):
        super(Rotation3D, self).__init__(rotation_matrix)
        if self.n_dim != 3:
            raise DimensionalityError("Rotation3D has to be built from a 3D"
                                      " rotation matrix")

    def axis_and_angle_of_rotation(self):
        """
        Decomposes this 3D rotation's rotation matrix into a angular rotation
        about an axis. The rotation is considered in a right handed sense.

        :return: (axis, angle_of_rotation)
        axis: A unit vector, the axis about which the rotation takes place
        angle_of_rotation: The angle in radians of the rotation about the
        axis.
        The angle is signed in a right handed sense.

        See http://en.wikipedia.org/wiki/Rotation_matrix#Determining_the_axis
        """
        eval_, evec = np.linalg.eig(self.rotation_matrix)
        real_eval_mask = np.isreal(eval_)
        real_eval = np.real(eval_[real_eval_mask])
        evec_with_real_eval = np.real_if_close(evec[:, real_eval_mask])
        error = 1e-7
        below_margin = np.abs(real_eval) < (1 + error)
        above_margin = (1 - error) < np.abs(real_eval)
        re_unit_eval_mask = np.logical_and(below_margin, above_margin)
        evec_with_real_unitary_eval = evec_with_real_eval[:, re_unit_eval_mask]
        # all the eigenvectors with real unitary eigenvalues are now all
        # equally 'valid' if multiple remain that probably means that this
        # rotation is actually a no op (i.e. rotate by 360 degrees about any
        #  axis is an invariant transform) but need to check this. For now,
        # just take the first
        if evec_with_real_unitary_eval.shape[1] != 1:
            # TODO confirm that multiple eigenvalues of 1 means the rotation
            #  does nothing
            return None, None
        axis = evec_with_real_unitary_eval[:, 0]
        axis /= np.sqrt((axis ** 2).sum())  # normalize to unit vector
        # to find the angle of rotation, build a new unit vector perpendicular
        # to the axis, and see how it rotates
        axis_temp_vector = axis - np.random.rand(axis.size)
        perpendicular_vector = np.cross(axis, axis_temp_vector)
        perpendicular_vector /= np.sqrt((perpendicular_vector ** 2).sum())
        transformed_vector = np.dot(self.rotation_matrix,
                                    perpendicular_vector)
        angle_of_rotation = np.arccos(
            np.dot(transformed_vector, perpendicular_vector))
        chirality_of_rotation = np.dot(axis, np.cross(perpendicular_vector,
                                                      transformed_vector))
        if chirality_of_rotation < 0:
            angle_of_rotation *= -1.0
        return axis, angle_of_rotation


def Scale(scale_factor, n_dim=None):
    """
    Factory function for producing Scale transforms.
    A UniformScale will be produced if:
    - A float scale_factor and a n_dim kwarg are provided
    - A ndarray scale_factor with shape (n_dim, ) is provided with all
    elements being the same
    A NonUniformScale will be provided if:
    - A ndarray scale_factor with shape (n_dim, ) is provided with at least
    two differing scale factors.
    """
    if n_dim is None:
        # scale_factor better be a numpy array then
        if np.allclose(scale_factor, scale_factor[0]):
            return UniformScale(scale_factor[0], scale_factor.shape[0])
        else:
            return NonUniformScale(scale_factor)
    else:
        return UniformScale(scale_factor, n_dim)


class NonUniformScale(DiscreteAffineTransform, AffineTransform):
    """
    An n_dim scale transform, with a scale component for each dimension.
    """
    def __init__(self, scale):
        """ The scale must be a 1-d ndarray of shape (n_dim, )
        :param scale: A vector specifying the scale factor to be applied
        along each axis.
        """
        homogeneous_matrix = np.eye(scale.size + 1)
        np.fill_diagonal(homogeneous_matrix, scale)
        homogeneous_matrix[-1, -1] = 1
        super(NonUniformScale, self).__init__(homogeneous_matrix)

    @property
    def scale(self):
        return self.homogeneous_matrix.diagonal()[:-1]

    @property
    def inverse(self):
        return NonUniformScale(1.0/self.scale)

    def _transform_str(self):
        message = 'NonUniformScale by %s ' % self.scale
        return message


class UniformScale(DiscreteAffineTransform, SimilarityTransform):
    """
    An n_dim similarity scale transform, with a single scale component
    applied to all dimensions.
    """
    def __init__(self, scale, n_dim):
        """ The scale must be a 1-d ndarray of shape (n_dim, )
        :param scale: A vector specifying the scale factor to be applied
        along each axis.
        """
        homogeneous_matrix = np.eye(n_dim + 1)
        np.fill_diagonal(homogeneous_matrix, scale)
        homogeneous_matrix[-1, -1] = 1
        super(UniformScale, self).__init__(homogeneous_matrix)

    @property
    def scale(self):
        return self.homogeneous_matrix.diagonal()[0]

    @property
    def inverse(self):
        return UniformScale(1.0/self.scale, self.n_dim)

    def _transform_str(self):
        message = 'UniformScale by %f ' % self.scale
        return message


class Translation(DiscreteAffineTransform, SimilarityTransform):
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

    @property
    def inverse(self):
        return Translation(-self.translation)

    def _transform_str(self):
        message = 'Translate by %s ' % self.translation
        return message
