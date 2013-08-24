import abc
import copy
from pybug.transform.base import Transform
from pybug.exceptions import DimensionalityError
import pybug.matlab as matlab
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
    def linear_component(self):
        """
        Returns just the linear transform component of this affine
        transform.
        """
        return self.homogeneous_matrix[:-1, :-1]

    @property
    def translation_component(self):
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

    def _apply(self, x, **kwargs):
        """
        Applies this transform to a new set of vectors
        :param x: A (n_points, n_dims) ndarray to apply this transform to.

        :return: The transformed version of x
        """
        return np.dot(x, self.linear_component.T) + self.translation_component

    def compose(self, affine_transform):
        """
        Chains this affine transform with another one,
        producing a new affine transform
        :param affine_transform: Transform to be applied FOLLOWING self
        :return: the resulting affine transform
        """
        # note we dot this way as we have our data in the transposed
        # representation to normal
        return AffineTransform(np.dot(self.homogeneous_matrix,
                                      affine_transform.homogeneous_matrix))

    def decompose(self):
        """
        Uses an SVD to decompose this transform into discrete Affine
        Transforms.

        :return transforms: A list of a DiscreteAffineTransforms that are
        equivalent to this affine transform, s.t.
        reduce(lambda x,y: x.chain(y), self.decompose()) == self
        """
        U, S, V = np.linalg.svd(self.linear_component)
        rotation_2 = Rotation(U)
        rotation_1 = Rotation(V)
        scale = Scale(S)
        translation = Translation(self.translation_component)
        return [rotation_1, scale, rotation_2, translation]

    def jacobian(self, points):
        """
        Computes the Jacobian of the transform w.r.t the parameters. This is
        constant for affine transforms.

        The Jacobian generated (for 2D) is of the form::

            x 0 y 0 1 0
            0 x 0 y 0 1

        This maintains a parameter order of::

          W(x;p) = [1 + p1  p3      p5] [x]
                   [p2      1 + p4  p6] [y]
                                        [1]

        :return dW/dp: A n_points x n_params x n_dims ndarray representing
            the Jacobian of the transform.
        """
        n_points, points_n_dim = points.shape
        if points_n_dim != self.n_dim:
            raise DimensionalityError(
                "Trying to sample jacobian in incorrect dimensions "
                "(transform is {0}D, sampling at {1}D)".format(
                    self.n_dim, points_n_dim))
        # prealloc the jacobian
        jac = np.zeros((n_points, self.n_parameters, self.n_dim))
        # a mask that we can apply at each iteration
        dim_mask = np.eye(self.n_dim, dtype=np.bool)

        for i, s in enumerate(range(0, self.n_dim * self.n_dim, self.n_dim)):
            # i is current axis
            # s is slicing offset
            # make a mask for a single points jacobian
            full_mask = np.zeros((self.n_parameters, self.n_dim), dtype=bool)
            # fill the mask in for the ith axis
            full_mask[slice(s, s + self.n_dim)] = dim_mask
            # assign the ith axis points to this mask, broadcasting over all
            # points
            jac[:, full_mask] = points[:, i][..., None]
        # finally, just repeat the same but for the ones at the end
        full_mask = np.zeros((self.n_parameters, self.n_dim), dtype=bool)
        full_mask[slice(s + self.n_dim, s + 2 * self.n_dim)] = dim_mask
        jac[:, full_mask] = 1
        return jac

    def jacobian_points(self, points):
        """
        Computes the Jacobian of the transform wrt the points to which
        the transform is applied to. This is constant for affine transforms.

        The Jacobian for a given point (for 2D) is of the form::

            Jx = [(1 + a),     -b  ]
            Jy = [   b,     (1 + a)]
            J =  [Jx, Jy] = [[(1 + a), -b], [b, (1 + a)]]

        where a and b come from:

            W(x;p) = [1 + a   -b      tx] [x]
                     [b       1 + a   ty] [y]
                                          [1]

        :return dW/dx: A n_points x n_dims x n_dims ndarray representing
            the Jacobian of the transform wrt the points to which the
            transform is applied to.
        """
        return self.linear_component[None, ...]

    @property
    def n_parameters(self):
        """
        n_dim * (n_dim + 1) parameters - every element of the matrix bar the
        homogeneous part
        2D Affine: 6 parameters::

            [p1, p3, p5]
            [p2, p4, p6]

        3D Affine: 12 parameters::
            [p1, p4, p7, p10]
            [p2, p5, p8, p11]
            [p3, p6, p9, p12]
        """
        return self.n_dim * (self.n_dim + 1)

    def as_vector(self):
        """
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. This does not
        include the homogeneous part of the warp. Note that it flattens using
        Fortran ordering, to stay consistent with Matlab.
        """
        params = self.homogeneous_matrix - np.eye(self.n_dim + 1)
        return params[:self.n_dim, :].flatten(order='F')

    @classmethod
    def from_vector(cls, p):
        # n.b. generally, from_vector should be an instance method. However,
        # as Python class methods can be called on any instance,
        # we are free to implement the from_vector method as a class method
        # where appropriate, as is the case in AffineTransform. This means
        # we can use from_vector as a constructor to the class in addition
        # to it's usual role in building novel instances where some kind of
        # state needs to be stolen from a pre-existing instance (hence the
        # need for this to in general be an instance method).
        if p.shape[0] is 6:  # 2D affine
            homo_matrix = np.eye(3)
            homo_matrix[:2, :] += matlab.reshape(p, [2, 3])
            return AffineTransform(homo_matrix)
        elif p.shape[0] is 12:  # 3D affine
            homo_matrix = np.eye(4)
            homo_matrix[:3, :] += matlab.reshape(p, [3, 4])
            return AffineTransform(homo_matrix)
        else:
            raise DimensionalityError("Affine Transforms can only be 2D or "
                                      "3D")

    @property
    def inverse(self):
        return AffineTransform(np.linalg.inv(self.homogeneous_matrix))

    @classmethod
    def estimate(cls, source, target):
        return cls(cls._estimate(source, target))

    @classmethod
    def _estimate(cls, source, target):
        """
        Infers the affine transform relating two vectors with the same
        dimensionality.

        The affine transform is defined as:

            X = a*x + b*y + tx
            Y = c*x + d*y + ty

        These equations can be transformed to the following form::

            a*x + b*y + tx - X = 0
            c*x + d*y + ty - Y = 0

        which can be written in matrix form as:

            A x = 0

        where::

            A   = [[x y 1 0 0 0 X]
                   [0 0 0 x y 1 Y]
                    ...
                    ...
                  ]
            x.T = [a b tx c d ty c6]

        Using total least-squares, the solution of the previous homogeneous
        system of equations is the right singular vector of A which
        corresponds to the smallest singular value normalized by the
        coefficient c6.

        :param source: A n_points x n_dims ndarray.
        :param target: A n_points x n_dims ndarray.
        :return: The AffineTransform object relating the previous two vectors.
        """
        n_dim = source.shape[1]
        if n_dim == 2:
            n_points = source.shape[0]
            xs = source[:, 0]
            ys = source[:, 1]
            xd = target[:, 0]
            yd = target[:, 1]

            # parameters: a, b, tx, c, d, ty
            A = np.zeros((n_points * 2, 7))
            # a
            A[:n_points, 0] = xs
            # b
            A[:n_points, 1] = ys
            # tx
            A[:n_points, 2] = 1
            # c
            A[n_points:, 3] = xs
            # d
            A[n_points:, 4] = ys
            # ty
            A[n_points:, 5] = 1
            # target
            A[:n_points, 6] = xd
            A[n_points:, 6] = yd
            # the parameters of the affine transform are given by the least
            # significant right singular vector normalized by coefficient c6.
            _, _, V = np.linalg.svd(A)
            a, b, tx, c, d, ty = - V[-1, :-1] / V[-1, -1]
            # build homogeneous matrix
            homogeneous_matrix = np.array([[a, b, tx],
                                           [c, d, ty],
                                           [0, 0,  1]])
            return homogeneous_matrix
        elif n_dim == 3:
            raise NotImplementedError("3D affine transforms cannot be "
                                      "inferred yet.")
        else:
            raise DimensionalityError("Only 2D and 3D affine transforms "
                                      "are currently supported.")


class SimilarityTransform(AffineTransform):
    """
    Specialist version of an AffineTransform that is guaranteed to be a
    Similarity transform.
    """

    def __init__(self, homogeneous_matrix):
        #TODO check that I am a similarity transform
        super(SimilarityTransform, self).__init__(homogeneous_matrix)

    @property
    def n_parameters(self):
        """
        2D Similarity: 4 parameters::

            [(1 + a), -b,      tx]
            [b,       (1 + a), ty]

        3D Similarity: Currently not supported
        :return:
        """
        if self.n_dim == 2:
            return 4
        elif self.n_dim == 3:
            raise NotImplementedError("3D similarity transforms cannot be "
                                      "vectorized yet.")
        else:
            raise DimensionalityError("Only 2D and 3D Similarity transforms "
                                      "are currently supported.")

    def jacobian(self, points):
        """
        Computes the Jacobian of the transform w.r.t the parameters.

        The Jacobian generated (for 2D) is of the form::

            x -y 1 0
            y  x 0 1

        This maintains a parameter order of::

          W(x;p) = [1 + a  -b   ] [x] + tx
                   [b      1 + a] [y] + ty

        :param points: The points to calculate the jacobian over
        :return dW/dp: A n_points x n_params x n_dims ndarray representing
            the Jacobian of the transform.
        :raises: DimensionalityError if ``points_n_dim != self.n_dim`` or
            transform is not 2D
        """
        n_points, points_n_dim = points.shape
        if points_n_dim != self.n_dim:
            raise DimensionalityError('Trying to sample jacobian in incorrect '
                                      'dimensions (transform is {0}D, '
                                      'sampling at {1}D)'.format(self.n_dim,
                                                                 points_n_dim))
        elif self.n_dim != 2:
            # TODO: implement 3D Jacobian
            raise DimensionalityError("Only the Jacobian of a 2D similarity "
                                      "transform is currently supported.")

        # prealloc the jacobian
        jac = np.zeros((n_points, self.n_parameters, self.n_dim))
        ones = np.ones_like(points)

        # Build a mask and apply it to the points to build the jacobian
        # Do this for each paramter - [a, b, tx, ty] respectively
        self._apply_jacobian_mask(jac, np.array([1, 1]), 0, points)
        self._apply_jacobian_mask(jac, np.array([-1, 1]), 1, points[:, ::-1])
        self._apply_jacobian_mask(jac, np.array([1, 0]), 2, ones)
        self._apply_jacobian_mask(jac, np.array([0, 1]), 3, ones)

        return jac

    def _apply_jacobian_mask(self, jac, param_mask, row_index, points):
        # make a mask for a single points jacobian
        full_mask = np.zeros((self.n_parameters, self.n_dim), dtype=np.bool)
        # fill the mask in for the ith axis
        full_mask[row_index] = [True, True]
        # assign the ith axis points to this mask, broadcasting over all
        # points
        jac[:, full_mask] = points * param_mask

    def as_vector(self):
        """
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. The parameters
        are output in the order [a, b, tx, ty].
        """
        n_dim = self.n_dim
        if n_dim == 2:
            params = self.homogeneous_matrix - np.eye(n_dim + 1)
            # Pick off a, b, tx, ty
            params = params[:n_dim, :].flatten(order='F')
            # Pick out a, b, tx, ty
            return params[[0, 1, 4, 5]]
        elif n_dim == 3:
            raise NotImplementedError("3D similarity transforms cannot be "
                                      "vectorized yet.")
        else:
            raise DimensionalityError("Only 2D and 3D Similarity transforms "
                                      "are currently supported.")

    @classmethod
    def from_vector(cls, p):
        # See affine from_vector with regards to classmethod decorator
        if p.shape[0] == 4:
            homo = np.eye(3)
            homo[0, 0] += p[0]
            homo[1, 1] += p[0]
            homo[0, 1] = -p[1]
            homo[1, 0] = p[1]
            homo[:2, 2] = p[2:]
            return SimilarityTransform(homo)
        elif p.shape[0] == 7:
            raise NotImplementedError("3D similarity transforms cannot be "
                                      "vectorized yet.")
        else:
            raise DimensionalityError("Only 2D and 3D Similarity transforms "
                                      "are currently supported.")

    def compose(self, transform):
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
            return super(SimilarityTransform, self).compose(transform)

    @classmethod
    def _estimate(cls, source, target):
        """
        Infers the affine transform relating two vectors with the same
        dimensionality.

        The affine transform is defined as:

            X = a*x - b*y + tx
            Y = b*x + a*y + ty

        These equations can be transformed to the following form::

            a*x - b*y + tx - X = 0
            b*x + a*y + ty - Y = 0

        which can be written in matrix form as:

            A x = 0

        where::

            A   = [[x -y 1 0 X]
                   [y  x 0 1 Y]
                    ...
                    ...
                  ]
            x.T = [a b tx ty c4]

        Using total least-squares, the solution of the previous homogeneous
        system of equations is the right singular vector of A which
        corresponds to the smallest singular value normalized by the
        coefficient c4.

        :param source: A n_points x n_dims ndarray.
        :param target: A n_points x n_dims ndarray.
        :return: The SimilarityTransform object relating the previous two
            vectors.
        """
        n_dim = source.shape[1]
        if n_dim == 2:
            n_points = source.shape[0]
            xs = source[:, 0]
            ys = source[:, 1]
            xd = target[:, 0]
            yd = target[:, 1]
            # parameters: a, b, tx, ty
            A = np.zeros((n_points * 2, 5))
            # a
            A[:n_points, 0] = xs
            A[n_points:, 0] = ys
            # b
            A[:n_points, 1] = -ys
            A[n_points:, 1] = xs
            # tx
            A[:n_points, 2] = 1
            # ty
            A[n_points:, 3] = 1
            # target
            A[:n_points, 4] = xd
            A[n_points:, 4] = yd
            # the parameters of the similarity transform are given by the
            # least significant right singular vector normalized by
            # coefficient c4.
            _, _, V = np.linalg.svd(A)
            a, b, tx, ty = - V[-1, :-1] / V[-1, -1]
            # build homogeneous matrix
            homogeneous_matrix = np.array([[a, -b, tx],
                                           [b,  a, ty],
                                           [0,  0,  1]])
            return homogeneous_matrix
        elif n_dim == 3:
            raise NotImplementedError("3D similarity transforms cannot be "
                                      "inferred yet.")
        else:
            raise DimensionalityError("Only 2D and 3D similarity transforms "
                                      "are currently supported.")

    @property
    def inverse(self):
        return SimilarityTransform(np.linalg.inv(self.homogeneous_matrix))


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
        return self.linear_component

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

    @property
    def n_parameters(cls):
        """
        1 parameter - [theta] - The angle of rotation around [0, 0, 1]
        """
        return 1

    def as_vector(self):
        """
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. The parameters
        are output in the order [theta].
        """
        return self.axis_and_angle_of_rotation()[1]

    @classmethod
    def from_vector(cls, p):
        # See affine from_vector with regards to classmethod decorator
        return Rotation2D(np.array([[np.cos(p), -np.sin(p)],
                                    [np.sin(p), np.cos(p)]]))

    @classmethod
    def _estimate(cls, source, target):
        homogeneous_matrix = super(Rotation2D, cls)._estimate(source, target)
        similarity = SimilarityTransform(homogeneous_matrix)
        r1, s, r2, t = similarity.decompose()
        return r1.compose(r2).homogeneous_matrix[:-1, :-1]


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

    @property
    def n_parameters(self):
        """
        Not currently implemented
        """
        raise NotImplementedError('3D rotations do not support vectorisation '
                                  'yet.')

    def as_vector(self):
        """
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. The parameters
        are output in the order [TODO: fill me in].
        """
        # TODO: Implement 3D rotation vectorisation
        raise NotImplementedError('3D rotations do not support vectorisation '
                                  'yet.')

    @classmethod
    def from_vector(cls, p):
        # See affine from_vector with regards to classmethod decorator
        # TODO: Implement 3D rotation vectorisation
        raise NotImplementedError('3D rotations do not support vectorisation '
                                  'yet.')


def Scale(scale_factor, n_dim=None):
    """
    Factory function for producing Scale transforms. Zero scale factors are not
    permitted

    A UniformScale will be produced if:
        - A float scale_factor and a n_dim kwarg are provided
        - A ndarray scale_factor with shape (n_dim, ) is provided with all
    elements being the same

    A NonUniformScale will be provided if:
        - A ndarray scale_factor with shape (n_dim, ) is provided with at least
        two differing scale factors.

    :param scale_factor: Either an ndarray of scales for each dimensions or a
        single scalar value to be applied across each dimension
    :param n_dim: The dimensionality of the output transform
    :raises: ValueError if any of the scale factors is zero
    :returns: Either a UniformScale or a NonUniformScale
    """
    if not np.all(scale_factor):
        raise ValueError('Having a zero in one of the scales is invalid')

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

    @property
    def n_parameters(self):
        """
        n_dim parameters - [scale_x, scale_y, ....] - The scalar values
        representing the scale across each dimension
        """
        return self.scale.shape[0]

    def as_vector(self):
        """
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. The parameters
        are output in the order [sx, sy, ...].
        """
        return self.scale

    @classmethod
    def from_vector(cls, p):
        # See affine from_vector with regards to classmethod decorator
        return NonUniformScale(p)

    @classmethod
    def _estimate(cls, source, target):
        homogeneous_matrix = super(Translation, cls)._estimate(source, target)
        n_dim = source[1]
        if n_dim == 2:
            scale = np.zeros((1, 2))
            scale[0] = homogeneous_matrix[0, 0]
            scale[1] = homogeneous_matrix[1, 1]
        elif n_dim == 3:
            scale = np.zeros((1, 3))
            scale[0] = homogeneous_matrix[0, 0]
            scale[1] = homogeneous_matrix[1, 1]
            scale[2] = homogeneous_matrix[2, 2]
        return scale


def UniformScale(scale, n_dim):
    """
    Factory function for producing UniformScale objects. A single scale and the
    number of dimensions required is expected. Currently, only 2D and 3D
    Uniform Scale objects are supported.

    :param scale: A scalar value representing the scale across each axis
    :param n_dim: The number of dimensions for the transform
    :return: Either a
        :class:`UniformScale2D <pybug.transform.affine.UniformScale2D>` or a
        :class:`UniformScale3D <pybug.transform.affine.UniformScale3D>`
    """
    if n_dim == 2:
        return UniformScale2D(scale)
    elif n_dim == 3:
        return UniformScale3D(scale)
    else:
        raise DimensionalityError('Only 2D or 3D UniformScale transforms are '
                                  'currently supported.')


class AbstractUniformScale(DiscreteAffineTransform, SimilarityTransform):
    """
    An abstract similarity scale transform, with a single scale component
    applied to all dimensions.
    """

    __metaclass__ = abc.ABCMeta

    @property
    def scale(self):
        return self.homogeneous_matrix.diagonal()[0]

    def _transform_str(self):
        message = 'UniformScale by %f ' % self.scale
        return message

    @property
    def n_parameters(self):
        """
        1 parameter - scale - The scalar value representing the scale across
        each dimension
        """
        return 1

    def as_vector(self):
        """
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. The parameters
        are output in the order [s].
        """
        return self.scale


class UniformScale2D(AbstractUniformScale):
    """
    An 2D similarity scale transform, with a single scale component
    applied to all dimensions.

    :param scale: A scaler value indicating the scale across each axis
    """

    def __init__(self, scale):
        homogeneous_matrix = np.eye(3)
        np.fill_diagonal(homogeneous_matrix, scale)
        homogeneous_matrix[-1, -1] = 1
        super(UniformScale2D, self).__init__(homogeneous_matrix)

    @property
    def inverse(self):
        return UniformScale2D(1.0/self.scale)

    @classmethod
    def from_vector(cls, p):
        # See affine from_vector with regards to classmethod decorator
        return UniformScale2D(p)

    @classmethod
    def _estimate(cls, source, target):
        homogeneous_matrix = super(UniformScale2D, cls)._estimate(source,
                                                                  target)
        scale = homogeneous_matrix[0]
        return scale


class UniformScale3D(AbstractUniformScale):
    """
    An 3D similarity scale transform, with a single scale component
    applied to all dimensions.

    :param scale: A scaler value indicating the scale across each axis
    """

    def __init__(self, scale):
        homogeneous_matrix = np.eye(4)
        np.fill_diagonal(homogeneous_matrix, scale)
        homogeneous_matrix[-1, -1] = 1
        super(UniformScale3D, self).__init__(homogeneous_matrix)

    @property
    def inverse(self):
        return UniformScale3D(1.0/self.scale)

    @classmethod
    def from_vector(cls, p):
        # See affine from_vector with regards to classmethod decorator
        return UniformScale3D(p)


class Translation(DiscreteAffineTransform, SimilarityTransform):
    """
    An ``n_dim`` translation transform.
    """

    def __init__(self, translation):
        r"""
        Creates a translation transformation object. Expects a translation
        vector of length ``n_dim - 1``.

        :param translation: a 1D vector of length n_dim (i.e.
            if you want to make a 3d translation you must specify the
            translation in each dimension explicitly)
        :type translation: ndarray [``n_dim``]
        """
        homogeneous_matrix = np.eye(translation.shape[0] + 1)
        homogeneous_matrix[:-1, -1] = translation
        super(Translation, self).__init__(homogeneous_matrix)

    @property
    def inverse(self):
        return Translation(-self.translation_component)

    def _transform_str(self):
        message = 'Translate by %s ' % self.translation_component
        return message

    @property
    def n_parameters(self):
        """
        n_dim parameters - [tx, ty, ...] - The translation along each axis
        :return:
        """
        return self.n_dim

    def as_vector(self):
        """
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. The parameters
        are output in the order [tx, ty].
        """
        return self.homogeneous_matrix[:self.n_dim, self.n_dim]

    @classmethod
    def from_vector(cls, p):
        # See affine from_vector with regards to classmethod decorator
        return Translation(p)

    @classmethod
    def _estimate(cls, source, target):
        homogeneous_matrix = super(Translation, cls)._estimate(source, target)
        translation = homogeneous_matrix[:-1, -1]
        return translation
