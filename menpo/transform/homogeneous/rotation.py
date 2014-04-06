import abc
import numpy as np

from menpo.exception import DimensionalityError

from .affine import DiscreteAffineTransform
from .similarity import Similarity


def Rotation(rotation_matrix):
    r"""
    Factory function for producing :class:`AbstractRotation` transforms.

    Parameters
    ----------
    rotation_matrix : (D, D) ndarray
        A square legal 2D or 3D rotation matrix

    Returns
    -------
    rotation : :class:`Rotation2D` or :class:`Rotation3D`
        A 2D or 3D rotation transform

    Raises
    ------
    DimensionalityError
        Only 2D and 3D transforms are supported.
    """
    if rotation_matrix.shape[0] == 2:
        return Rotation2D(rotation_matrix)
    elif rotation_matrix.shape[0] == 3:
        return Rotation3D(rotation_matrix)
    else:
        raise DimensionalityError("Can only construct 2D or 3D Rotations")
        # TODO build rotations about axis, euler angles etc
        # see http://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
        # for details.


class AbstractRotation(DiscreteAffineTransform, Similarity):
    r"""
    Abstract ``n_dims`` rotation transform.

    Parameters
    ----------
    rotation_matrix : (D, D) ndarray
        A valid, square rotation matrix
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, rotation_matrix):
        #TODO check that I am a valid rotation
        h_matrix = np.eye(rotation_matrix.shape[0] + 1)
        h_matrix[:-1, :-1] = rotation_matrix
        Similarity.__init__(self, h_matrix)

    @property
    def rotation_matrix(self):
        r"""
        The rotation matrix.

        :type: (D, D) ndarray
        """
        return self.linear_component

    @property
    def inverse(self):
        r"""
        The inverse rotation matrix.

        :type: (D, D) ndarray
        """
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
        r"""
        Abstract method for computing the axis and angle of rotation.

        Returns
        -------
        axis : (D,) ndarray
            The unit vector representing the axis of rotation
        angle_of_rotation : double
            The angle in radians of the rotation about the axis. The angle is
            signed in a right handed sense.
        """
        pass


class Rotation2D(AbstractRotation):
    r"""
    A 2-dimensional rotation. Parametrised by a single parameter, ``theta``,
    which represents the right-handed rotation around ``[0, 0, 1]``.

    Parameters
    ----------
    rotation_matrix : (2, 2) ndarray
        The 2D rotation matrix.

    Raises
    ------
    DimensionalityError
        Only 2D rotation matrices are supported.
    """

    def __init__(self, rotation_matrix):
        super(Rotation2D, self).__init__(rotation_matrix)
        if self.n_dims != 2:
            raise DimensionalityError("Rotation2D has to be built from a 2D"
                                      " rotation matrix")

    def axis_and_angle_of_rotation(self):
        r"""
        Decomposes this 2D rotation's rotation matrix into a angular rotation
        The rotation is considered in a right handed sense. The axis is, by
        definition, [0, 0, 1].

        Returns
        -------
        axis : (2,) ndarray
            The vector representing the axis of rotation
        angle_of_rotation : double
            The angle in radians of the rotation about the axis. The angle is
            signed in a right handed sense.
        """
        axis = np.array([0, 0, 1])
        test_vector = np.array([1, 0])
        transformed_vector = np.dot(self.rotation_matrix,
                                    test_vector)
        angle_of_rotation = np.arccos(np.dot(transformed_vector, test_vector))
        return axis, angle_of_rotation

    @property
    def n_parameters(self):
        r"""
        The number of parameters: 1

        :type: int
        """
        return 1

    def as_vector(self):
        r"""
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. The parameters
        are output in the order [theta].

        +----------+--------------------------------------------+
        |parameter | definition                                 |
        +==========+============================================+
        |theta     | The angle of rotation around ``[0, 0, 1]`` |
        +----------+--------------------------------------------+

        Returns
        -------
        theta : double
            Angle of rotation around axis. Right-handed.
        """
        return self.axis_and_angle_of_rotation()[1]

    def from_vector_inplace(self, p):
        r"""
        Returns an instance of the transform from the given parameters,
        expected to be in Fortran ordering.

        Supports rebuilding from 2D parameter sets.

        2D Rotation: 1 parameter::

            [theta]

        Parameters
        ----------
        p : (1,) ndarray
            The array of parameters.

        Returns
        -------
        transform : :class:`Rotation2D`
            The transform initialised to the given parameters.
        """
        self.h_matrix[:2, :2] = np.array([[np.cos(p), -np.sin(p)],
                                          [np.sin(p), np.cos(p)]])

    @classmethod
    def identity(cls):
        return Rotation2D(np.eye(2))


class Rotation3D(AbstractRotation):
    r"""
    A 3-dimensional rotation. **Currently no parametrisation is implemented**.

    Parameters
    ----------
    rotation_matrix : (D, D) ndarray
        The 3D rotation matrix.

    Raises
    ------
    DimensionalityError
        Only 3D rotation matrices are supported.
    """

    def __init__(self, rotation_matrix):
        super(Rotation3D, self).__init__(rotation_matrix)
        if self.n_dims != 3:
            raise DimensionalityError("Rotation3D has to be built from a 3D"
                                      " rotation matrix")

    def axis_and_angle_of_rotation(self):
        r"""
        Decomposes this 3D rotation's rotation matrix into a angular rotation
        about an axis. The rotation is considered in a right handed sense.

        Returns
        -------
        axis : (3,) ndarray
            A unit vector, the axis about which the rotation takes place
        angle_of_rotation : double
            The angle in radians of the rotation about the ``axis``.
            The angle is signed in a right handed sense.

        References
        ----------
        .. [1] http://en.wikipedia.org/wiki/Rotation_matrix#Determining_the_axis
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
        r"""
        Not yet implemented.

        Raises
        -------
        NotImplementedError
            Not yet implemented.
        """
        # TODO: Implement 3D rotation vectorisation
        raise NotImplementedError('3D rotations do not support vectorisation '
                                  'yet.')

    def as_vector(self):
        r"""
        Not yet implemented.

        Raises
        -------
        NotImplementedError
            Not yet implemented.
        """
        # TODO: Implement 3D rotation vectorisation
        raise NotImplementedError('3D rotations do not support vectorisation '
                                  'yet.')

    def from_vector_inplace(self, p):
        r"""
        Not yet implemented.

        Raises
        -------
        NotImplementedError
            Not yet implemented.
        """
        # See affine from_vector with regards to classmethod decorator
        # TODO: Implement 3D rotation vectorisation
        raise NotImplementedError('3D rotations do not support vectorisation '
                                  'yet.')

    @classmethod
    def identity(cls):
        return Rotation3D(np.eye(3))
