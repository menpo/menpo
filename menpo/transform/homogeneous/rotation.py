import abc
import numpy as np

from .base import HomogFamilyAlignment
from .affine import DiscreteAffine
from .similarity import Similarity


def optimal_rotation_matrix(source, target):
    r"""
    Performs an SVD on the corrolation matrix to find an optimal rotation
    between source and target

    Parameters
    ----------

    source: :class:`menpo.shape.PointCloud`
        The source points to be aligned

    target: :class:`menpo.shape.PointCloud`
        The target points to be aligned

    Returns
    -------

    ndarray
        The optimal square rotation matrix
    """
    correlation = np.dot(target.points.T, source.points)
    U, D, Vt = np.linalg.svd(correlation)
    return np.dot(U, Vt)


# TODO build rotations about axis, euler angles etc
# see http://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
# for details

class Rotation(DiscreteAffine, Similarity):
    r"""
    Abstract `n_dims` rotation transform.

    Parameters
    ----------
    rotation_matrix : (D, D) `ndarray`
        A valid, square rotation matrix
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, rotation_matrix, skip_checks=False):
        h_matrix = np.eye(rotation_matrix.shape[0] + 1)
        Similarity.__init__(self, h_matrix, copy=False, skip_checks=True)
        self.set_rotation_matrix(rotation_matrix, skip_checks=skip_checks)

    @classmethod
    def from_2d_ccw_angle(cls, theta, degrees=True):
        r"""
        Convenience constructor for 2D CCW rotations about the origin

        Parameters
        ----------
        theta : `float`
            The angle of rotation about the origin
        degrees : `bool`, optional
            If ``True`` theta is interpreted as a degree. If ``False``, theta is
            interpreted as radians.

        Returns
        -------
        rotation : :map:`Rotation`
            A 2D rotation transform.
        """
        if degrees:
            # convert to radians
            theta = theta * np.pi / 180.0
        return Rotation(np.array([[np.cos(theta), -np.sin(theta)],
                                  [np.sin(theta),  np.cos(theta)]]))

    @classmethod
    def identity(cls, n_dims):
        return Rotation(np.eye(n_dims))

    @property
    def rotation_matrix(self):
        r"""
        The rotation matrix.

        :type: (D, D) `ndarray`
        """
        return self.linear_component

    def set_rotation_matrix(self, value, skip_checks=False):
        if not skip_checks:
            shape = value.shape
            if len(shape) != 2 and shape[0] != shape[1]:
                raise ValueError("You need to provide a square rotation matrix")
            # The update better be the same size
            elif self.n_dims != shape[0]:
                raise ValueError("Trying to update the rotation "
                                 "matrix to a different dimension")
            # TODO actually check I am a valid rotation
            # TODO slightly dodgy here accessing _h_matrix
        self._h_matrix[:-1, :-1] = value

    def _transform_str(self):
        axis, rad_angle_of_rotation = self.axis_and_angle_of_rotation()
        if axis is None:
            return "NO OP"
        angle_of_rot = (rad_angle_of_rotation * 180.0) / np.pi
        message = ('CCW Rotation of {:.1f} degrees '
                   'about {}'.format(angle_of_rot,axis))
        return message

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
        if self.n_dims == 2:
            return self._axis_and_angle_of_rotation_2d()
        elif self.n_dims == 3:
            return self._axis_and_angle_of_rotation_3d()

    def _axis_and_angle_of_rotation_2d(self):
        r"""
        Decomposes this Rotation's rotation matrix into a angular rotation
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

    def _axis_and_angle_of_rotation_3d(self):
        r"""
        Decomposes this 3D rotation's rotation matrix into a angular rotation
        about an axis. The rotation is considered in a right handed sense.

        Returns
        -------
        axis : (3,) ndarray
            A unit vector, the axis about which the rotation takes place
        angle_of_rotation : double
            The angle in radians of the rotation about the `axis`.
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
        raise NotImplementedError("Rotations are not yet vectorizable")

    def _as_vector(self):
        r"""
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as deltas from the identity warp. The parameters
        are output in the order [theta].

        +----------+--------------------------------------------+
        |parameter | definition                                 |
        +==========+============================================+
        |theta     | The angle of rotation around `[0, 0, 1]`   |
        +----------+--------------------------------------------+

        Returns
        -------
        theta : double
            Angle of rotation around axis. Right-handed.
        """
        # TODO vectorizable rotations
        raise NotImplementedError("Rotations are not yet vectorizable")

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
        raise NotImplementedError("Rotations are not yet vectorizable")

    @property
    def composes_inplace_with(self):
        return Rotation

    def pseudoinverse(self):
        r"""
        The inverse rotation matrix.

        :type: (D, D) ndarray
        """
        return Rotation(np.linalg.inv(self.rotation_matrix), skip_checks=True)


class AlignmentRotation(HomogFamilyAlignment, Rotation):

    def __init__(self, source, target):
        HomogFamilyAlignment.__init__(self, source, target)
        Rotation.__init__(self, optimal_rotation_matrix(source, target))

    def set_rotation_matrix(self, value, skip_checks=False):
        Rotation.set_rotation_matrix(self, value, skip_checks=skip_checks)
        self._sync_target_from_state()

    def _sync_state_from_target(self):
        r = optimal_rotation_matrix(self.source, self.target)
        Rotation.set_rotation_matrix(self, r, skip_checks=True)

    def as_non_alignment(self):
        r"""Returns a copy of this rotation without it's alignment nature.

        Returns
        -------
        transform : :map:`Rotation`
            A version of this rotation with the same transform behavior but
            without the alignment logic.
        """
        return Rotation(self.rotation_matrix, skip_checks=True)
