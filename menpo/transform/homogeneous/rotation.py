# Parts of this code taken from:
#
# Copyright (c) 2006-2015, Christoph Gohlke
# Copyright (c) 2006-2015, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
import numpy as np

from .base import HomogFamilyAlignment
from .affine import DiscreteAffine
from .similarity import Similarity


def optimal_rotation_matrix(source, target, allow_mirror=False):
    r"""
    Performs an SVD on the correlation matrix to find an optimal rotation
    between `source` and `target`.

    Parameters
    ----------
    source: :map:`PointCloud`
        The source points to be aligned
    target: :map:`PointCloud`
        The target points to be aligned
    allow_mirror : `bool`, optional
        If ``True``, the Kabsch algorithm check is not performed, and mirroring
        of the Rotation matrix is permitted.

    Returns
    -------
    rotation : `ndarray`
        The optimal square rotation matrix.
    """
    correlation = np.dot(target.points.T, source.points)
    U, D, Vt = np.linalg.svd(correlation)
    R = np.dot(U, Vt)

    if not allow_mirror:
        # d = sgn(det(V * Ut))
        d = np.sign(np.linalg.det(R))
        if d < 0:
            E = np.eye(U.shape[0])
            E[-1, -1] = d
            # R = U * E * Vt, E = [[1, 0, 0], [0, 1, 0], [0, 0, d]] for 2D
            R = np.dot(U, np.dot(E, Vt))
    return R


# TODO build rotations about axis, euler angles etc
# see http://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
# for details

class Rotation(DiscreteAffine, Similarity):
    r"""
    Abstract `n_dims` rotation transform.

    Parameters
    ----------
    rotation_matrix : ``(n_dims, n_dims)`` `ndarray`
        A valid, square rotation matrix
    skip_checks : `bool`, optional
        If ``True`` avoid sanity checks on ``rotation_matrix`` for performance.
    """

    def __init__(self, rotation_matrix, skip_checks=False):
        h_matrix = np.eye(rotation_matrix.shape[0] + 1)
        Similarity.__init__(self, h_matrix, copy=False, skip_checks=True)
        self.set_rotation_matrix(rotation_matrix, skip_checks=skip_checks)

    @classmethod
    def init_identity(cls, n_dims):
        r"""
        Creates an identity transform.

        Parameters
        ----------
        n_dims : `int`
            The number of dimensions.

        Returns
        -------
        identity : :class:`Rotation`
            The identity matrix transform.
        """
        return Rotation(np.eye(n_dims))

    @classmethod
    def init_from_2d_ccw_angle(cls, theta, degrees=True):
        r"""
        Convenience constructor for 2D CCW rotations about the origin.

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
            theta = np.deg2rad(theta)
        return Rotation(np.array([[np.cos(theta), -np.sin(theta)],
                                  [np.sin(theta),  np.cos(theta)]]),
                        skip_checks=True)

    @classmethod
    def init_3d_from_quaternion(cls, q):
        r"""
        Convenience constructor for 3D rotations based on quaternion parameters.

        Parameters
        ----------
        q : ``(4,)`` `ndarray`
            The quaternion parameters.

        Returns
        -------
        rotation : :map:`Rotation`
            A 3D rotation transform.
        """
        r = cls.init_identity(n_dims=3)
        return r.from_vector(q)

    @classmethod
    def init_from_3d_ccw_angle_around_x(cls, theta, degrees=True):
        r"""
        Convenience constructor for 3D CCW rotations around the x axis

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
            A 3D rotation transform.
        """
        if degrees:
            theta = np.deg2rad(theta)
        return Rotation(np.array([[ 1,             0,              0],
                                  [ 0, np.cos(theta), -np.sin(theta)],
                                  [ 0, np.sin(theta), np.cos(theta)]]),
                        skip_checks=True)

    @classmethod
    def init_from_3d_ccw_angle_around_y(cls, theta, degrees=True):
        r"""
        Convenience constructor for 3D CCW rotations around the y axis

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
            A 3D rotation transform.
        """
        if degrees:
            theta = np.deg2rad(theta)
        return Rotation(np.array([[ np.cos(theta), 0, np.sin(theta)],
                                  [             0, 1,             0],
                                  [-np.sin(theta), 0, np.cos(theta)]]),
                        skip_checks=True)

    @classmethod
    def init_from_3d_ccw_angle_around_z(cls, theta, degrees=True):
        r"""
        Convenience constructor for 3D CCW rotations around the z axis

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
            A 3D rotation transform.
        """
        if degrees:
            theta = np.deg2rad(theta)
        return Rotation(np.array([[ np.cos(theta), -np.sin(theta), 0],
                                  [ np.sin(theta),  np.cos(theta), 0],
                                  [             0,              0, 1]]),
                        skip_checks=True)

    @property
    def rotation_matrix(self):
        r"""
        The rotation matrix.

        :type: ``(n_dims, n_dims)`` `ndarray`
        """
        return self.linear_component

    def set_rotation_matrix(self, value, skip_checks=False):
        r"""
        Sets the rotation matrix.

        Parameters
        ----------
        value : ``(n_dims, n_dims)`` `ndarray`
            The new rotation matrix.
        skip_checks : `bool`, optional
            If ``True`` avoid sanity checks on ``value`` for performance.
        """
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
        axis, radians_of_rotation = self.axis_and_angle_of_rotation()
        if axis is None:
            return "NO OP"
        degrees_of_rotation = np.rad2deg(radians_of_rotation)
        message = ('CCW Rotation of {:.1f} degrees '
                   'about {}'.format(degrees_of_rotation, axis))
        return message

    def axis_and_angle_of_rotation(self):
        r"""
        Abstract method for computing the axis and angle of rotation.

        Returns
        -------
        axis : ``(n_dims,)`` `ndarray`
            The unit vector representing the axis of rotation
        angle_of_rotation : `float`
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
        definition, `[0, 0, 1]`.

        Returns
        -------
        axis : ``(2,)`` `ndarray`
            The vector representing the axis of rotation
        angle_of_rotation : `float`
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
        axis : ``(3,)`` `ndarray`
            A unit vector, the axis about which the rotation takes place
        angle_of_rotation : `float`
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
        r"""
        Number of parameters of Rotation. Only 3D rotations are currently
        supported.

        Returns
        -------
        n_parameters : `int`
            The transform parameters. Only 3D rotations are currently
            supported which are parametrized with quaternions.

        Raises
        ------
        DimensionalityError, NotImplementedError
            Non-3D Rotations are not yet vectorizable
        """
        if self.n_dims == 3:
            # Quaternion parameters
            return 4
        else:
            raise NotImplementedError("Non-3D Rotations are not yet "
                                      "vectorizable")

    def _as_vector(self):
        r"""
        Return the parameters of the transform as a 1D array. These parameters
        are parametrised as quaternions. Only 3D transforms are currently
        supported.

        Returns
        -------
        q : ``(4,)`` `ndarray`
            The 4 quaternion parameters.

        Raises
        ------
        DimensionalityError, NotImplementedError
            Non-3D Rotations are not yet vectorizable
        """
        if self.n_dims == 3:
            m00 = self.h_matrix[0, 0]
            m01 = self.h_matrix[0, 1]
            m02 = self.h_matrix[0, 2]
            m10 = self.h_matrix[1, 0]
            m11 = self.h_matrix[1, 1]
            m12 = self.h_matrix[1, 2]
            m20 = self.h_matrix[2, 0]
            m21 = self.h_matrix[2, 1]
            m22 = self.h_matrix[2, 2]
            # symmetric matrix K
            K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                          [m01+m10,     m11-m00-m22, 0.0,         0.0],
                          [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                          [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
            K /= 3.0
            # Quaternion is eigenvector of K that corresponds to largest
            # eigenvalue
            w, V = np.linalg.eigh(K)
            q = V[[3, 0, 1, 2], np.argmax(w)]
            if q[0] < 0.0:
                q = -q
            return q
        else:
            raise NotImplementedError("Non-3D Rotations are not yet "
                                      "vectorizable")

    def _from_vector_inplace(self, p):
        r"""
        Returns an instance of the transform from the given parameters
        expressed in quaternions. Currently only 3D rotations are supported.

        Parameters
        ----------
        p : ``(4,)`` `ndarray`
            The array of quaternion parameters.

        Returns
        -------
        transform : :map:`Rotation`
            The transform initialised to the given parameters.

        Raises
        ------
        DimensionalityError, NotImplementedError
            Non-3D Rotations are not yet vectorizable
        ValueError
            Expected 4 quaternion parameters; got {} instead.
        """
        if self.n_dims == 3:
            if len(p) == 4:
                n = np.dot(p, p)
                # epsilon for testing whether a number is close to zero
                if n < np.finfo(float).eps * 4.0:
                    return np.identity(4)
                p = p * np.sqrt(2.0 / n)
                p = np.outer(p, p)
                rotation = np.array(
                    [[1.-p[2, 2]-p[3, 3],    p[1, 2]-p[3, 0],    p[1, 3]+p[2, 0]],
                     [   p[1, 2]+p[3, 0], 1.-p[1, 1]-p[3, 3],    p[2, 3]-p[1, 0]],
                     [   p[1, 3]-p[2, 0],    p[2, 3]+p[1, 0], 1.-p[1, 1]-p[2, 2]]])
                self.set_rotation_matrix(rotation, skip_checks=True)
            else:
                raise ValueError("Expected 4 quaternion parameters; got {} "
                                 "instead.".format(len(p)))
        else:
            raise NotImplementedError("Non-3D rotations are not yet "
                                      "vectorizable")

    @property
    def composes_inplace_with(self):
        r"""
        :class:`Rotation` can swallow composition with any other
        :class:`Rotation`.
        """
        return Rotation

    def pseudoinverse(self):
        r"""
        The inverse rotation matrix.

        :type: :class:`Rotation`
        """
        return Rotation(np.linalg.inv(self.rotation_matrix), skip_checks=True)


class AlignmentRotation(HomogFamilyAlignment, Rotation):
    r"""
    Constructs an :class:`Rotation` by finding the optimal rotation transform to
    align `source` to `target`.

    Parameters
    ----------
    source : :map:`PointCloud`
        The source pointcloud instance used in the alignment
    target : :map:`PointCloud`
        The target pointcloud instance used in the alignment
    allow_mirror : `bool`, optional
        If ``True``, the Kabsch algorithm check is not performed, and mirroring
        of the Rotation matrix is permitted.
    """

    def __init__(self, source, target, allow_mirror=False):
        HomogFamilyAlignment.__init__(self, source, target)
        Rotation.__init__(self, optimal_rotation_matrix(
            source, target, allow_mirror=allow_mirror))
        self.allow_mirror = allow_mirror

    def set_rotation_matrix(self, value, skip_checks=False):
        r"""
        Sets the rotation matrix.

        Parameters
        ----------
        value : ``(n_dims, n_dims)`` `ndarray`
            The new rotation matrix.
        skip_checks : `bool`, optional
            If ``True`` avoid sanity checks on ``value`` for performance.
        """
        Rotation.set_rotation_matrix(self, value, skip_checks=skip_checks)
        self._sync_target_from_state()

    def _sync_state_from_target(self):
        r = optimal_rotation_matrix(self.source, self.target,
                                    allow_mirror=self.allow_mirror)
        Rotation.set_rotation_matrix(self, r, skip_checks=True)

    def as_non_alignment(self):
        r"""
        Returns a copy of this rotation without its alignment nature.

        Returns
        -------
        transform : :map:`Rotation`
            A version of this rotation with the same transform behavior but
            without the alignment logic.
        """
        return Rotation(self.rotation_matrix, skip_checks=True)
