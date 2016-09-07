import numpy as np

from .homogeneous import Translation, UniformScale, Rotation, Similarity, Affine


def scale_about_centre(obj, scale):
    r"""
    Return a Homogeneous Transform that implements scaling an object about
    its centre. The given object must be transformable and must implement
    a method to provide the object centre.

    Parameters
    ----------
    obj : :map:`Transformable`
        A transformable object that has the ``centre`` method.
    scale : `float` or ``(n_dims,)`` `ndarray`
        The scale factor as defined in the :map:`Scale` documentation.

    Returns
    -------
    transform : :map:`Homogeneous`
        A homogeneous transform that implements the scaling.
    """
    rescale = Similarity.init_identity(obj.n_dims)

    s = UniformScale(scale, obj.n_dims, skip_checks=True)
    t = Translation(-obj.centre(), skip_checks=True)
    # Translate to origin, scale, then translate back
    rescale.compose_before_inplace(t)
    rescale.compose_before_inplace(s)
    rescale.compose_before_inplace(t.pseudoinverse())
    return rescale


def rotate_ccw_about_centre(obj, theta, degrees=True):
    r"""
    Return a Homogeneous Transform that implements rotating an object
    counter-clockwise about its centre. The given object must be transformable
    and must implement a method to provide the object centre.

    Parameters
    ----------
    obj : :map:`Transformable`
        A transformable object that has the ``centre`` method.
    theta : `float`
        The angle of rotation clockwise about the origin.
    degrees : `bool`, optional
        If ``True`` theta is interpreted as degrees. If ``False``, theta is
        interpreted as radians.

    Returns
    -------
    transform : :map:`Homogeneous`
        A homogeneous transform that implements the rotation.
    """
    rotate_ccw = Similarity.init_identity(obj.n_dims)

    r = Rotation.init_from_2d_ccw_angle(theta, degrees=degrees)
    t = Translation(-obj.centre(), skip_checks=True)
    # Translate to origin, rotate counter-clockwise, then translate back
    rotate_ccw.compose_before_inplace(t)
    rotate_ccw.compose_before_inplace(r)
    rotate_ccw.compose_before_inplace(t.pseudoinverse())
    return rotate_ccw


def create_2d_shear_transform(phi, psi, degrees=True):
    r"""
    Return a 2D shear Affine Transform.

    Parameters
    ----------
    phi : `float`
        The angle of shearing in the X direction.
    psi : `float`
        The angle of shearing in the Y direction.
    degrees : `bool`, optional
        If ``True``, then phi and psi are interpreted as degrees. If ``False``
        they are interpreted as radians.

    Returns
    -------
    transform : :map:`Affine`
        An affine transform that implements the shearing.
    """
    # Parse angles
    if degrees:
        phi = phi * np.pi / 180.
        psi = psi * np.pi / 180.
    # Create shear matrix
    h_matrix = np.eye(3)
    h_matrix[0, 1] = np.tan(phi)
    h_matrix[1, 0] = np.tan(psi)
    return Affine(h_matrix)


def shear_about_centre(obj, phi, psi, degrees=True):
    r"""
    Return a Homogeneous Transform that implements shearing (distorting) an
    object about its centre. The given object must be transformable and must
    implement a method to provide the object centre.

    Parameters
    ----------
    obj : :map:`Transformable`
        A transformable object that has the ``centre`` method.
    phi : `float`
        The angle of shearing in the X direction.
    psi : `float`
        The angle of shearing in the Y direction.
    degrees : `bool`, optional
        If ``True``, then phi and psi are interpreted as degrees. If ``False``
        they are interpreted as radians.

    Returns
    -------
    transform : :map:`Homogeneous`
        A homogeneous transform that implements the shearing.


    Raises
    ------
    ValueError
        Shearing can only be applied on 2D objects
    """
    if obj.n_dims != 2:
        raise ValueError('Shearing can only be applied on 2D objects')
    # Create shearing and translation transforms
    a = create_2d_shear_transform(phi, psi, degrees=degrees)
    t = Translation(-obj.centre(), skip_checks=True)
    # Translate to origin, skew, then translate back
    shear = Similarity.init_identity(obj.n_dims)
    shear.compose_before_inplace(t)
    shear.compose_before_inplace(a)
    shear.compose_before_inplace(t.pseudoinverse())
    return shear
