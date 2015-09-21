from .homogeneous import Translation, UniformScale, Rotation, Similarity


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
