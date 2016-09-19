from functools import reduce

from .homogeneous import (Translation, UniformScale, Rotation, Affine,
                          Homogeneous)


def transform_about_centre(obj, transform):
    r"""
    Return a Transform that implements transforming an object about
    its centre. The given object must be transformable and must implement
    a method to provide the object centre. More precisely, the object will be
    translated to the origin (according to it's centre), transformed, and then
    translated back to it's previous position.

    Parameters
    ----------
    obj : :map:`Transformable`
        A transformable object that has the ``centre`` method.
    transform : :map:`ComposableTransform`
        A composable transform.

    Returns
    -------
    transform : :map:`Homogeneous`
        A homogeneous transform that implements the scaling.
    """
    to_origin = Translation(-obj.centre(), skip_checks=True)
    back_to_centre = Translation(obj.centre(), skip_checks=True)

    # Fast path - compose in-place in order to ensure only a single matrix
    # is returned
    if isinstance(transform, Homogeneous):
        # Translate to origin, transform, then translate back
        return to_origin.compose_before(transform).compose_before(back_to_centre)
    else:  # Fallback to transform chain
        return reduce(lambda a, b: a.compose_before(b),
                      [to_origin, transform, back_to_centre])


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
    s = UniformScale(scale, obj.n_dims, skip_checks=True)
    return transform_about_centre(obj, s)


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
    if obj.n_dims != 2:
        raise ValueError('CCW rotation is currently only supported for '
                         '2D objects')
    r = Rotation.init_from_2d_ccw_angle(theta, degrees=degrees)
    return transform_about_centre(obj, r)


def shear_about_centre(obj, phi, psi, degrees=True):
    r"""
    Return an affine transform that implements shearing (distorting) an
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
    transform : :map:`Affine`
        An affine transform that implements the shearing.

    Raises
    ------
    ValueError
        Shearing can only be applied on 2D objects
    """
    if obj.n_dims != 2:
        raise ValueError('Shearing is currently only supported for 2D objects')
    s = Affine.init_from_2d_shear(phi, psi, degrees=degrees)
    return transform_about_centre(obj, s)
