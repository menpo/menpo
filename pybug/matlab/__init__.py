from numpy import gradient as np_gradient, reshape as np_reshape


def gradient(f, *varargs):
    """
    Return the gradient of an N-dimensional array.

    The gradient is computed using central differences in the interior and
    first differences at the boundaries. The returned gradient hence has the
     same shape as the input array. This matches
    Matlab's functionality, which is quoted as:

    "The first output FX is always the gradient along the 2nd
    dimension of F, going across columns.  The second output FY is always
    the gradient along the 1st dimension of F, going across rows.  For the
    third output FZ and the outputs that follow, the Nth output is the
    gradient along the Nth dimension of F."

    :param f:array_like
        An N-dimensional array containing samples of a scalar function.
    :param varargs: scalars
        0, 1, or N scalars specifying the sample distances in each direction,
        that is: dx, dy, dz, ... The default distance is 1.
    :return: ndarray
        N arrays of the same shape as f giving the derivative of f with respect
         to each dimension. In order to match Matlab,
         the first output is along the second dimension (dF/dx for images)
         and the second output is along the first dimension (dF/dy).
    """
    gradients = np_gradient(f, *varargs)
    if len(f.shape) > 1:
        gradients[:2] = gradients[1::-1]
    return gradients


def reshape(a, newshape):
    """
    Gives a new shape to an array without changing its data. Assumes Fortran
    ordering to match Matlab.

    :param a: array_like
        Array to be reshaped.
    :param newshape: int or tuple of ints
        The new shape should be compatible with the original shape. If an
        integer, then the result will be a 1-D array of that length. One
        shape dimension can be -1. In this case, the value is inferred from
        the length of the array and remaining dimensions.
    :return: ndarray
        This will be a new view object if possible; otherwise, it will be a
        copy.
    """
    return np_reshape(a, newshape, order='F')
