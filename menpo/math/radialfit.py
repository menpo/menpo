import numpy as np
from collections import namedtuple

optimise = None  # expensive, from scipy

RadialFitResult = namedtuple('RadialFitResult', ['centre', 'radius'])


def radial_fit(p):
    """
    Find the least squares radial fitting a set of ND points.

    Parameters
    ----------
    p : ``(N, D)`` `ndarray`
        Points to find find centre of

    Returns
    -------
    centre_i : (D,) `ndarray`
        The ND coordinates of the centre of the circle.
    r_i : `float`
        The radius of the circle.

    References
    ----------
    .. [1] http://www.scipy.org/Cookbook/Least_Squares_Circle

    """
    global optimise
    if optimise is None:
        from scipy import optimize  # expensive

    def error(tuple_c, x):
        c = np.array(tuple_c)
        err = r(x, c)
        return err - err.mean()

    r = lambda x, c: np.sqrt(np.sum((x - c) ** 2, axis=1))
    av_r = lambda x, c: np.mean(r(x, c))
    c_est = np.mean(p, axis=0)
    c_i, ier = optimize.leastsq(error, c_est, args=p)
    return RadialFitResult(centre=c_i, radius=av_r(p, c_i))
