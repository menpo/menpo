import numpy as np
from scipy import optimize


def circle_fit(coords):
    """
    Find the least squares circle fitting a set of 2D points `(x,y)`.

    Parameters
    ----------
    coords : (N, 2) ndarray
        Set of `x` and `y` coordinates.

    Returns
    -------
    centre_i : (2,)
        The 2D coordinates of the centre of the circle.
    r_i : double
        The radius of the circle.

    References
    ----------
    .. [1] http://www.scipy.org/Cookbook/Least_Squares_Circle
    """

    def r_sq_of_circle(coords, centre):
        return np.mean(np.sum((coords - centre) ** 2, axis=1))

    def residuals(p, x, y):
        x_c, y_c = p
        err = np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)
        return err - err.mean()

    c_est = np.mean(coords, axis=0)
    #r_sq = r_sq_of_circle(coords, c_est)

    centre_i, ier = optimize.leastsq(residuals, c_est,
                                     args=(coords[:, 0], coords[:, 1]))
    r_i = np.sqrt(r_sq_of_circle(coords, centre_i))
    return centre_i, r_i

