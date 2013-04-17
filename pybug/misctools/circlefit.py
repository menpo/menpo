import numpy as np
from scipy import optimize


def circle_fit(coords):
    """ Adapted from:
    http://www.scipy.org/Cookbook/Least_Squares_Circle
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

#coords = np.array([[ 36.,  14.],
#                   [ 36.,  10.],
#                   [ 19.,  28.],
#                   [ 18.,  31.],
#                   [ 33.,  18.],
#                   [ 26.,  26.]])
#
