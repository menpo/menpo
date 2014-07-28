import numpy as np

from .landmark import ASFImporter, PTSImporter


class ImageASFImporter(ASFImporter):
    r"""
    Implements the :meth:`_build_points` method for images. Here, `y` is the
    first axis.

    Parameters
    ----------
    filepath : string
        Absolute filepath of the landmarks
    """

    def __init__(self, filepath):
        super(ImageASFImporter, self).__init__(filepath)

    def _build_points(self, xs, ys):
        """
        For images, `axis 0 = ys` and `axis 1 = xs`. Therefore, return the
        appropriate points array ordering.

        Parameters
        ----------
        xs : (N,) ndarray
            Row vector of `x` coordinates
        ys : (N,) ndarray
            Row vector of `y` coordinates

        Returns
        -------
        points : (N, 2) ndarray
            Array with `ys` as the first axis: `[ys; xs]`
        """
        return np.hstack([ys, xs])


class ImagePTSImporter(PTSImporter):
    r"""
    Implements the :meth:`_build_points` method for images. Here, `y` is the
    first axis.

    Parameters
    ----------
    filepath : string
        Absolute filepath of the landmarks
    """

    def __init__(self, filepath):
        super(ImagePTSImporter, self).__init__(filepath)

    def _build_points(self, xs, ys):
        """
        For images, `axis 0 = ys` and `axis 1 = xs`. Therefore, return the
        appropriate points array ordering.

        Parameters
        ----------
        xs : (N,) ndarray
            Row vector of `x` coordinates
        ys : (N,) ndarray
            Row vector of `y` coordinates

        Returns
        -------
        points : (N, 2) ndarray
            Array with `ys` as the first axis: `[ys; xs]`
        """
        return np.hstack([ys, xs])
