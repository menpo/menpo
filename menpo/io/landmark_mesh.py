from menpo.io.landmark import PTSImporter
import numpy as np


class MeshPTSImporter(PTSImporter):
    r"""
    Implements the :meth:`_build_points` method for meshes. Here, `x` is the
    first axis.

    Parameters
    ----------
    filepath : string
        Absolute filepath of the landmarks
    """

    def __init__(self, filepath):
        super(MeshPTSImporter, self).__init__(filepath)

    def _build_points(self, xs, ys):
        """
        For meshes, `axis 0 = xs` and `axis 1 = ys`. Therefore, return the
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
            Array with `xs` as the first axis: `[xs; ys]`
        """
        return np.hstack([xs, ys])
