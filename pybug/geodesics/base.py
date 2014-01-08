import collections
from pybug.geodesics import kirsanov
from pybug.geodesics.exceptions import TriMeshGeodesicsError


class TriMeshGeodesics(object):
    r"""
    A number of geodesics algorithms for TriMeshes, including
    an exact geodesic method.

    Parameters
    ----------
    points : (N, D) ndarray
        The cartesian points that make up the mesh
    trilist : (M, 3) ndarray
        The triangulation of the given points
    """

    def __init__(self, points, trilist):
        self._kirsanov = kirsanov.KirsanovGeodesics(points, trilist)
        self.points = points
        self.trilist = trilist

    @property
    def n_points(self):
        r"""
        The number of points in the mesh

        :type: (N, D) ndarray
        """
        return self.points.shape[0]

    def geodesics(self, source_indexes, method='exact'):
        r"""
        Calculate the geodesic distance for all points from the
        given ``source_indexes``.

        Parameters
        -----------
        source_indexes : (N,) list
            List of indexes to calculate the geodesics for
        method : {'exact'}
            The method using to calculate the geodesics. Only the 'exact'
            method is currently supported

            Default: exact

        Returns
        -------
        TODO: document these?
        phi : UNKNOWN
        best_source : UNKNOWN

        Raises
        -------
        TriMeshGeodesicsError
            When indexes are out of the range of the number of points
        """
        if not isinstance(source_indexes, collections.Iterable):
            source_indexes = [source_indexes]
        if not all(0 <= i < self.n_points for i in source_indexes):
            raise TriMeshGeodesicsError('Invalid indexes ' +
                                        '(all must be in range  '
                                        '0 <= i < n_points)')
        return self._kirsanov.geodesics(source_indexes, method)
