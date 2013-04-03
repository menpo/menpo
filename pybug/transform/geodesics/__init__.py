import kirsanov

class TriMeshGeodesicsError(Exception):
    pass

class TriMeshGeodesics(object):
    """A number of geodesics algorithms for TriMeshes, including
    an exact geodesic method.
    """
    def __init__(self, points, trilist):
        self._kirsanov = kirsanov.KirsanovGeodesics(points, trilist)
        self.points = points
        self.trilist = trilist

    @property
    def n_points(self):
        return self.points.shape[0]
        

    def geodesics(self, source_indexes, method='exact'):
        """Calculate the geodesic distance for all points from the 
        source_indexes. kwarg 'method' can be used to choose the algorithm
        (default method='exact' giving exact geodesics)
        """
        if type(source_indexes) == int:
            source_indexes = [source_indexes]
        if not all(i >= 0 and i < self.n_points for i in source_indexes):
            raise TriMeshGeodesicsError('Invalid indexes '\
                    + '(all must be in range  0 <= i < n_points)')
        return self._kirsanov.geodesics(source_indexes, method)

