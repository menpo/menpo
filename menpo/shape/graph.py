import numpy as np

from . import PointCloud
from .adjacency import mask_adjacency_array, reindex_adjacency_array
from ..visualize.base import PointGraphViewer


class Graph(object):

    def __init__(self, adjacency_array, copy=True):
        if adjacency_array.shape[1] != 2:
            raise ValueError('Adjacency array must contain the sets of '
                             'connected edges and thus must have shape '
                             '(n_edges, 2).')

        if copy:
            self.adjacency_array = adjacency_array.copy()
        else:
            self.adjacency_array = adjacency_array

    @property
    def n_edges(self):
        return self.adjacency_array.shape[0]


class PointGraph(Graph, PointCloud):

    def __init__(self, points, adjacency_array, copy=True):
        Graph.__init__(self, adjacency_array, copy=copy)
        PointCloud.__init__(self, points, copy=copy)

    def from_mask(self, mask):
        """
        A 1D boolean array with the same number of elements as the number of
        points in the PointGraph. This is then broadcast across the dimensions
        of the PointGraph and returns a new PointGraph containing only those
        points that were ``True`` in the mask.

        Parameters
        ----------
        mask : ``(n_points,)`` `ndarray`
            1D array of booleans

        Returns
        -------
        pointgraph : :map:`PointGraph`
            A new pointgraph that has been masked.

        Raises
        ------
        ValueError
            Mask must have same number of points as pointgraph.
        """
        if mask.shape[0] != self.n_points:
            raise ValueError('Mask must be a 1D boolean array of the same '
                             'number of entries as points in this PointGraph.')

        pg = self.copy()
        if np.all(mask):  # Shortcut for all true masks
            return pg
        else:
            masked_adj = mask_adjacency_array(mask, pg.adjacency_array)
            pg.adjacency_array = reindex_adjacency_array(masked_adj)
            pg.points = pg.points[mask, :]
            return pg

    def _view(self, figure_id=None, new_figure=False, **kwargs):
        return PointGraphViewer(figure_id, new_figure,
                                self.points,
                                self.adjacency_array).render(**kwargs)
