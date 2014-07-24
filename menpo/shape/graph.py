from . import PointCloud
from menpo.visualize.base import PointGraphViewer
import numpy as np


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
            # Find the indices that have been asked to be removed
            indices_to_remove = np.nonzero(~mask)[0]
            # Set intersection to find any rows containing those elements,
            # reshape back in to the same size as adjacency array
            entries_to_remove = np.in1d(pg.adjacency_array,
                                        indices_to_remove).reshape([-1, 2])
            # Only keep those entries that are not flagged for removal
            indices_to_keep = ~entries_to_remove.any(axis=1)
            adj = pg.adjacency_array[indices_to_keep, :]

            # Create a mapping vector (reindex the adjacency array)
            remap_vector = np.arange(np.max(adj) + 1)
            unique_values = np.unique(adj)
            remap_vector[unique_values] = np.arange(unique_values.shape[0])

            # Apply the mask
            pg.adjacency_array = remap_vector[adj]
            pg.points = pg.points[mask, :]
            return pg

    def _view(self, figure_id=None, new_figure=False, **kwargs):
        return PointGraphViewer(figure_id, new_figure,
                                self.points,
                                self.adjacency_array).render(**kwargs)
