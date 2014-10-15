import abc

import numpy as np

from . import PointCloud
from .adjacency import mask_adjacency_array, reindex_adjacency_array
from menpo.visualize import PointGraphViewer


class Graph(object):
    r"""
    Abstract class for Graph definitions and manipulation.

    Parameters
    -----------
    adjacency_array : ``(n_edges, 2, )`` `ndarray`
        The Adjacency Array of the graph, i.e. an array containing the sets of
        the graph's edges. The numbering of vertices is assumed to start from 0.

        For an undirected graph, the order of an edge's vertices doesn't matter,
        for example:
               |---0---|        adjacency_array = ndarray([[0, 1],
               |       |                                   [0, 2],
               |       |                                   [1, 2],
               1-------2                                   [1, 3],
               |       |                                   [2, 4],
               |       |                                   [3, 4],
               3-------4                                   [3, 5]])
               |
               5

        For a directed graph, we assume that the vertices in the first column of
        the adjacency_array are the fathers and the vertices in the second
        column of the adjacency_array are the children, for example:
               |-->0<--|        adjacency_array = ndarray([[1, 0],
               |       |                                   [2, 0],
               |       |                                   [1, 2],
               1<----->2                                   [2, 1],
               |       |                                   [1, 3],
               v       v                                   [2, 4],
               3------>4                                   [3, 4],
               |                                           [3, 5]])
               v
               5

    copy : `bool`, optional
        If ``False``, the ``adjacency_list`` will not be copied on assignment.

    Raises
    ------
    ValueError
        Adjacency list must contain the sets of connected edges and thus must
        have shape (n_edges, 2).
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, adjacency_array, copy=True):
        if adjacency_array.shape[1] != 2:
            raise ValueError('Adjacency list must contain the sets of '
                             'connected edges and thus must have shape '
                             '(n_edges, 2).')

        if copy:
            self.adjacency_array = adjacency_array.copy()
        else:
            self.adjacency_array = adjacency_array
        self.adjacency_list = self._get_adjacency_list()

    @property
    def n_edges(self):
        r"""
        Returns the number of the graph's edges.

        :type: `int`
        """
        return self.adjacency_array.shape[0]

    @property
    def n_vertices(self):
        r"""
        Returns the number of the graph's vertices.

        :type: `int`
        """
        return self.adjacency_array.max() + 1

    @abc.abstractmethod
    def get_adjacency_matrix(self):
        r"""
        Returns the Adjacency Matrix of the graph, i.e. the boolean ndarray that
        with True and False if there is an edge connecting the two vertices or
        not respectively.

        :type: ``(n_vertices, n_vertices, )`` `ndarray`
        """
        pass

    @abc.abstractmethod
    def _get_adjacency_list(self):
        r"""
        Returns the Adjacency List of the graph, i.e. a list of length
        n_vertices that for each vertex has a list of the vertex neighbours.
        If the graph is directed, the neighbours are children.

        :type: `list` of `lists` of len n_vertices
        """
        pass

    def find_path(self, start, end, path=[]):
        r"""
        Returns a list with the first path (without cycles) found from start
        vertex to end vertex.

        :type: `list`
        """
        path = path + [start]
        if start == end:
            return path
        if start > self.n_vertices - 1 or start < 0:
            return None
        for v in self.adjacency_list[start]:
            if v not in path:
                newpath = self.find_path(v, end, path)
                if newpath:
                    return newpath
        return None

    def find_all_paths(self, start, end, path=[]):
        r"""
        Returns a list of lists with all the paths (without cycles) found from
        start vertex to end vertex.

        :type: `list` of `list`
        """
        path = path + [start]
        if start == end:
            return [path]
        if start > self.n_vertices - 1 or start < 0:
            return []
        paths = []
        for v in self.adjacency_list[start]:
            if v not in path:
                newpaths = self.find_all_paths(v, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths

    def n_paths(self, start, end):
        r"""
        Returns the number of all the paths (without cycles) existing from
        start vertex to end vertex.

        :type: `int`
        """
        return len(self.find_all_paths(start, end))

    def find_shortest_path(self, start, end, path=[]):
        r"""
        Returns a list with the shortest path (without cycles) found from start
        vertex to end vertex.

        :type: `list`
        """
        path = path + [start]
        if start == end:
            return path
        if start > self.n_vertices - 1 or start < 0:
            return None
        shortest = None
        for v in self.adjacency_list[start]:
            if v not in path:
                newpath = self.find_shortest_path(v, end, path)
                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
        return shortest

    def tojson(self):
        r"""
        Convert the graph to a dictionary JSON representation.

        Returns
        -------
        dictionary with 'adjacency_array' key. Suitable or use in the by the
        `json` standard library package.
        """
        return {'adjacency_array': self.adjacency_array.tolist()}


class UndirectedGraph(Graph):
    r"""
    Class for Undirected Graph definition and manipulation.
    """
    def get_adjacency_matrix(self):
        adjacency_mat = np.zeros((self.n_vertices, self.n_vertices),
                                 dtype=np.bool)
        for e in range(self.n_edges):
            v1 = self.adjacency_array[e, 0]
            v2 = self.adjacency_array[e, 1]
            adjacency_mat[v1, v2] = True
            adjacency_mat[v2, v1] = True

        return adjacency_mat

    def _get_adjacency_list(self):
        adjacency_list = [[] for _ in range(self.n_vertices)]
        for e in range(self.n_edges):
            v1 = self.adjacency_array[e, 0]
            v2 = self.adjacency_array[e, 1]
            adjacency_list[v1].append(v2)
            adjacency_list[v2].append(v1)
        return adjacency_list

    def neighbours(self, vertex):
        r"""
        Returns the neighbours of the selected vertex.

        :type: `list`
        """
        return self.adjacency_list[vertex]


class DirectedGraph(Graph):
    r"""
    Class for Directed Graph definition and manipulation.
    """
    def get_adjacency_matrix(self):
        adjacency_mat = np.zeros((self.n_vertices, self.n_vertices),
                                 dtype=np.bool)
        for e in range(self.n_edges):
            father = self.adjacency_array[e, 0]
            child = self.adjacency_array[e, 1]
            adjacency_mat[father, child] = True
        return adjacency_mat

    def _get_adjacency_list(self):
        adjacency_list = [[] for _ in range(self.n_vertices)]
        for e in range(self.n_edges):
            v1 = self.adjacency_array[e, 0]
            v2 = self.adjacency_array[e, 1]
            adjacency_list[v1].append(v2)
        return adjacency_list

    def children(self, vertex):
        r"""
        Returns the children of the selected vertex.

        :type: `list`
        """
        return self.adjacency_list[vertex]


class Tree(DirectedGraph):
    r"""
    Class for Tree definitions and manipulation.

    Parameters
    -----------
    adjacency_array : ``(n_edges, 2, )`` `ndarray`
        The Adjacency Array of the graph, i.e. an array containing the sets of
        the tree's edges. The numbering of vertices is assumed to start from 0.

        We assume that the vertices in the first column of the adjacency_array
        are the fathers and the vertices in the second column of the
        adjacency_array are the children, for example:

                   0            adjacency_array = ndarray([[0, 1],
                   |                                       [0, 2],
                ___|___                                    [1, 3],
               1       2                                   [1, 4],
               |       |                                   [2, 5],
              _|_      |                                   [3, 6],
             3   4     5                                   [4, 7],
             |   |     |                                   [5, 8]])
             |   |     |
             6   7     8

    root_vertex : `int`
        The vertex that will be considered as root.

    copy : `bool`, optional
        If ``False``, the ``adjacency_list`` will not be copied on assignment.

    Raises
    ------
    ValueError
        The provided edges do not represent a tree.
    ValueError
        The root_vertex must be between 0 and n_vertices-1.
    """
    def __init__(self, adjacency_array, root_vertex, copy=True):
        super(Tree, self).__init__(adjacency_array, copy=copy)
        if not self.n_edges == self.n_vertices - 1:
            raise ValueError('The provided edges do not represent a tree.')
        if root_vertex > self.n_vertices - 1 or root_vertex < 0:
            raise ValueError('The root_vertex must be between '
                             '0 and {}.'.format(self.n_vertices-1))
        self.root_vertex = root_vertex
        self.predecessors_list = self._get_predecessors_list()

    def _get_predecessors_list(self):
        r"""
        Returns the Predecessors List of the tree, i.e. a list of length
        n_vertices that for each vertex it has its parent. The value of the
        root vertex is None.

        :type: `list` of len n_vertices
        """
        predecessors_list = [None] * self.n_vertices
        for e in range(self.n_edges):
            v1 = self.adjacency_array[e, 0]
            v2 = self.adjacency_array[e, 1]
            predecessors_list[v2] = v1
        return predecessors_list

    def depth_of_vertex(self, vertex):
        r"""
        Returns the depth of the specified vertex.

        :type: `int`
        """
        parent = vertex
        depth = 0
        while not parent == self.root_vertex:
            current = parent
            parent = self.predecessors_list[current]
            depth += 1
        return depth

    @property
    def maximum_depth(self):
        r"""
        Returns the maximum depth of the tree.

        :type: `int`
        """
        max_depth = -1
        for v in range(self.n_vertices):
            max_depth = max([max_depth, self.depth_of_vertex(v)])
        return max_depth

    def vertices_at_depth(self, depth):
        r"""
        Returns a list of vertices at the specified depth.

        :type: `list` of `int`
        """
        ver = []
        for v in range(self.n_vertices):
            if self.depth_of_vertex(v) == depth:
                ver.append(v)
        return ver

    def n_vertices_at_depth(self, depth):
        r"""
        Returns the number of vertices at the specified depth.

        :type: `int`
        """
        n_ver = 0
        for v in range(self.n_vertices):
            if self.depth_of_vertex(v) == depth:
                n_ver += 1
        return n_ver

    def is_leaf(self, vertex):
        r"""
        Returns whether the vertex is a leaf.

        :type: `bool`
        """
        return len(self.children(vertex)) == 0

    def leafs(self):
        r"""
        Returns a list with the leafs of the tree.

        :type: `int`
        """
        leafs = []
        for v in range(self.n_vertices):
            if self.is_leaf(v):
                leafs.append(v)
        return leafs

    def n_leafs(self):
        r"""
        Returns the number of leafs of the tree.

        :type: `int`
        """
        n_leafs = 0
        for v in range(self.n_vertices):
            if self.is_leaf(v):
                n_leafs += 1
        return n_leafs


class PointGraph(UndirectedGraph, PointCloud):

    def __init__(self, points, adjacency_array, copy=True):
        UndirectedGraph.__init__(self, adjacency_array, copy=copy)
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

    def tojson(self):
        r"""
        Convert this `PointGraph` to a dictionary JSON representation.

        Returns
        -------
        dictionary with 'points' and 'adjacency_array' keys. Both are lists
        suitable or use in the by the `json` standard library package.
        """
        json_dict = PointCloud.tojson(self)
        json_dict.update(Graph.tojson(self))
        return json_dict

    def _view(self, figure_id=None, new_figure=False, **kwargs):
        return PointGraphViewer(figure_id, new_figure,
                                self.points,
                                self.adjacency_array).render(**kwargs)
