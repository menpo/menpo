import numpy as np
from scipy.sparse import csgraph, csr_matrix, triu

from . import PointCloud


class Graph(object):
    r"""
    Abstract class for Graph definitions and manipulation.

    Parameters
    ----------
    adjacency_matrix : ``(n_vertices, n_vertices, )`` `ndarray` or `csr_matrix`
        The adjacency matrix of the graph in which the rows represent source
        vertices and columns represent destination vertices. The non-edges must
        be represented with zeros and the edges can have a weight value.

        The adjacency matrix of an undirected graph must be symmetric.
    directed : `bool`
        If ``True``, the graph is considered directed. If ``False``, the graph
        is considered undirected.
    copy : `bool`, optional
        If ``False``, the ``adjacency_matrix`` will not be copied on assignment.
    skip_checks : `bool`, optional
        If ``True``, no checks will be performed.

    Raises
    ------
    ValueError
        adjacency_matrix must be either a numpy.ndarray or a
        scipy.sparse.csr_matrix.
    ValueError
        Graph must have at least one vertex.
    ValueError
        adjacency_matrix must be square (n_vertices, n_vertices, ),
        ({adjacency_matrix.shape[0]}, {adjacency_matrix.shape[1]}) given
        instead.
    ValueError
        The adjacency matrix of an undirected graph must be symmetric.

    Examples
    --------
    The adjacency matrix of the following undirected graph ::

        |---0---|
        |       |
        |       |
        1-------2
        |       |
        |       |
        3-------4
        |
        |
        5

    can be defined as ::

        import numpy as np
        adjacency_matrix = np.array([[0, 1, 1, 0, 0, 0],
                                     [1, 0, 1, 1, 0, 0],
                                     [1, 1, 0, 0, 1, 0],
                                     [0, 1, 0, 0, 1, 1],
                                     [0, 0, 1, 1, 0, 0],
                                     [0, 0, 0, 1, 0, 0]])

    or ::

        from scipy.sparse import csr_matrix
        adjacency_matrix = csr_matrix(
                            ([1] * 14,
                             ([0, 1, 0, 2, 1, 2, 1, 3, 2, 4, 3, 4, 3, 5],
                              [1, 0, 2, 0, 2, 1, 3, 1, 4, 2, 4, 3, 5, 3])),
                            shape=(6, 6))


    The adjacency matrix of the following directed graph ::

        |-->0<--|
        |       |
        |       |
        1<----->2
        |       |
        v       v
        3------>4
        |
        v
        5

    can be represented as ::

        import numpy as np
        adjacency_matrix = np.array([[0, 0, 0, 0, 0, 0],
                                     [1, 0, 1, 1, 0, 0],
                                     [1, 1, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 1, 1],
                                     [0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0]])

    or ::

        from scipy.sparse import csr_matrix
        adjacency_matrix = csr_matrix(([1] * 8, ([1, 2, 1, 2, 1, 2, 3, 3],
                                                 [0, 0, 2, 1, 3, 4, 4, 5])),
                                      shape=(6, 6))

    Finally, the adjacency matrix of the following graph with isolated
    vertices ::

            0---|
                |
                |
        1       2
                |
                |
        3-------4


        5

    can be defined as ::

        import numpy as np
        adjacency_matrix = np.array([[0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 1, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 0]])

    or ::

        from scipy.sparse import csr_matrix
        adjacency_matrix = csr_matrix(([1] * 6, ([0, 2, 2, 4, 3, 4],
                                                 [2, 0, 4, 2, 4, 3])),
                                      shape=(6, 6))
    """
    def __init__(self, adjacency_matrix, copy=True, skip_checks=False):
        # check if adjacency_matrix is numpy.ndarray or scipy.sparse.csr_matrix
        if isinstance(adjacency_matrix, np.ndarray):
            # it is numpy.ndarray, thus convert it to scipy.sparse.csr_matrix
            adjacency_matrix = csr_matrix(adjacency_matrix)
        elif not (isinstance(adjacency_matrix, np.ndarray) or
                  isinstance(adjacency_matrix, csr_matrix)):
            raise ValueError('adjacency_matrix must be either a numpy.ndarray'
                             'or a scipy.sparse.csr_matrix.')

        if not skip_checks:
            # check that adjacency_matrix has expected shape
            if adjacency_matrix.shape[0] == 0:
                raise ValueError('Graph must have at least one vertex.')
            elif adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
                raise ValueError('adjacency_matrix must be square '
                                 '(n_vertices, n_vertices, ), ({}, {}) given '
                                 'instead'.format(adjacency_matrix.shape[0],
                                                  adjacency_matrix.shape[1]))

            # check if adjacency matrix of undirected graph is symmetric
            if not self._directed and not _is_symmetric(adjacency_matrix):
                raise ValueError('The adjacency matrix of an undirected graph '
                                 'must be symmetric.')

        # store adjacency_matrix
        if copy:
            self.adjacency_matrix = adjacency_matrix.copy()
        else:
            self.adjacency_matrix = adjacency_matrix

    @classmethod
    def init_from_edges(cls, edges, n_vertices, skip_checks=False):
        r"""
        Initialize graph from edges array.

        Parameters
        ----------
        edges : ``(n_edges, 2, )`` `ndarray`
            The `ndarray` of edges, i.e. all the pairs of vertices that are
            connected with an edge.
        n_vertices : `int`
            The total number of vertices, assuming that the numbering of
            vertices starts from ``0``. ``edges`` and ``n_vertices`` can be
            defined in a way to set isolated vertices.
        skip_checks : `bool`, optional
            If ``True``, no checks will be performed.

        Examples
        --------
        The following undirected graph ::

            |---0---|
            |       |
            |       |
            1-------2
            |       |
            |       |
            3-------4
            |
            |
            5

        can be defined as ::

            from menpo.shape import UndirectedGraph
            import numpy as np
            edges = np.array([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1],
                              [1, 3], [3, 1], [2, 4], [4, 2], [3, 4], [4, 3],
                              [3, 5], [5, 3]])
            graph = UndirectedGraph.init_from_edges(edges, n_vertices=6)


        The following directed graph ::

            |-->0<--|
            |       |
            |       |
            1<----->2
            |       |
            v       v
            3------>4
            |
            v
            5

        can be represented as ::

            from menpo.shape import DirectedGraph
            import numpy as np
            edges = np.array([[1, 0], [2, 0], [1, 2], [2, 1], [1, 3], [2, 4],
                              [3, 4], [3, 5]])
            graph = DirectedGraph.init_from_edges(edges, n_vertices=6)

        Finally, the following graph with isolated vertices ::

                0---|
                    |
                    |
            1       2
                    |
                    |
            3-------4


            5

        can be defined as ::

            from menpo.shape import UndirectedGraph
            import numpy as np
            edges = np.array([[0, 2], [2, 0], [2, 4], [4, 2], [3, 4], [4, 3]])
            graph = UndirectedGraph.init_from_edges(edges, n_vertices=6)

        """
        adjacency_matrix = _convert_edges_to_adjacency_matrix(edges, n_vertices)
        return cls(adjacency_matrix, copy=False, skip_checks=skip_checks)

    @property
    def vertices(self):
        r"""
        Returns the `list` of vertices.

        :type: `list`
        """
        return range(self.adjacency_matrix.shape[0])

    @property
    def n_vertices(self):
        r"""
        Returns the number of vertices.

        :type: `int`
        """
        return self.adjacency_matrix.shape[0]

    @property
    def edges(self):
        r"""
        Returns the `ndarray` of edges, i.e. all the pairs of vertices that are
        connected with an edge.

        :type: ``(n_edges, 2, )`` `ndarray`
        """
        pass

    @property
    def n_edges(self):
        r"""
        Returns the number of edges.

        :type: `int`
        """
        return self.edges.shape[0]

    def isolated_vertices(self):
        r"""
        Returns the isolated vertices of the graph (if any), i.e. the vertices
        that have no edge connections.

        Returns
        -------
        isolated_vertices : `list`
            A `list` of the isolated vertices. If there aren't any, it returns
            an empty `list`.
        """
        return _isolated_vertices(self.adjacency_matrix)

    def has_isolated_vertices(self):
        r"""
        Whether the graph has any isolated vertices, i.e. vertices with no edge
        connections.

        Returns
        -------
        has_isolated_vertices : `bool`
            ``True`` if the graph has at least one isolated vertex.
        """
        return len(self.isolated_vertices()) > 0

    def get_adjacency_list(self):
        r"""
        Returns the adjacency list of the graph, i.e. a `list` of length
        ``n_vertices`` that for each vertex has a `list` of the vertex
        neighbours. If the graph is directed, the neighbours are children.

        Returns
        -------
        adjacency_list : `list` of `list` of length ``n_vertices``
            The adjacency list of the graph.
        """
        # initialize list with empty lists
        adjacency_list = [[] for _ in range(self.n_vertices)]

        # get rows/columns of edges
        rows, cols = self.adjacency_matrix.nonzero()

        # store them accordingly
        for i in range(rows.shape[0]):
            from_v = rows[i]
            to_v = cols[i]
            adjacency_list[from_v].append(to_v)
        return adjacency_list

    def is_edge(self, vertex_1, vertex_2, skip_checks=False):
        r"""
        Whether there is an edge between the provided vertices.

        Parameters
        ----------
        vertex_1 : `int`
            The first selected vertex. Parent if the graph is directed.
        vertex_2 : `int`
            The second selected vertex. Child if the graph is directed.
        skip_checks : `bool`, optional
            If ``False``, the given vertices will be checked.

        Returns
        -------
        is_edge : `bool`
            ``True`` if there is an edge connecting ``vertex_1`` and
            ``vertex_2``.

        Raises
        ------
        ValueError
            The vertex must be between 0 and {n_vertices-1}.
        """
        if not skip_checks:
            self._check_vertex(vertex_1)
            self._check_vertex(vertex_2)
        return self.adjacency_matrix[vertex_1, vertex_2] != 0

    def find_path(self, start, end, method='bfs', skip_checks=False):
        r"""
        Returns a `list` with the first path (without cycles) found from the
        ``start`` vertex to the ``end`` vertex. It can employ either depth-first
        search or breadth-first search.

        Parameters
        ----------
        start : `int`
            The vertex from which the path starts.
        end : `int`
            The vertex to which the path ends.
        method : {``bfs``, ``dfs``}, optional
            The method to be used.
        skip_checks : `bool`, optional
            If ``True``, then input arguments won't pass through checks. Useful
            for efficiency.

        Returns
        -------
        path : `list`
            The path's vertices.

        Raises
        ------
        ValueError
            Method must be either bfs or dfs.
        """
        # checks
        if not skip_checks:
            self._check_vertex(start)
            self._check_vertex(end)

        # search
        if method == 'bfs':
            nodes, predecessors = csgraph.breadth_first_order(
                self.adjacency_matrix, start, directed=self._directed,
                return_predecessors=True)
        elif method == 'dfs':
            nodes, predecessors = csgraph.depth_first_order(
                self.adjacency_matrix, start, directed=self._directed,
                return_predecessors=True)
        else:
            raise ValueError('Method must be either bfs or dfs.')

        # get path
        if predecessors[end] == -9999:
            path = []
        else:
            path = [end]
            i = None
            while i != start:
                i = predecessors[path[-1]]
                path.append(i)
            path.reverse()
        return path

    def find_all_paths(self, start, end, path=[]):
        r"""
        Returns a list of lists with all the paths (without cycles) found from
        start vertex to end vertex.

        Parameters
        ----------
        start : `int`
            The vertex from which the paths start.
        end : `int`
            The vertex from which the paths end.
        path : `list`, optional
            An existing path to append to.

        Returns
        -------
        paths : `list` of `list`
            The list containing all the paths from start to end.
        """
        if path is None:
            path = []
        path = path + [start]
        if start == end:
            return [path]
        if start > self.n_vertices - 1 or start < 0:
            return []
        paths = []
        for v in list(self.adjacency_matrix[start, :].nonzero()[1]):
            if v not in path:
                newpaths = self.find_all_paths(v, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths

    def n_paths(self, start, end):
        r"""
        Returns the number of all the paths (without cycles) existing from
        start vertex to end vertex.

        Parameters
        ----------
        start : `int`
            The vertex from which the paths start.
        end : `int`
            The vertex from which the paths end.

        Returns
        -------
        paths : `int`
            The paths' numbers.
        """
        return len(self.find_all_paths(start, end))

    def find_all_shortest_paths(self, algorithm='auto', unweighted=False):
        r"""
        Returns the distances and predecessors arrays of the graph's shortest
        paths.

        Parameters
        ----------
        algorithm : 'str', see below, optional
            The algorithm to be used. Possible options are:

            ================ =========================================
            'dijkstra'       Dijkstra's algorithm with Fibonacci heaps
            'bellman-ford'   Bellman-Ford algorithm
            'johnson'        Johnson's algorithm
            'floyd-warshall' Floyd-Warshall algorithm
            'auto'           Select the best among the above
            ================ =========================================

        unweighted : `bool`, optional
            If ``True``, then find unweighted distances. That is, rather than
            finding the path between each vertex such that the sum of weights is
            minimized, find the path such that the number of edges is minimized.

        Returns
        -------
        distances : ``(n_vertices, n_vertices,)`` `ndarray`
            The matrix of distances between all graph vertices.
            ``distances[i,j]`` gives the shortest distance from vertex ``i`` to
            vertex ``j`` along the graph.
        predecessors : ``(n_vertices, n_vertices,)`` `ndarray`
            The matrix of predecessors, which can be used to reconstruct the
            shortest paths. Each entry ``predecessors[i, j]`` gives the index of
            the previous vertex in the path from vertex ``i`` to vertex ``j``.
            If no path exists between vertices ``i`` and ``j``, then
            ``predecessors[i, j] = -9999``.
        """
        # find costs and predecessors of all shortest paths
        return csgraph.shortest_path(self.adjacency_matrix,
                                     directed=self._directed, method=algorithm,
                                     unweighted=unweighted,
                                     return_predecessors=True)

    def find_shortest_path(self, start, end, algorithm='auto', unweighted=False,
                           skip_checks=False):
        r"""
        Returns a `list` with the shortest path (without cycles) found from
        ``start`` vertex to ``end`` vertex.

        Parameters
        ----------
        start : `int`
            The vertex from which the path starts.
        end : `int`
            The vertex to which the path ends.
        algorithm : 'str', see below, optional
            The algorithm to be used. Possible options are:

            ================ =========================================
            'dijkstra'       Dijkstra's algorithm with Fibonacci heaps
            'bellman-ford'   Bellman-Ford algorithm
            'johnson'        Johnson's algorithm
            'floyd-warshall' Floyd-Warshall algorithm
            'auto'           Select the best among the above
            ================ =========================================

        unweighted : `bool`, optional
            If ``True``, then find unweighted distances. That is, rather than
            finding the path such that the sum of weights is minimized, find
            the path such that the number of edges is minimized.
        skip_checks : `bool`, optional
            If ``True``, then input arguments won't pass through checks. Useful
            for efficiency.

        Returns
        -------
        path : `list`
            The shortest path's vertices, including ``start`` and ``end``. If
            there was not path connecting the vertices, then an empty `list` is
            returned.
        distance : `int` or `float`
            The distance (cost) of the path from ``start`` to ``end``.
        """
        # checks
        if not skip_checks:
            self._check_vertex(start)
            self._check_vertex(end)

        # find distances and predecessors of all shortest paths
        (distances, predecessors) = self.find_all_shortest_paths(
            algorithm=algorithm, unweighted=unweighted)

        # retrieve shortest path and its distance
        if predecessors[start, end] < 0:
            path = []
            distance = np.inf
        else:
            path = [end]
            distance = 0
            i = None
            while i != start:
                i = predecessors[start, path[-1]]
                path.append(i)
                distance += distances[start, path[-1]]
            path.reverse()
        return path, distance

    def has_cycles(self):
        r"""
        Checks if the graph has at least one cycle.

        Returns
        -------
        has_cycles : `bool`
            ``True`` if the graph has cycles.
        """
        return _has_cycles(self.get_adjacency_list(), self._directed)

    def is_tree(self):
        r"""
        Checks if the graph is tree.

        Returns
        -------
        is_true : `bool`
            If the graph is a tree.
        """
        return not self.has_cycles() and self.n_edges == self.n_vertices - 1

    def _check_vertex(self, vertex):
        r"""
        Checks that a given vertex is valid.

        Parameters
        ----------
        vertex : `int`
            Index of a given vertex.

        Raises
        ------
        ValueError
            The vertex must be between 0 and {n_vertices-1}.
        """
        if vertex > self.n_vertices - 1 or vertex < 0:
            raise ValueError('The vertex must be between '
                             '0 and {}.'.format(self.n_vertices - 1))


class UndirectedGraph(Graph):
    r"""
    Class for Undirected Graph definition and manipulation.

    Parameters
    ----------
    adjacency_matrix : ``(n_vertices, n_vertices, )`` `ndarray` or `csr_matrix`
        The adjacency matrix of the graph. The non-edges must be represented
        with zeros and the edges can have a weight value.

        :Note: ``adjacency_matrix`` must be symmetric.
    copy : `bool`, optional
        If ``False``, the ``adjacency_matrix`` will not be copied on assignment.
    skip_checks : `bool`, optional
        If ``True``, no checks will be performed.

    Raises
    ------
    ValueError
        adjacency_matrix must be either a numpy.ndarray or a
        scipy.sparse.csr_matrix.
    ValueError
        Graph must have at least two vertices.
    ValueError
        adjacency_matrix must be square (n_vertices, n_vertices, ),
        ({adjacency_matrix.shape[0]}, {adjacency_matrix.shape[1]}) given
        instead.
    ValueError
        The adjacency matrix of an undirected graph must be symmetric.

    Examples
    --------
    The following undirected graph ::

        |---0---|
        |       |
        |       |
        1-------2
        |       |
        |       |
        3-------4
        |
        |
        5

    can be defined as ::

        import numpy as np
        adjacency_matrix = np.array([[0, 1, 1, 0, 0, 0],
                                     [1, 0, 1, 1, 0, 0],
                                     [1, 1, 0, 0, 1, 0],
                                     [0, 1, 0, 0, 1, 1],
                                     [0, 0, 1, 1, 0, 0],
                                     [0, 0, 0, 1, 0, 0]])
        graph = UndirectedGraph(adjacency_matrix)

    or ::

        from scipy.sparse import csr_matrix
        adjacency_matrix = csr_matrix(
                            ([1] * 14,
                             ([0, 1, 0, 2, 1, 2, 1, 3, 2, 4, 3, 4, 3, 5],
                              [1, 0, 2, 0, 2, 1, 3, 1, 4, 2, 4, 3, 5, 3])),
                            shape=(6, 6))
        graph = UndirectedGraph(adjacency_matrix)

    The adjacency matrix of the following graph with isolated vertices ::

            0---|
                |
                |
        1       2
                |
                |
        3-------4


        5

    can be defined as ::

        import numpy as np
        adjacency_matrix = np.array([[0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 1, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 0]])
        graph = UndirectedGraph(adjacency_matrix)

    or ::

        from scipy.sparse import csr_matrix
        adjacency_matrix = csr_matrix(([1] * 6, ([0, 2, 2, 4, 3, 4],
                                                 [2, 0, 4, 2, 4, 3])),
                                      shape=(6, 6))
        graph = UndirectedGraph(adjacency_matrix)
    """
    def __init__(self, adjacency_matrix, copy=True, skip_checks=False):
        self._directed = False
        super(UndirectedGraph, self).__init__(adjacency_matrix, copy=copy,
                                              skip_checks=skip_checks)

    @classmethod
    def init_from_edges(cls, edges, n_vertices, skip_checks=False):
        r"""
        Initialize graph from edges array.

        Parameters
        ----------
        edges : ``(n_edges, 2, )`` `ndarray`
            The `ndarray` of edges, i.e. all the pairs of vertices that are
            connected with an edge.
        n_vertices : `int`
            The total number of vertices, assuming that the numbering of
            vertices starts from ``0``. ``edges`` and ``n_vertices`` can be
            defined in a way to set isolated vertices.
        skip_checks : `bool`, optional
            If ``True``, no checks will be performed.

        Examples
        --------
        The following undirected graph ::

            |---0---|
            |       |
            |       |
            1-------2
            |       |
            |       |
            3-------4
            |
            |
            5

        can be defined as ::

            from menpo.shape import UndirectedGraph
            import numpy as np
            edges = np.array([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1],
                              [1, 3], [3, 1], [2, 4], [4, 2], [3, 4], [4, 3],
                              [3, 5], [5, 3]])
            graph = UndirectedGraph.init_from_edges(edges, n_vertices=6)

        Finally, the following graph with isolated vertices ::

                0---|
                    |
                    |
            1       2
                    |
                    |
            3-------4


            5

        can be defined as ::

            from menpo.shape import UndirectedGraph
            import numpy as np
            edges = np.array([[0, 2], [2, 0], [2, 4], [4, 2], [3, 4], [4, 3]])
            graph = UndirectedGraph.init_from_edges(edges, n_vertices=6)

        """
        adjacency_matrix = _convert_edges_to_symmetric_adjacency_matrix(
            edges, n_vertices)
        return cls(adjacency_matrix, copy=False, skip_checks=skip_checks)

    @property
    def edges(self):
        return np.vstack(triu(self.adjacency_matrix).nonzero()).T

    def neighbours(self, vertex, skip_checks=False):
        r"""
        Returns the neighbours of the selected vertex.

        Parameters
        ----------
        vertex : `int`
            The selected vertex.
        skip_checks : `bool`, optional
            If ``False``, the given vertex will be checked.

        Returns
        -------
        neighbours : `list`
            The list of neighbours.

        Raises
        ------
        ValueError
            The vertex must be between 0 and {n_vertices-1}.
        """
        # check given vertex
        if not skip_checks:
            self._check_vertex(vertex)
        return list(self.adjacency_matrix[vertex, :].nonzero()[1])

    def n_neighbours(self, vertex, skip_checks=False):
        r"""
        Returns the number of neighbours of the selected vertex.

        Parameters
        ----------
        vertex : `int`
            The selected vertex.
        skip_checks : `bool`, optional
            If ``False``, the given vertex will be checked.

        Returns
        -------
        n_neighbours : `int`
            The number of neighbours.

        Raises
        ------
        ValueError
            The vertex must be between 0 and {n_vertices-1}.
        """
        return len(self.neighbours(vertex, skip_checks=skip_checks))

    def minimum_spanning_tree(self, root_vertex):
        r"""
        Returns the minimum spanning tree of the graph using Kruskal's
        algorithm.

        Parameters
        ----------
        root_vertex : `int`
            The vertex that will be set as root in the output MST.

        Returns
        -------
        mst : :map:`Tree`
            The computed minimum spanning tree.

        Raises
        ------
        ValueError
            Cannot compute minimum spanning tree of a graph with isolated
            vertices
        """
        # check if graph has isolated vertices
        if self.has_isolated_vertices():
            raise ValueError('Cannot compute minimum spanning tree of a graph '
                             'with isolated vertices.')
        # Compute MST. It returns an undirected graph.
        mst_adjacency = csgraph.minimum_spanning_tree(self.adjacency_matrix)
        # Get directed tree from the above undirected graph using DFS.
        mst_adjacency = csgraph.depth_first_tree(mst_adjacency, root_vertex,
                                                 directed=False)
        return Tree(mst_adjacency, root_vertex)

    def __str__(self):
        isolated = ''
        if self.has_isolated_vertices():
            isolated = " ({} isolated)".format(len(self.isolated_vertices()))
        return "Undirected graph of {} vertices{} and {} " \
               "edges.".format(self.n_vertices, isolated, self.n_edges)


class DirectedGraph(Graph):
    r"""
    Class for Directed Graph definition and manipulation.

    Parameters
    ----------
    adjacency_matrix : ``(n_vertices, n_vertices, )`` `ndarray` or `csr_matrix`
        The adjacency matrix of the graph in which the rows represent source
        vertices and columns represent destination vertices. The non-edges must
        be represented with zeros and the edges can have a weight value.
    copy : `bool`, optional
        If ``False``, the ``adjacency_matrix`` will not be copied on assignment.
    skip_checks : `bool`, optional
        If ``True``, no checks will be performed.

    Raises
    ------
    ValueError
        adjacency_matrix must be either a numpy.ndarray or a
        scipy.sparse.csr_matrix.
    ValueError
        Graph must have at least two vertices.
    ValueError
        adjacency_matrix must be square (n_vertices, n_vertices, ),
        ({adjacency_matrix.shape[0]}, {adjacency_matrix.shape[1]}) given
        instead.

    Examples
    --------
    The following directed graph ::

        |-->0<--|
        |       |
        |       |
        1<----->2
        |       |
        v       v
        3------>4
        |
        v
        5

    can be defined as ::

        import numpy as np
        adjacency_matrix = np.array([[0, 0, 0, 0, 0, 0],
                                     [1, 0, 1, 1, 0, 0],
                                     [1, 1, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 1, 1],
                                     [0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0]])
        graph = DirectedGraph(adjacency_matrix)

    or ::

        from scipy.sparse import csr_matrix
        adjacency_matrix = csr_matrix(([1] * 8, ([1, 2, 1, 2, 1, 2, 3, 3],
                                                 [0, 0, 2, 1, 3, 4, 4, 5])),
                                      shape=(6, 6))
        graph = DirectedGraph(adjacency_matrix)

    The following graph with isolated vertices ::

            0<--|
                |
                |
        1       2
                |
                v
        3------>4


        5

    can be defined as ::

        import numpy as np
        adjacency_matrix = np.array([[0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0]])
        graph = DirectedGraph(adjacency_matrix)

    or ::

        from scipy.sparse import csr_matrix
        adjacency_matrix = csr_matrix(([1] * 3, ([2, 2, 3], [0, 4, 4])),
                                      shape=(6, 6))
        graph = DirectedGraph(adjacency_matrix)
    """
    def __init__(self, adjacency_matrix, copy=True, skip_checks=False):
        self._directed = True
        super(DirectedGraph, self).__init__(adjacency_matrix, copy=copy,
                                            skip_checks=skip_checks)

    @property
    def edges(self):
        return np.vstack(self.adjacency_matrix.nonzero()).T

    def children(self, vertex, skip_checks=False):
        r"""
        Returns the children of the selected vertex.

        Parameters
        ----------
        vertex : `int`
            The selected vertex.
        skip_checks : `bool`, optional
            If ``False``, the given vertex will be checked.

        Returns
        -------
        children : `list`
            The list of children.

        Raises
        ------
        ValueError
            The vertex must be between 0 and {n_vertices-1}.
        """
        if not skip_checks:
            self._check_vertex(vertex)
        return list(self.adjacency_matrix[vertex, :].nonzero()[1])

    def n_children(self, vertex, skip_checks=False):
        r"""
        Returns the number of children of the selected vertex.

        Parameters
        ----------
        vertex : `int`
            The selected vertex.

        Returns
        -------
        n_children : `int`
            The number of children.
        skip_checks : `bool`, optional
            If ``False``, the given vertex will be checked.

        Raises
        ------
        ValueError
            The vertex must be in the range ``[0, n_vertices - 1]``.
        """
        return len(self.children(vertex, skip_checks=skip_checks))

    def parents(self, vertex, skip_checks=False):
        r"""
        Returns the parents of the selected vertex.

        Parameters
        ----------
        vertex : `int`
            The selected vertex.
        skip_checks : `bool`, optional
            If ``False``, the given vertex will be checked.

        Returns
        -------
        parents : `list`
            The list of parents.

        Raises
        ------
        ValueError
            The vertex must be in the range ``[0, n_vertices - 1]``.
        """
        if not skip_checks:
            self._check_vertex(vertex)
        return list(self.adjacency_matrix[:, vertex].nonzero()[0])

    def n_parents(self, vertex, skip_checks=False):
        r"""
        Returns the number of parents of the selected vertex.

        Parameters
        ----------
        vertex : `int`
            The selected vertex.
        skip_checks : `bool`, optional
            If ``False``, the given vertex will be checked.

        Returns
        -------
        n_parents : `int`
            The number of parents.

        Raises
        ------
        ValueError
            The vertex must be in the range ``[0, n_vertices - 1]``.
        """
        return len(self.parents(vertex, skip_checks=skip_checks))

    def __str__(self):
        isolated = ''
        if self.has_isolated_vertices():
            isolated = " ({} isolated)".format(len(self.isolated_vertices()))
        return "Directed graph of {} vertices{} and {} " \
               "edges.".format(self.n_vertices, isolated, self.n_edges)


class Tree(DirectedGraph):
    r"""
    Class for Tree definitions and manipulation.

    Parameters
    ----------
    adjacency_matrix : ``(n_vertices, n_vertices, )`` `ndarray` or `csr_matrix`
        The adjacency matrix of the tree in which the rows represent parents
        and columns represent children. The non-edges must be represented with
        zeros and the edges can have a weight value.

        :Note: A tree must not have isolated vertices.
    root_vertex : `int`
        The vertex to be set as root.
    copy : `bool`, optional
        If ``False``, the ``adjacency_matrix`` will not be copied on assignment.
    skip_checks : `bool`, optional
        If ``True``, no checks will be performed.

    Raises
    ------
    ValueError
        adjacency_matrix must be either a numpy.ndarray or a
        scipy.sparse.csr_matrix.
    ValueError
        Graph must have at least two vertices.
    ValueError
        adjacency_matrix must be square (n_vertices, n_vertices, ),
        ({adjacency_matrix.shape[0]}, {adjacency_matrix.shape[1]}) given
        instead.
    ValueError
        The provided edges do not represent a tree.
    ValueError
        The root_vertex must be in the range ``[0, n_vertices - 1]``.
    ValueError
        The combination of adjacency matrix and root vertex is not valid. BFS
        returns a different tree.

    Examples
    --------
    The following tree ::

               0
               |
            ___|___
           1       2
           |       |
          _|_      |
         3   4     5
         |   |     |
         |   |     |
         6   7     8

    can be defined as ::

        import numpy as np
        adjacency_matrix = np.array([[0, 1, 1, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        tree = Tree(adjacency_matrix, root_vertex=0)

    or ::

        from scipy.sparse import csr_matrix
        adjacency_matrix = csr_matrix(([1] * 8, ([0, 0, 1, 1, 2, 3, 4, 5],
                                                 [1, 2, 3, 4, 5, 6, 7, 8])),
                                      shape=(9, 9))
        tree = Tree(adjacency_matrix, root_vertex=0)
    """
    def __init__(self, adjacency_matrix, root_vertex, copy=True,
                 skip_checks=False):
        super(Tree, self).__init__(adjacency_matrix, copy=copy,
                                   skip_checks=skip_checks)

        if not skip_checks:
            # check if the provided tree has isolated vertices
            if self.has_isolated_vertices():
                raise ValueError('A tree cannot have isolated vertices.')
            # check if provided adjacency_matrix represents a tree
            if not self.is_tree():
                raise ValueError('The provided edges do not represent a tree.')
            # check if root_vertex is valid
            self._check_vertex(root_vertex)
            # check if the tree is properly defined given the root
            if not np.allclose(
                    csgraph.breadth_first_tree(self.adjacency_matrix,
                                               root_vertex,
                                               directed=True).nonzero(),
                    self.adjacency_matrix.nonzero()):
                raise ValueError('The combination of adjacency matrix and root '
                                 'vertex is not valid. BFS returns a different '
                                 'tree.')

        # store root and predecessors list
        self.root_vertex = root_vertex
        self.predecessors_list = self._get_predecessors_list()

    @classmethod
    def init_from_edges(cls, edges, n_vertices, root_vertex, copy=True,
                        skip_checks=False):
        r"""
        Construct a :map:`Tree` from edges array.

        Parameters
        ----------
        edges : ``(n_edges, 2, )`` `ndarray`
            The `ndarray` of edges, i.e. all the pairs of vertices that are
            connected with an edge.
        n_vertices : `int`
            The total number of vertices, assuming that the numbering of
            vertices starts from ``0``. ``edges`` and ``n_vertices`` can be
            defined in a way to set isolated vertices.
        root_vertex : `int`
            That vertex that will be set as root.
        copy : `bool`, optional
            If ``False``, the ``adjacency_matrix`` will not be copied on
            assignment.
        skip_checks : `bool`, optional
            If ``True``, no checks will be performed.

        Examples
        --------
        The following tree ::

                   0
                   |
                ___|___
               1       2
               |       |
              _|_      |
             3   4     5
             |   |     |
             |   |     |
             6   7     8

        can be defined as ::

            from menpo.shape import PointTree
            import numpy as np
            points = np.array([[30, 30], [10, 20], [50, 20], [0, 10], [20, 10],
                               [50, 10], [0, 0], [20, 0], [50, 0]])
            edges = np.array([[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 6],
                              [4, 7], [5, 8]])
            tree = PointTree.init_from_edges(points, edges, root_vertex=0)
        """
        adjacency_matrix = _convert_edges_to_adjacency_matrix(edges, n_vertices)
        return cls(adjacency_matrix, root_vertex=root_vertex, copy=copy,
                   skip_checks=skip_checks)

    def _get_predecessors_list(self):
        r"""
        Returns the predecessors list of the tree, i.e. a `list` of length
        ``n_vertices`` that stores the parent for each vertex. The value of the
        root vertex is ``None``.

        :type: `list` of length ``n_vertices``
        """
        # initialize list with None
        predecessors_list = [None] * self.n_vertices

        # get rows/columns of edges
        parents, children = self.adjacency_matrix.nonzero()

        # store them accordingly
        for i in range(children.shape[0]):
            parent = parents[i]
            child = children[i]
            predecessors_list[child] = parent
        return predecessors_list

    def depth_of_vertex(self, vertex, skip_checks=False):
        r"""
        Returns the depth of the specified vertex.

        Parameters
        ----------
        vertex : `int`
            The selected vertex.
        skip_checks : `bool`, optional
            If ``False``, the given vertex will be checked.

        Returns
        -------
        depth : `int`
            The depth of the selected vertex.

        Raises
        ------
        ValueError
            The vertex must be in the range ``[0, n_vertices - 1]``.
        """
        if not skip_checks:
            self._check_vertex(vertex)
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
        all_depths = [self.depth_of_vertex(v) for v in range(self.n_vertices)]
        return np.max(all_depths)

    def vertices_at_depth(self, depth):
        r"""
        Returns a list of vertices at the specified depth.

        Parameters
        ----------
        depth : `int`
            The selected depth.

        Returns
        -------
        vertices : `list`
            The vertices that lie in the specified depth.
        """
        ver = []
        for v in range(self.n_vertices):
            if self.depth_of_vertex(v) == depth:
                ver.append(v)
        return ver

    def n_vertices_at_depth(self, depth):
        r"""
        Returns the number of vertices at the specified depth.

        Parameters
        ----------
        depth : `int`
            The selected depth.

        Returns
        -------
        n_vertices : `int`
            The number of vertices that lie in the specified depth.
        """
        n_ver = 0
        for v in range(self.n_vertices):
            if self.depth_of_vertex(v) == depth:
                n_ver += 1
        return n_ver

    def is_leaf(self, vertex, skip_checks=False):
        r"""
        Whether the vertex is a leaf.

        Parameters
        ----------
        vertex : `int`
            The selected vertex.
        skip_checks : `bool`, optional
            If ``False``, the given vertex will be checked.

        Returns
        -------
        is_leaf : `bool`
            If ``True``, then selected vertex is a leaf.

        Raises
        ------
        ValueError
            The vertex must be in the range ``[0, n_vertices - 1]``.
        """
        if not skip_checks:
            self._check_vertex(vertex)
        return len(self.children(vertex)) == 0

    @property
    def leaves(self):
        r"""
        Returns a `list` with the all leaves of the tree.

        :type: `list`
        """
        leaves = []
        for v in range(self.n_vertices):
            if self.is_leaf(v):
                leaves.append(v)
        return leaves

    @property
    def n_leaves(self):
        r"""
        Returns the number of leaves of the tree.

        :type: `int`
        """
        return len(self.leaves)

    def parent(self, vertex, skip_checks=False):
        r"""
        Returns the parent of the selected vertex.

        Parameters
        ----------
        vertex : `int`
            The selected vertex.
        skip_checks : `bool`, optional
            If ``False``, the given vertex will be checked.

        Returns
        -------
        parent : `int`
            The parent vertex.

        Raises
        ------
        ValueError
            The vertex must be in the range ``[0, n_vertices - 1]``.
        """
        if not skip_checks:
            self._check_vertex(vertex)
        return self.predecessors_list[vertex]

    def __str__(self):
        return "Tree of depth {} with {} vertices and {} leaves.".format(
            self.maximum_depth, self.n_vertices, self.n_leaves)


class PointGraph(Graph, PointCloud):
    r"""
    Class for defining a Graph with geometry.

    Parameters
    ----------
    points : ``(n_vertices, n_dims, )`` `ndarray`
        The array of point locations.
    adjacency_matrix : ``(n_vertices, n_vertices, )`` `ndarray` or `csr_matrix`
        The adjacency matrix of the graph in which the rows represent source
        vertices and columns represent destination vertices. The non-edges must
        be represented with zeros and the edges can have a weight value.

        The adjacency matrix of an undirected graph must be symmetric.
    directed : `bool`
        If ``True``, the graph is considered directed. If ``False``, the graph
        is considered undirected.
    copy : `bool`, optional
        If ``False``, the ``adjacency_matrix`` will not be copied on assignment.
    skip_checks : `bool`, optional
        If ``True``, no checks will be performed.

    Raises
    ------
    ValueError
        adjacency_matrix must be either a numpy.ndarray or a
        scipy.sparse.csr_matrix.
    ValueError
        Graph must have at least two vertices.
    ValueError
        adjacency_matrix must be square (n_vertices, n_vertices, ),
        ({adjacency_matrix.shape[0]}, {adjacency_matrix.shape[1]}) given
        instead.
    ValueError
        The adjacency matrix of an undirected graph must be symmetric.
    ValueError
        A point for each graph vertex needs to be passed. Got {} points instead
        of {}

    Examples
    --------
    The adjacency matrix of the following undirected graph ::

        |---0---|
        |       |
        |       |
        1-------2
        |       |
        |       |
        3-------4
        |
        |
        5

    can be defined as ::

        import numpy as np
        adjacency_matrix = np.array([[0, 1, 1, 0, 0, 0],
                                     [1, 0, 1, 1, 0, 0],
                                     [1, 1, 0, 0, 1, 0],
                                     [0, 1, 0, 0, 1, 1],
                                     [0, 0, 1, 1, 0, 0],
                                     [0, 0, 0, 1, 0, 0]])

    or ::

        from scipy.sparse import csr_matrix
        adjacency_matrix = csr_matrix(
                            ([1] * 14,
                             ([0, 1, 0, 2, 1, 2, 1, 3, 2, 4, 3, 4, 3, 5],
                              [1, 0, 2, 0, 2, 1, 3, 1, 4, 2, 4, 3, 5, 3])),
                            shape=(6, 6))


    The adjacency matrix of the following directed graph ::

        |-->0<--|
        |       |
        |       |
        1<----->2
        |       |
        v       v
        3------>4
        |
        v
        5

    can be represented as ::

        import numpy as np
        adjacency_matrix = np.array([[0, 0, 0, 0, 0, 0],
                                     [1, 0, 1, 1, 0, 0],
                                     [1, 1, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 1, 1],
                                     [0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0]])

    or ::

        from scipy.sparse import csr_matrix
        adjacency_matrix = csr_matrix(([1] * 8, ([1, 2, 1, 2, 1, 2, 3, 3],
                                                 [0, 0, 2, 1, 3, 4, 4, 5])),
                                      shape=(6, 6))

    Finally, the adjacency matrix of the following graph with isolated
    vertices ::

            0---|
                |
                |
        1       2
                |
                |
        3-------4


        5

    can be defined as ::

        import numpy as np
        adjacency_matrix = np.array([[0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 1, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 0]])

    or ::

        from scipy.sparse import csr_matrix
        adjacency_matrix = csr_matrix(([1] * 6, ([0, 2, 2, 4, 3, 4],
                                                 [2, 0, 4, 2, 4, 3])),
                                      shape=(6, 6))
    """
    def __init__(self, points, adjacency_matrix, copy=True, skip_checks=False):
        if not skip_checks:
            # check the number of points
            _check_n_points(points, adjacency_matrix)
        Graph.__init__(self, adjacency_matrix, copy=copy,
                       skip_checks=skip_checks)
        PointCloud.__init__(self, points, copy=copy)

    @classmethod
    def init_from_edges(cls, points, edges, copy=True, skip_checks=False):
        r"""
        Construct a PointGraph from edges array.

        Parameters
        ----------
        points : ``(n_vertices, n_dims, )`` `ndarray`
            The array of point locations.
        edges : ``(n_edges, 2, )`` `ndarray`
            The `ndarray` of edges, i.e. all the pairs of vertices that are
            connected with an edge.
        copy : `bool`, optional
            If ``False``, the ``adjacency_matrix`` will not be copied on
            assignment.
        skip_checks : `bool`, optional
            If ``True``, no checks will be performed.

        Examples
        --------
        The following undirected graph ::

            |---0---|
            |       |
            |       |
            1-------2
            |       |
            |       |
            3-------4
            |
            |
            5

        can be defined as ::

            from menpo.shape import PointUndirectedGraph
            import numpy as np
            points = np.array([[10, 30], [0, 20], [20, 20], [0, 10], [20, 10],
                               [0, 0]])
            edges = np.array([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1],
                              [1, 3], [3, 1], [2, 4], [4, 2], [3, 4], [4, 3],
                              [3, 5], [5, 3]])
            graph = PointUndirectedGraph.init_from_edges(points, edges)


        The following directed graph ::

            |-->0<--|
            |       |
            |       |
            1<----->2
            |       |
            v       v
            3------>4
            |
            v
            5

        can be represented as ::

            from menpo.shape import PointDirectedGraph
            import numpy as np
            points = np.array([[10, 30], [0, 20], [20, 20], [0, 10], [20, 10],
                               [0, 0]])
            edges = np.array([[1, 0], [2, 0], [1, 2], [2, 1], [1, 3], [2, 4],
                              [3, 4], [3, 5]])
            graph = PointDirectedGraph.init_from_edges(points, edges)

        Finally, the following graph with isolated vertices ::

                0---|
                    |
                    |
            1       2
                    |
                    |
            3-------4


            5

        can be defined as ::

            from menpo.shape import PointUndirectedGraph
            import numpy as np
            points = np.array([[10, 30], [0, 20], [20, 20], [0, 10], [20, 10],
                               [0, 0]])
            edges = np.array([[0, 2], [2, 0], [2, 4], [4, 2], [3, 4], [4, 3]])
            graph = PointUndirectedGraph.init_from_edges(points, edges)

        """
        adjacency_matrix = _convert_edges_to_adjacency_matrix(edges,
                                                              points.shape[0])
        return cls(points, adjacency_matrix, copy=copy, skip_checks=skip_checks)

    def tojson(self):
        r"""
        Convert this PointGraph to a dictionary representation suitable for
        inclusion in the LJSON landmark format.

        Returns
        -------
        json : `dict`
            Dictionary with ``points`` and ``connectivity`` keys.
        """
        json_dict = PointCloud.tojson(self)
        json_dict['connectivity'] = self.edges.tolist()
        return json_dict

    def _view_2d(self, figure_id=None, new_figure=False, image_view=True,
                 render_lines=True, line_colour='r',
                 line_style='-', line_width=1.,
                 render_markers=True, marker_style='o', marker_size=20,
                 marker_face_colour='k', marker_edge_colour='k',
                 marker_edge_width=1., render_numbering=False,
                 numbers_horizontal_align='center',
                 numbers_vertical_align='bottom',
                 numbers_font_name='sans-serif', numbers_font_size=10,
                 numbers_font_style='normal', numbers_font_weight='normal',
                 numbers_font_colour='k', render_axes=True,
                 axes_font_name='sans-serif', axes_font_size=10,
                 axes_font_style='normal', axes_font_weight='normal',
                 axes_x_limits=None, axes_y_limits=None, axes_x_ticks=None,
                 axes_y_ticks=None, figure_size=(10, 8), label=None):
        r"""
        Visualization of the PointGraph in 2D.

        Returns
        -------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        image_view : `bool`, optional
            If ``True`` the PointGraph will be viewed as if it is in the image
            coordinate system.
        render_lines : `bool`, optional
            If ``True``, the edges will be rendered.
        line_colour : See Below, optional
            The colour of the lines.
            Example options::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        line_style : ``{'-', '--', '-.', ':'}``, optional
            The style of the lines.
        line_width : `float`, optional
            The width of the lines.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : See Below, optional
            The style of the markers. Example options ::

                {., ,, o, v, ^, <, >, +, x, D, d, s, p, *, h, H, 1, 2, 3, 4, 8}

        marker_size : `int`, optional
            The size of the markers in points^2.
        marker_face_colour : See Below, optional
            The face (filling) colour of the markers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_colour : See Below, optional
            The edge colour of the markers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_width : `float`, optional
            The width of the markers' edge.
        render_numbering : `bool`, optional
            If ``True``, the landmarks will be numbered.
        numbers_horizontal_align : ``{center, right, left}``, optional
            The horizontal alignment of the numbers' texts.
        numbers_vertical_align : ``{center, top, bottom, baseline}``, optional
            The vertical alignment of the numbers' texts.
        numbers_font_name : See Below, optional
            The font of the numbers. Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        numbers_font_size : `int`, optional
            The font size of the numbers.
        numbers_font_style : ``{normal, italic, oblique}``, optional
            The font style of the numbers.
        numbers_font_weight : See Below, optional
            The font weight of the numbers.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                semibold, demibold, demi, bold, heavy, extra bold, black}

        numbers_font_colour : See Below, optional
            The font colour of the numbers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : See Below, optional
            The font of the axes.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : {``normal``, ``italic``, ``oblique``}, optional
            The font style of the axes.
        axes_font_weight : See Below, optional
            The font weight of the axes.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                semibold, demibold, demi, bold, heavy, extra bold, black}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the PointGraph as a percentage of the PointGraph's
            width. If `tuple` or `list`, then it defines the axis limits. If
            ``None``, then the limits are set automatically.
        axes_y_limits : (`float`, `float`) `tuple` or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the PointGraph as a percentage of the PointGraph's
            height. If `tuple` or `list`, then it defines the axis limits. If
            ``None``, then the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) `tuple` or ``None``, optional
            The size of the figure in inches.
        label : `str`, optional
            The name entry in case of a legend.

        Returns
        -------
        viewer : :map:`PointGraphViewer2d`
            The viewer object.
        """
        from menpo.visualize import PointGraphViewer2d
        renderer = PointGraphViewer2d(figure_id, new_figure, self.points,
                                      self.edges)
        renderer.render(
            image_view=image_view, render_lines=render_lines,
            line_colour=line_colour, line_style=line_style,
            line_width=line_width, render_markers=render_markers,
            marker_style=marker_style, marker_size=marker_size,
            marker_face_colour=marker_face_colour,
            marker_edge_colour=marker_edge_colour,
            marker_edge_width=marker_edge_width,
            render_numbering=render_numbering,
            numbers_horizontal_align=numbers_horizontal_align,
            numbers_vertical_align=numbers_vertical_align,
            numbers_font_name=numbers_font_name,
            numbers_font_size=numbers_font_size,
            numbers_font_style=numbers_font_style,
            numbers_font_weight=numbers_font_weight,
            numbers_font_colour=numbers_font_colour, render_axes=render_axes,
            axes_font_name=axes_font_name, axes_font_size=axes_font_size,
            axes_font_style=axes_font_style, axes_font_weight=axes_font_weight,
            axes_x_limits=axes_x_limits, axes_y_limits=axes_y_limits,
            axes_x_ticks=axes_x_ticks, axes_y_ticks=axes_y_ticks,
            figure_size=figure_size, label=label)
        return renderer

    def _view_3d(self, figure_id=None, new_figure=False):
        r"""
        Visualization of the PointGraph in 3D.

        Parameters
        ----------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.

        Returns
        -------
        viewer : PointGraphViewer3d
            The Menpo3D viewer object.
        """
        try:
            from menpo3d.visualize import PointGraphViewer3d
            return PointGraphViewer3d(figure_id, new_figure, self.points,
                                      self.edges).render()
        except ImportError:
            from menpo.visualize import Menpo3dMissingError
            raise Menpo3dMissingError()

    def view_widget(self, browser_style='buttons', figure_size=(10, 8),
                    style='coloured'):
        r"""
        Visualization of the PointGraph using an interactive widget.

        Parameters
        ----------
        browser_style : {``'buttons'``, ``'slider'``}, optional
            It defines whether the selector of the objects will have the form of
            plus/minus buttons or a slider.
        figure_size : (`int`, `int`) `tuple`, optional
            The initial size of the rendered figure.
        style : {``'coloured'``, ``'minimal'``}, optional
            If ``'coloured'``, then the style of the widget will be coloured. If
            ``minimal``, then the style is simple using black and white colours.
        """
        try:
            from menpowidgets import visualize_pointclouds
            visualize_pointclouds(self, figure_size=figure_size, style=style,
                                  browser_style=browser_style)
        except ImportError:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()


class PointUndirectedGraph(PointGraph, UndirectedGraph):
    r"""
    Class for defining an Undirected Graph with geometry.

    Parameters
    ----------
    points : ``(n_vertices, n_dims, )`` `ndarray`
        The array of point locations.
    adjacency_matrix : ``(n_vertices, n_vertices, )`` `ndarray` or `csr_matrix`
        The adjacency matrix of the graph. The non-edges must be represented
        with zeros and the edges can have a weight value.

        :Note: ``adjacency_matrix`` must be symmetric.
    copy : `bool`, optional
        If ``False``, the ``adjacency_matrix`` will not be copied on assignment.
    skip_checks : `bool`, optional
        If ``True``, no checks will be performed.

    Raises
    ------
    ValueError
        A point for each graph vertex needs to be passed. Got ``n_points``
        points instead of ``n_vertices``.
    ValueError
        adjacency_matrix must be either a numpy.ndarray or a
        scipy.sparse.csr_matrix.
    ValueError
        Graph must have at least two vertices.
    ValueError
        adjacency_matrix must be square (n_vertices, n_vertices, ),
        ({adjacency_matrix.shape[0]}, {adjacency_matrix.shape[1]}) given
        instead.
    ValueError
        The adjacency matrix of an undirected graph must be symmetric.

    Examples
    --------
    The following undirected graph ::

        |---0---|
        |       |
        |       |
        1-------2
        |       |
        |       |
        3-------4
        |
        |
        5

    can be defined as ::

        import numpy as np
        adjacency_matrix = np.array([[0, 1, 1, 0, 0, 0],
                                     [1, 0, 1, 1, 0, 0],
                                     [1, 1, 0, 0, 1, 0],
                                     [0, 1, 0, 0, 1, 1],
                                     [0, 0, 1, 1, 0, 0],
                                     [0, 0, 0, 1, 0, 0]])
        points = np.array([[10, 30], [0, 20], [20, 20], [0, 10], [20, 10],
                           [0, 0]])
        graph = PointUndirectedGraph(points, adjacency_matrix)

    or ::

        from scipy.sparse import csr_matrix
        adjacency_matrix = csr_matrix(
                            ([1] * 14,
                             ([0, 1, 0, 2, 1, 2, 1, 3, 2, 4, 3, 4, 3, 5],
                              [1, 0, 2, 0, 2, 1, 3, 1, 4, 2, 4, 3, 5, 3])),
                            shape=(6, 6))
        points = np.array([[10, 30], [0, 20], [20, 20], [0, 10], [20, 10],
                           [0, 0]])
        graph = PointUndirectedGraph(points, adjacency_matrix)

    The adjacency matrix of the following graph with isolated vertices ::

            0---|
                |
                |
        1       2
                |
                |
        3-------4


        5

    can be defined as ::

        import numpy as np
        adjacency_matrix = np.array([[0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 1, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 0]])
        points = np.array([[10, 30], [0, 20], [20, 20], [0, 10], [20, 10],
                           [0, 0]])
        graph = PointUndirectedGraph(points, adjacency_matrix)

    or ::

        from scipy.sparse import csr_matrix
        adjacency_matrix = csr_matrix(([1] * 6, ([0, 2, 2, 4, 3, 4],
                                                 [2, 0, 4, 2, 4, 3])),
                                      shape=(6, 6))
        points = np.array([[10, 30], [0, 20], [20, 20], [0, 10], [20, 10],
                           [0, 0]])
        graph = PointUndirectedGraph(points, adjacency_matrix)
    """
    def __init__(self, points, adjacency_matrix, copy=True, skip_checks=False):
        self._directed = False
        super(PointUndirectedGraph, self).__init__(points, adjacency_matrix,
                                                   copy=copy,
                                                   skip_checks=skip_checks)

    @classmethod
    def init_from_edges(cls, points, edges, copy=True, skip_checks=False):
        r"""
        Construct a :map:`PointUndirectedGraph` from edges array.

        Parameters
        ----------
        points : ``(n_vertices, n_dims, )`` `ndarray`
            The array of point locations.
        edges : ``(n_edges, 2, )`` `ndarray`
            The `ndarray` of edges, i.e. all the pairs of vertices that are
            connected with an edge.
        copy : `bool`, optional
            If ``False``, the ``adjacency_matrix`` will not be copied on
            assignment.
        skip_checks : `bool`, optional
            If ``True``, no checks will be performed.

        Examples
        --------
        The following undirected graph ::

            |---0---|
            |       |
            |       |
            1-------2
            |       |
            |       |
            3-------4
            |
            |
            5

        can be defined as ::

            from menpo.shape import PointUndirectedGraph
            import numpy as np
            points = np.array([[10, 30], [0, 20], [20, 20], [0, 10], [20, 10],
                               [0, 0]])
            edges = np.array([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1],
                              [1, 3], [3, 1], [2, 4], [4, 2], [3, 4], [4, 3],
                              [3, 5], [5, 3]])
            graph = PointUndirectedGraph.init_from_edges(points, edges)


        Finally, the following graph with isolated vertices ::

                0---|
                    |
                    |
            1       2
                    |
                    |
            3-------4


            5

        can be defined as ::

            from menpo.shape import PointUndirectedGraph
            import numpy as np
            points = np.array([[10, 30], [0, 20], [20, 20], [0, 10], [20, 10],
                               [0, 0]])
            edges = np.array([[0, 2], [2, 0], [2, 4], [4, 2], [3, 4], [4, 3]])
            graph = PointUndirectedGraph.init_from_edges(points, edges)

        """
        adjacency_matrix = _convert_edges_to_symmetric_adjacency_matrix(
            edges, points.shape[0])
        return cls(points, adjacency_matrix, copy=copy, skip_checks=skip_checks)

    def from_mask(self, mask):
        """
        A 1D boolean array with the same number of elements as the number of
        points in the `PointUndirectedGraph`. This is then broadcast across
        the dimensions of the `PointUndirectedGraph` and returns a new
        `PointUndirectedGraph` containing only those points that were ``True``
        in the mask.

        Parameters
        ----------
        mask : ``(n_vertices,)`` `ndarray`
            1D array of booleans

        Returns
        -------
        pointgraph : :map:`PointUndirectedGraph`
            A new pointgraph that has been masked.

        Raises
        ------
        ValueError
            Mask must be a 1D boolean array of the same number of entries as
            points in this PointUndirectedGraph.
        """
        if mask.shape[0] != self.n_points:
            raise ValueError('Mask must be a 1D boolean array of the same '
                             'number of entries as points in this '
                             'PointUndirectedGraph.')

        if np.all(mask):  # Shortcut for all true masks
            return self.copy()
        else:
            # Get new adjacency_matrix and points
            (adjacency_matrix, points) = _mask_adjacency_matrix_and_points(
                mask, self.adjacency_matrix.todense().copy(),
                self.points.copy())
            return PointUndirectedGraph(points, adjacency_matrix, copy=False,
                                        skip_checks=False)

    def minimum_spanning_tree(self, root_vertex):
        r"""
        Returns the minimum spanning tree of the graph using Kruskal's
        algorithm.

        Parameters
        ----------
        root_vertex : `int`
            The vertex that will be set as root in the output MST.

        Returns
        -------
        mst : :map:`PointTree`
            The computed minimum spanning tree with the `points` of `self`.

        Raises
        ------
        ValueError
            Cannot compute minimum spanning tree of a graph with isolated
            vertices
        """
        # check if graph has isolated vertices
        if self.has_isolated_vertices():
            raise ValueError('Cannot compute minimum spanning tree of a graph '
                             'with isolated vertices.')
        # Compute MST. It returns an undirected graph.
        mst_adjacency = csgraph.minimum_spanning_tree(self.adjacency_matrix)
        # Get directed tree from the above undirected graph using DFS.
        mst_adjacency = csgraph.depth_first_tree(mst_adjacency, root_vertex,
                                                 directed=False)
        # remove isolated vertices from the points
        return PointTree(self.points, mst_adjacency, root_vertex, copy=True)


class PointDirectedGraph(PointGraph, DirectedGraph):
    r"""
    Class for defining a directed graph with geometry.

    Parameters
    ----------
    points : ``(n_vertices, n_dims)`` `ndarray`
        The array representing the points.
    adjacency_matrix : ``(n_vertices, n_vertices, )`` `ndarray` or `csr_matrix`
        The adjacency matrix of the graph in which the rows represent source
        vertices and columns represent destination vertices. The non-edges must
        be represented with zeros and the edges can have a weight value.
    copy : `bool`, optional
        If ``False``, the ``adjacency_matrix`` will not be copied on assignment.
    skip_checks : `bool`, optional
        If ``True``, no checks will be performed.

    Raises
    ------
    ValueError
        A point for each graph vertex needs to be passed. Got {n_points} points
        instead of {n_vertices}.
    ValueError
        adjacency_matrix must be either a numpy.ndarray or a
        scipy.sparse.csr_matrix.
    ValueError
        Graph must have at least two vertices.
    ValueError
        adjacency_matrix must be square (n_vertices, n_vertices, ),
        ({adjacency_matrix.shape[0]}, {adjacency_matrix.shape[1]}) given
        instead.

    Examples
    --------
    The following directed graph ::

        |-->0<--|
        |       |
        |       |
        1<----->2
        |       |
        v       v
        3------>4
        |
        v
        5

    can be defined as ::

        import numpy as np
        adjacency_matrix = np.array([[0, 0, 0, 0, 0, 0],
                                     [1, 0, 1, 1, 0, 0],
                                     [1, 1, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 1, 1],
                                     [0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0]])
        points = np.array([[10, 30], [0, 20], [20, 20], [0, 10], [20, 10],
                           [0, 0]])
        graph = PointDirectedGraph(points, adjacency_matrix)

    or ::

        from scipy.sparse import csr_matrix
        adjacency_matrix = csr_matrix(([1] * 8, ([1, 2, 1, 2, 1, 2, 3, 3],
                                                 [0, 0, 2, 1, 3, 4, 4, 5])),
                                      shape=(6, 6))
        points = np.array([[10, 30], [0, 20], [20, 20], [0, 10], [20, 10],
                           [0, 0]])
        graph = PointDirectedGraph(points, adjacency_matrix)

    The following graph with isolated vertices ::

            0<--|
                |
                |
        1       2
                |
                v
        3------>4


        5

    can be defined as ::

        import numpy as np
        adjacency_matrix = np.array([[0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0],
                                     [1, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0]])
        points = np.array([[10, 30], [0, 20], [20, 20], [0, 10], [20, 10],
                           [0, 0]])
        graph = PointDirectedGraph(points, adjacency_matrix)

    or ::

        from scipy.sparse import csr_matrix
        adjacency_matrix = csr_matrix(([1] * 3, ([2, 2, 3], [0, 4, 4])),
                                      shape=(6, 6))
        points = np.array([[10, 30], [0, 20], [20, 20], [0, 10], [20, 10],
                           [0, 0]])
        graph = PointDirectedGraph(points, adjacency_matrix)
    """
    def __init__(self, points, adjacency_matrix, copy=True, skip_checks=False):
        self._directed = True
        super(PointDirectedGraph, self).__init__(points, adjacency_matrix,
                                                 copy=copy,
                                                 skip_checks=skip_checks)

    def relative_location_edge(self, parent, child):
        r"""
        Returns the relative location between the provided vertices. That is
        if vertex j is the parent and vertex i is its child and vector l
        denotes the coordinates of a vertex, then

        ::

            l_i - l_j = [[x_i], [y_i]] - [[x_j], [y_j]] =
                      = [[x_i - x_j], [y_i - y_j]]

        Parameters
        ----------
        parent : `int`
            The first selected vertex which is considered as the parent.
        child : `int`
            The second selected vertex which is considered as the child.

        Returns
        -------
        relative_location : ``(2,)`` `ndarray`
            The relative location vector.

        Raises
        ------
        ValueError
            Vertices ``parent`` and ``child`` are not connected with an edge.
        """
        if not self.is_edge(parent, child):
            raise ValueError('Vertices {} and {} are not connected '
                             'with an edge.'.format(parent, child))
        return self.points[child, ...] - self.points[parent, ...]

    def relative_locations(self):
        r"""
        Returns the relative location between the vertices of each edge. If
        vertex j is the parent and vertex i is its child and vector l denotes
        the coordinates of a vertex, then:

        ::

                    l_i - l_j = [[x_i], [y_i]] - [[x_j], [y_j]] =
                              = [[x_i - x_j], [y_i - y_j]]

        Returns
        -------
        relative_locations : ``(n_vertexes, 2)`` `ndarray`
            The relative locations vector.
        """
        parents = list(self.adjacency_matrix.nonzero()[0])
        children = list(self.adjacency_matrix.nonzero()[1])
        return self.points[children] - self.points[parents]

    def from_mask(self, mask):
        """
        A 1D boolean array with the same number of elements as the number of
        points in the `PointDirectedGraph`. This is then broadcast across the
        dimensions of the `PointDirectedGraph` and returns a new
        `PointDirectedGraph` containing only those points that were ``True`` in
        the mask.

        Parameters
        ----------
        mask : ``(n_points,)`` `ndarray`
            1D array of booleans

        Returns
        -------
        pointgraph : :map:`PointDirectedGraph`
            A new pointgraph that has been masked.

        Raises
        ------
        ValueError
            Mask must be a 1D boolean array of the same number of entries as
            points in this PointDirectedGraph.
        """
        if mask.shape[0] != self.n_points:
            raise ValueError('Mask must be a 1D boolean array of the same '
                             'number of entries as points in this '
                             'PointDirectedGraph.')

        if np.all(mask):  # Shortcut for all true masks
            return self.copy()
        else:
            # Get new adjacency_matrix and points
            (adjacency_matrix, points) = _mask_adjacency_matrix_and_points(
                mask, self.adjacency_matrix.todense().copy(),
                self.points.copy())
            return PointDirectedGraph(points, adjacency_matrix, copy=False,
                                      skip_checks=False)


class PointTree(PointDirectedGraph, Tree):
    r"""
    Class for defining a Tree with geometry.

    Parameters
    ----------
    points : ``(n_vertices, n_dims)`` `ndarray`
        The array representing the points.
    adjacency_matrix : ``(n_vertices, n_vertices, )`` `ndarray` or `csr_matrix`
        The adjacency matrix of the tree in which the rows represent parents
        and columns represent children. The non-edges must be represented with
        zeros and the edges can have a weight value.

        :Note: A tree must not have isolated vertices.
    root_vertex : `int`
        The vertex to be set as root.
    copy : `bool`, optional
        If ``False``, the ``adjacency_matrix`` will not be copied on assignment.
    skip_checks : `bool`, optional
        If ``True``, no checks will be performed.

    Raises
    ------
    ValueError
        A point for each graph vertex needs to be passed. Got {n_points} points
        instead of {n_vertices}.
    ValueError
        adjacency_matrix must be either a numpy.ndarray or a
        scipy.sparse.csr_matrix.
    ValueError
        Graph must have at least two vertices.
    ValueError
        adjacency_matrix must be square (n_vertices, n_vertices, ),
        ({adjacency_matrix.shape[0]}, {adjacency_matrix.shape[1]}) given
        instead.
    ValueError
        The provided edges do not represent a tree.
    ValueError
        The root_vertex must be in the range ``[0, n_vertices - 1]``.
    ValueError
        The combination of adjacency matrix and root vertex is not valid. BFS
        returns a different tree.

    Examples
    --------
    The following tree ::

               0
               |
            ___|___
           1       2
           |       |
          _|_      |
         3   4     5
         |   |     |
         |   |     |
         6   7     8

    can be defined as ::

        import numpy as np
        adjacency_matrix = np.array([[0, 1, 1, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 1, 1, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        points = np.array([[30, 30], [10, 20], [50, 20], [0, 10], [20, 10],
                           [50, 10], [0, 0], [20, 0], [50, 0]])
        tree = PointTree(points, adjacency_matrix, root_vertex=0)

    or ::

        from scipy.sparse import csr_matrix
        adjacency_matrix = csr_matrix(([1] * 8, ([0, 0, 1, 1, 2, 3, 4, 5],
                                                 [1, 2, 3, 4, 5, 6, 7, 8])),
                                      shape=(9, 9))
        points = np.array([[30, 30], [10, 20], [50, 20], [0, 10], [20, 10],
                           [50, 10], [0, 0], [20, 0], [50, 0]])
        tree = PointTree(points, adjacency_matrix, root_vertex=0)
    """
    def __init__(self, points, adjacency_matrix, root_vertex, copy=True,
                 skip_checks=False):
        super(PointTree, self).__init__(points, adjacency_matrix, copy=copy,
                                        skip_checks=skip_checks)
        Tree.__init__(self, adjacency_matrix, root_vertex, copy=copy,
                      skip_checks=skip_checks)

    @classmethod
    def init_from_edges(cls, points, edges, root_vertex, copy=True,
                        skip_checks=False):
        r"""
        Construct a :map:`PointTree` from edges array.

        Parameters
        ----------
        points : ``(n_vertices, n_dims, )`` `ndarray`
            The array of point locations.
        edges : ``(n_edges, 2, )`` `ndarray`
            The `ndarray` of edges, i.e. all the pairs of vertices that are
            connected with an edge.
        root_vertex : `int`
            That vertex that will be set as root.
        copy : `bool`, optional
            If ``False``, the ``adjacency_matrix`` will not be copied on
            assignment.
        skip_checks : `bool`, optional
            If ``True``, no checks will be performed.

        Examples
        --------
        The following tree ::

                   0
                   |
                ___|___
               1       2
               |       |
              _|_      |
             3   4     5
             |   |     |
             |   |     |
             6   7     8

        can be defined as ::

            from menpo.shape import PointTree
            import numpy as np
            points = np.array([[30, 30], [10, 20], [50, 20], [0, 10], [20, 10],
                               [50, 10], [0, 0], [20, 0], [50, 0]])
            edges = np.array([[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 6],
                              [4, 7], [5, 8]])
            tree = PointTree.init_from_edges(points, edges, root_vertex=0)
        """
        adjacency_matrix = _convert_edges_to_adjacency_matrix(edges,
                                                              points.shape[0])
        return cls(points, adjacency_matrix, root_vertex=root_vertex,
                   copy=copy, skip_checks=skip_checks)

    def from_mask(self, mask):
        """
        A 1D boolean array with the same number of elements as the number of
        points in the `PointTree`. This is then broadcast across the dimensions
        of the `PointTree` and returns a new `PointTree` containing only those
        points that were ``True`` in the mask.

        Parameters
        ----------
        mask : ``(n_points,)`` `ndarray`
            1D array of booleans

        Returns
        -------
        pointtree : :map:`PointTree`
            A new pointtree that has been masked.

        Raises
        ------
        ValueError
            Mask must be a 1D boolean array of the same number of entries as
            points in this PointTree.
        ValueError
            Cannot remove root vertex.
        """
        if mask.shape[0] != self.n_points:
            raise ValueError('Mask must be a 1D boolean array of the same '
                             'number of entries as points in this PointTree.')

        if np.all(mask):  # Shortcut for all true masks
            return self.copy()
        else:
            # Impossible to remove root vertex
            if not mask[self.root_vertex]:
                raise ValueError('Cannot remove root vertex.')
            # Get new adjacency_matrix and points
            (adjacency_matrix, points) = _mask_adjacency_matrix_and_points(
                mask, self.adjacency_matrix.todense().copy(),
                self.points.copy())
            root_vertex = self.root_vertex - np.sum(~mask[:self.root_vertex])
            # iteratively find isolated vertices and remove them
            n_components, labels = csgraph.connected_components(
                adjacency_matrix, directed=True)
            while n_components > 1:
                label_to_keep = labels[root_vertex]
                mask = labels == label_to_keep
                (adjacency_matrix, points) = _mask_adjacency_matrix_and_points(
                    mask, adjacency_matrix, points)
                root_vertex = root_vertex - np.sum(~mask[:root_vertex])
                n_components, labels = csgraph.connected_components(
                    adjacency_matrix, directed=True)
            return PointTree(points, adjacency_matrix, root_vertex=root_vertex,
                             copy=False, skip_checks=False)


def _is_symmetric(array):
    r"""
    Check if an array is symmetric.

    Parameters
    ----------
    array : `ndarray` or `scipy.sparse.csr_matrix`
        The array to check.

    Returns
    -------
    is_symmetric : `bool`
        ``True`` if the array is symmetric.
    """
    return np.allclose(array.transpose().nonzero(), array.nonzero())


def _check_n_points(points, adjacency_matrix):
    r"""
    Checks whether the ``points`` array and the ``adjacency_matrix`` have the
    same number of points.

    Parameters
    ----------
    points : ``(n_vertices, n_dims,)`` `ndarray`
        Points array.
    adjacency_matrix : ``(n_vertices, n_vertices,)`` `ndarray`
        The adjacency matrix.

    Raises
    ------
    ValueError
        A point for each graph vertex needs to be passed. Got {} points instead
        of {}
    """
    if not points.shape[0] == adjacency_matrix.shape[0]:
        raise ValueError('A point for each graph vertex needs to be passed. '
                         'Got {} points instead '
                         'of {}'.format(points.shape[0],
                                        adjacency_matrix.shape[0]))


def _has_cycles(adjacency_list, directed):
    r"""
    Function that checks if the provided directed graph has cycles using a Depth
    First Search (DFS).

    Parameters
    ----------
    adjacency_list : `list` of `list` of length ``n_vertices``
        The adjacency list of the graph.
    directed : `bool`
        Defines if the provided graph is directed or not.

    Returns
    -------
    has_cycles : `bool`
        Whether the graph has cycles.
    """
    def dfs(node, entered, exited, tree_edges, back_edges):
        if node not in entered:
            entered.add(node)
            for y in adjacency_list[node]:
                if y not in entered:
                    tree_edges[y] = node
                elif (not directed and tree_edges.get(node, None) != y
                      or directed and y not in exited):
                    back_edges.setdefault(y, set()).add(node)
                dfs(y, entered, exited, tree_edges, back_edges)
            exited.add(node)
        return tree_edges, back_edges
    for x in range(len(adjacency_list)):
        if dfs(x, entered=set(), exited=set(), tree_edges={}, back_edges={})[1]:
            return True
    else:
        return False


def _mask_adjacency_matrix_and_points(mask, adjacency_matrix, points):
    r"""
    Function that masks a provided adjacency matrix and points array.

    Parameters
    ----------
    mask : ``(n_vertices,)`` `ndarray`
        1D array of booleans
    adjacency_matrix : ``(n_vertices, n_vertices,)`` `ndarray`
        The adjacency matrix.
    points : ``(n_vertices, n_dims)`` `ndarray`
        The array representing the points.

    Returns
    -------
    adjacency_matrix : `ndarray`
        The masked adjacency matrix.
    points : `ndarray`
        The masked points array.

    Raises
    ------
    ValueError
        The provided mask deletes all edges.
    """
    # Find the indices that have been asked to be removed
    indices_to_remove = np.nonzero(~mask)[0]
    # Remove rows and columns from adjacency matrix
    adjacency_matrix = np.delete(adjacency_matrix, indices_to_remove, 0)
    adjacency_matrix = np.delete(adjacency_matrix, indices_to_remove, 1)
    if adjacency_matrix.size == 0:
        raise ValueError('The provided mask deletes all edges.')
    # remove rows from points
    points = points[mask, :]
    return adjacency_matrix, points


def _isolated_vertices(adjacency_matrix):
    all_vertices = set(range(adjacency_matrix.shape[0]))
    # find the set difference between {0, 1, ..., n_vertices} and the set
    # of rows (columns) that have at least one non-zero element.
    rows = all_vertices.difference(set(adjacency_matrix.nonzero()[0]))
    cols = all_vertices.difference(set(adjacency_matrix.nonzero()[1]))
    return list(rows.intersection(cols))


def _convert_edges_to_adjacency_matrix(edges, n_vertices):
    r"""
    Converts an edges array to and adjacency matrix.

    Parameters
    ----------
    edges : ``(n_edges, 2, )`` `ndarray` or ``None``
        The `ndarray` of edges, i.e. all the pairs of vertices that are
        connected with an edge.
    n_vertices : `int`
        The total number of vertices, assuming that the numbering of
        vertices starts from ``0``. ``edges`` and ``n_vertices`` can be
        defined in a way to set isolated vertices.

    Returns
    -------
    adjacency_matrix : ``(n_vertices, n_vertices, )`` `csr_matrix`
        The adjacency matrix of the graph in which the rows represent source
        vertices and columns represent destination vertices.
    """
    if isinstance(edges, list):
        edges = np.array(edges)
    if edges is None or edges.shape[0] == 0:
        # create adjacency with zeros
        return csr_matrix((n_vertices, n_vertices), dtype=np.int)
    else:
        # create sparse adjacency
        return csr_matrix(([1] * edges.shape[0], (edges[:, 0], edges[:, 1])),
                          shape=(n_vertices, n_vertices))


def _convert_edges_to_symmetric_adjacency_matrix(edges, n_vertices):
    r"""
    Converts an edges array to and adjacency matrix.

    Parameters
    ----------
    edges : ``(n_edges, 2, )`` `ndarray` or ``None``
        The `ndarray` of edges, i.e. all the pairs of vertices that are
        connected with an edge.
    n_vertices : `int`
        The total number of vertices, assuming that the numbering of
        vertices starts from ``0``. ``edges`` and ``n_vertices`` can be
        defined in a way to set isolated vertices.

    Returns
    -------
    adjacency_matrix : ``(n_vertices, n_vertices, )`` `csr_matrix`
        The adjacency matrix of the graph in which the rows represent source
        vertices and columns represent destination vertices.
    """
    if isinstance(edges, list):
        edges = np.array(edges)
    if edges is None or edges.shape[0] == 0:
        # create adjacency with zeros
        adjacency_matrix = csr_matrix((n_vertices, n_vertices), dtype=np.int)
    else:
        rows = np.hstack((edges[:, 0], edges[:, 1]))
        cols = np.hstack((edges[:, 1], edges[:, 0]))
        adjacency_matrix = csr_matrix(([1] * rows.shape[0], (rows, cols)),
                                      shape=(n_vertices, n_vertices))
        adjacency_matrix[adjacency_matrix.nonzero()] = 1
    return adjacency_matrix
