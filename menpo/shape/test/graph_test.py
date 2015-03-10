import numpy as np
from numpy.testing import assert_allclose
from nose.tools import raises
from scipy.sparse import csr_matrix

from menpo.shape import (UndirectedGraph, DirectedGraph, Tree,
                         PointUndirectedGraph, PointDirectedGraph, PointTree)

# Points for point graphs
points = np.array([[10, 30], [0, 20], [20, 20], [0, 10], [20, 10], [0, 0]])
points2 = np.array([[30, 30], [10, 20], [50, 20], [0, 10], [20, 10],
                    [50, 10], [0, 0], [20, 0], [50, 0]])
point = np.array([[10, 10]])

# Define undirected graph and pointgraph
adj_undirected = np.array([[0, 1, 1, 0, 0, 0],
                           [1, 0, 1, 1, 0, 0],
                           [1, 1, 0, 0, 1, 0],
                           [0, 1, 0, 0, 1, 1],
                           [0, 0, 1, 1, 0, 0],
                           [0, 0, 0, 1, 0, 0]])
g_undirected = UndirectedGraph(adj_undirected)
pg_undirected = PointUndirectedGraph(points, adj_undirected)

# Define directed graph and pointgraph
adj_directed = csr_matrix(([1] * 8, ([1, 2, 1, 2, 1, 2, 3, 3],
                                     [0, 0, 2, 1, 3, 4, 4, 5])), shape=(6, 6))
g_directed = DirectedGraph(adj_directed)
pg_directed = PointDirectedGraph(points, adj_directed)

# Define tree and pointtree
adj_tree = np.array([[0, 1, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0]])
g_tree = Tree(adj_tree, 0)
pg_tree = PointTree(points2, adj_tree, 0)

# Define undirected graph and pointgraph with isolated vertices
adj_isolated = csr_matrix(([1] * 6, ([0, 2, 2, 4, 3, 4], [2, 0, 4, 2, 4, 3])),
                          shape=(6, 6))
g_isolated = UndirectedGraph(adj_isolated)
pg_isolated = PointUndirectedGraph(points, adj_isolated)

# Define undirected graph and pointgraph with a single vertex
adj_single = np.array([[0]])
g_single = DirectedGraph(adj_single)
pg_single = PointDirectedGraph(point, adj_single)



@raises(ValueError)
def test_create_graph_exception():
    adj = np.array([[0, 1, 3],
                    [0, 2, 4]])
    UndirectedGraph(adj)


@raises(ValueError)
def test_create_tree_exception_1():
    adj = csr_matrix(([1] * 8, ([0, 0, 1, 1, 2, 3, 4, 5, 0],
                                [1, 2, 3, 4, 5, 6, 7, 8, 4])), shape=(9, 9))
    Tree(adj, 0)


@raises(ValueError)
def test_create_tree_exception_2():
    Tree(adj_tree, 20)


@raises(ValueError)
def test_create_tree_exception_3():
    adj = csr_matrix(([1] * 8, ([0, 0, 1, 1, 2, 3, 4, 5],
                                [1, 2, 3, 4, 5, 6, 7, 8])), shape=(10, 10))
    Tree(adj, 0)


def test_init_from_edges():
    g = PointDirectedGraph.init_from_edges(
        points, np.array([[1, 0], [2, 0], [1, 2], [2, 1], [1, 3], [2, 4],
                          [3, 4], [3, 5]]))
    assert (pg_directed.adjacency_matrix - g.adjacency_matrix).nnz == 0
    g = PointUndirectedGraph.init_from_edges(
        points, np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 4],
                          [3, 5]]))
    assert (pg_undirected.adjacency_matrix - g.adjacency_matrix).nnz == 0
    g = PointUndirectedGraph.init_from_edges(
        points, np.array([[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1],
                          [1, 3], [3, 1], [2, 4], [4, 2], [3, 4], [4, 3],
                          [3, 5], [5, 3]]))
    assert (pg_undirected.adjacency_matrix - g.adjacency_matrix).nnz == 0
    g = PointTree.init_from_edges(
        points2, np.array([[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [3, 6],
                           [4, 7], [5, 8]]), root_vertex=0)
    assert (pg_tree.adjacency_matrix - g.adjacency_matrix).nnz == 0
    g = PointUndirectedGraph.init_from_edges(
        points, np.array([[0, 2], [2, 4], [3, 4]]))
    assert (pg_isolated.adjacency_matrix - g.adjacency_matrix).nnz == 0
    g = PointDirectedGraph.init_from_edges(point, np.array([]))
    assert (pg_single.adjacency_matrix - g.adjacency_matrix).nnz == 0


def test_n_edges():
    assert_allclose(g_directed.n_edges, 8)
    assert_allclose(g_undirected.n_edges, 7)
    assert_allclose(g_tree.n_edges, 8)
    assert_allclose(g_isolated.n_edges, 3)
    assert_allclose(g_single.n_edges, 0)


def test_edges():
    assert_allclose(g_directed.edges, np.array([[1, 0], [1, 2], [1, 3], [2, 0],
                                                [2, 1], [2, 4], [3, 4],
                                                [3, 5]]))
    assert_allclose(g_undirected.edges, np.array([[0, 1], [0, 2], [1, 2],
                                                  [1, 3], [2, 4], [3, 4],
                                                  [3, 5]]))
    assert_allclose(g_tree.edges, np.array([[0, 1], [0, 2], [1, 3], [1, 4],
                                            [2, 5], [3, 6], [4, 7], [5, 8]]))
    assert_allclose(g_isolated.edges, np.array([[0, 2], [2, 4], [3, 4]]))
    assert_allclose(g_single.edges, np.empty((0, 2)))


def test_n_vertices():
    assert_allclose(g_directed.n_vertices, 6)
    assert_allclose(g_undirected.n_vertices, 6)
    assert_allclose(g_tree.n_vertices, 9)
    assert_allclose(g_isolated.n_vertices, 6)
    assert_allclose(g_single.n_vertices, 1)


def test_vertices():
    assert (list(g_directed.vertices) == [0, 1, 2, 3, 4, 5])
    assert (list(g_undirected.vertices) == [0, 1, 2, 3, 4, 5])
    assert (list(g_tree.vertices) == [0, 1, 2, 3, 4, 5, 6, 7, 8])
    assert (list(g_isolated.vertices) == [0, 1, 2, 3, 4, 5])
    assert (list(g_single.vertices) == [0])


def test_isolated_vertices():
    assert (g_directed.isolated_vertices() == [])
    assert not g_directed.has_isolated_vertices()
    assert (g_undirected.isolated_vertices() == [])
    assert not g_undirected.has_isolated_vertices()
    assert (g_tree.isolated_vertices() == [])
    assert not g_tree.has_isolated_vertices()
    assert (g_isolated.isolated_vertices() == [1, 5])
    assert g_isolated.has_isolated_vertices()
    assert (g_single.isolated_vertices() == [0])
    assert g_single.has_isolated_vertices()


def test_adjacency_list():
    assert (g_directed.get_adjacency_list() ==
            [[], [0, 2, 3], [0, 1, 4], [4, 5], [], []])
    assert (g_undirected.get_adjacency_list() ==
            [[1, 2], [0, 2, 3], [0, 1, 4], [1, 4, 5], [2, 3], [3]])
    assert (g_tree.get_adjacency_list() ==
            [[1, 2], [3, 4], [5], [6], [7], [8], [], [], []])
    assert (g_isolated.get_adjacency_list() ==
            [[2], [], [0, 4], [4], [2, 3], []])
    assert (g_single.get_adjacency_list() == [[]])


def test_n_paths():
    assert_allclose(g_directed.n_paths(1, 4), 2)
    assert_allclose(g_undirected.n_paths(0, 5), 4)
    assert_allclose(g_tree.n_paths(2, 6), 0)
    assert_allclose(g_isolated.n_paths(3, 2), 1)
    assert_allclose(g_single.n_paths(0, 0), 1)


def test_find_all_paths():
    assert (g_directed.find_all_paths(1, 4) == [[1, 2, 4], [1, 3, 4]])
    assert (g_undirected.find_all_paths(0, 5) ==
            [[0, 1, 2, 4, 3, 5], [0, 1, 3, 5], [0, 2, 1, 3, 5],
             [0, 2, 4, 3, 5]])
    assert (g_tree.find_all_paths(2, 6) == [])
    assert (g_isolated.find_all_paths(1, 5) == [])
    assert (g_single.find_all_paths(0, 10) == [])


def test_find_path():
    assert (g_directed.find_all_paths(1, 4)[0] == g_directed.find_path(1, 4))
    assert (g_undirected.find_all_paths(0, 5)[1] ==
            g_undirected.find_path(0, 5))
    assert (g_tree.find_path(2, 6) == [])
    assert (g_isolated.find_path(4, 1) == [])
    assert (g_single.find_path(0, 0) == [])


def test_find_shortest_path():
    assert (g_directed.find_shortest_path(1, 0) == ([1, 0], 0.0))
    assert (g_undirected.find_shortest_path(5, 0) == ([5, 3, 1, 0], 3.0))
    assert (g_tree.find_shortest_path(1, 7) == ([1, 4, 7], 1.0))
    assert (g_isolated.find_shortest_path(3, 0) == ([3, 4, 2, 0], 3.0))
    assert (g_single.find_shortest_path(0, 0) == ([], np.inf))


def test_neighbours_children_parent():
    assert (g_directed.children(3) == [4, 5])
    assert (g_directed.n_children(0) == 0)
    assert (g_directed.parents(0) == [1, 2])
    assert (g_directed.n_parents(3) == 1)

    assert (g_undirected.neighbours(1) == [0, 2, 3])
    assert (g_undirected.n_neighbours(3) == 3)

    assert (g_tree.children(5) == [8])
    assert (g_tree.n_children(6) == 0)
    assert (g_tree.parent(4) == [1])

    assert (g_isolated.neighbours(1) == [])
    assert (g_isolated.n_neighbours(4) == 2)

    assert (g_single.children(0) == [])
    assert (g_single.n_children(0) == 0)
    assert (g_single.parents(0) == [])
    assert (g_single.n_parents(0) == 0)


def test_tree():
    assert (g_tree.maximum_depth == 3)
    assert (g_tree.depth_of_vertex(0) == 0)
    assert (g_tree.depth_of_vertex(4) == 2)
    assert (g_tree.n_vertices_at_depth(2) == 3)
    assert g_tree.is_leaf(7)
    assert not g_tree.is_leaf(5)
    assert (g_tree.leaves == [6, 7, 8])
    assert (g_tree.n_leaves == 3)


def test_minimum_spanning_tree():
    adjacency_matrix = np.array([[0, 11, 13, 12],
                                 [11, 0, 0, 14],
                                 [13, 0, 0, 10],
                                 [12, 14, 10, 0]])
    g = UndirectedGraph(adjacency_matrix)
    t = g.minimum_spanning_tree(root_vertex=0)
    assert t.n_edges == 3
    assert_allclose(t.adjacency_matrix.todense(),
                    csr_matrix(([11., 12., 10.], ([0, 0, 3], [1, 3, 2])),
                               shape=(4, 4)).todense())
    assert t.get_adjacency_list() == [[1, 3], [], [], [2]]
    assert t.predecessors_list == [None, 0, 3, 0]


def test_is_edge():
    assert g_directed.is_edge(2, 4)
    assert not g_directed.is_edge(3, 1)
    assert not g_directed.is_edge(5, 0)

    assert g_undirected.is_edge(2, 1)
    assert g_undirected.is_edge(1, 2)
    assert not g_undirected.is_edge(5, 0)

    assert g_tree.is_edge(4, 7)
    assert not g_tree.is_edge(6, 3)
    assert not g_tree.is_edge(8, 1)

    assert g_isolated.is_edge(0, 2)
    assert g_isolated.is_edge(4, 2)
    assert not g_isolated.is_edge(1, 5)

    assert not g_single.is_edge(0, 0)


def test_is_tree():
    assert not g_undirected.is_tree()
    assert not g_directed.is_tree()
    assert g_tree.is_tree()
    assert not g_isolated.is_tree()
    assert g_single.is_tree()


def test_from_mask():
    assert (pg_directed.from_mask(
        np.array([False, False, True,
                  True, True, False])).get_adjacency_list() == [[2], [2], []])
    assert (pg_undirected.from_mask(
        np.array([True, False, True,
                  True, True, False])).get_adjacency_list() ==
            [[1], [0, 3], [3], [1, 2]])
    assert (pg_tree.from_mask(
        np.array([True, False, True, True, True,
                  True, True, True, False])).get_adjacency_list() ==
            [[1], [2], []])
    assert (pg_isolated.from_mask(
        np.array([True, True, False,
                  True, True, False])).get_adjacency_list() ==
            [[], [], [3], [2]])
    assert (pg_single.from_mask(np.array([True])).get_adjacency_list() == [[]])


@raises(ValueError)
def test_from_mask_errors():
    pg_directed.from_mask(np.array([False, False, False, False, False, False]))
    pg_undirected.from_mask(np.array([False, False, False, False, False, True]))
    pg_tree.from_mask(np.array([False, True, True, True, True, True, True,
                                True, True]))
    pg_isolated.from_mask(np.array([True, True, False, True, False, True]))
    pg_single.from_mask(np.array([False]))


def test_relative_locations():
    assert_allclose(pg_tree.relative_location_edge(5, 8), np.array([0, -10]))
    assert_allclose(pg_tree.relative_locations(), np.array([[-20, -10],
                                                            [20, -10],
                                                            [-10, -10],
                                                            [10, -10],
                                                            [0, -10],
                                                            [0, -10],
                                                            [0, -10],
                                                            [0, -10]]))


@raises(ValueError)
def test_relative_locations():
    pg_tree.relative_location_edge(8, 5)
    pg_tree.relative_location_edge(0, 6)
