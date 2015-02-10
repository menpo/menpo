import numpy as np
from numpy.testing import assert_allclose
from nose.tools import raises

from menpo.shape import UndirectedGraph, DirectedGraph, Tree

# Define adjacency arrays
adj_undirected = np.array([[0, 1],
                           [0, 2],
                           [1, 2],
                           [1, 3],
                           [2, 4],
                           [3, 4],
                           [3, 5]])
g_undirected = UndirectedGraph(adj_undirected)
adj_directed = np.array([[1, 0],
                         [2, 0],
                         [1, 2],
                         [2, 1],
                         [1, 3],
                         [2, 4],
                         [3, 4],
                         [3, 5]])
g_directed = DirectedGraph(adj_directed)
adj_tree = np.array([[0, 1],
                     [0, 2],
                     [1, 3],
                     [1, 4],
                     [2, 5],
                     [3, 6],
                     [4, 7],
                     [5, 8]])
g_tree = Tree(adj_tree, 0)


@raises(ValueError)
def test_create_graph_exception():
    adj = np.array([[0, 1, 3],
                    [0, 2, 4]])
    UndirectedGraph(adj)


@raises(ValueError)
def test_create_tree_exception_1():
    adj = np.array([[0, 1],
                    [0, 2],
                    [1, 3],
                    [1, 4],
                    [2, 5],
                    [3, 6],
                    [4, 7],
                    [5, 8],
                    [0, 4]])
    Tree(adj, 0)


@raises(ValueError)
def test_create_tree_exception_2():
    Tree(adj_tree, 20)


def test_n_edges():
    assert_allclose(g_directed.n_edges, 8)
    assert_allclose(g_undirected.n_edges, 7)
    assert_allclose(g_tree.n_edges, 8)


def test_n_vertices():
    assert_allclose(g_directed.n_vertices, 6)
    assert_allclose(g_undirected.n_vertices, 6)
    assert_allclose(g_tree.n_vertices, 9)


def test_adjacency_list():
    assert (g_directed.adjacency_list ==
            [[], [0, 2, 3], [0, 1, 4], [4, 5], [], []])
    assert (g_undirected.adjacency_list ==
            [[1, 2], [0, 2, 3], [0, 1, 4], [1, 4, 5], [2, 3], [3]])
    assert (g_tree.adjacency_list ==
            [[1, 2], [3, 4], [5], [6], [7], [8], [], [], []])


def test_adjacency_matrix():
    assert_allclose(g_directed.get_adjacency_matrix(),
                    np.array([[False, False, False, False, False, False],
                              [True, False, True, True, False, False],
                              [True, True, False, False, True, False],
                              [False, False, False, False, True, True],
                              [False, False, False, False, False, False],
                              [False, False, False, False, False, False]]))
    assert_allclose(g_undirected.get_adjacency_matrix(),
                    np.array([[False, True, True, False, False, False],
                              [True, False, True, True, False, False],
                              [True, True, False, False, True, False],
                              [False, True, False, False, True, True],
                              [False, False, True, True, False, False],
                              [False, False, False, True, False, False]]))
    assert_allclose(g_tree.get_adjacency_matrix(),
                    np.array([[False, True, True, False, False, False, False, False, False],
                              [False, False, False, True, True, False, False, False, False],
                              [False, False, False, False, False, True, False, False, False],
                              [False, False, False, False, False, False, True, False, False],
                              [False, False, False, False, False, False, False, True, False],
                              [False, False, False, False, False, False, False, False, True],
                              [False, False, False, False, False, False, False, False, False],
                              [False, False, False, False, False, False, False, False, False],
                              [False, False, False, False, False, False, False, False, False]]))


def test_n_paths():
    assert_allclose(g_directed.n_paths(1, 4), 2)
    assert_allclose(g_undirected.n_paths(0, 5), 4)
    assert_allclose(g_tree.n_paths(2, 6), 0)


def test_find_all_paths():
    assert (g_directed.find_all_paths(1, 4) == [[1, 2, 4], [1, 3, 4]])
    assert (g_undirected.find_all_paths(0, 5) ==
            [[0, 1, 2, 4, 3, 5], [0, 1, 3, 5], [0, 2, 1, 3, 5], [0, 2, 4, 3, 5]])
    assert (g_tree.find_all_paths(2, 6) == [])


def test_find_path():
    assert (g_directed.find_all_paths(1, 4)[0] == g_directed.find_path(1, 4))
    assert (g_undirected.find_all_paths(0, 5)[0] == g_undirected.find_path(0, 5))
    assert (g_tree.find_path(2, 6) is None)


def test_find_shortest_path():
    assert (g_directed.find_shortest_path(1, 0) == [1, 0])
    assert (g_undirected.find_shortest_path(5, 0) == [5, 3, 1, 0])
    assert (g_tree.find_shortest_path(1, 7) == [1, 4, 7])


def test_neighbours_children_parent():
    assert (g_directed.children(3) == [4, 5])
    assert (g_directed.n_children(0) == 0)
    assert (g_directed.parent(0) == [1, 2])
    assert (g_directed.n_parent(3) == 1)

    assert (g_undirected.neighbours(1) == [0, 2, 3])
    assert (g_undirected.n_neighbours(3) == 3)

    assert (g_tree.children(5) == [8])
    assert (g_tree.n_children(6) == 0)
    assert (g_tree.parent(4) == [1])


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
    adjacency_array = np.array([[3, 1],
                                [2, 3],
                                [0, 3],
                                [2, 0],
                                [0, 1]])
    weights = np.array([[0, 11, 13, 12],
                        [11, 0, 0, 14],
                        [13, 0, 0, 10],
                        [12, 14, 10, 0]])
    g = UndirectedGraph(adjacency_array)
    t = g.minimum_spanning_tree(weights, root_vertex=0)
    assert t.n_edges == 3
    assert_allclose(t.adjacency_array, np.array([[0, 1], [0, 3], [3, 2]]))
    assert t.adjacency_list == [[1, 3], [], [], [2]]
    assert_allclose(t.get_adjacency_matrix(),
                    np.array([[False, True, False, True],
                              [False, False, False, False],
                              [False, False, False, False],
                              [False, False, True, False]]))
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


def test_is_tree():
    assert not g_undirected.is_tree()
    assert not g_directed.is_tree()
    assert g_tree.is_tree()
