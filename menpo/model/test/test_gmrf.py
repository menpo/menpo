import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from menpo.shape import PointCloud, DirectedGraph, UndirectedGraph
from menpo.math import as_matrix

from .. import GMRFModel, GMRFVectorModel


def _compute_sum_cost_block_sparse(samples, test_sample, graph,
                                   n_features_per_vertex, subtract_mean, mode):
    # create ndarray with data
    data = as_matrix(samples, length=None, return_template=False)
    # initialize cost
    cost = 0.
    # for loop over the graph's edges
    for e in graph.edges:
        v1 = e[0]
        v2 = e[1]
        v1_from = v1 * n_features_per_vertex
        v1_to = (v1 + 1) * n_features_per_vertex
        v2_from = v2 * n_features_per_vertex
        v2_to = (v2 + 1) * n_features_per_vertex
        # slice data and test vector
        y = test_sample.as_vector()
        if mode == 'concatenation':
            x = np.hstack((data[:, v1_from:v1_to], data[:, v2_from:v2_to]))
            y = np.hstack((y[v1_from:v1_to], y[v2_from:v2_to]))
        else:
            x = data[:, v1_from:v1_to] - data[:, v2_from:v2_to]
            y = y[v1_from:v1_to] - y[v2_from:v2_to]
        # compute mean and covariance
        cov = np.linalg.inv(np.cov(x.T))
        mean = np.mean(x, axis=0)
        # compute and sum cost
        if subtract_mean:
            v = y - mean
        else:
            v = y
        cost += v.dot(cov).T.dot(v)
    return cost


def _compute_sum_cost_block_diagonal(samples, test_sample, graph,
                                     n_features_per_vertex, subtract_mean):
    # create ndarray with data
    data = as_matrix(samples, length=None, return_template=False)
    # initialize cost
    cost = 0.
    # for loop over the graph's edges
    for v1 in graph.vertices:
        v1_from = v1 * n_features_per_vertex
        v1_to = (v1 + 1) * n_features_per_vertex
        # slice data and test vector
        y = test_sample.as_vector()
        x = data[:, v1_from:v1_to]
        y = y[v1_from:v1_to]
        # compute mean and covariance
        cov = np.linalg.inv(np.cov(x.T))
        mean = np.mean(x, axis=0)
        # compute and sum cost
        if subtract_mean:
            v = y - mean
        else:
            v = y
        cost += v.dot(cov).T.dot(v)
    return cost


def test_mahalanobis_distance():
    # arguments values
    mode_values = ['concatenation', 'subtraction']
    n_features_per_vertex_values = [2, 3]
    sparse_values = [True, False]
    subtract_mean_values = [True, False]
    n_components_values = [None, 30]

    # create graph
    n_vertices = 6
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])
    graphs = [DirectedGraph.init_from_edges(edges, n_vertices),
              UndirectedGraph(np.zeros((n_vertices, n_vertices)))]

    for n_features_per_vertex in n_features_per_vertex_values:
        # create samples
        n_samples = 50
        samples = []
        for i in range(n_samples):
            samples.append(PointCloud(np.random.rand(n_vertices,
                                                     n_features_per_vertex)))
        test_sample = PointCloud(np.random.rand(n_vertices,
                                                n_features_per_vertex))

        for graph in graphs:
            for mode in mode_values:
                for sparse in sparse_values:
                    for n_components in n_components_values:
                        # train GMRF
                        gmrf = GMRFModel(
                            samples, graph, mode=mode, sparse=sparse,
                            n_components=n_components, dtype=np.float64)

                        for subtract_mean in subtract_mean_values:
                            # compute costs
                            if graph.n_edges == 0:
                                cost1 = _compute_sum_cost_block_diagonal(
                                    samples, test_sample, graph,
                                    n_features_per_vertex, subtract_mean)
                            else:
                                cost1 = _compute_sum_cost_block_sparse(
                                    samples, test_sample, graph,
                                    n_features_per_vertex, subtract_mean, mode)
                            cost2 = gmrf.mahalanobis_distance(
                                test_sample, subtract_mean=subtract_mean)
                            assert_almost_equal(cost1, cost2)


def test_increment():
    # arguments values
    mode_values = ['concatenation', 'subtraction']
    n_features_per_vertex_values = [2, 3]
    sparse_values = [True, False]
    n_components_values = [None, 30]

    # create graph
    n_vertices = 6
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])
    graphs = [DirectedGraph.init_from_edges(edges, n_vertices),
              UndirectedGraph(np.zeros((n_vertices, n_vertices)))]

    for n_features_per_vertex in n_features_per_vertex_values:
        # create samples
        n_samples = 100
        samples = []
        for i in range(n_samples):
            samples.append(np.random.rand(n_vertices * n_features_per_vertex))

        for graph in graphs:
            for mode in mode_values:
                for sparse in sparse_values:
                    for n_components in n_components_values:
                        # Incremental GMRF
                        gmrf1 = GMRFVectorModel(
                            samples[:50], graph, mode=mode, sparse=sparse,
                            n_components=n_components, dtype=np.float64,
                            incremental=True)
                        gmrf1.increment(samples[50::])

                        # Non incremental GMRF
                        gmrf2 = GMRFVectorModel(
                            samples, graph, mode=mode, sparse=sparse,
                            n_components=n_components, dtype=np.float64)

                        # Compare
                        if sparse:
                            assert_array_almost_equal(gmrf1.precision.todense(),
                                                      gmrf2.precision.todense())
                        assert_array_almost_equal(gmrf1.mean_vector,
                                                  gmrf2.mean_vector)
