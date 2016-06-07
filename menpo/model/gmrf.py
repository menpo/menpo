from functools import partial
import numpy as np
from scipy.sparse import bsr_matrix

from menpo.base import name_of_callable
from menpo.math import as_matrix
from menpo.shape import UndirectedGraph
from menpo.visualize import print_progress, bytes_str, print_dynamic


def _covariance_matrix_inverse(cov_mat, n_components):
    if n_components is None:
        return np.linalg.inv(cov_mat)
    else:
        try:
            s, v, d = np.linalg.svd(cov_mat)
            s = s[:, :n_components]
            v = v[:n_components]
            d = d[:n_components, :]
            return s.dot(np.diag(1/v)).dot(d)
        except:
            return np.linalg.inv(cov_mat)


def _create_sparse_precision(X, graph, n_features, n_features_per_vertex,
                             mode='concatenation', dtype=np.float32,
                             n_components=None, bias=0,
                             return_covariances=False, verbose=False):
    # check mode argument
    if mode not in ['concatenation', 'subtraction']:
        raise ValueError("mode must be either ''concatenation'' "
                         "or ''subtraction''; {} is given.".format(mode))

    # Initialize arrays
    all_blocks = np.zeros((graph.n_edges * 4,
                           n_features_per_vertex, n_features_per_vertex),
                          dtype=dtype)
    if return_covariances:
        if mode == 'concatenation':
            cov_shape = (graph.n_edges,
                         2 * n_features_per_vertex, 2 * n_features_per_vertex)
        else:
            cov_shape = (graph.n_edges,
                         n_features_per_vertex, n_features_per_vertex)
        all_covariances = np.zeros(cov_shape, dtype=dtype)
    columns = np.zeros(graph.n_edges * 4)
    rows = np.zeros(graph.n_edges * 4)

    # Print information if asked
    if verbose:
        edges = print_progress(range(graph.n_edges), n_items=graph.n_edges,
                               prefix='Precision per edge',
                               end_with_newline=False)
    else:
        edges = range(graph.n_edges)

    # Compute covariance matrix for each edge, invert it and store it
    count = -1
    for e in edges:
        # edge vertices
        v1 = graph.edges[e, 0]
        v2 = graph.edges[e, 1]

        # find indices in data matrix
        v1_from = v1 * n_features_per_vertex
        v1_to = (v1 + 1) * n_features_per_vertex
        v2_from = v2 * n_features_per_vertex
        v2_to = (v2 + 1) * n_features_per_vertex

        # data concatenation
        if mode == 'concatenation':
            edge_data = X[:, list(range(v1_from, v1_to)) +
                             list(range(v2_from, v2_to))]
        else:
            edge_data = X[:, v1_from:v1_to] - X[:, v2_from:v2_to]

        # compute covariance matrix
        covmat = np.cov(edge_data, rowvar=0, bias=bias)
        if return_covariances:
            all_covariances[e] = covmat

        # invert it
        covmat = _covariance_matrix_inverse(covmat, n_components)

        # store it
        if mode == 'concatenation':
            # v1, v1
            count += 1
            all_blocks[count] = \
                covmat[:n_features_per_vertex, :n_features_per_vertex]
            rows[count] = v1
            columns[count] = v1
            # v2, v2
            count += 1
            all_blocks[count] = \
                covmat[n_features_per_vertex::, n_features_per_vertex::]
            rows[count] = v2
            columns[count] = v2
            # v1, v2
            count += 1
            all_blocks[count] = \
                covmat[:n_features_per_vertex, n_features_per_vertex::]
            rows[count] = v1
            columns[count] = v2
            # v2, v1
            count += 1
            all_blocks[count] = \
                covmat[n_features_per_vertex::, :n_features_per_vertex]
            rows[count] = v2
            columns[count] = v1
        else:
            # v1, v1
            count += 1
            all_blocks[count] = covmat
            rows[count] = v1
            columns[count] = v1
            # v2, v2
            count += 1
            all_blocks[count] = covmat
            rows[count] = v2
            columns[count] = v2
            # v1, v2
            count += 1
            all_blocks[count] = -covmat
            rows[count] = v1
            columns[count] = v2
            # v2, v1
            count += 1
            all_blocks[count] = -covmat
            rows[count] = v2
            columns[count] = v1

    # sort rows, columns and all_blocks
    rows_arg_sort = rows.argsort()
    columns = columns[rows_arg_sort]
    all_blocks = all_blocks[rows_arg_sort]
    rows = rows[rows_arg_sort]

    # create indptr
    n_rows = graph.n_vertices
    indptr = np.zeros(n_rows + 1)
    for i in range(n_rows):
        inds, = np.where(rows == i)
        if inds.size == 0:
            indptr[i + 1] = indptr[i]
        else:
            indptr[i] = inds[0]
            indptr[i + 1] = inds[-1] + 1

    # create block sparse matrix
    if return_covariances:
        return (bsr_matrix((all_blocks, columns, indptr),
                           shape=(n_features, n_features), dtype=dtype),
                all_covariances)
    else:
        return bsr_matrix((all_blocks, columns, indptr),
                          shape=(n_features, n_features), dtype=dtype)


def _create_dense_precision(X, graph, n_features, n_features_per_vertex,
                            mode='concatenation', dtype=np.float32,
                            n_components=None, bias=0,
                            return_covariances=False, verbose=False):
    # check mode argument
    if mode not in ['concatenation', 'subtraction']:
        raise ValueError("mode must be either ''concatenation'' "
                         "or ''subtraction''; {} is given.".format(mode))

    # Initialize precision
    precision = np.zeros((n_features, n_features), dtype=dtype)
    if return_covariances:
        if mode == 'concatenation':
            cov_shape = (graph.n_edges,
                         2 * n_features_per_vertex, 2 * n_features_per_vertex)
        else:
            cov_shape = (graph.n_edges,
                         n_features_per_vertex, n_features_per_vertex)
        all_covariances = np.zeros(cov_shape, dtype=dtype)

    # Print information if asked
    if verbose:
        print_dynamic('Allocated precision matrix of size {}'.format(
            bytes_str(precision.nbytes)))
        edges = print_progress(range(graph.n_edges), n_items=graph.n_edges,
                               prefix='Precision per edge',
                               end_with_newline=False)
    else:
        edges = range(graph.n_edges)

    # Compute covariance matrix for each edge, invert it and store it
    for e in edges:
        # edge vertices
        v1 = graph.edges[e, 0]
        v2 = graph.edges[e, 1]

        # find indices in data matrix
        v1_from = v1 * n_features_per_vertex
        v1_to = (v1 + 1) * n_features_per_vertex
        v2_from = v2 * n_features_per_vertex
        v2_to = (v2 + 1) * n_features_per_vertex

        # data concatenation
        if mode == 'concatenation':
            edge_data = X[:, list(range(v1_from, v1_to)) +
                             list(range(v2_from, v2_to))]
        else:
            edge_data = X[:, v1_from:v1_to] - X[:, v2_from:v2_to]

        # compute covariance matrix
        covmat = np.cov(edge_data, rowvar=0, bias=bias)
        if return_covariances:
            all_covariances[e] = covmat

        # invert it
        covmat = _covariance_matrix_inverse(covmat, n_components)

        # store it
        if mode == 'concatenation':
            # v1, v1
            precision[v1_from:v1_to, v1_from:v1_to] += \
                covmat[:n_features_per_vertex, :n_features_per_vertex]
            # v2, v2
            precision[v2_from:v2_to, v2_from:v2_to] += \
                covmat[n_features_per_vertex::, n_features_per_vertex::]
            # v1, v2
            precision[v1_from:v1_to, v2_from:v2_to] = \
                covmat[:n_features_per_vertex, n_features_per_vertex::]
            # v2, v1
            precision[v2_from:v2_to, v1_from:v1_to] = \
                covmat[n_features_per_vertex::, :n_features_per_vertex]
        elif mode == 'subtraction':
            # v1, v2
            precision[v1_from:v1_to, v2_from:v2_to] = -covmat
            # v2, v1
            precision[v2_from:v2_to, v1_from:v1_to] = -covmat
            # v1, v1
            precision[v1_from:v1_to, v1_from:v1_to] += covmat
            # v2, v2
            precision[v2_from:v2_to, v2_from:v2_to] += covmat

    # return covariances
    if return_covariances:
        return precision, all_covariances
    else:
        return precision


def _create_sparse_diagonal_precision(X, graph, n_features,
                                      n_features_per_vertex,
                                      dtype=np.float32, n_components=None,
                                      bias=0, return_covariances=False,
                                      verbose=False):
    # initialize covariances matrix
    all_blocks = np.zeros((graph.n_vertices,
                           n_features_per_vertex, n_features_per_vertex),
                          dtype=dtype)
    if return_covariances:
        all_covariances = np.zeros(
            (graph.n_vertices, n_features_per_vertex, n_features_per_vertex),
            dtype=dtype)
    columns = np.zeros(graph.n_vertices)
    rows = np.zeros(graph.n_vertices)

    # Print information if asked
    if verbose:
        vertices = print_progress(
            range(graph.n_vertices), n_items=graph.n_vertices,
            prefix='Precision per vertex', end_with_newline=False)
    else:
        vertices = range(graph.n_vertices)

    # Compute covariance matrix for each patch
    for v in vertices:
        # find indices in target precision matrix
        i_from = v * n_features_per_vertex
        i_to = (v + 1) * n_features_per_vertex

        # compute covariance
        covmat = np.cov(X[:, i_from:i_to], rowvar=0, bias=bias)
        if return_covariances:
            all_covariances[v] = covmat

        # invert it
        all_blocks[v] = _covariance_matrix_inverse(covmat, n_components)

        # store the inverse covariance and its locations
        rows[v] = v
        columns[v] = v

    # sort rows, columns and all_blocks
    rows_arg_sort = rows.argsort()
    columns = columns[rows_arg_sort]
    all_blocks = all_blocks[rows_arg_sort]
    rows = rows[rows_arg_sort]

    # create indptr
    n_rows = graph.n_vertices
    indptr = np.zeros(n_rows + 1)
    for i in range(n_rows):
        inds, = np.where(rows == i)
        if inds.size == 0:
            indptr[i + 1] = indptr[i]
        else:
            indptr[i] = inds[0]
            indptr[i + 1] = inds[-1] + 1

    # create block sparse matrix
    if return_covariances:
        return (bsr_matrix((all_blocks, columns, indptr),
                           shape=(n_features, n_features), dtype=dtype),
                all_covariances)
    else:
        return bsr_matrix((all_blocks, columns, indptr),
                          shape=(n_features, n_features), dtype=dtype)


def _create_dense_diagonal_precision(X, graph, n_features,
                                     n_features_per_vertex,
                                     dtype=np.float32, n_components=None,
                                     bias=0, return_covariances=False,
                                     verbose=False):
    # Initialize precision
    precision = np.zeros((n_features, n_features), dtype=dtype)
    if return_covariances:
        all_covariances = np.zeros(
            (graph.n_vertices, n_features_per_vertex, n_features_per_vertex),
            dtype=dtype)
    if verbose:
        print_dynamic('Allocated precision matrix of size {}'.format(
            bytes_str(precision.nbytes)))

    # Print information if asked
    if verbose:
        vertices = print_progress(
            range(graph.n_vertices), n_items=graph.n_vertices,
            prefix='Precision per vertex', end_with_newline=False)
    else:
        vertices = range(graph.n_vertices)

    # Compute covariance matrix for each patch
    for v in vertices:
        # find indices in target precision matrix
        i_from = v * n_features_per_vertex
        i_to = (v + 1) * n_features_per_vertex

        # compute covariance
        covmat = np.cov(X[:, i_from:i_to], rowvar=0, bias=bias)
        if return_covariances:
            all_covariances[v] = covmat

        # invert it
        covmat = _covariance_matrix_inverse(covmat, n_components)

        # insert to precision matrix
        precision[i_from:i_to, i_from:i_to] = covmat

    # return covariances
    if return_covariances:
        return precision, all_covariances
    else:
        return precision


def _increment_sparse_precision(X, mean_vector, covariances, n, graph,
                                n_features, n_features_per_vertex,
                                mode='concatenation', dtype=np.float32,
                                n_components=None, bias=0, verbose=False):
    # check mode argument
    if mode not in ['concatenation', 'subtraction']:
        raise ValueError("mode must be either ''concatenation'' "
                         "or ''subtraction''; {} is given.".format(mode))

    # Initialize arrays
    all_blocks = np.zeros((graph.n_edges * 4,
                           n_features_per_vertex, n_features_per_vertex),
                          dtype=dtype)
    columns = np.zeros(graph.n_edges * 4)
    rows = np.zeros(graph.n_edges * 4)

    # Print information if asked
    if verbose:
        edges = print_progress(range(graph.n_edges), n_items=graph.n_edges,
                               prefix='Precision per edge',
                               end_with_newline=False)
    else:
        edges = range(graph.n_edges)

    # Compute covariance matrix for each edge, invert it and store it
    count = -1
    for e in edges:
        # edge vertices
        v1 = graph.edges[e, 0]
        v2 = graph.edges[e, 1]

        # find indices in data matrix
        v1_from = v1 * n_features_per_vertex
        v1_to = (v1 + 1) * n_features_per_vertex
        v2_from = v2 * n_features_per_vertex
        v2_to = (v2 + 1) * n_features_per_vertex

        # data concatenation
        if mode == 'concatenation':
            edge_data = X[:, list(range(v1_from, v1_to)) +
                             list(range(v2_from, v2_to))]
            m = mean_vector[list(range(v1_from, v1_to)) +
                            list(range(v2_from, v2_to))]
        else:
            edge_data = X[:, v1_from:v1_to] - X[:, v2_from:v2_to]
            m = mean_vector[v1_from:v1_to] - mean_vector[v2_from:v2_to]

        # increment
        _, covariances[e] = _increment_multivariate_gaussian_cov(
            edge_data, m, covariances[e], n, bias=bias)

        # invert it
        covmat = _covariance_matrix_inverse(covariances[e], n_components)

        # store it
        if mode == 'concatenation':
            # v1, v1
            count += 1
            all_blocks[count] = \
                covmat[:n_features_per_vertex, :n_features_per_vertex]
            rows[count] = v1
            columns[count] = v1
            # v2, v2
            count += 1
            all_blocks[count] = \
                covmat[n_features_per_vertex::, n_features_per_vertex::]
            rows[count] = v2
            columns[count] = v2
            # v1, v2
            count += 1
            all_blocks[count] = \
                covmat[:n_features_per_vertex, n_features_per_vertex::]
            rows[count] = v1
            columns[count] = v2
            # v2, v1
            count += 1
            all_blocks[count] = \
                covmat[n_features_per_vertex::, :n_features_per_vertex]
            rows[count] = v2
            columns[count] = v1
        else:
            # v1, v1
            count += 1
            all_blocks[count] = covmat
            rows[count] = v1
            columns[count] = v1
            # v2, v2
            count += 1
            all_blocks[count] = covmat
            rows[count] = v2
            columns[count] = v2
            # v1, v2
            count += 1
            all_blocks[count] = -covmat
            rows[count] = v1
            columns[count] = v2
            # v2, v1
            count += 1
            all_blocks[count] = -covmat
            rows[count] = v2
            columns[count] = v1

    # sort rows, columns and all_blocks
    rows_arg_sort = rows.argsort()
    columns = columns[rows_arg_sort]
    all_blocks = all_blocks[rows_arg_sort]
    rows = rows[rows_arg_sort]

    # create indptr
    n_rows = graph.n_vertices
    indptr = np.zeros(n_rows + 1)
    for i in range(n_rows):
        inds, = np.where(rows == i)
        if inds.size == 0:
            indptr[i + 1] = indptr[i]
        else:
            indptr[i] = inds[0]
            indptr[i + 1] = inds[-1] + 1

    # create block sparse matrix
    return (bsr_matrix((all_blocks, columns, indptr),
                       shape=(n_features, n_features), dtype=dtype),
            covariances)


def _increment_dense_precision(X, mean_vector, covariances, n, graph,
                               n_features, n_features_per_vertex,
                               mode='concatenation', dtype=np.float32,
                               n_components=None, bias=0, verbose=False):
    # check mode argument
    if mode not in ['concatenation', 'subtraction']:
        raise ValueError("mode must be either ''concatenation'' "
                         "or ''subtraction''; {} is given.".format(mode))

    # Initialize precision
    precision = np.zeros((n_features, n_features), dtype=dtype)

    # Print information if asked
    if verbose:
        print_dynamic('Allocated precision matrix of size {}'.format(
            bytes_str(precision.nbytes)))
        edges = print_progress(range(graph.n_edges), n_items=graph.n_edges,
                               prefix='Precision per edge',
                               end_with_newline=False)
    else:
        edges = range(graph.n_edges)

    # Compute covariance matrix for each edge, invert it and store it
    for e in edges:
        # edge vertices
        v1 = graph.edges[e, 0]
        v2 = graph.edges[e, 1]

        # find indices in data matrix
        v1_from = v1 * n_features_per_vertex
        v1_to = (v1 + 1) * n_features_per_vertex
        v2_from = v2 * n_features_per_vertex
        v2_to = (v2 + 1) * n_features_per_vertex

        # data concatenation
        if mode == 'concatenation':
            edge_data = X[:, list(range(v1_from, v1_to)) +
                             list(range(v2_from, v2_to))]
            m = mean_vector[list(range(v1_from, v1_to)) +
                            list(range(v2_from, v2_to))]
        else:
            edge_data = X[:, v1_from:v1_to] - X[:, v2_from:v2_to]
            m = mean_vector[v1_from:v1_to] - mean_vector[v2_from:v2_to]

        # increment
        _, covariances[e] = _increment_multivariate_gaussian_cov(
            edge_data, m, covariances[e], n, bias=bias)

        # invert it
        covmat = _covariance_matrix_inverse(covariances[e], n_components)

        # store it
        if mode == 'concatenation':
            # v1, v1
            precision[v1_from:v1_to, v1_from:v1_to] += \
                covmat[:n_features_per_vertex, :n_features_per_vertex]
            # v2, v2
            precision[v2_from:v2_to, v2_from:v2_to] += \
                covmat[n_features_per_vertex::, n_features_per_vertex::]
            # v1, v2
            precision[v1_from:v1_to, v2_from:v2_to] = \
                covmat[:n_features_per_vertex, n_features_per_vertex::]
            # v2, v1
            precision[v2_from:v2_to, v1_from:v1_to] = \
                covmat[n_features_per_vertex::, :n_features_per_vertex]
        elif mode == 'subtraction':
            # v1, v2
            precision[v1_from:v1_to, v2_from:v2_to] = -covmat
            # v2, v1
            precision[v2_from:v2_to, v1_from:v1_to] = -covmat
            # v1, v1
            precision[v1_from:v1_to, v1_from:v1_to] += covmat
            # v2, v2
            precision[v2_from:v2_to, v2_from:v2_to] += covmat

    # return covariances
    return precision, covariances


def _increment_sparse_diagonal_precision(X, mean_vector, covariances, n, graph,
                                         n_features, n_features_per_vertex,
                                         dtype=np.float32, n_components=None,
                                         bias=0, verbose=False):
    # initialize covariances matrix
    all_blocks = np.zeros((graph.n_vertices,
                           n_features_per_vertex, n_features_per_vertex),
                          dtype=dtype)
    columns = np.zeros(graph.n_vertices)
    rows = np.zeros(graph.n_vertices)

    # Print information if asked
    if verbose:
        vertices = print_progress(
            range(graph.n_vertices), n_items=graph.n_vertices,
            prefix='Precision per vertex', end_with_newline=False)
    else:
        vertices = range(graph.n_vertices)

    # Compute covariance matrix for each patch
    for v in vertices:
        # find indices in target precision matrix
        i_from = v * n_features_per_vertex
        i_to = (v + 1) * n_features_per_vertex

        # get data
        edge_data = X[:, i_from:i_to]
        m = mean_vector[i_from:i_to]

        # increment
        _, covariances[v] = _increment_multivariate_gaussian_cov(
            edge_data, m, covariances[v], n, bias=bias)

        # invert it
        all_blocks[v] = _covariance_matrix_inverse(covariances[v], n_components)

        # store the inverse covariance and its locations
        rows[v] = v
        columns[v] = v

    # sort rows, columns and all_blocks
    rows_arg_sort = rows.argsort()
    columns = columns[rows_arg_sort]
    all_blocks = all_blocks[rows_arg_sort]
    rows = rows[rows_arg_sort]

    # create indptr
    n_rows = graph.n_vertices
    indptr = np.zeros(n_rows + 1)
    for i in range(n_rows):
        inds, = np.where(rows == i)
        if inds.size == 0:
            indptr[i + 1] = indptr[i]
        else:
            indptr[i] = inds[0]
            indptr[i + 1] = inds[-1] + 1

    # create block sparse matrix
    return (bsr_matrix((all_blocks, columns, indptr),
                       shape=(n_features, n_features), dtype=dtype),
            covariances)


def _increment_dense_diagonal_precision(X, mean_vector, covariances, n, graph,
                                        n_features, n_features_per_vertex,
                                        dtype=np.float32, n_components=None,
                                        bias=0, verbose=False):
    # Initialize precision
    precision = np.zeros((n_features, n_features), dtype=dtype)

    # Print information if asked
    if verbose:
        print_dynamic('Allocated precision matrix of size {}'.format(
            bytes_str(precision.nbytes)))
        vertices = print_progress(
            range(graph.n_vertices), n_items=graph.n_vertices,
            prefix='Precision per vertex', end_with_newline=False)
    else:
        vertices = range(graph.n_vertices)

    # Compute covariance matrix for each patch
    for v in vertices:
        # find indices in target precision matrix
        i_from = v * n_features_per_vertex
        i_to = (v + 1) * n_features_per_vertex

        # get data
        edge_data = X[:, i_from:i_to]
        m = mean_vector[i_from:i_to]

        # increment
        _, covariances[v] = _increment_multivariate_gaussian_cov(
            edge_data, m, covariances[v], n, bias=bias)

        # invert it
        precision[i_from:i_to, i_from:i_to] = _covariance_matrix_inverse(
            covariances[v], n_components)

    # return covariances
    return precision, covariances


def _increment_multivariate_gaussian_mean(X, m, n):
    # Get new number of samples
    new_n = X.shape[0]

    # Update mean vector
    # m_{new} = (n m + \sum_{i=1}^{n_{new}} x_i) / (n + n_{new})
    # where: m       -> old mean vector
    #        n_{new} -> new number of samples
    #        n       -> old number of samples
    #        x_i     -> new data vectors
    return (n * m + np.sum(X, axis=0)) / (n + new_n)


def _increment_multivariate_gaussian_cov(X, m, S, n, bias=0):
    # Get new number of samples
    new_n = X.shape[0]

    # Update mean vector
    # m_{new} = (n m + \sum_{i=1}^{n_{new}} x_i) / (n + n_{new})
    # where: m_{new} -> new mean vector
    #        m       -> old mean vector
    #        n_{new} -> new number of samples
    #        n       -> old number of samples
    #        x_i     -> new data vectors
    new_m = _increment_multivariate_gaussian_mean(X, m, n)

    # Select the normalization value
    if bias == 1:
        k = n
    elif bias == 0:
        k = n - 1
    else:
        raise ValueError("bias must be either 0 or 1")

    # Update covariance matrix
    # S__{new} = (k S + n m^T m + X^T X - (n + n_{new}) m_{new}^T m_{new})
    #                                                            / (k + n_{new})
    m1 = n * m[None, :].T.dot(m[None, :])
    m2 = (n + new_n) * new_m[None, :].T.dot(new_m[None, :])
    new_S = (k * S + m1 + X.T.dot(X) - m2) / (k + new_n)

    return new_m, new_S


class GMRFVectorModel(object):
    r"""
    Trains a Gaussian Markov Random Field (GMRF).

    Parameters
    ----------
    samples : `ndarray` or `list` or `iterable` of `ndarray`
        List or iterable of numpy arrays to build the model from, or an
        existing data matrix.
    graph : :map:`UndirectedGraph` or :map:`DirectedGraph` or :map:`Tree`
        The graph that defines the relations between the features.
    n_samples : `int`, optional
        If provided then ``samples``  must be an iterator that yields
        ``n_samples``. If not provided then samples has to be a `list` (so we
        know how large the data matrix needs to be).
    mode : ``{'concatenation', 'subtraction'}``, optional
        Defines the feature vector of each edge. Assuming that
        :math:`\mathbf{x}_i` and :math:`\mathbf{x}_j` are the feature vectors
        of two adjacent vertices (:math:`i,j:(v_i,v_j)\in E`), then the edge's
        feature vector in the case of ``'concatenation'`` is

        .. math::
           \left[{\mathbf{x}_i}^T, {\mathbf{x}_j}^T\right]^T

        and in the case of ``'subtraction'``

        .. math::
           \mathbf{x}_i - \mathbf{x}_j

    n_components : `int` or ``None``, optional
        When ``None`` (default), the covariance matrix of each edge is inverted
        using `np.linalg.inv`. If `int`, it is inverted using truncated SVD
        using the specified number of compnents.
    dtype : `numpy.dtype`, optional
        The data type of the GMRF's precision matrix. For example, it can be set
        to `numpy.float32` for single precision or to `numpy.float64` for double
        precision. Depending on the size of the precision matrix, this option can
        you a lot of memory.
    sparse : `bool`, optional
        When ``True``, the GMRF's precision matrix has type
        `scipy.sparse.bsr_matrix`, otherwise it is a `numpy.array`.
    bias : `int`, optional
        Default normalization is by ``(N - 1)``, where ``N`` is the number of
        observations given (unbiased estimate). If `bias` is 1, then
        normalization is by ``N``. These values can be overridden by using
        the keyword ``ddof`` in numpy versions >= 1.5.
    incremental : `bool`, optional
        This argument must be set to ``True`` in case the user wants to
        incrementally update the GMRF. Note that if ``True``, the model
        occupies 2x memory.
    verbose : `bool`, optional
        If ``True``, the progress of the model's training is printed.

    Notes
    -----
    Let us denote a graph as :math:`G=(V,E)`, where
    :math:`V=\{v_i,v_2,\ldots, v_{|V|}\}` is the set of :math:`|V|` vertices and
    there is an edge :math:`(v_i,v_j)\in E` for each pair of connected vertices.
    Let us also assume that we have a set of random variables
    :math:`X=\{X_i\}, \forall i:v_i\in V`, which represent an abstract feature
    vector of length :math:`k` extracted from each vertex :math:`v_i`, i.e.
    :math:`\mathbf{x}_i,i:v_i\in V`.

    A GMRF is described by an undirected graph, where the vertexes stand for
    random variables and the edges impose statistical constraints on these
    random variables. Thus, the GMRF models the set of random variables with
    a multivariate normal distribution

    .. math::
       p(X=\mathbf{x}|G)\sim\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\Sigma})

    We denote by :math:`\mathbf{Q}` the block-sparse precision matrix that is
    the inverse of the covariance matrix :math:`\boldsymbol{\Sigma}`, i.e.
    :math:`\mathbf{Q}=\boldsymbol{\Sigma}^{-1}`.  By applying the GMRF we make
    the assumption that the random variables satisfy the three Markov
    properties (pairwise, local and global) and that the blocks of the
    precision matrix that correspond to disjoint vertexes are zero, i.e.

    .. math::
       \mathbf{Q}_{ij}=\mathbf{0}_{k\times k},\forall i,j:(v_i,v_j)\notin E

    References
    ----------
    .. [1] H. Rue, and L. Held. "Gaussian Markov random fields: theory and
       applications," CRC Press, 2005.
    .. [2] E. Antonakos, J. Alabort-i-Medina, and S. Zafeiriou. "Active
       Pictorial Structures", IEEE International Conference on Computer Vision
       & Pattern Recognition (CVPR), Boston, MA, USA, pp. 5435-5444, June 2015.
    """
    def __init__(self, samples, graph, n_samples=None, mode='concatenation',
                 n_components=None, dtype=np.float64, sparse=True, bias=0,
                 incremental=False, verbose=False):
        # Generate data matrix
        # (n_samples, n_features)
        data, self.n_samples = self._data_to_matrix(samples, n_samples)

        # n_features and n_features_per_vertex
        self.n_features = data.shape[1]
        self.n_features_per_vertex = int(self.n_features / graph.n_vertices)

        # Assign arguments
        self.graph = graph
        self.mode = mode
        self.n_components = n_components
        self.sparse = sparse
        self.dtype = dtype
        self.bias = bias
        self.is_incremental = incremental

        # Compute mean vector
        self.mean_vector = np.mean(data, axis=0)

        # Select correct method to create the precision matrix based on the
        # graph type and the sparse flag
        if self.graph.n_edges == 0:
            if self.sparse:
                constructor = _create_sparse_diagonal_precision
            else:
                constructor = _create_dense_diagonal_precision
        else:
            if self.sparse:
                constructor = partial(_create_sparse_precision, mode=self.mode)
            else:
                constructor = partial(_create_dense_precision, mode=self.mode)

        # Create the precision matrix and optionally store the covariance
        # matrices
        if self.is_incremental:
            self.precision, self._covariance_matrices = constructor(
                data, self.graph, self.n_features, self.n_features_per_vertex,
                dtype=self.dtype, n_components=self.n_components, bias=self.bias,
                return_covariances=self.is_incremental, verbose=verbose)
        else:
            self._covariance_matrices = None
            self.precision = constructor(
                data, self.graph, self.n_features, self.n_features_per_vertex,
                dtype=self.dtype, n_components=self.n_components, bias=self.bias,
                return_covariances=self.is_incremental, verbose=verbose)

    def _data_to_matrix(self, data, n_samples):
        # build a data matrix from all the samples
        if n_samples is None:
            n_samples = len(data)
        # Assumed data is ndarray of (n_samples, n_features) or list of samples
        if not isinstance(data, np.ndarray):
            # Make sure we have an array, slice of the number of requested
            # samples
            data = np.array(data)[:n_samples]
        return data, n_samples

    def mean(self):
        r"""
        Return the mean of the model. For this model, returns the same result
        as ``mean_vector``.

        :type: `ndarray`
        """
        return self.mean_vector

    def increment(self, samples, n_samples=None, verbose=False):
        r"""
        Update the mean and precision matrix of the GMRF by updating the
        distributions of all the edges.

        Parameters
        ----------
        samples : `ndarray` or `list` or `iterable` of `ndarray`
            List or iterable of numpy arrays to build the model from, or an
            existing data matrix.
        n_samples : `int`, optional
            If provided then ``samples``  must be an iterator that yields
            ``n_samples``. If not provided then samples has to be a
            list (so we know how large the data matrix needs to be).
        verbose : `bool`, optional
            If ``True``, the progress of the model's incremental update is
            printed.
        """
        # Check if it can be incrementally updated
        if not self.is_incremental:
            raise ValueError('GMRF cannot be incrementally updated.')

        # Build a data matrix from the new samples
        data, _ = self._data_to_matrix(samples, n_samples)

        # Increment the model
        self._increment(data=data, verbose=verbose)

    def _increment(self, data, verbose):
        # Empty memory
        self.precision = 0

        # Select correct method to create the precision matrix based on the
        # graph type and the sparse flag
        if self.graph.n_edges == 0:
            if self.sparse:
                constructor = _increment_sparse_diagonal_precision
            else:
                constructor = _increment_dense_diagonal_precision
        else:
            if self.sparse:
                constructor = partial(_increment_sparse_precision,
                                      mode=self.mode)
            else:
                constructor = partial(_increment_dense_precision,
                                      mode=self.mode)

        # Create the precision matrix and optionally store the covariance
        # matrices
        self.precision, self._covariance_matrices = constructor(
            data, self.mean_vector, self._covariance_matrices, self.n_samples,
            self.graph, self.n_features, self.n_features_per_vertex,
            dtype=self.dtype, n_components=self.n_components, bias=self.bias,
            verbose=verbose)

        # Update mean and number of samples
        self.mean_vector = _increment_multivariate_gaussian_mean(
            data, self.mean_vector, self.n_samples)
        self.n_samples += data.shape[0]

    def mahalanobis_distance(self, samples, subtract_mean=True,
                             square_root=False):
        r"""
        Compute the mahalanobis distance given a sample :math:`\mathbf{x}` or an
        array of samples :math:`\mathbf{X}`, i.e.

        .. math::
           \sqrt{(\mathbf{x}-\boldsymbol{\mu})^T \mathbf{Q} (\mathbf{x}-\boldsymbol{\mu})}
           \text{ or }
           \sqrt{(\mathbf{X}-\boldsymbol{\mu})^T \mathbf{Q} (\mathbf{X}-\boldsymbol{\mu})}

        Parameters
        ----------
        samples : `ndarray`
            A single data vector or an array of multiple data vectors.
        subtract_mean : `bool`, optional
            When ``True``, the mean vector is subtracted from the data vector.
        square_root : `bool`, optional
            If ``False``, the mahalanobis distance gets squared.
        """
        samples, _ = self._data_to_matrix(samples, None)
        if len(samples.shape) == 1:
            samples = samples[..., None].T
        return self._mahalanobis_distance(samples=samples,
                                          subtract_mean=subtract_mean,
                                          square_root=square_root)

    def _mahalanobis_distance(self, samples, subtract_mean, square_root):
        # we assume that samples is an ndarray of n_samples x n_features

        # create data matrix
        if subtract_mean:
            n_samples = samples.shape[0]
            samples = samples - np.tile(self.mean_vector[..., None],
                                        n_samples).T

        # compute mahalanobis per sample
        if self.sparse:
            # if sparse, unfortunately the einstein sum is not implemented
            tmp = self.precision.dot(samples.T)
            d = samples.dot(tmp)
            d = np.diag(d)
        else:
            # if dense, then the einstein sum is much faster
            d = np.einsum('ij,ij->i', np.dot(samples, self.precision), samples)

        # if only one sample, then return a scalar
        if d.shape[0] == 1:
            d = d[0]

        # square root
        if square_root:
            return np.sqrt(d)
        else:
            return d

    def principal_components_analysis(self, max_n_components=None):
        r"""
        Returns a :map:`PCAVectorModel` with the Principal Components.

        Note that the eigenvalue decomposition is applied directly on the
        precision matrix and then the eigenvalues are inverted.

        Parameters
        ----------
        max_n_components : `int` or ``None``, optional
            The maximum number of principal components. If ``None``, all the
            components are returned.

        Returns
        -------
        pca : :map:`PCAVectorModel`
            The PCA model.
        """
        from .pca import PCAVectorModel
        return PCAVectorModel.init_from_covariance_matrix(
            C=self.precision, mean=self.mean_vector, n_samples=self.n_samples,
            centred=True, is_inverse=True, max_n_components=max_n_components)

    @property
    def _str_title(self):
        r"""
        Returns a string containing the name of the model.

        :type: `str`
        """
        tmp = 'a'
        if isinstance(self.graph, UndirectedGraph):
            tmp = 'an'
        return "GMRF model on {} {}".format(tmp, self.graph)

    def __str__(self):
        incremental_str = (' - Can be incrementally updated.' if
                           self.is_incremental else ' - Cannot be '
                                                    'incrementally updated.')
        svd_str = (' - # SVD components:        {}'.format(self.n_components)
                   if self.n_components is not None else ' - No ' 'SVD used.')
        _Q_sparse = 'scipy.sparse' if self.sparse else 'numpy.array'
        q_str = ' - Q is stored as {} with {} precision'.format(
            _Q_sparse, name_of_callable(self.dtype))
        mode_str = ('concatenated' if self.mode == 'concatenation' else
                    'subtracted')
        str_out = 'Gaussian MRF Model \n' \
                  ' - {}\n' \
                  ' - The data of the vertexes of each edge are {}.\n' \
                  '{}\n' \
                  ' - # variables (vertexes):  {}\n' \
                  ' - # features per variable: {}\n' \
                  ' - # features in total:     {}\n' \
                  '{}\n' \
                  ' - # samples:               {}\n' \
                  '{}\n'.format(
            self.graph.__str__(), mode_str, q_str, self.graph.n_vertices,
            self.n_features_per_vertex, self.n_features, svd_str,
            self.n_samples, incremental_str)
        return str_out


class GMRFModel(GMRFVectorModel):
    r"""
    Trains a Gaussian Markov Random Field (GMRF).

    Parameters
    ----------
    samples : `list` or `iterable` of :map:`Vectorizable`
        List or iterable of samples to build the model from.
    graph : :map:`UndirectedGraph` or :map:`DirectedGraph` or :map:`Tree`
        The graph that defines the relations between the features.
    n_samples : `int`, optional
        If provided then ``samples``  must be an iterator that yields
        ``n_samples``. If not provided then samples has to be a `list` (so we
        know how large the data matrix needs to be).
    mode : ``{'concatenation', 'subtraction'}``, optional
        Defines the feature vector of each edge. Assuming that
        :math:`\mathbf{x}_i` and :math:`\mathbf{x}_j` are the feature vectors
        of two adjacent vertices (:math:`i,j:(v_i,v_j)\in E`), then the edge's
        feature vector in the case of ``'concatenation'`` is

        .. math::
           \left[{\mathbf{x}_i}^T, {\mathbf{x}_j}^T\right]^T

        and in the case of ``'subtraction'``

        .. math::
           \mathbf{x}_i - \mathbf{x}_j

    n_components : `int` or ``None``, optional
        When ``None`` (default), the covariance matrix of each edge is inverted
        using `np.linalg.inv`. If `int`, it is inverted using truncated SVD
        using the specified number of compnents.
    dtype : `numpy.dtype`, optional
        The data type of the GMRF's precision matrix. For example, it can be set
        to `numpy.float32` for single precision or to `numpy.float64` for double
        precision. Depending on the size of the precision matrix, this option can
        you a lot of memory.
    sparse : `bool`, optional
        When ``True``, the GMRF's precision matrix has type
        `scipy.sparse.bsr_matrix`, otherwise it is a `numpy.array`.
    bias : `int`, optional
        Default normalization is by ``(N - 1)``, where ``N`` is the number of
        observations given (unbiased estimate). If `bias` is 1, then
        normalization is by ``N``. These values can be overridden by using
        the keyword ``ddof`` in numpy versions >= 1.5.
    incremental : `bool`, optional
        This argument must be set to ``True`` in case the user wants to
        incrementally update the GMRF. Note that if ``True``, the model
        occupies 2x memory.
    verbose : `bool`, optional
        If ``True``, the progress of the model's training is printed.

    Notes
    -----
    Let us denote a graph as :math:`G=(V,E)`, where
    :math:`V=\{v_i,v_2,\ldots, v_{|V|}\}` is the set of :math:`|V|` vertices and
    there is an edge :math:`(v_i,v_j)\in E` for each pair of connected vertices.
    Let us also assume that we have a set of random variables
    :math:`X=\{X_i\}, \forall i:v_i\in V`, which represent an abstract feature
    vector of length :math:`k` extracted from each vertex :math:`v_i`, i.e.
    :math:`\mathbf{x}_i,i:v_i\in V`.

    A GMRF is described by an undirected graph, where the vertexes stand for
    random variables and the edges impose statistical constraints on these
    random variables. Thus, the GMRF models the set of random variables with
    a multivariate normal distribution

    .. math::
       p(X=\mathbf{x}|G)\sim\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\Sigma})

    We denote by :math:`\mathbf{Q}` the block-sparse precision matrix that is
    the inverse of the covariance matrix :math:`\boldsymbol{\Sigma}`, i.e.
    :math:`\mathbf{Q}=\boldsymbol{\Sigma}^{-1}`.  By applying the GMRF we make
    the assumption that the random variables satisfy the three Markov
    properties (pairwise, local and global) and that the blocks of the
    precision matrix that correspond to disjoint vertexes are zero, i.e.

    .. math::
       \mathbf{Q}_{ij}=\mathbf{0}_{k\times k},\forall i,j:(v_i,v_j)\notin E

    References
    ----------
    .. [1] H. Rue, and L. Held. "Gaussian Markov random fields: theory and
       applications," CRC Press, 2005.
    .. [2] E. Antonakos, J. Alabort-i-Medina, and S. Zafeiriou. "Active
       Pictorial Structures", IEEE International Conference on Computer Vision
       & Pattern Recognition (CVPR), Boston, MA, USA, pp. 5435-5444, June 2015.
    """
    def __init__(self, samples, graph, mode='concatenation', n_components=None,
                 dtype=np.float64, sparse=True, n_samples=None, bias=0,
                 incremental=False, verbose=False):
        # Build a data matrix from all the samples
        data, self.template_instance = as_matrix(
            samples, length=n_samples, return_template=True, verbose=verbose)
        n_samples = data.shape[0]

        GMRFVectorModel.__init__(self, data, graph, mode=mode,
                                 n_components=n_components, dtype=dtype,
                                 sparse=sparse, n_samples=n_samples, bias=bias,
                                 incremental=incremental, verbose=verbose)

    def mean(self):
        r"""
        Return the mean of the model.

        :type: :map:`Vectorizable`
        """
        return self.template_instance.from_vector(self.mean_vector)

    def increment(self, samples, n_samples=None, verbose=False):
        r"""
        Update the mean and precision matrix of the GMRF by updating the
        distributions of all the edges.

        Parameters
        ----------
        samples : `list` or `iterable` of :map:`Vectorizable`
            List or iterable of samples to build the model from.
        n_samples : `int`, optional
            If provided then ``samples``  must be an iterator that yields
            ``n_samples``. If not provided then samples has to be a
            list (so we know how large the data matrix needs to be).
        verbose : `bool`, optional
            If ``True``, the progress of the model's incremental update is
            printed.
        """
        # Check if it can be incrementally updated
        if not self.is_incremental:
            raise ValueError('GMRF cannot be incrementally updated.')

        # Build a data matrix from the new samples
        data = as_matrix(samples, length=n_samples, verbose=verbose)

        # Increment the model
        self._increment(data=data, verbose=verbose)

    def mahalanobis_distance(self, samples, subtract_mean=True,
                             square_root=False):
        r"""
        Compute the mahalanobis distance given a sample :math:`\mathbf{x}` or an
        array of samples :math:`\mathbf{X}`, i.e.

        .. math::
           \sqrt{(\mathbf{x}-\boldsymbol{\mu})^T \mathbf{Q} (\mathbf{x}-\boldsymbol{\mu})}
           \text{ or }
           \sqrt{(\mathbf{X}-\boldsymbol{\mu})^T \mathbf{Q} (\mathbf{X}-\boldsymbol{\mu})}

        Parameters
        ----------
        samples : :map:`Vectorizable` or `list` of :map:`Vectorizable`
            The new data sample or a list of samples.
        subtract_mean : `bool`, optional
            When ``True``, the mean vector is subtracted from the data vector.
        square_root : `bool`, optional
            If ``False``, the mahalanobis distance gets squared.
        """
        if isinstance(samples, list):
            samples = as_matrix(samples, length=None,
                                return_template=False, verbose=False)
        else:
            samples = samples.as_vector()[..., None].T
        return self._mahalanobis_distance(samples=samples,
                                          subtract_mean=subtract_mean,
                                          square_root=square_root)

    def principal_components_analysis(self, max_n_components=None):
        r"""
        Returns a :map:`PCAModel` with the Principal Components.

        Note that the eigenvalue decomposition is applied directly on the
        precision matrix and then the eigenvalues are inverted.

        Parameters
        ----------
        max_n_components : `int` or ``None``, optional
            The maximum number of principal components. If ``None``, all the
            components are returned.

        Returns
        -------
        pca : :map:`PCAModel`
            The PCA model.
        """
        from .pca import PCAModel
        return PCAModel.init_from_covariance_matrix(
            C=self.precision, mean=self.mean(), n_samples=self.n_samples,
            centred=True, is_inverse=True, max_n_components=max_n_components)
