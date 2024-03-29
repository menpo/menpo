import numpy as np
import scipy.sparse as sparse

from . import (
    PointCloud,
    UndirectedGraph,
    DirectedGraph,
    Tree,
    TriMesh,
    PointUndirectedGraph,
    PointDirectedGraph,
    PointTree,
)


def stencil_grid(stencil, shape, dtype=None, format=None):
    """Construct a sparse matrix form a local matrix stencil

    This function is useful for building sparse adjacency matrices according
    to a specific connectivity pattern.

    This function is borrowed from the PyAMG project, under the permission of
    the MIT license:

    The MIT License (MIT)

    Copyright (c) 2008-2015 PyAMG Developers

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.

    The original version of this file can be found here:

    https://github.com/pyamg/pyamg/blob/621d63411895898660e5ea078840118905bec061/pyamg/gallery/stencil.py

    This file has been modified to fit the style standards of the Menpo
    project.

    Parameters
    ----------
    S : `ndarray`
        Matrix stencil stored in N-d array
    grid : `tuple`
        Tuple containing the N shape dimensions (shape)
    dtype : `np.dtype`, optional
        Numpy data type of the result
    format : `str`, optional
        Sparse matrix format to return, e.g. "csr", "coo", etc.

    Returns
    -------
    A : sparse matrix
        Sparse matrix which represents the operator given by applying
        stencil stencil at each vertex of a regular shape with given dimensions.

    Notes
    -----
    The shape vertices are enumerated as ``arange(prod(shape)).reshape(shape)``.
    This implies that the last shape dimension cycles fastest, while the
    first dimension cycles slowest.  For example, if ``shape=(2,3)`` then the
    shape vertices are ordered as ``(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)``.

    This coincides with the ordering used by the NumPy functions
    ``ndenumerate()`` and ``mgrid()``.

    Raises
    ------
    ValueError
        If the stencil shape is not odd.
    ValueError
        If the stencil dimension does not equal the number of shape dimensions
    ValueError
        If the shape dimensions are not all positive

    Examples
    --------
    >>> import numpy as np
    >>> from menpo.shape import stencil_grid
    >>> stencil = [[0,-1,0],[-1,4,-1],[0,-1,0]]  # 2D Poisson stencil
    >>> shape = (3, 3)                           # 2D shape with shape 3x3
    >>> A = stencil_grid(stencil, shape, dtype=float, format='csr')
    >>> A.todense()
    matrix([[ 4., -1.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
            [-1.,  4., -1.,  0., -1.,  0.,  0.,  0.,  0.],
            [ 0., -1.,  4.,  0.,  0., -1.,  0.,  0.,  0.],
            [-1.,  0.,  0.,  4., -1.,  0., -1.,  0.,  0.],
            [ 0., -1.,  0., -1.,  4., -1.,  0., -1.,  0.],
            [ 0.,  0., -1.,  0., -1.,  4.,  0.,  0., -1.],
            [ 0.,  0.,  0., -1.,  0.,  0.,  4., -1.,  0.],
            [ 0.,  0.,  0.,  0., -1.,  0., -1.,  4., -1.],
            [ 0.,  0.,  0.,  0.,  0., -1.,  0., -1.,  4.]])

    >>> stencil = [[0,1,0],[1,0,1],[0,1,0]]  # 2D Lattice Connectivity
    >>> shape = (3, 3)                       # 2D shape with shape 3x3
    >>> A = stencil_grid(stencil, shape, dtype=float, format='csr')
    >>> A.todense()
    matrix([[ 0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
            [ 1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.],
            [ 0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
            [ 1.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.],
            [ 0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.],
            [ 0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  1.],
            [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.],
            [ 0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  1.],
            [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.]])

    """
    stencil = np.asarray(stencil, dtype=dtype)
    shape = tuple(shape)

    if not (np.asarray(stencil.shape) % 2 == 1).all():
        raise ValueError("all stencil dimensions must be odd")

    if len(shape) != np.ndim(stencil):
        raise ValueError(
            "stencil dimension must equal number of shape\
                          dimensions"
        )

    if min(shape) < 1:
        raise ValueError("shape dimensions must be positive")

    N_v = np.prod(shape)  # number of vertices in the mesh
    N_s = (stencil != 0).sum()  # number of nonzero stencil entries

    # diagonal offsets
    diags = np.zeros(N_s, dtype=int)

    # compute index offset of each dof within the stencil
    strides = np.cumprod([1] + list(reversed(shape)))[:-1]
    indices = tuple(i.copy() for i in stencil.nonzero())
    for i, s in zip(indices, stencil.shape):
        i -= s // 2
        # i = (i - s) // 2
        # i = i // 2
        # i = i - (s // 2)
    for stride, coords in zip(strides, reversed(indices)):
        diags += stride * coords

    data = stencil[stencil != 0].repeat(N_v).reshape(N_s, N_v)

    indices = np.vstack(indices).T

    # zero boundary connections
    for index, diag in zip(indices, data):
        diag = diag.reshape(shape)
        for n, i in enumerate(index):
            if i > 0:
                s = [slice(None)] * len(shape)
                s[n] = slice(0, i)
                diag[tuple(s)] = 0
            elif i < 0:
                s = [slice(None)] * len(shape)
                s[n] = slice(i, None)
                diag[tuple(s)] = 0

    # remove diagonals that lie outside matrix
    mask = abs(diags) < N_v
    if not mask.all():
        diags = diags[mask]
        data = data[mask]

    # sum duplicate diagonals
    if len(np.unique(diags)) != len(diags):
        new_diags = np.unique(diags)
        new_data = np.zeros((len(new_diags), data.shape[1]), dtype=data.dtype)

        for dia, dat in zip(diags, data):
            n = np.searchsorted(new_diags, dia)
            new_data[n, :] += dat

        diags = new_diags
        data = new_data

    return sparse.dia_matrix((data, diags), shape=(N_v, N_v)).asformat(format)


def _get_points_and_number_of_vertices(shape):
    if isinstance(shape, PointCloud):
        return shape.points, shape.n_points
    else:
        raise ValueError("shape must be PointCloud instance.")


def _get_star_graph_edges(vertices_list, root_vertex):
    edges = []
    for v in vertices_list:
        if v != root_vertex:
            edges.append([root_vertex, v])
    return edges


def _get_complete_graph_edges(vertices_list):
    n_vertices = len(vertices_list)
    edges = []
    for i in range(n_vertices - 1):
        k = i + 1
        for j in range(k, n_vertices, 1):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            edges.append([v1, v2])
    return edges


def _get_chain_graph_edges(vertices_list, closed):
    n_vertices = len(vertices_list)
    edges = []
    for i in range(n_vertices - 1):
        k = i + 1
        v1 = vertices_list[i]
        v2 = vertices_list[k]
        edges.append([v1, v2])
    if closed:
        v1 = vertices_list[-1]
        v2 = vertices_list[0]
        edges.append([v1, v2])
    return edges


def empty_graph(shape, return_pointgraph=True):
    r"""
    Returns an empty graph given the landmarks configuration of a shape
    instance.

    Parameters
    ----------
    shape : :map:`PointCloud` or subclass
        The shape instance that defines the landmarks configuration based on
        which the graph will be created.
    return_pointgraph : `bool`, optional
        If ``True``, then a :map:`PointUndirectedGraph` instance will be
        returned. If ``False``, then an :map:`UndirectedGraph` instance will be
        returned.

    Returns
    -------
    graph : :map:`UndirectedGraph` or :map:`PointUndirectedGraph`
        The generated graph.
    """
    # get points and number of vertices
    points, n_vertices = _get_points_and_number_of_vertices(shape)

    # create empty edges
    edges = None

    # return graph
    if return_pointgraph:
        return PointUndirectedGraph.init_from_edges(
            points, edges, n_vertices, skip_checks=True
        )
    else:
        return UndirectedGraph.init_from_edges(edges, n_vertices, skip_checks=True)


def star_graph(shape, root_vertex, graph_cls=PointTree):
    r"""
    Returns a star graph given the landmarks configuration of a shape instance.

    Parameters
    ----------
    shape : :map:`PointCloud` or subclass
        The shape instance that defines the landmarks configuration based on
        which the graph will be created.
    root_vertex : `int`
        The root of the star tree.
    graph_cls : `Graph` or `PointGraph` subclass
        The output graph type.
        Possible options are ::

            {:map:`UndirectedGraph`, :map:`DirectedGraph`, :map:`Tree`,
             :map:`PointUndirectedGraph`, :map:`PointDirectedGraph`,
             :map:`PointTree`}

    Returns
    -------
    graph : `Graph` or `PointGraph` subclass
        The generated graph.

    Raises
    ------
    ValueError
        graph_cls must be UndirectedGraph, DirectedGraph, Tree,
        PointUndirectedGraph, PointDirectedGraph or PointTree.
    """
    # get points and number of vertices
    points, n_vertices = _get_points_and_number_of_vertices(shape)

    # create star graph edges
    edges = _get_star_graph_edges(range(n_vertices), root_vertex)

    # return graph
    if graph_cls == Tree:
        return graph_cls.init_from_edges(
            edges=edges,
            n_vertices=n_vertices,
            root_vertex=root_vertex,
            skip_checks=True,
        )
    elif graph_cls == PointTree:
        return graph_cls.init_from_edges(
            points=points, edges=edges, root_vertex=root_vertex, skip_checks=True
        )
    elif graph_cls == UndirectedGraph or graph_cls == DirectedGraph:
        return graph_cls.init_from_edges(
            edges=edges, n_vertices=n_vertices, skip_checks=True
        )
    elif graph_cls == PointUndirectedGraph or graph_cls == PointDirectedGraph:
        return graph_cls.init_from_edges(points=points, edges=edges, skip_checks=True)
    else:
        raise ValueError(
            "graph_cls must be UndirectedGraph, DirectedGraph, "
            "Tree, PointUndirectedGraph, PointDirectedGraph or "
            "PointTree."
        )


def complete_graph(shape, graph_cls=PointUndirectedGraph):
    r"""
    Returns a complete graph given the landmarks configuration of a shape
    instance.

    Parameters
    ----------
    shape : :map:`PointCloud` or subclass
        The shape instance that defines the landmarks configuration based on
        which the graph will be created.
    graph_cls : `Graph` or `PointGraph` subclass
        The output graph type.
        Possible options are ::

            {:map:`UndirectedGraph`, :map:`DirectedGraph`,
             :map:`PointUndirectedGraph`, :map:`PointDirectedGraph`}

    Returns
    -------
    graph : `Graph` or `PointGraph` subclass
        The generated graph.

    Raises
    ------
    ValueError
        graph_cls must be UndirectedGraph, DirectedGraph, PointUndirectedGraph
        or PointDirectedGraph.
    """
    # get points and number of vertices
    points, n_vertices = _get_points_and_number_of_vertices(shape)

    # create complete graph edges
    edges = _get_complete_graph_edges(range(n_vertices))

    # return graph
    if graph_cls == UndirectedGraph or graph_cls == DirectedGraph:
        return graph_cls.init_from_edges(
            edges=edges, n_vertices=n_vertices, skip_checks=True
        )
    elif graph_cls == PointUndirectedGraph or graph_cls == PointDirectedGraph:
        return graph_cls.init_from_edges(points=points, edges=edges, skip_checks=True)
    else:
        raise ValueError(
            "graph_cls must be UndirectedGraph, DirectedGraph, "
            "PointUndirectedGraph or PointDirectedGraph."
        )


def chain_graph(shape, graph_cls=PointDirectedGraph, closed=False):
    r"""
    Returns a chain graph given the landmarks configuration of a shape instance.

    Parameters
    ----------
    shape : :map:`PointCloud` or subclass
        The shape instance that defines the landmarks configuration based on
        which the graph will be created.
    graph_cls : `Graph` or `PointGraph` subclass
        The output graph type.
        Possible options are ::

            {:map:`UndirectedGraph`, :map:`DirectedGraph`, :map:`Tree`,
             :map:`PointUndirectedGraph`, :map:`PointDirectedGraph`,
             :map:`PointTree`}

    closed : `bool`, optional
        If ``True``, then the chain will be closed (i.e. edge between the
        first and last vertices).

    Returns
    -------
    graph : `Graph` or `PointGraph` subclass
        The generated graph.

    Raises
    ------
    ValueError
        A closed chain graph cannot be a Tree or PointTree instance.
    ValueError
        graph_cls must be UndirectedGraph, DirectedGraph, Tree,
        PointUndirectedGraph, PointDirectedGraph or PointTree.
    """
    # get points and number of vertices
    points, n_vertices = _get_points_and_number_of_vertices(shape)

    # create chain graph edges
    edges = _get_chain_graph_edges(range(n_vertices), closed=closed)

    # return graph
    if graph_cls == Tree:
        if closed:
            raise ValueError("A closed chain graph cannot be a Tree " "instance.")
        else:
            return graph_cls.init_from_edges(
                edges=edges, n_vertices=n_vertices, root_vertex=0, skip_checks=True
            )
    elif graph_cls == PointTree:
        if closed:
            raise ValueError("A closed chain graph cannot be a PointTree " "instance.")
        else:
            return graph_cls.init_from_edges(
                points=points, edges=edges, root_vertex=0, skip_checks=True
            )
    elif graph_cls == UndirectedGraph or graph_cls == DirectedGraph:
        return graph_cls.init_from_edges(
            edges=edges, n_vertices=n_vertices, skip_checks=True
        )
    elif graph_cls == PointUndirectedGraph or graph_cls == PointDirectedGraph:
        return graph_cls.init_from_edges(points=points, edges=edges, skip_checks=True)
    else:
        raise ValueError(
            "graph_cls must be UndirectedGraph, DirectedGraph, "
            "Tree, PointUndirectedGraph, PointDirectedGraph or "
            "PointTree."
        )


def delaunay_graph(shape, return_pointgraph=True):
    r"""
    Returns a graph with the edges being generated by Delaunay triangulation.

    Parameters
    ----------
    shape : :map:`PointCloud` or subclass
        The shape instance that defines the landmarks configuration based on
        which the graph will be created.
    return_pointgraph : `bool`, optional
        If ``True``, then a :map:`PointUndirectedGraph` instance will be
        returned. If ``False``, then an :map:`UndirectedGraph` instance will be
        returned.

    Returns
    -------
    graph : :map:`UndirectedGraph` or :map:`PointUndirectedGraph`
        The generated graph.
    """
    # get TriMesh instance that estimates the Delaunay triangulation
    if isinstance(shape, PointCloud):
        trimesh = TriMesh(shape.points)
        n_vertices = shape.n_points
        points = shape.points
    else:
        raise ValueError("shape must be a PointCloud instance or subclass.")

    # get edges
    edges = trimesh.edge_indices()

    # return graph
    if return_pointgraph:
        return PointUndirectedGraph.init_from_edges(
            points=points, edges=edges, skip_checks=True
        )
    else:
        return UndirectedGraph.init_from_edges(
            edges=edges, n_vertices=n_vertices, skip_checks=True
        )
