from menpo.landmark import LandmarkGroup
from . import (PointCloud, UndirectedGraph, DirectedGraph, Tree, TriMesh,
               PointUndirectedGraph, PointDirectedGraph, PointTree)


def _get_points_and_number_of_vertices(shape):
    if isinstance(shape, LandmarkGroup):
        return shape.lms.points, shape.n_landmarks
    elif isinstance(shape, PointCloud):
        return shape.points, shape.n_points
    else:
        raise ValueError("shape must be either a LandmarkGroup or a "
                         "PointCloud instance.")


def _get_star_graph_edges(vertices_list, root_vertex):
    edges = []
    for v in vertices_list:
        if v != root_vertex:
            edges.append([root_vertex, v])
    return edges


def _get_complete_graph_edges(vertices_list):
    n_vertices = len(vertices_list)
    edges = []
    for i in range(n_vertices-1):
        k = i + 1
        for j in range(k, n_vertices, 1):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            edges.append([v1, v2])
    return edges


def _get_chain_graph_edges(vertices_list, closed):
    n_vertices = len(vertices_list)
    edges = []
    for i in range(n_vertices-1):
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
    shape : :map:`PointCloud` or :map:`LandmarkGroup` or subclass
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
        return PointUndirectedGraph.init_from_edges(points, edges, n_vertices,
                                                    skip_checks=True)
    else:
        return UndirectedGraph.init_from_edges(edges, n_vertices,
                                               skip_checks=True)


def star_graph(shape, root_vertex, graph_cls=PointTree):
    r"""
    Returns a star graph given the landmarks configuration of a shape instance.

    Parameters
    ----------
    shape : :map:`PointCloud` or :map:`LandmarkGroup` or subclass
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
        return graph_cls.init_from_edges(edges=edges, n_vertices=n_vertices,
                                         root_vertex=root_vertex,
                                         skip_checks=True)
    elif graph_cls == PointTree:
        return graph_cls.init_from_edges(points=points, edges=edges,
                                         root_vertex=root_vertex,
                                         skip_checks=True)
    elif graph_cls == UndirectedGraph or graph_cls == DirectedGraph:
        return graph_cls.init_from_edges(edges=edges, n_vertices=n_vertices,
                                         skip_checks=True)
    elif graph_cls == PointUndirectedGraph or graph_cls == PointDirectedGraph:
        return graph_cls.init_from_edges(points=points, edges=edges,
                                         skip_checks=True)
    else:
        raise ValueError("graph_cls must be UndirectedGraph, DirectedGraph, "
                         "Tree, PointUndirectedGraph, PointDirectedGraph or "
                         "PointTree.")


def complete_graph(shape, graph_cls=PointUndirectedGraph):
    r"""
    Returns a complete graph given the landmarks configuration of a shape
    instance.

    Parameters
    ----------
    shape : :map:`PointCloud` or :map:`LandmarkGroup` or subclass
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
        return graph_cls.init_from_edges(edges=edges, n_vertices=n_vertices,
                                         skip_checks=True)
    elif graph_cls == PointUndirectedGraph or graph_cls == PointDirectedGraph:
        return graph_cls.init_from_edges(points=points, edges=edges,
                                         skip_checks=True)
    else:
        raise ValueError("graph_cls must be UndirectedGraph, DirectedGraph, "
                         "PointUndirectedGraph or PointDirectedGraph.")


def chain_graph(shape, graph_cls=PointDirectedGraph, closed=False):
    r"""
    Returns a chain graph given the landmarks configuration of a shape instance.

    Parameters
    ----------
    shape : :map:`PointCloud` or :map:`LandmarkGroup` or subclass
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
            raise ValueError("A closed chain graph cannot be a Tree "
                             "instance.")
        else:
            return graph_cls.init_from_edges(edges=edges, n_vertices=n_vertices,
                                             root_vertex=0, skip_checks=True)
    elif graph_cls == PointTree:
        if closed:
            raise ValueError("A closed chain graph cannot be a PointTree "
                             "instance.")
        else:
            return graph_cls.init_from_edges(points=points, edges=edges,
                                             root_vertex=0, skip_checks=True)
    elif graph_cls == UndirectedGraph or graph_cls == DirectedGraph:
        return graph_cls.init_from_edges(edges=edges, n_vertices=n_vertices,
                                         skip_checks=True)
    elif graph_cls == PointUndirectedGraph or graph_cls == PointDirectedGraph:
        return graph_cls.init_from_edges(points=points, edges=edges,
                                         skip_checks=True)
    else:
        raise ValueError("graph_cls must be UndirectedGraph, DirectedGraph, "
                         "Tree, PointUndirectedGraph, PointDirectedGraph or "
                         "PointTree.")


def delaunay_graph(shape, return_pointgraph=True):
    r"""
    Returns a graph with the edges being generated by Delaunay triangulation.

    Parameters
    ----------
    shape : :map:`PointCloud` or :map:`LandmarkGroup` or subclass
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
    if isinstance(shape, LandmarkGroup):
        trimesh = TriMesh(shape.lms.points)
        n_vertices = shape.n_landmarks
        points = shape.lms.points
    elif isinstance(shape, PointCloud):
        trimesh = TriMesh(shape.points)
        n_vertices = shape.n_points
        points = shape.points
    else:
        raise ValueError("shape must be either a LandmarkGroup or a "
                         "PointCloud instance.")

    # get edges
    edges = trimesh.edge_indices()

    # return graph
    if return_pointgraph:
        return PointUndirectedGraph.init_from_edges(
            points=points, edges=edges, skip_checks=True)
    else:
        return UndirectedGraph.init_from_edges(
            edges=edges, n_vertices=n_vertices, skip_checks=True)
