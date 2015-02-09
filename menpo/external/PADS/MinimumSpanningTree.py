"""MinimumSpanningTree.py

Kruskal's algorithm for minimum spanning trees. D. Eppstein, April 2006.

NOTE: This file is different from the original to fit with menpo's structure.
"""
from UnionFind import UnionFind

from menpo.shape import UndirectedGraph


def MinimumSpanningTree(graph, weights):
    """
    Return the minimum spanning tree of an undirected graph G.
    G should be represented in such a way that iter(G) lists its
    vertices, iter(G[u]) lists the neighbors of u, G[u][v] gives the
    length of edge u,v, and G[u][v] should always equal G[v][u].
    The tree is returned as a list of edges.
    """
    if not isinstance(graph, UndirectedGraph):
        raise ValueError("Provided graph is not an UndirectedGraph.")
    for vertex in range(graph.n_vertices):
        for child in graph.adjacency_list[vertex]:
            if weights[vertex, child] != weights[child, vertex]:
                raise ValueError("Assymetric weights provided.")

    # Kruskal's algorithm: sort edges by weight, and add them one at a time.
    # We use Kruskal's algorithm, first because it is very simple to
    # implement once UnionFind exists, and second, because the only slow
    # part (the sort) is sped up by being built in to Python.
    subtrees = UnionFind()
    tree = []
    for W, u, v in sorted((weights[u][v], u, v) for u in range(graph.n_vertices)
                          for v in graph.adjacency_list[u]):
        if subtrees[u] != subtrees[v]:
            tree.append((u, v))
            subtrees.union(u, v)
    return tree

