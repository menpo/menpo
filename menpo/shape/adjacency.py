import numpy as np


def mask_adjacency_array(mask, adjacency_array):
    # Find the indices that have been asked to be removed
    indices_to_remove = np.nonzero(~mask)[0]
    # Set intersection to find any rows containing those elements,
    # reshape back in to the same size as adjacency array
    entries_to_remove = np.in1d(adjacency_array, indices_to_remove)
    entries_to_remove = entries_to_remove.reshape([-1,
                                                   adjacency_array.shape[1]])
    # Only keep those entries that are not flagged for removal
    indices_to_keep = ~entries_to_remove.any(axis=1)
    return adjacency_array[indices_to_keep, :]


def mask_adjacency_array_tree(mask, adjacency_array, adjacency_list,
                              predecessors_list, root_vertex):
    # Find the indices that have been asked to be removed
    vertices_to_remove = np.nonzero(~mask)[0]
    # Impossible to remove root vertex
    if root_vertex in vertices_to_remove:
        raise ValueError('If root vertex is removed, then there is no tree!')
    # Get list of edges to be removed
    edges_to_be_removed = []
    for v in vertices_to_remove:
        edges_to_be_removed.append([predecessors_list[v], v])
        _remove_tree_edge(adjacency_list, v, edges_to_be_removed)
    # Return ndarray of edges to keep, i.e. new adjacency_array
    return np.array(list_diff(adjacency_array.tolist(), edges_to_be_removed))


list_diff = lambda l1, l2: [x for x in l1 if x not in l2]


def _remove_tree_edge(adjacency_list, vertex, edges_to_be_removed):
    for c in adjacency_list[vertex]:
        edges_to_be_removed.append([vertex, c])
        _remove_tree_edge(adjacency_list, c, edges_to_be_removed)


def reindex_adjacency_array(adjacency_array):
    remap_vector = np.arange(np.max(adjacency_array) + 1)
    unique_values = np.unique(adjacency_array)
    remap_vector[unique_values] = np.arange(unique_values.shape[0])

    # Apply the mask
    return remap_vector[adjacency_array]
