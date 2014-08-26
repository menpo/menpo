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


def reindex_adjacency_array(adjacency_array):
    remap_vector = np.arange(np.max(adjacency_array) + 1)
    unique_values = np.unique(adjacency_array)
    remap_vector[unique_values] = np.arange(unique_values.shape[0])

    # Apply the mask
    return remap_vector[adjacency_array]
