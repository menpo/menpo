from collections import OrderedDict
import numpy as np
from pytest import raises
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from menpo.shape import LabelledPointUndirectedGraph, PointUndirectedGraph
from menpo.testing import is_same_array


points = np.ones((10, 3))
adjacency_matrix = csr_matrix((10, 10))
mask_dict = OrderedDict([('all', np.ones(10, dtype=np.bool))])
lower = np.array([True, True, True, True, True, True,
                  False, False, False, False], dtype=np.bool)
upper = np.array([False, False, False, False, False, False,
                  True, True, True, True], dtype=np.bool)
mask_dict_2 = OrderedDict([('lower', lower), ('upper', upper)])
mask_dict_3 = OrderedDict([('all', np.ones(10, dtype=np.bool)),
                           ('lower', lower), ('upper', upper)])


def test_LabelledPointUndirectedGraph_copy_true():
    lgroup = LabelledPointUndirectedGraph(points, adjacency_matrix, mask_dict)
    assert not is_same_array(lgroup.points, points)
    assert lgroup._labels_to_masks is not mask_dict
    assert lgroup.adjacency_matrix is not adjacency_matrix


def test_LabelledPointUndirectedGraph_copy_false():
    lgroup = LabelledPointUndirectedGraph(points, adjacency_matrix, mask_dict, copy=False)
    assert is_same_array(lgroup.points, points)
    assert lgroup._labels_to_masks is mask_dict
    assert lgroup.adjacency_matrix is adjacency_matrix


def test_LabelledPointUndirectedGraph_copy_method():
    lgroup = LabelledPointUndirectedGraph(points, adjacency_matrix, mask_dict)
    lgroup_copy = lgroup.copy()

    assert not is_same_array(lgroup_copy.points, lgroup.points)
    # Check the mask dictionary is deepcopied properly
    assert lgroup._labels_to_masks is not lgroup_copy._labels_to_masks
    masks = zip(lgroup_copy._labels_to_masks.values(),
                lgroup._labels_to_masks.values())
    for ms in masks:
        assert ms[0] is not ms[1]


def test_LabelledPointUndirectedGraph_iterate_labels():
    lgroup = LabelledPointUndirectedGraph(points, adjacency_matrix, mask_dict)

    for l in lgroup.labels:
        assert l == 'all'
    assert len(lgroup.labels) == 1


def test_LabelledPointUndirectedGraph_get_label():
    pcloud = LabelledPointUndirectedGraph(points, adjacency_matrix, mask_dict_2)

    assert isinstance(pcloud, PointUndirectedGraph)
    assert pcloud.get_label('lower').n_points == 6
    assert pcloud.get_label('upper').n_points == 4


def test_LabelledPointUndirectedGraph_add_label():
    lgroup = LabelledPointUndirectedGraph(points, adjacency_matrix, mask_dict_2)

    new_lgroup = lgroup.add_label('lower2', [0, 1])
    assert not is_same_array(new_lgroup.points, lgroup.points)

    lower_pcloud = new_lgroup.get_label('lower2')
    assert lower_pcloud.n_points == 2
    assert_allclose(lower_pcloud.points[0, :], [1., 1., 1.])
    assert_allclose(lower_pcloud.points[1, :], [1., 1., 1.])


def test_LabelledPointUndirectedGraph_add_ordered_labels():
    lgroup = LabelledPointUndirectedGraph(points, adjacency_matrix, mask_dict_2)

    labels = lgroup.labels
    assert labels[0] == 'lower'
    assert labels[1] == 'upper'

    new_lgroup = lgroup.add_label('A', [0, 1])
    new_labels = new_lgroup.labels

    assert new_labels[2] == 'A'


def test_LabelledPointUndirectedGraph_remove_label():
    lgroup = LabelledPointUndirectedGraph(points, adjacency_matrix, mask_dict_3)

    new_lgroup = lgroup.remove_label('lower')

    assert 'all' in new_lgroup.labels
    assert 'lower' not in new_lgroup.labels
    assert 'all' in lgroup.labels


def test_LabelledPointUndirectedGraph_remove_label_unlabelled():
    lgroup = LabelledPointUndirectedGraph(points, adjacency_matrix, mask_dict_2)

    with raises(ValueError):
        lgroup.remove_label('lower')


def test_LabelledPointUndirectedGraph_create_unlabelled():
    points = np.array([[0, 1], [2, 3], [4, 5]])
    adjacency_matrix = csr_matrix((3, 3))
    mask_dict = OrderedDict([('all', np.zeros(3, dtype=np.bool))])

    with raises(ValueError):
        LabelledPointUndirectedGraph(points, adjacency_matrix, mask_dict)


def test_LabelledPointUndirectedGraph_pass_non_ordered_dict():
    points = np.array([[0, 1], [2, 3], [4, 5]])
    adjacency_matrix = csr_matrix((3, 3))
    mask_dict = {'all': np.ones(3, dtype=np.bool)}

    with raises(ValueError):
        LabelledPointUndirectedGraph(points, adjacency_matrix, mask_dict)


def test_LabelledPointUndirectedGraph_create_no_mask():
    points = np.array([[0, 1], [2, 3], [4, 5]])
    adjacency_matrix = csr_matrix((3, 3))
    mask_dict = None

    with raises(ValueError):
        LabelledPointUndirectedGraph(points, adjacency_matrix, mask_dict)


def test_LabelledPointUndirectedGraph_create_incorrect_shape():
    points = np.array([[0, 1], [2, 3], [4, 5]])
    adjacency_matrix = csr_matrix((3, 3))
    mask_dict = OrderedDict([('all', np.ones(5, dtype=np.bool))])

    with raises(ValueError):
        LabelledPointUndirectedGraph(points, adjacency_matrix, mask_dict)


def test_LabelledPointUndirectedGraph_with_labels():
    lgroup = LabelledPointUndirectedGraph(points, adjacency_matrix, mask_dict_2)

    new_lgroup = lgroup.with_labels('lower')

    assert new_lgroup.n_labels == 1
    assert new_lgroup.n_points == 6
    assert 'lower' in new_lgroup.labels

    new_lgroup = lgroup.with_labels(['lower'])

    assert new_lgroup.n_labels == 1
    assert new_lgroup.n_points == 6
    assert 'lower' in new_lgroup.labels


def test_LabelledPointUndirectedGraph_without_labels():
    lgroup = LabelledPointUndirectedGraph(points, adjacency_matrix, mask_dict_2)

    new_lgroup = lgroup.without_labels('upper')

    assert new_lgroup.n_labels == 1
    assert new_lgroup.n_points == 6
    assert 'lower' in new_lgroup.labels
    assert 'upper' not in new_lgroup.labels

    new_lgroup = lgroup.without_labels(['upper'])

    assert new_lgroup.n_labels == 1
    assert new_lgroup.n_points == 6
    assert 'lower' in new_lgroup.labels
    assert 'upper' not in new_lgroup.labels


def test_LabelledPointUndirectedGraph_str():
    lgroup = LabelledPointUndirectedGraph(points, adjacency_matrix, mask_dict_2)

    out_str = lgroup.__str__()
    assert len(out_str) > 0


def test_LabelledPointUndirectedGraph_create_with_all_label():
    lgroup = LabelledPointUndirectedGraph.init_with_all_label(points,
                                                              adjacency_matrix)

    assert lgroup.n_labels == 1
    assert 'all' in lgroup.labels


def test_LabelledPointUndirectedGraph_has_nan_values():
    points2 = points.copy()
    points2[0, 0] = np.nan
    lgroup = LabelledPointUndirectedGraph.init_with_all_label(points2,
                                                              adjacency_matrix)
    assert lgroup.has_nan_values()
