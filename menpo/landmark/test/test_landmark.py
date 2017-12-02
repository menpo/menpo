from collections import OrderedDict
import numpy as np
from scipy.sparse import csr_matrix
from pytest import raises
from numpy.testing import assert_allclose, assert_equal

from menpo.landmark import LandmarkManager
from menpo.shape import PointCloud, LabelledPointUndirectedGraph
from menpo.testing import is_same_array


points = np.ones((10, 3))
adjacency_matrix = csr_matrix((10, 10))
mask_dict = OrderedDict([('all', np.ones(10, dtype=np.bool))])


def test_LandmarkManager_set_LabelledPointUndirectedGraph():
    lgroup = LabelledPointUndirectedGraph(points, adjacency_matrix, mask_dict)

    man = LandmarkManager()
    man['test_set'] = lgroup
    assert not is_same_array(man['test_set'].points,
                             lgroup.points)
    assert man['test_set']._labels_to_masks is not lgroup._labels_to_masks


def test_LandmarkManager_set_None_key():
    lgroup = LabelledPointUndirectedGraph.init_with_all_label(points, adjacency_matrix)

    man = LandmarkManager()
    with raises(ValueError):
        man[None] = lgroup


def test_LandmarkManager_set_pointcloud():
    pcloud = PointCloud(points)

    man = LandmarkManager()
    man['test_set'] = pcloud

    new_pcloud = man['test_set']
    assert new_pcloud is not pcloud
    assert isinstance(new_pcloud, PointCloud)


def test_LandmarkManager_copy_method():
    pcloud = PointCloud(points)

    man = LandmarkManager()
    man['test_set'] = pcloud
    man_copy = man.copy()

    assert man_copy['test_set'] is not man['test_set']
    assert not is_same_array(man_copy['test_set'].points,
                             man['test_set'].points)


def test_LandmarkManager_set_PointCloud_not_copy_target():
    pcloud = PointCloud(points)

    man = LandmarkManager()
    man['test_set'] = pcloud
    assert not is_same_array(man['test_set'].points,
                             pcloud.points)
    assert_allclose(man['test_set'].points, np.ones([10, 3]))


def test_LandmarkManager_iterate():
    pcloud = PointCloud(points)

    man = LandmarkManager()
    man['test_set'] = pcloud
    man['test_set2'] = pcloud
    man['test_set3'] = pcloud

    for l in man:
        assert not is_same_array(man[l].points, pcloud.points)
    assert len(man) == 3


def test_LandmarkManager_get():
    lgroup = LabelledPointUndirectedGraph(points, adjacency_matrix, mask_dict)

    man = LandmarkManager()
    man._landmark_groups['test_set'] = lgroup

    assert(man['test_set'] is lgroup)
    assert is_same_array(man['test_set'].points, lgroup.points)


def test_LandmarkManager_set():
    lgroup = LabelledPointUndirectedGraph(points, adjacency_matrix, mask_dict)

    man = LandmarkManager()
    man['test_set'] = lgroup

    assert_allclose(man._landmark_groups['test_set'].points,
                    lgroup.points)
    assert man._landmark_groups['test_set'].n_labels == 1


def test_LandmarkManager_del():
    pcloud = PointCloud(points)

    man = LandmarkManager()
    man['test_set'] = pcloud

    del man['test_set']

    assert man.n_groups == 0


def test_LandmarkManager_group_order():
    points = np.ones((10, 3))
    pcloud = PointCloud(points, copy=False)
    target = PointCloud(points)

    man = LandmarkManager()
    man['test_set'] = pcloud.copy()
    man['abc'] = pcloud.copy()
    man['def'] = pcloud.copy()
    assert_equal(list(man._landmark_groups.keys()), ['test_set', 'abc', 'def'])
    # check that after deleting and inserting the order will remain the same.
    del man['test_set']
    man['tt'] = pcloud.copy()
    assert_equal(list(man._landmark_groups.keys()), ['abc', 'def', 'tt'])


def test_LandmarkManager_in():
    pcloud = PointCloud(points)

    man = LandmarkManager()
    man['test_set'] = pcloud

    assert 'test_set' in man


def test_LandmarkManager_str():
    pcloud = PointCloud(points)

    man = LandmarkManager()
    man['test_set'] = pcloud

    out_str = man.__str__()
    assert len(out_str) > 0
