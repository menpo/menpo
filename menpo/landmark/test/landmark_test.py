import numpy as np
from menpo.landmark import LandmarkGroup, LandmarkManager
from menpo.shape import PointCloud
from menpo.testing import is_same_array


def test_LandmarkGroup_copy_true():
    points = np.ones((10, 3))
    mask_dict = {'all': np.ones(10, dtype=np.bool)}
    pcloud = PointCloud(points, copy=False)
    lgroup = LandmarkGroup(None, 'label', pcloud, mask_dict)
    assert (not is_same_array(lgroup._pointcloud.points, points))
    assert (lgroup._labels_to_masks is not mask_dict)
    assert (lgroup._pointcloud is not pcloud)


def test_LandmarkGroup_copy_false():
    points = np.ones((10, 3))
    mask_dict = {'all': np.ones(10, dtype=np.bool)}
    pcloud = PointCloud(points, copy=False)
    lgroup = LandmarkGroup(None, 'label', pcloud, mask_dict, copy=False)
    assert (is_same_array(lgroup._pointcloud.points, points))
    assert (lgroup._labels_to_masks is mask_dict)
    assert (lgroup._pointcloud is pcloud)


def test_LandmarkManager_set_not_copy_target():
    points = np.ones((10, 3))
    mask_dict = {'all': np.ones(10, dtype=np.bool)}
    pcloud = PointCloud(points, copy=False)
    target = PointCloud(points)
    lgroup = LandmarkGroup(None, 'label', pcloud, mask_dict, copy=False)

    man = LandmarkManager(target)
    man['test_set'] = lgroup
    assert (not is_same_array(man['test_set']._pointcloud.points,
                              lgroup._pointcloud.points))
    assert (man['test_set'] is not lgroup._labels_to_masks)
    assert (man['test_set']._target is target)
