import numpy as np
from nose.tools import assert_equal
from numpy.testing import assert_allclose

from menpo.landmark import LandmarkGroup, LandmarkManager
from menpo.shape import PointCloud
from menpo.testing import is_same_array


def test_LandmarkGroup_copy_true():
    points = np.ones((10, 3))
    mask_dict = {'all': np.ones(10, dtype=np.bool)}
    pcloud = PointCloud(points, copy=False)
    lgroup = LandmarkGroup(None, 'label', pcloud, mask_dict)
    assert (not is_same_array(lgroup.lms.points, points))
    assert (lgroup._labels_to_masks is not mask_dict)
    assert (lgroup.lms is not pcloud)


def test_LandmarkGroup_copy_false():
    points = np.ones((10, 3))
    mask_dict = {'all': np.ones(10, dtype=np.bool)}
    pcloud = PointCloud(points, copy=False)
    lgroup = LandmarkGroup(None, 'label', pcloud, mask_dict, copy=False)
    assert (is_same_array(lgroup._pointcloud.points, points))
    assert (lgroup._labels_to_masks is mask_dict)
    assert (lgroup.lms is pcloud)


def test_LandmarkManager_set_LandmarkGroup():
    points = np.ones((10, 3))
    mask_dict = {'all': np.ones(10, dtype=np.bool)}
    pcloud = PointCloud(points, copy=False)
    target = PointCloud(points)
    lgroup = LandmarkGroup(None, 'label', pcloud, mask_dict, copy=False)

    man = LandmarkManager(target)
    man['test_set'] = lgroup
    assert (not is_same_array(man['test_set'].lms.points,
                              lgroup.lms.points))
    assert_equal(man['test_set'].group_label, 'test_set')
    assert_allclose(man['test_set']['all'].lms.points, np.ones([10, 3]))
    assert (man['test_set']._labels_to_masks is not lgroup._labels_to_masks)
    assert (man['test_set']._target is target)


def test_LandmarkManager_set_pointcloud():
    points = np.ones((10, 3))
    pcloud = PointCloud(points, copy=False)
    target = PointCloud(points, copy=True)

    man = LandmarkManager(target)
    man['test_set'] = pcloud

    lgroup = man['test_set']
    assert (lgroup._target is target)
    assert (lgroup.lms is not pcloud)
    assert_allclose(lgroup._labels_to_masks['all'], np.ones(10, dtype=np.bool))


def test_landmarkgroup_copy_method():
    points = np.ones((10, 3))
    mask_dict = {'all': np.ones(10, dtype=np.bool)}
    pcloud = PointCloud(points, copy=False)
    target = PointCloud(points)

    lgroup = LandmarkGroup(target, 'label', pcloud, mask_dict, copy=False)
    lgroup_copy = lgroup.copy()

    assert (not is_same_array(lgroup_copy.lms.points,
                              lgroup.lms.points))
    assert (lgroup_copy._target is lgroup._target)
    # Check the mask dictionary is deepcopied properly
    assert (lgroup._labels_to_masks is not lgroup_copy._labels_to_masks)
    masks = zip(lgroup_copy._labels_to_masks.values(),
                lgroup._labels_to_masks.values())
    for ms in masks:
        assert (ms[0] is not ms[1])


def test_LandmarkManager_copy_method():
    points = np.ones((10, 3))
    pcloud = PointCloud(points, copy=False)
    target = PointCloud(points, copy=True)

    man = LandmarkManager(target)
    man['test_set'] = pcloud
    man_copy = man.copy()

    assert (man_copy._target is man._target)
    assert (man_copy['test_set'] is not man['test_set'])
    assert (not is_same_array(man_copy['test_set'].lms.points,
                              man['test_set'].lms.points))
    assert_equal(man['test_set'].group_label, 'test_set')


def test_LandmarkManager_set_PointCloud_not_copy_target():
    points = np.ones((10, 3))
    pcloud = PointCloud(points, copy=False)
    target = PointCloud(points)

    man = LandmarkManager(target)
    man['test_set'] = pcloud
    print(man['test_set'])
    assert (not is_same_array(man['test_set'].lms.points,
                              pcloud.points))
    assert_allclose(man['test_set']['all'].lms.points, np.ones([10, 3]))
    assert_equal(man['test_set'].group_label, 'test_set')
    assert (man['test_set']._target is target)


def test_LandmarkManager_iterate():
    points = np.ones((10, 3))
    pcloud = PointCloud(points, copy=False)
    target = PointCloud(points)

    man = LandmarkManager(target)
    man['test_set'] = pcloud

    for l in man:
        assert_equal(l, 'test_set')


def test_LandmarkGroup_iterate():
    points = np.ones((10, 3))
    mask_dict = {'all': np.ones(10, dtype=np.bool)}
    pcloud = PointCloud(points, copy=False)
    target = PointCloud(points)

    lgroup = LandmarkGroup(target, 'label', pcloud, mask_dict, copy=False)

    for l in lgroup:
        assert_equal(l, 'all')


def test_LandmarkManager_get():
    points = np.ones((10, 3))
    pcloud = PointCloud(points, copy=False)
    target = PointCloud(points)
    mask_dict = {'all': np.ones(10, dtype=np.bool)}

    lgroup = LandmarkGroup(target, 'label', pcloud, mask_dict, copy=False)

    man = LandmarkManager(target)
    man._landmark_groups['test_set'] = lgroup

    assert_equal(man['test_set'], lgroup)


def test_LandmarkManager_set():
    points = np.ones((10, 3))
    pcloud = PointCloud(points, copy=False)
    target = PointCloud(points)
    mask_dict = {'all': np.ones(10, dtype=np.bool)}

    lgroup = LandmarkGroup(target, 'label', pcloud, mask_dict, copy=False)

    man = LandmarkManager(target)
    man['test_set'] = lgroup

    assert_equal(man._landmark_groups['test_set'].group_label,
                 'test_set')
    assert (man._landmark_groups['test_set']._target is target)
    assert_allclose(man._landmark_groups['test_set'].lms.points,
                    lgroup.lms.points)
    assert_equal(man._landmark_groups['test_set'].n_labels, 1)


def test_LandmarkManager_del():
    points = np.ones((10, 3))
    pcloud = PointCloud(points, copy=False)
    target = PointCloud(points)

    man = LandmarkManager(target)
    man['test_set'] = pcloud

    del man['test_set']

    assert_equal(man.n_groups, 0)


def test_LandmarkManager_in():
    points = np.ones((10, 3))
    pcloud = PointCloud(points, copy=False)
    target = PointCloud(points)

    man = LandmarkManager(target)
    man['test_set'] = pcloud

    assert ('test_set' in man)
