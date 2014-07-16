import numpy as np

from menpo.image import Image
from menpo.shape import PointCloud, TriMesh, TexturedTriMesh, ColouredTriMesh
from menpo.testing import is_same_array


def test_trimesh_copy():
    points = np.ones([10, 3])
    trilist = np.ones([10, 3])
    landmarks = PointCloud(np.ones([3, 3]), copy=False)

    tmesh = TriMesh(points, trilist=trilist, copy=False)
    tmesh.landmarks['test'] = landmarks
    tmesh_copy = tmesh.copy()

    assert (not is_same_array(tmesh_copy.points, tmesh.points))
    assert (not is_same_array(tmesh_copy.trilist, tmesh.trilist))
    assert (not is_same_array(tmesh_copy.landmarks['test'].lms.points,
                              tmesh.landmarks['test'].lms.points))


def test_colouredtrimesh_copy():
    points = np.ones([10, 3])
    colours = np.ones([10, 3])
    trilist = np.ones([10, 3])
    landmarks = PointCloud(np.ones([3, 3]), copy=False)

    ctmesh = ColouredTriMesh(points, trilist=trilist, colours=colours,
                             copy=False)
    ctmesh.landmarks['test'] = landmarks
    ctmesh_copy = ctmesh.copy()

    assert (not is_same_array(ctmesh_copy.points, ctmesh.points))
    assert (not is_same_array(ctmesh_copy.trilist, ctmesh.trilist))
    assert (not is_same_array(ctmesh_copy.colours, ctmesh.colours))
    assert (not is_same_array(ctmesh_copy.landmarks['test'].lms.points,
                              ctmesh.landmarks['test'].lms.points))


def test_texturedtrimesh_copy():
    points = np.ones([10, 3])
    tcoords = np.ones([10, 3])
    trilist = np.ones([10, 3])
    landmarks = PointCloud(np.ones([3, 3]), copy=False)
    landmarks_im = PointCloud(np.ones([3, 2]), copy=False)
    pixels = np.ones([10, 10, 1])
    texture = Image(pixels, copy=False)
    texture.landmarks['test_im'] = landmarks_im

    ttmesh = TexturedTriMesh(points, tcoords, texture, trilist=trilist,
                             copy=False)
    ttmesh.landmarks['test'] = landmarks
    ttmesh_copy = ttmesh.copy()

    assert (not is_same_array(ttmesh_copy.points, ttmesh.points))
    assert (not is_same_array(ttmesh_copy.trilist, ttmesh.trilist))
    assert (not is_same_array(ttmesh_copy.tcoords.points,
                              ttmesh.tcoords.points))
    assert (not is_same_array(ttmesh_copy.texture.pixels,
                              ttmesh.texture.pixels))
    assert (not is_same_array(
        ttmesh_copy.texture.landmarks['test_im'].lms.points,
        ttmesh.texture.landmarks['test_im'].lms.points))
    assert (not is_same_array(ttmesh_copy.landmarks['test'].lms.points,
                              ttmesh.landmarks['test'].lms.points))
