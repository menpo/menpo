import numpy as np

from menpo.landmark.labels.human import face
from menpo.shape import PointCloud


def _cloud():
    return PointCloud(np.random.RandomState(0).rand(68, 2))


def test_the_49_point_trimesh_is_the_66_point_one_without_the_jaw():
    # it used to be a copy of the 66 point triangle list, which refers to points that 49 do not have
    pc = _cloud()
    mesh66, _ = face.face_ibug_68_to_face_ibug_66_trimesh(pc, return_mapping=True)
    mesh49, _ = face.face_ibug_68_to_face_ibug_49_trimesh(pc, return_mapping=True)
    assert mesh49.n_points == 49 and mesh66.n_points == 66
    assert mesh49.trilist.max() == 48
    # the 49 points are the last 49 of the 66 (the first 17 are the jaw)
    np.testing.assert_array_equal(mesh49.points, mesh66.points[17:])
    # and its triangles are the ones of the 66 that do not use the jaw
    uses_no_jaw = (mesh66.trilist >= 17).all(axis=1)
    no_repeat = np.array([len(set(t)) == 3 for t in mesh66.trilist.tolist()])
    expected = {tuple(sorted(t - 17)) for t in mesh66.trilist[uses_no_jaw & no_repeat]}
    assert {tuple(sorted(t)) for t in mesh49.trilist.tolist()} == expected
    assert (mesh49.tri_areas() > 0).all()


def test_the_51_point_trimesh_has_no_degenerate_triangles():
    mesh, _ = face.face_ibug_68_to_face_ibug_51_trimesh(_cloud(), return_mapping=True)
    assert mesh.n_points == 51 and (mesh.tri_areas() > 0).all()
