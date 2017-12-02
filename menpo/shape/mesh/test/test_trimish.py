import warnings
import numpy as np
from numpy.testing import assert_allclose
from menpo.image import Image, MaskedImage
from menpo.shape import TriMesh, TexturedTriMesh, ColouredTriMesh
from menpo.testing import is_same_array


def test_trimesh_creation():
    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 1, 0]])
    trilist = np.array([[0, 1, 3],
                        [1, 2, 3]])
    TriMesh(points, trilist=trilist)


def test_trimesh_init_2d_grid():
    tm = TriMesh.init_2d_grid([10, 10])
    assert tm.n_points == 100
    assert tm.n_dims == 2
    # 162 = 9 * 9 * 2
    assert_allclose(tm.trilist.shape, (162, 3))
    assert_allclose(tm.range(), [9, 9])


def test_trimesh_init_from_depth_image():
    fake_z = np.random.uniform(size=(10, 10))
    tm = TriMesh.init_from_depth_image(Image(fake_z))
    assert tm.n_points == 100
    assert tm.n_dims == 3
    assert_allclose(tm.range()[:2], [9, 9])
    assert tm.points[:, -1].max() <= 1.0
    assert tm.points[:, -1].min() >= 0.0


def test_trimesh_init_from_depth_image_masked():
    fake_z = np.random.uniform(size=(10, 10))
    mask = np.zeros(fake_z.shape, dtype=np.bool)
    mask[2:6, 2:6] = True
    im = MaskedImage(fake_z, mask=mask)
    tm = TriMesh.init_from_depth_image(im)
    assert tm.n_points == 16
    assert tm.n_dims == 3
    assert_allclose(tm.range()[:2], [3, 3])
    assert tm.points[:, -1].max() <= 1.0
    assert tm.points[:, -1].min() >= 0.0


def test_colouredtrimesh_init_2d_grid():
    tm = ColouredTriMesh.init_2d_grid([10, 10])
    assert tm.n_points == 100
    assert tm.n_dims == 2
    # 162 = 9 * 9 * 2
    assert_allclose(tm.trilist.shape, (162, 3))
    assert_allclose(tm.range(), [9, 9])


def test_colouredtrimesh_init_from_depth_image():
    fake_z = np.random.uniform(size=(10, 10))
    tm = ColouredTriMesh.init_from_depth_image(Image(fake_z))
    assert tm.n_points == 100
    assert tm.n_dims == 3
    assert_allclose(tm.range()[:2], [9, 9])
    assert tm.points[:, -1].max() <= 1.0
    assert tm.points[:, -1].min() >= 0.0


def test_colouredtrimesh_init_from_depth_image_masked():
    fake_z = np.random.uniform(size=(10, 10))
    mask = np.zeros(fake_z.shape, dtype=np.bool)
    mask[2:6, 2:6] = True
    im = MaskedImage(fake_z, mask=mask)
    tm = ColouredTriMesh.init_from_depth_image(im)
    assert tm.n_points == 16
    assert tm.n_dims == 3
    assert_allclose(tm.range()[:2], [3, 3])
    assert tm.points[:, -1].max() <= 1.0
    assert tm.points[:, -1].min() >= 0.0


def test_colouredtrimesh_init_from_depth_image_coloured():
    fake_z = np.random.uniform(size=(10, 10))
    fake_colours = np.random.uniform(size=(100, 3))
    tm = ColouredTriMesh.init_from_depth_image(Image(fake_z),
                                               colours=fake_colours)
    assert tm.n_points == 100
    assert tm.n_dims == 3
    assert tm.colours.shape == (100, 3)
    assert_allclose(tm.range()[:2], [9, 9])
    assert tm.points[:, -1].max() <= 1.0
    assert tm.points[:, -1].min() >= 0.0


def test_texturedtrimesh_init_2d_grid():
    tm = TexturedTriMesh.init_2d_grid([10, 10])
    assert tm.n_points == 100
    assert tm.n_dims == 2
    # 162 = 9 * 9 * 2
    assert_allclose(tm.trilist.shape, (162, 3))
    assert_allclose(tm.range(), [9, 9])


def test_texturedtrimesh_init_from_depth_image():
    fake_z = np.random.uniform(size=(10, 10))
    tm = TexturedTriMesh.init_from_depth_image(Image(fake_z))
    assert tm.n_points == 100
    assert tm.n_dims == 3
    assert_allclose(tm.range()[:2], [9, 9])
    assert tm.points[:, -1].max() <= 1.0
    assert tm.points[:, -1].min() >= 0.0


def test_texturedtrimesh_init_from_depth_image_masked():
    fake_z = np.random.uniform(size=(10, 10))
    mask = np.zeros(fake_z.shape, dtype=np.bool)
    mask[2:6, 2:6] = True
    im = MaskedImage(fake_z, mask=mask)
    tm = TexturedTriMesh.init_from_depth_image(im)
    assert tm.n_points == 16
    assert tm.n_dims == 3
    assert_allclose(tm.range()[:2], [3, 3])
    assert tm.points[:, -1].max() <= 1.0
    assert tm.points[:, -1].min() >= 0.0


def test_texturedtrimesh_init_from_depth_image_textured():
    fake_z = np.random.uniform(size=(10, 10))
    fake_tex = Image(fake_z)
    tm = TexturedTriMesh.init_from_depth_image(fake_tex, texture=fake_tex)
    assert tm.n_points == 100
    assert tm.n_dims == 3
    assert tm.texture.shape == (10, 10)
    assert_allclose(tm.range()[:2], [9, 9])
    assert tm.points[:, -1].max() <= 1.0
    assert tm.points[:, -1].min() >= 0.0


def test_trimesh_creation_copy_true():
    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 1, 0]])
    trilist = np.array([[0, 1, 3],
                        [1, 2, 3]])
    tm = TriMesh(points, trilist=trilist)
    assert (not is_same_array(tm.points, points))
    assert (not is_same_array(tm.trilist, trilist))


def test_trimesh_creation_copy_false():
    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 1, 0]])
    trilist = np.array([[0, 1, 3],
                        [1, 2, 3]])
    tm = TriMesh(points, trilist=trilist, copy=False)
    assert (is_same_array(tm.points, points))
    assert (is_same_array(tm.trilist, trilist))


def test_texturedtrimesh_creation_copy_false():
    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 1, 0]])
    trilist = np.array([[0, 1, 3],
                        [1, 2, 3]])
    pixels = np.ones([10, 10])
    tcoords = np.ones([4, 2])
    texture = Image(pixels, copy=False)
    ttm = TexturedTriMesh(points, tcoords, texture, trilist=trilist,
                          copy=False)
    assert (is_same_array(ttm.points, points))
    assert (is_same_array(ttm.trilist, trilist))
    assert (is_same_array(ttm.tcoords.points, tcoords))
    assert (is_same_array(ttm.texture.pixels, pixels))


def test_texturedtrimesh_creation_copy_true():
    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 1, 0]])
    trilist = np.array([[0, 1, 3],
                        [1, 2, 3]])
    pixels = np.ones([10, 10, 1])
    tcoords = np.ones([4, 2])
    texture = Image(pixels, copy=False)
    ttm = TexturedTriMesh(points, tcoords, texture, trilist=trilist,
                          copy=True)
    assert (not is_same_array(ttm.points, points))
    assert (not is_same_array(ttm.trilist, trilist))
    assert (not is_same_array(ttm.tcoords.points, tcoords))
    assert (not is_same_array(ttm.texture.pixels, pixels))


def test_colouredtrimesh_creation_copy_false():
    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 1, 0]])
    trilist = np.array([[0, 1, 3],
                        [1, 2, 3]])
    colours = np.ones([4, 13])
    ttm = ColouredTriMesh(points, trilist=trilist, colours=colours, copy=False)
    assert (is_same_array(ttm.points, points))
    assert (is_same_array(ttm.trilist, trilist))
    assert (is_same_array(ttm.colours, colours))


def test_colouredtrimesh_creation_copy_true():
    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 1, 0]])
    trilist = np.array([[0, 1, 3],
                        [1, 2, 3]])
    colours = np.ones([4, 13])
    ttm = ColouredTriMesh(points, trilist=trilist, colours=colours, copy=True)
    assert (not is_same_array(ttm.points, points))
    assert (not is_same_array(ttm.trilist, trilist))
    assert (not is_same_array(ttm.colours, colours))


def test_trimesh_creation_copy_warning():
    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 1, 0]])
    trilist = np.array([[0, 1, 3],
                        [1, 2, 3]], order='F')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        TriMesh(points, trilist=trilist, copy=False)
        assert len(w) == 1


def test_trimesh_n_dims():
    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 1, 0]])
    trilist = np.array([[0, 1, 3],
                        [1, 2, 3]])
    trimesh = TriMesh(points, trilist=trilist)
    assert(trimesh.n_dims == 3)


def test_trimesh_n_points():
    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 1, 0]])
    trilist = np.array([[0, 1, 3],
                        [1, 2, 3]])
    trimesh = TriMesh(points, trilist=trilist)
    assert(trimesh.n_points == 4)


def test_trimesh_n_tris():
    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 1, 0]])
    trilist = np.array([[0, 1, 3],
                        [1, 2, 3]])
    trimesh = TriMesh(points, trilist=trilist)
    assert(trimesh.n_tris == 2)


def test_trimesh_from_tri_mask():
    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 1, 0]])
    trilist = np.array([[0, 1, 3],
                        [1, 2, 3]])
    mask = np.zeros(2, dtype=np.bool)
    mask[0] = True
    trimesh = TriMesh(points, trilist=trilist).from_tri_mask(mask)
    assert(trimesh.n_tris == 1)
    assert(trimesh.n_points == 3)
    assert_allclose(trimesh.points, points[trilist[0]])


def test_trimesh_face_normals():
    points = np.array([[0.0, 0.0, -1.0],
                       [1.0, 0.0, 0.0],
                       [1.0, 1.0, 0.0],
                       [0.0, 1.0, 0.0]])
    trilist = np.array([[0, 1, 3],
                        [1, 2, 3]])
    expected_normals = np.array([[-np.sqrt(3)/3, -np.sqrt(3)/3, np.sqrt(3)/3],
                                 [-0, -0, 1]])
    trimesh = TriMesh(points, trilist=trilist)
    face_normals = trimesh.tri_normals()
    assert_allclose(face_normals, expected_normals)

    trimesh.points = trimesh.points.astype(np.float32)
    face_normals = trimesh.tri_normals()
    assert_allclose(face_normals, expected_normals)


def test_trimesh_vertex_normals():
    points = np.array([[0.0, 0.0, -1.0],
                       [1.0, 0.0, 0.0],
                       [1.0, 1.0, 0.0],
                       [0.0, 1.0, 0.0]])
    trilist = np.array([[0, 1, 3],
                        [1, 2, 3]])
    # 0 and 2 are the corner of the triangles and so the maintain the
    # face normals. The other two are the re-normalized vertices:
    # normalize(n0 + n2)
    expected_normals = np.array([[-np.sqrt(3)/3, -np.sqrt(3)/3, np.sqrt(3)/3],
                                 [-0.32505758,  -0.32505758, 0.88807383],
                                 [0, 0, 1],
                                 [-0.32505758,  -0.32505758, 0.88807383]])
    trimesh = TriMesh(points, trilist)
    vertex_normals = trimesh.vertex_normals()
    assert_allclose(vertex_normals, expected_normals)

    trimesh.points = trimesh.points.astype(np.float32)
    vertex_normals = trimesh.vertex_normals()
    assert_allclose(vertex_normals, expected_normals)
