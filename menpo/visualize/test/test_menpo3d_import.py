from collections import OrderedDict
from mock import patch, MagicMock
import numpy as np
from pytest import raises
from scipy.sparse import csr_matrix

from menpo.image import Image
from menpo.shape import (TriMesh, TexturedTriMesh, ColouredTriMesh, PointCloud,
                         LabelledPointUndirectedGraph)
from menpo.visualize import Menpo3dMissingError
from menpo.testing import surrogate


fake_triangle = np.array([[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]])
fake_trilist = np.array([[0, 1, 2]], dtype=np.uint32)
fake_texture = Image.init_blank([10, 10])
fake_tcoords = np.array([[0, 0], [0.5, 0.5], [1.0, 1.0]])
menpo3d_visualize_mock = MagicMock()
menpo3d_visualize_mock.side_effect = ImportError


@surrogate('menpo3d.visualize.TriMeshViewer3d')
@patch('menpo3d.visualize.TriMeshViewer3d', menpo3d_visualize_mock)
def test_trimesh_viewer():
    with raises(Menpo3dMissingError):
        TriMesh(fake_triangle, trilist=fake_trilist, copy=False).view()


@surrogate('menpo3d.visualize.TexturedTriMeshViewer3d')
@patch('menpo3d.visualize.TexturedTriMeshViewer3d', menpo3d_visualize_mock)
def test_textured_trimesh_viewer():
    with raises(Menpo3dMissingError):
        TexturedTriMesh(fake_triangle, fake_tcoords, fake_texture,
                        trilist=fake_trilist, copy=False).view()


@surrogate('menpo3d.visualize.ColouredTriMeshViewer3d')
@patch('menpo3d.visualize.ColouredTriMeshViewer3d', menpo3d_visualize_mock)
def test_coloured_trimesh_viewer():
    with raises(Menpo3dMissingError):
        ColouredTriMesh(fake_triangle, colours=fake_tcoords,
                        trilist=fake_trilist, copy=False).view()


@surrogate('menpo3d.visualize.PointCloudViewer3d')
@patch('menpo3d.visualize.PointCloudViewer3d', menpo3d_visualize_mock)
def test_pointcloud3d_viewer():
    with raises(Menpo3dMissingError):
        PointCloud(fake_triangle, copy=False).view()


@surrogate('menpo3d.visualize.LandmarkViewer3d')
@patch('menpo3d.visualize.LandmarkViewer3d', menpo3d_visualize_mock)
def test_landmark3d_viewer():
    adj_matrix = csr_matrix((3, 3))
    labels_map = OrderedDict([('all', np.ones(3, dtype=np.bool))])
    with raises(Menpo3dMissingError):
        LabelledPointUndirectedGraph(fake_triangle,
                                     adj_matrix,
                                     labels_map,
                                     copy=False).view()
