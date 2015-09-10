from collections import OrderedDict
from mock import patch, MagicMock
import numpy as np
from nose.tools import raises
from menpo.image import Image
from menpo.landmark import LandmarkGroup
from menpo.shape import TriMesh, TexturedTriMesh, ColouredTriMesh, PointCloud
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
@raises(Menpo3dMissingError)
def trimesh_viewer_test():
    TriMesh(fake_triangle, trilist=fake_trilist, copy=False).view()


@surrogate('menpo3d.visualize.TexturedTriMeshViewer3d')
@patch('menpo3d.visualize.TexturedTriMeshViewer3d', menpo3d_visualize_mock)
@raises(Menpo3dMissingError)
def textured_trimesh_viewer_test():
    TexturedTriMesh(fake_triangle, fake_tcoords, fake_texture,
                    trilist=fake_trilist, copy=False).view()


@surrogate('menpo3d.visualize.ColouredTriMeshViewer3d')
@patch('menpo3d.visualize.ColouredTriMeshViewer3d', menpo3d_visualize_mock)
@raises(Menpo3dMissingError)
def coloured_trimesh_viewer_test():
    ColouredTriMesh(fake_triangle, colours=fake_tcoords,
                    trilist=fake_trilist, copy=False).view()


@surrogate('menpo3d.visualize.PointCloudViewer3d')
@patch('menpo3d.visualize.PointCloudViewer3d', menpo3d_visualize_mock)
@raises(Menpo3dMissingError)
def pointcloud3d_viewer_test():
    PointCloud(fake_triangle, copy=False).view()


@surrogate('menpo3d.visualize.LandmarkViewer3d')
@patch('menpo3d.visualize.LandmarkViewer3d', menpo3d_visualize_mock)
@raises(Menpo3dMissingError)
def landmark3d_viewer_test():
    LandmarkGroup(PointCloud(fake_triangle),
                  OrderedDict([('all', np.ones(3, dtype=np.bool))]),
                  copy=False).view()
