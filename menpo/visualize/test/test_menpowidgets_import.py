import numpy as np
from mock import patch, MagicMock
from pytest import raises
from scipy.sparse import csr_matrix

from menpo.image import Image
from menpo.visualize import MenpowidgetsMissingError
from menpo.shape import PointCloud, TriMesh, PointDirectedGraph, LabelledPointUndirectedGraph
from menpo.landmark import LandmarkManager
from menpo.testing import surrogate

pcloud2d = np.zeros([1, 2])
triangle_pcloud2d = np.array([[0, 0], [0, 1], [1, 0.]])
menpowidgets_mock = MagicMock()
menpowidgets_mock.side_effect = ImportError


@surrogate('menpowidgets.visualize_images')
@patch('menpowidgets.visualize_images', menpowidgets_mock)
def test_image_view_widget():
    with raises(MenpowidgetsMissingError):
        Image.init_blank((5, 5)).view_widget()


@surrogate('menpowidgets.visualize_pointclouds')
@patch('menpowidgets.visualize_pointclouds', menpowidgets_mock)
def test_pointcloud_view_widget():
    with raises(MenpowidgetsMissingError):
        PointCloud(pcloud2d).view_widget()


@surrogate('menpowidgets.visualize_landmarkgroups')
@patch('menpowidgets.visualize_landmarkgroups', menpowidgets_mock)
def test_labelledpointundirectedgraph_view_widget():
    adj_matrix = csr_matrix((1, 1))
    with raises(MenpowidgetsMissingError):
        LabelledPointUndirectedGraph.init_with_all_label(
            pcloud2d, adj_matrix).view_widget()


@surrogate('menpowidgets.visualize_landmarks')
@patch('menpowidgets.visualize_landmarks', menpowidgets_mock)
def test_landmarkmanager_view_widget():
    l = LandmarkManager()
    l['test'] = PointCloud(pcloud2d)
    with raises(MenpowidgetsMissingError):
        l.view_widget()


@surrogate('menpowidgets.visualize_patches')
@patch('menpowidgets.visualize_patches', menpowidgets_mock)
def test_landmarkmanager_view_widget():
    l = LandmarkManager()
    l['test'] = PointCloud(pcloud2d)
    with raises(MenpowidgetsMissingError):
        l.view_widget()


@surrogate('menpowidgets.visualize_pointclouds')
@patch('menpowidgets.visualize_pointclouds', menpowidgets_mock)
def test_trimesh_view_widget():
    with raises(MenpowidgetsMissingError):
        TriMesh(triangle_pcloud2d).view_widget()


@surrogate('menpowidgets.visualize_pointclouds')
@patch('menpowidgets.visualize_pointclouds', menpowidgets_mock)
def test_pointgraph_view_widget():
    with raises(MenpowidgetsMissingError):
        PointDirectedGraph.init_from_edges(triangle_pcloud2d,
                                           [[0, 1], [1, 2]]).view_widget()
