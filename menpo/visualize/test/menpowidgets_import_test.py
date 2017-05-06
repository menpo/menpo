import numpy as np
from mock import patch, MagicMock
from nose.tools import raises
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
@raises(MenpowidgetsMissingError)
def image_view_widget_test():
    Image.init_blank((5, 5)).view_widget()


@surrogate('menpowidgets.visualize_pointclouds')
@patch('menpowidgets.visualize_pointclouds', menpowidgets_mock)
@raises(MenpowidgetsMissingError)
def pointcloud_view_widget_test():
    PointCloud(pcloud2d).view_widget()


@surrogate('menpowidgets.visualize_landmarkgroups')
@patch('menpowidgets.visualize_landmarkgroups', menpowidgets_mock)
@raises(MenpowidgetsMissingError)
def labelledpointundirectedgraph_view_widget_test():
    adj_matrix = csr_matrix((1, 1))
    LabelledPointUndirectedGraph.init_with_all_label(
        pcloud2d, adj_matrix).view_widget()


@surrogate('menpowidgets.visualize_landmarks')
@patch('menpowidgets.visualize_landmarks', menpowidgets_mock)
@raises(MenpowidgetsMissingError)
def landmarkmanager_view_widget_test():
    l = LandmarkManager()
    l['test'] = PointCloud(pcloud2d)
    l.view_widget()


@surrogate('menpowidgets.visualize_patches')
@patch('menpowidgets.visualize_patches', menpowidgets_mock)
@raises(MenpowidgetsMissingError)
def landmarkmanager_view_widget_test():
    l = LandmarkManager()
    l['test'] = PointCloud(pcloud2d)
    l.view_widget()
    assert 1


@surrogate('menpowidgets.visualize_pointclouds')
@patch('menpowidgets.visualize_pointclouds', menpowidgets_mock)
@raises(MenpowidgetsMissingError)
def trimesh_view_widget_test():
    TriMesh(triangle_pcloud2d).view_widget()
    assert 1


@surrogate('menpowidgets.visualize_pointclouds')
@patch('menpowidgets.visualize_pointclouds', menpowidgets_mock)
@raises(MenpowidgetsMissingError)
def pointgraph_view_widget_test():
    PointDirectedGraph.init_from_edges(triangle_pcloud2d,
                                       [[0, 1], [1, 2]]).view_widget()
