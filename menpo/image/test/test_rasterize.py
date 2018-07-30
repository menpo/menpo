import numpy as np
from numpy.testing import assert_allclose

from menpo.image import Image
from menpo.image.rasterize import rasterize_landmarks_2d
from menpo.shape import PointCloud, PointUndirectedGraph

centre = PointCloud([[4.5, 4.5]])
line = PointUndirectedGraph(np.array([[2, 4.5], [8, 4.5]]),
                            adjacency_matrix=np.array([[0, 1], [1, 0]]))


def test_rasterize_matplotlib_basic():
    im = Image.init_blank([11, 11], fill=0, n_channels=1)
    im.landmarks['test'] = centre
    new_im = rasterize_landmarks_2d(im, group='test', render_lines=False,
                                    marker_style='.', marker_face_colour='r',
                                    marker_size=2, marker_edge_width=0,
                                    backend='matplotlib')
    assert new_im.n_channels == 3
    assert new_im.shape == (11, 11)
    assert_allclose(new_im.pixels[:, 5, 5], [255, 0, 0])


def test_rasterize_pillow_basic():
    im = Image.init_blank([11, 11], fill=0, n_channels=3)
    im.landmarks['test'] = centre
    new_im = rasterize_landmarks_2d(im, group='test', render_lines=False,
                                    marker_style='s', marker_face_colour='r',
                                    marker_size=1, marker_edge_width=0,
                                    backend='pillow')
    assert new_im.n_channels == 3
    assert new_im.shape == (11, 11)
    assert_allclose(new_im.pixels[0, 3:6, 3:6], 255)


def test_rasterize_pillow_basic_line():
    im = Image.init_blank([11, 11], fill=0, n_channels=3)
    im.landmarks['test'] = line
    new_im = rasterize_landmarks_2d(im, group='test', render_lines=True,
                                    line_width=1, line_colour='b',
                                    marker_style='s', marker_face_colour='r',
                                    marker_size=1, marker_edge_width=0,
                                    backend='pillow')
    assert new_im.n_channels == 3
    assert_allclose(new_im.pixels[0, 1:4, 3:6], 255)
    assert_allclose(new_im.pixels[0, 7:-1, 3:6], 255)
    assert_allclose(new_im.pixels[2, 4:7, 4], 255)
