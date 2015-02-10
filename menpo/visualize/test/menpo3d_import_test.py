from collections import OrderedDict
import numpy as np
from nose.tools import raises
from nose.plugins.attrib import attr
from menpo.image import Image
from menpo.landmark import LandmarkGroup
from menpo.shape import TriMesh, TexturedTriMesh, ColouredTriMesh, PointCloud


fake_triangle = np.array([[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]])
fake_trilist = np.array([[0, 1, 2]], dtype=np.uint32)
fake_texture = Image.blank([10, 10])
fake_tcoords = np.array([[0, 0], [0.5, 0.5], [1.0, 1.0]])


@raises(ImportError)
@attr('viewing')
def trimesh_viewer_test():
    TriMesh(fake_triangle, trilist=fake_trilist, copy=False).view()


@raises(ImportError)
@attr('viewing')
def textured_trimesh_viewer_test():
    TexturedTriMesh(fake_triangle, fake_tcoords, fake_texture,
                    trilist=fake_trilist, copy=False).view()


@raises(ImportError)
@attr('viewing')
def coloured_trimesh_viewer_test():
    ColouredTriMesh(fake_triangle, colours=fake_tcoords,
                    trilist=fake_trilist, copy=False).view()


@raises(ImportError)
@attr('viewing')
def pointcloud3d_viewer_test():
    PointCloud(fake_triangle, copy=False).view()


@raises(ImportError)
@attr('viewing')
def landmark3d_viewer_test():
    LandmarkGroup(PointCloud(fake_triangle),
                  OrderedDict([('all', np.ones(3, dtype=np.bool))]),
                  copy=False).view()
