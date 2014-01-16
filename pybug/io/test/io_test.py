from pybug.io import import_auto
from pybug.shape import TriMesh
from pybug import data_path_to


def test_assimp_obj_import():
    meshes = import_auto(data_path_to('bunny.obj'))
    mesh = meshes[0]
    assert(isinstance(mesh, TriMesh))
