from pybug.io import auto_import
from pybug.shape import TriMesh
from pybug import data_path_to


def test_assimp_obj_import():
    meshes = auto_import(data_path_to('bunny.obj'))
    mesh = meshes[0]
    assert(isinstance(mesh, TriMesh))
