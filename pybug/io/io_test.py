#from pybug.io.mesh import AssimpImporter
from pybug.io import auto_import
from pybug.shape import TriMesh


def test_assimp_obj_import():
    meshes = auto_import('./data/bunny.obj')
    mesh = meshes[0]
    assert(isinstance(mesh, TriMesh))
