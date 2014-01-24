import pybug.io as pio
from pybug.shape import TriMesh


def test_import_asset_bunny():
    mesh = pio.import_builtin_asset('bunny.obj')
    assert(isinstance(mesh, TriMesh))
