import tempfile
import os.path as p
from menpo.io.mesh.base import export_obj
from numpy.testing import assert_allclose, assert_equal

import menpo


def test_export_obj_textured():
    i = menpo.io.import_builtin_asset('james.obj')
    o_path = p.join(tempfile.gettempdir(), 'test.obj')
    img_path = p.join(tempfile.gettempdir(), 'test.jpg')
    export_obj(o_path, i)
    i.texture.as_PILImage().save(img_path)
    o = menpo.io.import_mesh(o_path)
    assert_allclose(i.points, o.points)
    assert_equal(i.trilist, o.trilist)
    assert_allclose(i.tcoords.points, o.tcoords.points)


def test_export_obj_nontextured():
    i = menpo.io.import_builtin_asset('bunny.obj')
    o_path = p.join(tempfile.gettempdir(), 'test.obj')
    export_obj(o_path, i)
    o = menpo.io.import_mesh(o_path)
    assert_allclose(i.points, o.points)
    assert_equal(i.trilist, o.trilist)
