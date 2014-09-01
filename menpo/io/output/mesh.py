from menpo.shape.mesh import TexturedTriMesh


def OBJExporter(file_handle, mesh):
    r"""
    Given a file handle to write in to (which should act like a Python `file`
    object), write out the mesh data. No value is returned..

    Note that this does not save out textures of textured images, and so should
    not be used in isolation.

    Parameters
    ----------

    file_handle : `str`
        The full path where the obj will be saved out.

    mesh : instance of :map:`TriMesh`
        Any subclass of :map:`TriMesh`. If :map:`TexturedTriMesh` texture
        coordinates will be saved out. Note that :map:`ColouredTriMesh`
        will only have shape data saved out, as .OBJ doesn't robustly support
        per-vertex colour information.

    """
    for v in mesh.points:
        file_handle.write('v {} {} {}\n'.format(*v))
    file_handle.write('\n')
    if isinstance(mesh, TexturedTriMesh):
        for tc in mesh.tcoords.points:
            file_handle.write('vt {} {}\n'.format(*tc))
        file_handle.write('\n')
        # triangulation of points and tcoords is identical
        for t in (mesh.trilist + 1):
            file_handle.write('f {0}/{0} {1}/{1} {2}/{2}\n'.format(*t))
    else:
        # no tcoords - so triangulation is straight forward
        for t in (mesh.trilist + 1):
            file_handle.write('f {} {} {}\n'.format(*t))
