import numpy as np
from pybug.rasterize.copengl import COpenGLRasterizer
from pybug.image import MaskedImage


# noinspection PyProtectedMember
class GLRasterizer(object):

    def __init__(self, width, height):
        self._opengl = COpenGLRasterizer(width, height)

    @classmethod
    def from_mayavi_viewer(cls, viewer=None):
        r"""Build a rasterizer that perfectly matches a Mayavi viewer. If no
        viewer is provided, the current viewer is used.
        """
        vm, p, view_dims = extract_state_from_mayavi_viewer(viewer=viewer)
        rasterizer = GLRasterizer(*view_dims)
        rasterizer.set_view_matrix(vm)
        rasterizer.set_model_matrix(np.eye(4))
        rasterizer.set_projection_matrix(p)
        return rasterizer

    @property
    def model_matrix(self):
        return self._opengl.get_model_matrix()

    @property
    def view_matrix(self):
        return self._opengl.get_view_matrix()

    @property
    def projection_matrix(self):
        return self._opengl.get_projection_matrix()

    def set_model_matrix(self, value):
        value = _verify_opengl_homogeneous_matrix(value)
        self._opengl.set_model_matrix(value)

    def set_view_matrix(self, value):
        value = _verify_opengl_homogeneous_matrix(value)
        self._opengl.set_view_matrix(value)

    def set_projection_matrix(self, value):
        value = _verify_opengl_homogeneous_matrix(value)
        self._opengl.set_projection_matrix(value)

    def rasterize(self, rasterizable):
        if rasterizable._rasterize_type_texture:
            # request the
            r = rasterizable._rasterize_generate_textured_mesh()
        elif rasterizable._rasterize_type_color():
            raise ValueError("Color Mesh rasterizations are not supported "
                             "yet")

    def set_matrices_from_mayavi_viewer(self, viewer=None):
        vm, p, _ = extract_state_from_mayavi_viewer(viewer=viewer)
        self.set_view_matrix(vm)
        self.set_model_matrix(np.eye(4))
        self.set_projection_matrix(p)

    def rasterize_with_f3v_interpolant(self, rasterizable,
                                       f3v_interpolant):
        if rasterizable._rasterize_type_texture:
            # request the textured info for rasterizing
            r = rasterizable._rasterize_generate_textured_mesh()
            return self._rasterize_texture_with_interp(r, f3v_interpolant)
        elif rasterizable._rasterize_type_color():
            raise ValueError("Color Mesh rasterizations are not supported "
                             "yet")

    def rasterize_with_shape_image(self, rasterizable):
        return self.rasterize_with_f3v_interpolant(rasterizable, None)

    def rasterize(self, rasterizable):
        return self.rasterize_with_shape_image(rasterizable)[0]

    def _rasterize_texture_with_interp(self, r, per_vertex_f3v=None):
        r"""Rasterizes a textured mesh along with it's interpolant data
        through OpenGL.

        Parameters
        ----------
        r : object
            Any object with fields named 'points', 'trilist', 'texture' and
            'tcoords' specifying the data that will be used to render. Such
            objects are handed out by the
            _rasterize_generate_textured_mesh method on Rasterizable
            subclasses
        per_vertex_f3v: ndarray, shape (n_points, 3)
            A matrix specifying arbitrary 3 floating point numbers per
            vertex. This data will be linearly interpolated across triangles
            and returned in the f3v image. If none, the shape information is used

        Returns
        -------
        image : MaskedImage
            The rasterized image returned from OpenGL. Note that the
            behavior of the rasterization is governed by the projection,
            rotation and view matrices that may be set on this class,
            as well as the width and height of the rasterization, which is
            determined on the creation of this class. The mask is True if a
            triangle is visable at that pixel in the output, and False if not.

        f3v_image : MaskedImage
            The rasterized image returned from OpenGL. Note that the
            behavior of the rasterization is governed by the projection,
            rotation and view matrices that may be set on this class,
            as well as the width and height of the rasterization, which is
            determined on the creation of this class.

        """
        points = np.require(r.points, dtype=np.float32, requirements='c')
        trilist = np.require(r.trilist, dtype=np.uint32, requirements='c')
        texture = np.require(r.texture, dtype=np.float32, requirements='c')
        tcoords = np.require(r.tcoords, dtype=np.float32, requirements='c')
        if per_vertex_f3v is None:
            per_vertex_f3v = points
        interp = np.require(per_vertex_f3v, dtype=np.float32, requirements='c')
        rgb_fb, f3v_fb = self._opengl.render_offscreen_rgb(
            points, interp, trilist, tcoords, texture)
        mask_array = rgb_fb[..., 3].astype(np.bool)
        return (MaskedImage(rgb_fb[..., :3].copy(), mask=mask_array),
                MaskedImage(f3v_fb.copy(), mask=mask_array))


def _verify_opengl_homogeneous_matrix(matrix):
    if matrix.shape != (4, 4):
        raise ValueError("OpenGL matrices must have shape (4,4)")
    return np.require(matrix, dtype=np.float32, requirements='C')


def extract_state_from_mayavi_viewer(viewer=None):
    r"""Retries a modelview and projection matrix along with the viewing dims
    for a mayavi viewer. If no viewer is provided, works off the current
    scene.
    """
    if viewer is None:
        import mayavi.mlab as mlab
        scene = mlab.gcf().scene
    else:
        scene = viewer.figure.None
    vm = scene.camera.view_transform_matrix.to_array().astype(np.float32)
    scene_size = tuple(scene.get_size())
    aspect_ratio = float(scene_size[0]) / float(scene_size[1])
    p = scene.camera.get_perspective_transform_matrix(
        aspect_ratio, -1, 1).to_array().astype(np.float32)
    return vm, p, scene_size
