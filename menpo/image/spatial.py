import abc
from copy import deepcopy
import numpy as np

from menpo.image.masked import MaskedImage
from menpo.visualize.base import ImageViewer, DepthImageHeightViewer


class AbstractSpatialImage(MaskedImage):
    r"""
    A 2D image that represents spatial data in some fashion in it's channel
    data. As a result, it contains a :class:`menpo.shape.mesh.base.TriMesh`,
    or, if a texture is provided, a
    :class:`menpo.shape.mesh.base.TexturedTriMesh`.

    Parameters
    -----------
    image_data: (M, N, ..., C) ndarray
        Array representing the spatial image pixels, with the last axis being
        channels.
    mask: (M, N, ..., L) boolean ndarray or :class:`BooleanImage`, optional
        A suitable mask for the spatial data

        Default: All true mask
    trilist: (n_tris, 3), ndarray, optional
        Triangle list for the trimesh. If None, the trilist is generation
        from all True points using Delaunay triangulation.

        Default: None
    tcoords: (n_true, 2), ndarray, optional
        Texture coordinates relating each True value of the mask to the
        texture space

        Default: If texture is provided, tcoords are generated on the
        assumption that the texture and the spatial data are in
        correspondence. If no texture, None.
    texture: :class:`Abstract2DImage` instance, optional
        A texture to be associated with the spatial data

        Default: None (no texture)
    """

    def __init__(self, image_data, mask=None, trilist=None,
                 tcoords=None, texture=None):
        super(AbstractSpatialImage, self).__init__(image_data, mask=mask)
        if self.n_dims != 2:
            raise ValueError("Trying to build an AbstractSpatialImage with {} "
                             "dimensions - has to be 2 dimensional"
            .format(self.n_dims))
        self._trilist = None
        self._tcoords = None
        self._texture = None
        self._mesh = None

        if trilist is not None:
            self._trilist = np.array(trilist, copy=True, order='C')
        if tcoords is not None:
            self._tcoords = deepcopy(tcoords)
        if texture is not None:
            self._texture = deepcopy(texture)

    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = self._create_mesh_from_shape()
        return self._mesh

    def rebuild_mesh(self):
        self._mesh = None

    def from_vector_inplace(self, vector):
        r"""
        Takes a flattened vector and updates this SpatialImage.

        Done by reshaping the vector to the correct pixels and channels.
        Note that the only region of the image that will be filled is the
        masked region.

        Calling this method will cause a cache invalidation on this spatial
        images' mesh.

        Parameters
        ----------
        vector : (``n_pixels``,)
            A flattened vector of all pixels and channels of an image.
        """
        MaskedImage.from_vector_inplace(self, vector)
        self.rebuild_mesh()

    @abc.abstractmethod
    def _generate_points(self):
        pass

    def _create_mesh_from_shape(self):
        r"""
        Creates a mesh from the spatial information.

        Parameters
        ----------
        trilist: (n_tris, 3), ndarray, optional
            Triangle list for the trimesh. If None, the trilist is generation
            from all True points using Delaunay triangulation.

            Default: None

        tcoords: (n_true, 2), ndarray, optional
            Texture coordinates relating each True value of the mask to the
            texture space

            Default: If texture is provided, tcoords are generated on the
            assumption that the texture and the spatial data are in
            correspondence. If no texture, None.

        texture: :class:`Abstract2DImage` instance, optional
            A texture to be associated with the spatial data

            Default: None (no texture)
        """
        from menpo.shape.mesh import TriMesh, TexturedTriMesh
        from scipy.spatial import Delaunay

        points = self._generate_points()

        trilist = self._trilist
        if trilist is None:
            # Delaunay the 2D surface.
            trilist = Delaunay(points[..., :2]).simplices

        if self._texture is None:
            return TriMesh(points, trilist)
        else:
            tcoords = self._tcoords
            if tcoords is None:
                tcoords = self.mask.true_indices.astype(np.float64)
                # scale to [0, 1]
                tcoords = tcoords / np.array(self.shape)
                # (s,t) = (y,x)
                tcoords = np.fliplr(tcoords)
                # move origin to top left
                tcoords[:, 1] = 1.0 - tcoords[:, 1]
            return TexturedTriMesh(points, trilist, tcoords, self._texture)

    def _view(self, figure_id=None, new_figure=False, mode='image',
              channels=None, masked=True, **kwargs):
        r"""
        View the image using the default image viewer. Before the image is
        rendered the depth values are normalised between 0 and 1. The range
        is then shifted so that the viewable range provides a reasonable
        contrast.

        Parameters
        ----------
        mode : {'image', 'mesh', 'height'}
            The manner in which to render the depth map.

            ========== =========================
            key        description
            ========== =========================
            image      View as a greyscale image
            mesh       View as a triangulated mesh
            height     View as a height map
            ========== =========================

            Default: 'image'

        Returns
        -------
        image_viewer : :class:`menpo.visualize.viewimage.ViewerImage`
            The viewer the image is being shown within
        """
        import scipy.stats

        pixels = self.pixels.copy()
        pixels[np.isinf(pixels)] = np.nan
        pixels = np.abs(pixels)
        pixels -= scipy.stats.nanmean(pixels)
        pixels /= np.nanmax(pixels)

        mask = None
        if masked:
            mask = self.mask.mask

        if mode is 'image':
            return ImageViewer(figure_id, new_figure,
                               self.n_dims, pixels,
                               channels=channels, mask=mask).render(**kwargs)
        if mode is 'mesh':
            return self.mesh._view(figure_id=figure_id, new_figure=new_figure,
                                   **kwargs)
        else:
            return self._view_extra(figure_id, new_figure, mode, mask,
                                    **kwargs)

    def _view_extra(self, figure_id, new_figure, mode, mask, **kwargs):
        if mode is 'height':
            return DepthImageHeightViewer(
                figure_id, new_figure,
                self.pixels[:, :, 2], mask=mask).render(**kwargs)
        else:
            raise ValueError("Supported mode values are: 'image', 'mesh'"
                             " and 'height'")


class ShapeImage(AbstractSpatialImage):
    r"""
    An image the represents a shape image. Due to the fact a shape image has
    an implicit spatial meaning, it also contains a
    :class:'menpo.shape.mesh.base.TriMesh`. This allows the shape image to be
    treated as an image, but expose an object that represents the shape
    as a mesh.

    Has to be a 2D image, and has to have exactly 3 channels for (X,Y,
    Z) spatial values.

    Parameters
    -----------
    image_data: (M, N, 3) ndarray
        Array representing the spatial image pixels, with the last axis being
        the spatial data per pixel.
    mask: (M, N) boolean ndarray or :class:`BooleanImage`, optional
        A suitable mask for the spatial data

        Default: All true mask
    trilist: (n_tris, 3), ndarray, optional
        Triangle list for the trimesh. If None, the trilist is generation
        from all True points using Delaunay triangulation.

        Default: None
    tcoords: (n_true, 2), ndarray, optional
        Texture coordinates relating each True value of the mask to the
        texture space

        Default: If texture is provided, tcoords are generated on the
        assumption that the texture and the spatial data are in
        correspondence. If no texture, None.
    texture: :class:`Abstract2DImage` instance, optional
        A texture to be associated with the spatial data

        Default: None (no texture)
    """

    def __init__(self, image_data, mask=None, trilist=None,
                 tcoords=None, texture=None):
        super(ShapeImage, self).__init__(image_data, mask, trilist, tcoords,
                                         texture)
        if self.n_channels != 3:
            raise ValueError("Trying to build a ShapeImage with {} channels "
                             "- has to have exactly 3 (for X, Y, "
                             "Z)".format(self.n_channels))

    @classmethod
    def blank(cls, shape, fill=0, dtype=np.float, mask=None, **kwargs):
        r"""
        Returns a blank ShapeImage

        Parameters
        ----------
        shape : tuple or list
            The shape of the image

        fill : int, optional
            The value to fill all pixels with

            Default: 0
        dtype: numpy datatype, optional
            The datatype of the image.

            Default: np.float
        mask: (M, N) boolean ndarray or :class:`BooleanImage`
            An optional mask that can be applied to the image. Has to have a
             shape equal to that of the image.

             Default: all True :class:`BooleanImage`

        Returns
        -------
        blank_image : :class:`ShapeImage`
            A new shape image of the requested size.
        """
        n_channels = kwargs.get('n_channels', 3)
        if n_channels != 3:
            raise ValueError('The number of channels of a ShapeImage must be '
                             'set to 3')
        blank_image = super(AbstractSpatialImage, cls).blank(
            shape, n_channels=n_channels, fill=fill, dtype=dtype, mask=mask)
        blank_image.pixels[:, :, :2] = np.reshape(
            blank_image.mask.all_indices, (shape[0], shape[1], 2))
        return blank_image

    def as_depth_image(self):
        """
        Convert the shape image to a depth image by stripping off the z-values.
        Copies all associated data including landmarks and texture information.

        Returns
        -------
        depth_image : :class:`DepthImage`
            The depth image created by taking the z-values of this shape image.
        """
        depth_image = DepthImage(deepcopy(self.pixels[:, :, 2]),
                                 deepcopy(self.mask),
                                 trilist=deepcopy(self._trilist),
                                 tcoords=deepcopy(self._tcoords),
                                 texture=deepcopy(self._texture))
        depth_image.landmarks = self.landmarks
        return depth_image

    def _generate_points(self):
        return self.masked_pixels


class DepthImage(AbstractSpatialImage):
    r"""
    An image the represents a depth image. Due to the fact a depth image has
    an implicit spatial meaning, a DepthImage also contains a
    :class:'menpo.shape.mesh.base.TriMesh`. This allows the depth image to be
    treated as an image, but expose an object that represents the depth
    as a mesh.

    Will have exactly 1 channel. The numpy array used to build the
    DepthImage is of shape (M, N) - it does not include the channel axis.

    In images, the origin is in the top-left and the first axis represents
    the y-direction (down the image). For ``DepthImage``, when accessing the
    mesh property, the origin is interpreted as the bottom-left and the first
    axis represents the x-direction (across the image). This is to maintain
    consistency with a right-handed coordinate scheme for meshes.

    Parameters
    -----------
    image_data: (M, N) ndarray
        Array representing the spatial image pixels. There is no channel
        axis - each pixel position stores a single depth value.
    mask: (M, N) boolean ndarray or :class:`BooleanImage`, optional
        A suitable mask for the spatial data

        Default: All true mask
    trilist: (n_tris, 3), ndarray, optional
        Triangle list for the trimesh. If None, the trilist is generation
        from all True points using Delaunay triangulation.

        Default: None
    tcoords: (n_true, 2), ndarray, optional
        Texture coordinates relating each True value of the mask to the
        texture space

        Default: If texture is provided, tcoords are generated on the
        assumption that the texture and the spatial data are in
        correspondence. If no texture, None.
    texture: :class:`Abstract2DImage` instance, optional
        A texture to be associated with the spatial data

        Default: None (no texture)
    """

    def __init__(self, image_data, mask=None, trilist=None,
                 tcoords=None, texture=None):
        super(DepthImage, self).__init__(image_data[..., None], mask, trilist,
                                         tcoords, texture)
        if self.n_channels != 1:
            raise ValueError("Trying to build a DepthImage with {} channels "
                             "- has to have exactly 1 (for Z values)"
            .format(self.n_channels))

    @classmethod
    def blank(cls, shape, fill=0, dtype=np.float, mask=None, **kwargs):
        r"""
        Returns a blank DepthImage

        Parameters
        ----------
        shape : tuple or list
            The shape of the image

        fill : int, optional
            The value to fill all pixels with

            Default: 0
        dtype: numpy datatype, optional
            The datatype of the image.

            Default: np.float
        mask: (M, N) boolean ndarray or :class:`BooleanImage`
            An optional mask that can be applied to the image. Has to have a
             shape equal to that of the image.

             Default: all True :class:`BooleanImage`

        Returns
        -------
        blank_image : :class:`DepthImage`
            A new depth image of the requested size.
        """
        n_channels = kwargs.get('n_channels', 1)
        if n_channels != 1:
            raise ValueError('The number of channels of a DepthImage must be '
                             'set to 1')
        return super(DepthImage, cls).blank(
            shape, n_channels=n_channels, fill=fill, dtype=dtype, mask=mask)

    @classmethod
    def _init_with_channel(cls, image_data_with_channel, mask):
        if image_data_with_channel.ndim != 3 or \
                        image_data_with_channel.shape[-1] != 1:
            raise ValueError("DepthImage must be constructed with 3 "
                             "dimensions and 1 channel.")
        return cls(image_data_with_channel[..., 0], mask)

    def _generate_points(self):
        # We need to make the axes consistent with a right-handed coordinate
        # scheme, so we flip the x and y axis (make the first axis the x)
        points = np.hstack((self.mask.true_indices[:, [1, 0]],
                            self.masked_pixels))
        # Then we need to ensure that the origin is in the bottom left,
        # so we flip the y-axis
        points[:, 1] = self.mask.height - points[:, 1]
        return points

    def _view_extra(self, figure_id, new_figure, mode, mask, **kwargs):
        r"""
        View the image using the default image viewer. Before the image is
        rendered the depth values are normalised between 0 and 1. The range
        is then shifted so that the viewable range provides a reasonable
        contrast.

        Parameters
        ----------
        mode : {'image', 'mesh', 'height'}
            The manner in which to render the depth map.

            ========== =========================
            key        description
            ========== =========================
            image      View as a greyscale image
            mesh       View as a triangulated mesh
            height     View as a height map
            ========== =========================

            Default: 'image'

        Returns
        -------
        image_viewer : :class:`menpo.visualize.viewimage.ViewerImage`
            The viewer the image is being shown within
        """
        if mode is 'height':
            return DepthImageHeightViewer(
                figure_id, new_figure,
                self.pixels[:, :, 0], mask=mask).render(**kwargs)
        else:
            raise ValueError("Supported mode values are: 'image', 'mesh'"
                             " and 'height'")
