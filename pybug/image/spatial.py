import numpy as np
from pybug.image.masked import MaskedNDImage
from pybug.visualize.base import ImageViewer, DepthImageHeightViewer


class AbstractSpatialImage(MaskedNDImage):
    r"""
    A 2D image that represents spatial data in some fashion in it's channel
    data. As a result, it contains a :class:'pybug.shape.mesh.base.TriMesh`
    """
    def __init__(self, image_data, mask=None, texture=None,
                 tcoords=None, trilist=None):
        super(AbstractSpatialImage, self).__init__(image_data, mask=mask)
        if self.n_dims != 2:
            raise ValueError("Trying to build an AbstractSpatialImage with {} "
                             "dimensions - has to be 2 dimensional"
                             .format(self.n_dims))
        self.mesh = self._create_mesh_from_shape(trilist, tcoords,
                                                 texture)

    def _generate_points(self):
        raise NotImplementedError()

    def _create_mesh_from_shape(self, trilist, tcoords, texture):
        from pybug.shape.mesh import TriMesh, TexturedTriMesh
        from scipy.spatial import Delaunay
        points = self._generate_points()
        if trilist is None:
            # Delaunay the 2D surface.
            trilist = Delaunay(points[..., :2]).simplices
        if texture is None:
            return TriMesh(points, trilist)
        else:
            if tcoords is None:
                tcoords = self.mask.true_indices.astype(np.float64)
                # scale to [0, 1]
                tcoords = tcoords / np.array(self.shape)
                # (s,t) = (y,x)
                tcoords = np.fliplr(tcoords)
                # move origin to top left
                tcoords[:, 1] = 1.0 - tcoords[:, 1]
            return TexturedTriMesh(points, trilist, tcoords, texture)

    def _view(self, figure_id=None, new_figure=False, mode='image',
              channel=None, masked=True, **kwargs):
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
        image_viewer : :class:`pybug.visualize.viewimage.ViewerImage`
            The viewer the image is being shown within
        """
        pixels = self.pixels.copy()
        pixels[np.isinf(pixels)] = np.nan
        pixels = np.abs(pixels)
        pixels /= np.nanmax(pixels)

        mask = None
        if masked:
            mask = self.mask.mask

        if mode is 'image':
            return ImageViewer(figure_id, new_figure,
                               self.n_dims, pixels,
                               channel=channel, mask=mask).render(**kwargs)
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
    :class:'pybug.shape.mesh.base.TriMesh`. This allows the shape image to be
    treated as an image, but expose an object that represents the shape
    as a mesh.

    Has to be a 2D image, and has to have exactly 3 channels for (X,Y,
    Z) spatial values.
    """

    def __init__(self, image_data, mask=None, texture=None, tcoords=None,
                 trilist=None):
        super(ShapeImage, self).__init__(image_data, mask, texture, tcoords,
                                         trilist)
        if self.n_channels != 3:
            raise ValueError("Trying to build a ShapeImage with {} channels "
                             "- has to have exactly 3 (for X, Y, "
                             "Z)".format(self.n_channels))

    def _generate_points(self):
        return self.masked_pixels


class DepthImage(AbstractSpatialImage):
    r"""
    An image the represents a depth image. Due to the fact a depth image has
    an implicit spatial meaning, a DepthImage also contains a
    :class:'pybug.shape.mesh.base.TriMesh`. This allows the depth image to be
    treated as an image, but expose an object that represents the depth
    as a mesh.

    Will have exactly 1 channel. The numpy array used to build the
    DepthImage is of shape (M, N) - it does not include the channel axis.
    """

    def __init__(self, image_data, mask=None, texture=None, tcoords=None,
                 trilist=None):
        super(DepthImage, self).__init__(image_data[..., None], mask,
                                         texture, tcoords, trilist)
        if self.n_channels != 1:
            raise ValueError("Trying to build a DepthImage with {} channels "
                             "- has to have exactly 1 (for Z values)"
                             .format(self.n_channels))

    @classmethod
    def _init_with_channel(cls, image_data_with_channel, mask):
        return cls(image_data_with_channel[..., 0], mask)

    def _generate_points(self):
        return np.hstack((self.mask.true_indices, self.masked_pixels))

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
        image_viewer : :class:`pybug.visualize.viewimage.ViewerImage`
            The viewer the image is being shown within
        """
        if mode is 'height':
            return DepthImageHeightViewer(
                figure_id, new_figure,
                self.pixels[:, :, 0], mask=mask).render(**kwargs)
        else:
            raise ValueError("Supported mode values are: 'image', 'mesh'"
                             " and 'height'")
