# This has to go above the default importers to prevent cyclical importing
import abc

import numpy as np
from scipy.misc import imrotate

from pybug.exception import DimensionalityError


class Renderer(object):
    r"""
    Abstract class for rendering visualizations. Framework specific
    implementations of these classes are made in order to separate
    implementation cleanly from the rest of the code.

    It is assumed that the renderers follow some form of stateful pattern for
    rendering to Figures. Therefore, the major interface for rendering involves
    providing a ``figure_id`` or a boolean about whether a new figure should
    be used. If neither are provided then the default state of the rendering
    engine is assumed to maintained.

    Providing a ``figure_id`` and ``new_figure == True`` is not a valid state.

    Parameters
    ----------
    figure_id : object
        A figure id. Could be any valid object that identifies
        a figure in a given framework (string, int, etc)
    new_figure : bool
        Whether the rendering engine should create a new figure.

    Raises
    ------
    ValueError
        It is not valid to provide a figure id AND request a new figure to
        be rendered on.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, figure_id, new_figure):
        if figure_id is not None and new_figure:
            raise ValueError("Conflicting arguments. figure_id cannot be "
                             "specified if the new_figure flag is True")

        self.figure_id = figure_id
        self.new_figure = new_figure
        self.figure = self.get_figure()

    def render(self, **kwargs):
        r"""
        Render the object on the figure given at instantiation.

        Parameters
        ----------
        kwargs : dict
            Passed through to specific rendering engine.

        Returns
        -------
        viewer : :class:`Renderer`
            Pointer to ``self``.
        """
        return self._render(**kwargs)

    @abc.abstractmethod
    def _render(self, **kwargs):
        r"""
        Abstract method to be overridden the renderer. This will implement the
        actual rendering code for a given object class.

        Parameters
        ----------
        kwargs : dict
            Options to be set when rendering.

        Returns
        -------
        viewer : :class:`Renderer`
            Pointer to ``self``.
        """
        pass

    @abc.abstractmethod
    def get_figure(self):
        r"""
        Abstract method for getting the correct figure to render on. Should
        also set the correct ``figure_id`` for the figure.

        Returns
        -------
        figure : object
            The figure object that the renderer will render on.
        """
        pass


class Viewable(object):
    r"""
    Abstract interface for objects that can visualize themselves.
    """

    __metaclass__ = abc.ABCMeta

    def view_on(self, figure_id, **kwargs):
        r"""
        View the object on a a specific figure specified by the given id.

        Parameters
        ----------
        figure_id : object
            A unique identifier for a figure.
        kwargs : dict
            Passed through to specific rendering engine.

        Returns
        -------
        viewer : :class:`Renderer`
            The renderer instantiated.
        """
        return self._view(figure_id=figure_id, **kwargs)

    def view_new(self, **kwargs):
        r"""
        View the object on a new figure.

        Parameters
        ----------
        kwargs : dict
            Passed through to specific rendering engine.

        Returns
        -------
        viewer : :class:`Renderer`
            The renderer instantiated.
        """
        return self._view(new_figure=True, **kwargs)

    def view(self, **kwargs):
        r"""
        View the object using the default rendering engine figure handling.
        For example, the default behaviour for Matplotlib is that all draw
        commands are applied to the same ``figure`` object.

        Parameters
        ----------
        kwargs : dict
            Passed through to specific rendering engine.

        Returns
        -------
        viewer : :class:`Renderer`
            The renderer instantiated.
        """
        return self._view(**kwargs)

    @abc.abstractmethod
    def _view(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        Abstract method to be overridden by viewable objects. This will
        instantiate a specific visualisation implementation

        Parameters
        ----------
        figure_id : object, optional
            A unique identifier for a figure.

            Default: ``None``
        new_figure : bool, optional
            Whether the rendering engine should create a new figure.

            Default: ``False``
        kwargs : dict
            Passed through to specific rendering engine.

        Returns
        -------
        viewer : :class:`Renderer`
            The renderer instantiated.
        """
        pass


from pybug.visualize.viewmayavi import MayaviPointCloudViewer3d, \
    MayaviTriMeshViewer3d, MayaviTexturedTriMeshViewer3d, \
    MayaviLandmarkViewer3d, MayaviVectorViewer3d, MayaviSurfaceViewer3d
from pybug.visualize.viewmatplotlib import MatplotlibImageViewer2d, \
    MatplotlibImageSubplotsViewer2d, MatplotlibPointCloudViewer2d, \
    MatplotlibLandmarkViewer2d, MatplotlibLandmarkViewer2dImage, \
    MatplotlibTriMeshViewer2d, MatplotlibAlignmentViewer2d

# Default importer types
PointCloudViewer2d = MatplotlibPointCloudViewer2d
PointCloudViewer3d = MayaviPointCloudViewer3d
TriMeshViewer2d = MatplotlibTriMeshViewer2d
TriMeshViewer3d = MayaviTriMeshViewer3d
TexturedTriMeshViewer3d = MayaviTexturedTriMeshViewer3d
LandmarkViewer3d = MayaviLandmarkViewer3d
LandmarkViewer2d = MatplotlibLandmarkViewer2d
LandmarkViewer2dImage = MatplotlibLandmarkViewer2dImage
ImageViewer2d = MatplotlibImageViewer2d
ImageSubplotsViewer2d = MatplotlibImageSubplotsViewer2d
VectorViewer3d = MayaviVectorViewer3d
DepthImageHeightViewer = MayaviSurfaceViewer3d
AlignmentViewer2d = MatplotlibAlignmentViewer2d


class LandmarkViewer(object):
    r"""
    Base Landmark viewer that abstracts away dimensionality

    Parameters
    ----------
    figure_id : object
        A figure id. Could be any valid object that identifies
        a figure in a given framework (string, int, etc)
    new_figure : bool
        Whether the rendering engine should create a new figure.
    group_label : string
        The main label of the landmark set.
    pointcloud : :class:`pybug.shape.pointcloud.PointCloud`
        The pointclouds representing the landmarks.
    labels_to_masks : dict(string, ndarray)
        A dictionary of labels to masks into the pointcloud that represent
        which points belong to the given label.
    target : :class:`pybug.landmarks.base.Landmarkable`
        The parent shape that we are drawing the landmarks for.
    """

    def __init__(self, figure_id, new_figure,
                 group_label, pointcloud, labels_to_masks, target):
        self.pointcloud = pointcloud
        self.group_label = group_label
        self.labels_to_masks = labels_to_masks
        self.target = target
        self.figure_id = figure_id
        self.new_figure = new_figure

    def render(self, **kwargs):
        r"""
        Select the correct type of landmark viewer for the given parent shape.

        Parameters
        ----------
        kwargs : dict
            Passed through to landmark viewer.

        Returns
        -------
        viewer : :class:`Renderer`
            The rendering object.

        Raises
        ------
        DimensionalityError
            Only 2D and 3D viewers are supported.
        """
        if self.pointcloud.n_dims == 2:
            from pybug.image.base import AbstractNDImage

            if isinstance(self.target, AbstractNDImage):
                return LandmarkViewer2dImage(
                    self.figure_id, self.new_figure,
                    self.group_label, self.pointcloud,
                    self.labels_to_masks).render(**kwargs)
            else:
                return LandmarkViewer2d(self.figure_id, self.new_figure,
                                        self.group_label, self.pointcloud,
                                        self.labels_to_masks).render(**kwargs)
        elif self.pointcloud.n_dims == 3:
            return LandmarkViewer3d(self.figure_id, self.new_figure,
                                    self.group_label, self.pointcloud,
                                    self.labels_to_masks).render(**kwargs)
        else:
            raise DimensionalityError("Only 2D and 3D landmarks are "
                                      "currently supported")


class PointCloudViewer(object):
    r"""
    Base PointCloud viewer that abstracts away dimensionality.

    Parameters
    ----------
    figure_id : object
        A figure id. Could be any valid object that identifies
        a figure in a given framework (string, int, etc)
    new_figure : bool
        Whether the rendering engine should create a new figure.
    points : (N, D) ndarray
        The points to render.
    """

    def __init__(self, figure_id, new_figure, points):
        self.figure_id = figure_id
        self.new_figure = new_figure
        self.points = points

    def render(self, **kwargs):
        r"""
        Select the correct type of pointcloud viewer for the given
        pointcloud dimensionality.

        Parameters
        ----------
        kwargs : dict
            Passed through to pointcloud viewer.

        Returns
        -------
        viewer : :class:`Renderer`
            The rendering object.

        Raises
        ------
        DimensionalityError
            Only 2D and 3D viewers are supported.
        """
        if self.points.shape[1] == 2:
            return PointCloudViewer2d(self.figure_id, self.new_figure,
                                      self.points).render(**kwargs)
        elif self.points.shape[1] == 3:
            return PointCloudViewer3d(self.figure_id, self.new_figure,
                                      self.points).render(**kwargs)
        else:
            raise DimensionalityError("Only 2D and 3D pointclouds are "
                                      "currently supported")


class ImageViewer(object):
    r"""
    Base Image viewer that abstracts away dimensionality. It can visualize
    multiple channels of an image in subplots.

    Parameters
    ----------
    figure_id : object
        A figure id. Could be any valid object that identifies
        a figure in a given framework (string, int, etc)
    new_figure : bool
        Whether the rendering engine should create a new figure.
    dimensions : {2, 3} int
        The number of dimensions in the image
    pixels : (N, D) ndarray
        The pixels to render.
    channels: int or list or 'all' or None
        A specific selection of channels to render. The user can choose either
        a single or multiple channels. If all, render all channels in subplot
        mode. If None and channels are less than 36, render them all. If None
        and channels are more than 36, render the first 36.

        Default: None
    mask: (N, D) ndarray
        A boolean mask to be applied to the image. All points outside the
        mask are set to 0.
    """

    def __init__(self, figure_id, new_figure, dimensions, pixels,
                 channels=None, mask=None):
        pixels = pixels.copy()
        self.figure_id = figure_id
        self.new_figure = new_figure
        self.dimensions = dimensions
        self.channels, pixels = self._parse_channels(channels, pixels, 36)
        self.pixels = self._masked_pixels(pixels, mask)

    def _parse_channels(self, channels, pixels, upper_limit):
        r"""
        Parse channels parameter. If channels is int or list, keep it as is. If
        channels is all, return a list of all the image's channels. If channels
        is None, return the minimum between an upper_limit and the image's
        number of channels.

        Parameters
        ----------
        channels: int or list or 'all' or None
            A specific selection of channels to render.
        pixels : (N, D) ndarray
            The image's pixels to render.
        upper_limit: int
            The upper limit of subplots for the channels=None case.
        """
        if channels is None:
            # Default number of channels to visualize
            channels = range(min(pixels.shape[2], upper_limit))
        elif channels is 'all':
            # Visualize all channels
            channels = range(pixels.shape[2])
        pixels = pixels[..., channels]
        return channels, pixels

    def _masked_pixels(self, pixels, mask):
        r"""
        Return the masked pixels using a given boolean mask. In order to make
        sure that the non-masked pixels are visualized in black, their value
        is set to the minimum between min(pixels) and 0.

        Parameters
        ----------
        pixels : (N, D) ndarray
            The image's pixels to render.
        mask: (N, D) ndarray
            A boolean mask to be applied to the image. All points outside the
            mask are set to 0. If mask is None, then the initial pixels are
            returned.
        """
        if mask is not None:
            pixels[~mask] = min(pixels.min(), 0)
        return pixels

    def render(self, **kwargs):
        r"""
        Select the correct type of image viewer for the given
        image dimensionality.

        Parameters
        ----------
        kwargs : dict
            Passed through to image viewer.

        Returns
        -------
        viewer : :class:`Renderer`
            The rendering object.

        Raises
        ------
        DimensionalityError
            Only 2D images are supported.
        """
        if self.dimensions == 2:
            from collections import Iterable

            if isinstance(self.channels, Iterable) or \
                            self.channels == 'all' or \
                    (self.channels is None and
                             self.pixels.shape[2] not in [1, 3]):
                return ImageSubplotsViewer2d(self.figure_id, self.new_figure,
                                             self.pixels).render(**kwargs)
            else:
                return ImageViewer2d(self.figure_id, self.new_figure,
                                     self.pixels).render(**kwargs)
        else:
            raise DimensionalityError("Only 2D images are currently supported")


class FeatureImageViewer(ImageViewer):
    r"""
    Base Feature Image viewer that plots a feature image either as glyph or
    multichannel image (using ImageViewer class).

    Parameters
    ----------
    figure_id : object
        A figure id. Could be any valid object that identifies
        a figure in a given framework (string, int, etc)
    new_figure : bool
        Whether the rendering engine should create a new figure.
    dimensions : {2, 3} int
        The number of dimensions in the image
    pixels : (N, D) ndarray
        The pixels to render.
    channels: int or list or 'all' or None
        A specific selection of channels to render. The user can choose either
        a single or multiple channels. If all, render all channels. If None,
        in the case of glyph=True, render the first min(pixels.shape[2], 9)
        and in the case of glyph=False subplot the first
        min(pixels.shape[2], 36).
    mask: (N, D) ndarray
        A boolean mask to be applied to the image. All points outside the
        mask are set to 0.
    glyph: bool
        Defines whether to plot the glyph image or the different channels in
        subplots.

        Default: True
    vectors_block_size: int
        Defines the size of each block with vectors of the glyph image.
    use_negative: bool
        If this flag is enabled and if the feature pixels have negative values
        and if in glyph mode, then there will be created an image containing
        the glyph of positive and negative values concatenated one on top
        of the other.

        Default: False
    """

    def __init__(self, figure_id, new_figure, dimensions, pixels,
                 channels=None, mask=None, glyph=True, vectors_block_size=10,
                 use_negative=False):
        pixels = pixels.copy()
        self.figure_id = figure_id
        self.new_figure = new_figure
        if glyph:
            channels, pixels = super(FeatureImageViewer, self).\
                _parse_channels(channels, pixels, 9)
            self.channels = 0
            pixels, mask = self._feature_glyph_image(pixels, mask,
                                                     vectors_block_size,
                                                     use_negative)
        else:
            self.channels, pixels = super(FeatureImageViewer, self).\
                _parse_channels(channels, pixels, 36)
        self.pixels = super(FeatureImageViewer, self)._masked_pixels(pixels,
                                                                     mask)
        self.dimensions = dimensions

    def _feature_glyph_image(self, feature_data, mask_data, vectors_block_size,
                             use_negative):
        r"""
        Create glyph of a feature image. If feature_data has negative values,
        the use_negative flag controls whether there will be created a glyph of
        both positive and negative values concatenated the one on top of the
        other.

        Parameters
        ----------
        feature_data : (N, D) ndarray
            The feature pixels to use.
        mask_data: (N, D) ndarray
            The original boolean mask corresponding to the image. All points
            outside the mask are set to 0.
        vectors_block_size: int
            Defines the size of each block with vectors of the glyph image.
        use_negative: bool
            Defines whether to take into account possible negative values of
            feature_data.
        """
        negative_weights = -feature_data
        scale = np.maximum(feature_data.max(), negative_weights.max())
        pos, mask_pos = self._create_feature_glyph(feature_data, mask_data,
                                                   vectors_block_size)
        pos = pos * 255 / scale
        glyph_image = pos
        mask_image = mask_pos
        if use_negative and feature_data.min() < 0:
            neg, mask_neg = self._create_feature_glyph(negative_weights,
                                                       mask_data,
                                                       vectors_block_size)
            neg = neg * 255 / scale
            glyph_image = np.concatenate((pos, neg))
            mask_image = np.concatenate((mask_pos, mask_neg))
        return glyph_image, mask_image

    def _create_feature_glyph(self, features, mask_data, vbs):
        r"""
        Create glyph of feature pixels.

        Parameters
        ----------
        features : (N, D) ndarray
            The feature pixels to use.
        mask: (N, D) ndarray
            The original boolean mask corresponding to the image. All points
            outside the mask are set to 0.
        vbs: int
            Defines the size of each block with vectors of the glyph image.
        """
        # vbs = Vector block size
        num_bins = features.shape[2]
        # construct a "glyph" for each orientation
        block_image_temp = np.zeros((vbs, vbs))
        # Create a vertical line of ones, to be the first vector
        block_image_temp[:, round(vbs / 2) - 1:round(vbs / 2) + 1] = 1
        block_im = np.zeros((block_image_temp.shape[0],
                             block_image_temp.shape[1],
                             num_bins))
        # First vector as calculated above
        block_im[:, :, 0] = block_image_temp
        # Number of bins rotations to create an 'asterisk' shape
        for i in range(1, num_bins):
            block_im[:, :, i] = imrotate(block_image_temp, -i * vbs)

        # make pictures of positive feature_data by adding up weighted glyphs
        features[features < 0] = 0

        if mask_data is not None:
            ones_mask = np.ones(block_im.shape[:2])
            mask_im = (ones_mask[None, None, :, :] *
                       np.squeeze(mask_data)[:, :, None, None])
            mask_im = np.bmat(mask_im.tolist()).astype(np.bool)
        else:
            mask_im = None

        glyph_im = np.sum(block_im[None, None, :, :, :] *
                          features[:, :, None, None, :], axis=-1)
        glyph_im = np.bmat(glyph_im.tolist())
        return glyph_im, mask_im


class TriMeshViewer(object):
    r"""
    Base TriMesh viewer that abstracts away dimensionality.

    Parameters
    ----------
    figure_id : object
        A figure id. Could be any valid object that identifies
        a figure in a given framework (string, int, etc)
    new_figure : bool
        Whether the rendering engine should create a new figure.
    points : (N, D) ndarray
        The points to render.
    trilist : (M, 3) ndarray
        The triangulation for the points.
    """

    def __init__(self, figure_id, new_figure, points, trilist):
        self.figure_id = figure_id
        self.new_figure = new_figure
        self.points = points
        self.trilist = trilist

    def render(self, **kwargs):
        r"""
        Select the correct type of trimesh viewer for the given
        trimesh dimensionality.

        Parameters
        ----------
        kwargs : dict
            Passed through to trimesh viewer.

        Returns
        -------
        viewer : :class:`Renderer`
            The rendering object.

        Raises
        ------
        DimensionalityError
            Only 2D and 3D viewers are supported.
        """
        if self.points.shape[1] == 2:
            return TriMeshViewer2d(self.figure_id, self.new_figure,
                                   self.points, self.trilist).render(**kwargs)

        elif self.points.shape[1] == 3:
            return TriMeshViewer3d(self.figure_id, self.new_figure,
                                   self.points, self.trilist).render(**kwargs)
        else:
            raise DimensionalityError("Only 2D and 3D TriMeshes are "
                                      "currently supported")
