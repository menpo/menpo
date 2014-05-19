# This has to go above the default importers to prevent cyclical importing
import abc

from collections import Iterable


class Renderer(object):
    r"""
    Abstract class for rendering visualizations. Framework specific
    implementations of these classes are made in order to separate
    implementation cleanly from the rest of the code.

    It is assumed that the renderers follow some form of stateful pattern for
    rendering to Figures. Therefore, the major interface for rendering involves
    providing a `figure_id` or a boolean about whether a new figure should
    be used. If neither are provided then the default state of the rendering
    engine is assumed to maintained.

    Providing a `figure_id` and `new_figure == True` is not a valid state.

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
            Pointer to `self`.
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
            Pointer to `self`.
        """
        pass

    @abc.abstractmethod
    def get_figure(self):
        r"""
        Abstract method for getting the correct figure to render on. Should
        also set the correct `figure_id` for the figure.

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
        commands are applied to the same `figure` object.

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

            Default: `None`
        new_figure : bool, optional
            Whether the rendering engine should create a new figure.

            Default: `False`
        kwargs : dict
            Passed through to specific rendering engine.

        Returns
        -------
        viewer : :class:`Renderer`
            The renderer instantiated.
        """
        pass


from menpo.visualize.viewmayavi import MayaviPointCloudViewer3d, \
    MayaviTriMeshViewer3d, MayaviTexturedTriMeshViewer3d, \
    MayaviLandmarkViewer3d, MayaviVectorViewer3d, MayaviSurfaceViewer3d, \
    MayaviColouredTriMeshViewer3d
from menpo.visualize.viewmatplotlib import MatplotlibImageViewer2d, \
    MatplotlibImageSubplotsViewer2d, MatplotlibPointCloudViewer2d, \
    MatplotlibLandmarkViewer2d, MatplotlibLandmarkViewer2dImage, \
    MatplotlibTriMeshViewer2d, MatplotlibAlignmentViewer2d, \
    MatplotlibGraphPlotter, MatplotlibMultiImageViewer2d, \
    MatplotlibMultiImageSubplotsViewer2d, MatplotlibFittingViewer2d, \
    MatplotlibFittingSubplotsViewer2d

# Default importer types
PointCloudViewer2d = MatplotlibPointCloudViewer2d
PointCloudViewer3d = MayaviPointCloudViewer3d
TriMeshViewer2d = MatplotlibTriMeshViewer2d
TriMeshViewer3d = MayaviTriMeshViewer3d
TexturedTriMeshViewer3d = MayaviTexturedTriMeshViewer3d
ColouredTriMeshViewer3d = MayaviColouredTriMeshViewer3d
LandmarkViewer3d = MayaviLandmarkViewer3d
LandmarkViewer2d = MatplotlibLandmarkViewer2d
LandmarkViewer2dImage = MatplotlibLandmarkViewer2dImage
ImageViewer2d = MatplotlibImageViewer2d
ImageSubplotsViewer2d = MatplotlibImageSubplotsViewer2d
VectorViewer3d = MayaviVectorViewer3d
AlignmentViewer2d = MatplotlibAlignmentViewer2d
GraphPlotter = MatplotlibGraphPlotter
MultiImageViewer2d = MatplotlibMultiImageViewer2d
MultiImageSubplotsViewer2d = MatplotlibMultiImageSubplotsViewer2d
FittingViewer2d = MatplotlibFittingViewer2d
FittingSubplotsViewer2d = MatplotlibFittingSubplotsViewer2d


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
    pointcloud : :class:`menpo.shape.pointcloud.PointCloud`
        The pointclouds representing the landmarks.
    labels_to_masks : dict(string, ndarray)
        A dictionary of labels to masks into the pointcloud that represent
        which points belong to the given label.
    target : :class:`menpo.landmarks.base.Landmarkable`
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
            from menpo.image.base import Image

            if isinstance(self.target, Image):
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
            raise ValueError("Only 2D and 3D landmarks are "
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
            raise ValueError("Only 2D and 3D pointclouds are "
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
        pixels, self.use_subplots = \
            self._parse_channels(channels, pixels)
        self.pixels = self._masked_pixels(pixels, mask)

    def _parse_channels(self, channels, pixels):
        r"""
        Parse channels parameter. If channels is int or list, keep it as is. If
        channels is all, return a list of all the image's channels. If channels
        is None, return the minimum between an upper_limit and the image's
        number of channels. If image is grayscale or RGB and channels is None,
        then do not plot channels in different subplots.

        Parameters
        ----------
        channels: int or list or 'all' or None
            A specific selection of channels to render.
        pixels : (N, D) ndarray
            The image's pixels to render.
        upper_limit: int
            The upper limit of subplots for the channels=None case.
        """
        # Flag to trigger ImageSubplotsViewer2d or ImageViewer2d
        use_subplots = True
        n_channels = pixels.shape[2]
        if channels is None:
            if n_channels == 1:
                pixels = pixels[..., 0]
                use_subplots = False
            elif n_channels == 3:
                use_subplots = False
        elif channels != 'all':
            if isinstance(channels, Iterable):
                pixels = pixels[..., channels]
            else:
                pixels = pixels[..., channels]
                use_subplots = False

        return pixels, use_subplots

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
            if self.use_subplots:
                return ImageSubplotsViewer2d(self.figure_id, self.new_figure,
                                             self.pixels).render(**kwargs)
            else:
                return ImageViewer2d(self.figure_id, self.new_figure,
                                     self.pixels).render(**kwargs)
        else:
            raise ValueError("Only 2D images are currently supported")


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
            raise ValueError("Only 2D and 3D TriMeshes "
                             "are currently supported")


class MultipleImageViewer(ImageViewer):

    def __init__(self, figure_id, new_figure, dimensions, pixels_list,
                 channels=None, mask=None):
        super(MultipleImageViewer, self).__init__(
            figure_id, new_figure, dimensions, pixels_list[0],
            channels=channels, mask=mask)
        pixels_list = [self._parse_channels(channels, p)[0]
                       for p in pixels_list]
        self.pixels_list = [self._masked_pixels(p, mask)
                            for p in pixels_list]

    def render(self, **kwargs):
        if self.dimensions == 2:
            if self.use_subplots:
                MultiImageSubplotsViewer2d(self.figure_id, self.new_figure,
                                           self.pixels_list).render(**kwargs)
            else:
                return MultiImageViewer2d(self.figure_id, self.new_figure,
                                          self.pixels_list).render(**kwargs)
        else:
            raise ValueError("Only 2D images are currently supported")


class FittingViewer(ImageViewer):

    def __init__(self, figure_id, new_figure, dimensions, pixels,
                 target_list, channels=None, mask=None):
        super(FittingViewer, self).__init__(
            figure_id, new_figure, dimensions, pixels,
            channels=channels, mask=mask)
        self.target_list = target_list

    def render(self, **kwargs):
        if self.dimensions == 2:
            if self.use_subplots:
                FittingSubplotsViewer2d(
                    self.figure_id, self.new_figure, self.pixels,
                    self.target_list).render(**kwargs)
            else:
                return FittingViewer2d(
                    self.figure_id, self.new_figure, self.pixels,
                    self.target_list).render(**kwargs)
        else:
            raise ValueError("Only 2D images are currently supported")
