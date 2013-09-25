import numpy as np
import abc
from pybug.visualize.base import Renderer


class MatplotlibRenderer(Renderer):
    r"""
    Abstract class for rendering visualizations using Matplotlib.

    Parameters
    ----------
    figure_id : int or ``None``
        A figure id or ``None``. ``None`` assumes we maintain the Matplotlib
        state machine and use ``plt.gcf()``.
    new_figure : bool
        If ``True``, creates a new figure to render on.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, figure_id, new_figure):
        super(MatplotlibRenderer, self).__init__(figure_id, new_figure)

    def get_figure(self):
        r"""
        Gets the figure specified by the combination of ``self.figure_id`` and
        ``self.new_figure``. If ``self.figure_id == None`` then ``plt.gcf()``
        is used. ``self.figure_id`` is also set to the correct id of the figure
        if a new figure is created.

        Returns
        -------
        figure : Matplotlib figure object
            The figure we will be rendering on.
        """
        import matplotlib.pyplot as plt
        if self.new_figure or self.figure_id is not None:
            self.figure = plt.figure(self.figure_id)
        else:
            self.figure = plt.gcf()

        self.figure_id = self.figure.number

        return self.figure


class MatplotlibImageViewer2d(MatplotlibRenderer):

    def __init__(self, figure_id, new_figure, image):
        super(MatplotlibImageViewer2d, self).__init__(figure_id, new_figure)
        self.image = image

    def _render(self, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        if len(self.image.shape) == 2 or self.image.shape[2] == 1:
            im = self.image
            im = im if len(im.shape) == 2 else im[..., 0]
            plt.imshow(im, cmap=cm.Greys_r, **kwargs)
        else:
            plt.imshow(self.image, **kwargs)

        return self


class MatplotlibPointCloudViewer2d(MatplotlibRenderer):

    def __init__(self, figure_id, new_figure, points):
        super(MatplotlibPointCloudViewer2d, self).__init__(figure_id,
                                                           new_figure)
        self.points = points

    def _render(self, image_view=False, cmap=None,
                      colour_array='b', label=None, **kwargs):
        import matplotlib.pyplot as plt
        # Flip x and y for viewing if points are tied to an image
        points = self.points[:, ::-1] if image_view else self.points
        plt.scatter(points[:, 0], points[:, 1], cmap=cmap,
                    c=colour_array, label=label)
        return self


class MatplotlibTriMeshViewer2d(MatplotlibRenderer):

    def __init__(self, figure_id, new_figure, points, trilist):
        super(MatplotlibTriMeshViewer2d, self).__init__(figure_id, new_figure)
        self.points = points
        self.trilist = trilist

    def _render(self, image_view=False, label=None, **kwargs):
        import matplotlib.pyplot as plt
        # Flip x and y for viewing if points are tied to an image
        points = self.points[:, ::-1] if image_view else self.points
        plt.triplot(points[:, 0], points[:, 1], self.trilist,
                    label=label, color='b')

        return self


class MatplotlibLandmarkViewer2d(MatplotlibRenderer):

    def __init__(self, figure_id, new_figure, label, landmark_dict):
        super(MatplotlibLandmarkViewer2d, self).__init__(figure_id, new_figure)
        self.label = label
        self.landmark_dict = landmark_dict

    def _plot_landmarks(self, include_labels, image_view, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        colours = kwargs.get('colours',
                             np.random.random([3, len(self.landmark_dict)]))
        halign = kwargs.get('halign', 'center')
        valign = kwargs.get('valign', 'bottom')
        size = kwargs.get('size', 10)

        # TODO: Should we enforce viewing landmarks with Matplotlib? How
        # do we do this?
        # Set the default colormap, assuming that the pointclouds are
        # also viewed using Matplotlib
        kwargs.setdefault('cmap', cm.jet)

        for i, (label, pc) in enumerate(self.landmark_dict.iteritems()):
            # Set kwargs assuming that the pointclouds are viewed using
            # Matplotlib
            kwargs['colour_array'] = [colours[:, i]] * pc.n_points
            kwargs['label'] = '{0}_{1}'.format(self.label, label)
            pc.view_on(self.figure_id, image_view=image_view, **kwargs)

            if include_labels:
                ax = plt.gca()
                points = pc.points[:, ::-1] if image_view else pc.points
                for i, p in enumerate(points):
                    ax.annotate(str(i), xy=(p[0], p[1]),
                                horizontalalignment=halign,
                                verticalalignment=valign, size=size)
                ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

    def _render(self, include_labels=True, **kwargs):
        self._plot_landmarks(include_labels, False, **kwargs)
        return self


class MatplotlibLandmarkViewer2dImage(MatplotlibLandmarkViewer2d):

    def __init__(self, figure_id, new_figure, label, landmark_dict):
        super(MatplotlibLandmarkViewer2dImage, self).__init__(
            figure_id, new_figure, label, landmark_dict)

    def _render(self, include_labels=True, **kwargs):
        self._plot_landmarks(include_labels, True, **kwargs)
        return self


class MatplotlibAlignmentViewer2d(MatplotlibRenderer):

    def __init__(self, figure_id, new_figure, alignment_transform):
        super(MatplotlibAlignmentViewer2d, self).__init__(figure_id,
                                                          new_figure)
        self.alignment_transform = alignment_transform

    def _render(self, image=False, **kwargs):
        r"""
        Visualize how points are affected by the warp in 2 dimensions.
        """
        from matplotlib import pyplot
        source = self.alignment_transform.source
        target = self.alignment_transform.target
        # a factor by which the minimum and maximum x and y values of the warp
        # will be increased by.
        x_margin_factor, y_margin_factor = 0.5, 0.5
        # the number of x and y samples to take
        n_x, n_y = 50, 50
        # {x y}_{min max} is the actual bounds on either source or target
        # landmarks
        x_min, y_min = np.vstack(
            [target.min(0), source.min(0)]).min(0)
        x_max, y_max = np.vstack(
            [target.max(0), source.max(0)]).max(0)
        x_margin = x_margin_factor * (x_max - x_min)
        y_margin = y_margin_factor * (y_max - y_min)
        # {x y}_{min max}_m is the bound once it has been grown by the factor
        # of the spread in that dimension
        x_min_m = x_min - x_margin
        x_max_m = x_max + x_margin
        y_min_m = y_min - y_margin
        y_max_m = y_max + y_margin
        # build sample points for the selected region
        x = np.linspace(x_min_m, x_max_m, n_x)
        y = np.linspace(y_min_m, y_max_m, n_y)
        xx, yy = np.meshgrid(x, y)
        sample_points = np.concatenate(
            [xx.reshape([-1, 1]), yy.reshape([-1, 1])], axis=1)
        warped_points = self.alignment_transform.apply(sample_points)
        delta = warped_points - sample_points
        # plot the sample points result
        x, y, = 0, 1
        if image:
            # if we are overlaying points onto an image,
            # we have to account for the fact that axis 0 is typically
            # called 'y' and axis 1 is typically called 'x'. Flip them here
            x, y = y, x
        pyplot.quiver(sample_points[:, x], sample_points[:, y], delta[:, x],
                      delta[:, y])
        delta = target - source
        # plot how the landmarks move from source to target
        pyplot.quiver(source[:, x], source[:, y], delta[:, x],
                      delta[:, y], angles='xy', scale_units='xy', scale=1)
        # rescale to the bounds
        pyplot.xlim((x_min_m, x_max_m))
        pyplot.ylim((y_min_m, y_max_m))
        if image:
            # if we are overlaying points on an image, axis0 (the 'y' axis)
            # is flipped.
            pyplot.gca().invert_yaxis()
