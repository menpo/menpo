import abc

import numpy as np

from menpo.visualize.base import Renderer


class MatplotlibRenderer(Renderer):
    r"""
    Abstract class for rendering visualizations using Matplotlib.

    Parameters
    ----------
    figure_id : int or `None`
        A figure id or `None`. `None` assumes we maintain the Matplotlib
        state machine and use `plt.gcf()`.
    new_figure : bool
        If `True`, creates a new figure to render on.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, figure_id, new_figure):
        super(MatplotlibRenderer, self).__init__(figure_id, new_figure)

    def get_figure(self):
        r"""
        Gets the figure specified by the combination of `self.figure_id` and
        `self.new_figure`. If `self.figure_id == None` then `plt.gcf()`
        is used. `self.figure_id` is also set to the correct id of the figure
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


class MatplotlibSubplots(object):

    __metaclass__ = abc.ABCMeta

    def _subplot_layout(self, num_subplots):
        if num_subplots < 2:
            return [1, 1]
        while self._is_prime(num_subplots) and num_subplots > 4:
            num_subplots += 1
        p = self._factor(num_subplots)
        if len(p) == 1:
            p.insert(0, 1)
            return p
        while len(p) > 2:
            if len(p) >= 4:
                p[0] = p[0] * p[-2]
                p[1] = p[1] * p[-1]
                del p[-2:]
            else:
                p[0] = p[0] * p[1]
                del p[1]
            p.sort()
            # Reformat if the column/row ratio is too large: we want a roughly
        # square design
        while (p[1] / p[0]) > 2.5:
            p = self._subplot_layout(num_subplots + 1)
        return p

    def _factor(self, n):
        gaps = [1, 2, 2, 4, 2, 4, 2, 4, 6, 2, 6]
        length, cycle = 11, 3
        f, fs, next_ind = 2, [], 0
        while f * f <= n:
            while n % f == 0:
                fs.append(f)
                n /= f
            f += gaps[next_ind]
            next_ind += 1
            if next_ind == length:
                next_ind = cycle
        if n > 1:
            fs.append(n)
        return fs

    def _is_prime(self, n):
        if n == 2 or n == 3:
            return True
        if n < 2 or n % 2 == 0:
            return False
        if n < 9:
            return True
        if n % 3 == 0:
            return False
        r = int(n ** 0.5)
        f = 5
        while f <= r:
            if n % f == 0:
                return False
            if n % (f + 2) == 0:
                return False
            f += 6
        return True


class MatplotlibImageViewer2d(MatplotlibRenderer):
    def __init__(self, figure_id, new_figure, image):
        super(MatplotlibImageViewer2d, self).__init__(figure_id, new_figure)
        self.image = image

    def _render(self, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        if len(self.image.shape) == 2:  # Single channels are viewed in Gray
            plt.imshow(self.image, cmap=cm.Greys_r, **kwargs)
        else:
            plt.imshow(self.image, **kwargs)

        return self


class MatplotlibImageSubplotsViewer2d(MatplotlibRenderer, MatplotlibSubplots):
    def __init__(self, figure_id, new_figure, image):
        super(MatplotlibImageSubplotsViewer2d, self).__init__(figure_id,
                                                              new_figure)
        self.image = image
        self.num_subplots = self.image.shape[2]
        self.plot_layout = self._subplot_layout(self.num_subplots)

    def _render(self, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        p = self.plot_layout
        for i in range(self.image.shape[2]):
            plt.subplot(p[0], p[1], 1 + i)
            # Hide the x and y labels
            plt.axis('off')
            plt.imshow(self.image[:, :, i], cmap=cm.Greys_r, **kwargs)
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
    def __init__(self, figure_id, new_figure, group_label, pointcloud,
                 labels_to_masks):
        super(MatplotlibLandmarkViewer2d, self).__init__(figure_id, new_figure)
        self.group_label = group_label
        self.pointcloud = pointcloud
        self.labels_to_masks = labels_to_masks

    def _plot_landmarks(self, include_labels, image_view, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        colours = kwargs.get(
            'colours', np.random.random([3, len(self.labels_to_masks)]))
        halign = kwargs.get('halign', 'center')
        valign = kwargs.get('valign', 'bottom')
        size = kwargs.get('size', 10)

        # TODO: Should we enforce viewing landmarks with Matplotlib? How
        # do we do this?
        # Set the default colormap, assuming that the pointclouds are
        # also viewed using Matplotlib
        kwargs.setdefault('cmap', cm.jet)

        sub_pointclouds = self._build_sub_pointclouds()

        for i, (label, pc) in enumerate(sub_pointclouds):
            # Set kwargs assuming that the pointclouds are viewed using
            # Matplotlib
            kwargs['colour_array'] = [colours[:, i]] * np.sum(pc.points)
            kwargs['label'] = '{0}_{1}'.format(self.group_label, label)
            pc.view_on(self.figure_id, image_view=image_view, **kwargs)

            if include_labels:
                ax = plt.gca()
                points = pc.points[:, ::-1] if image_view else pc.points
                for i, p in enumerate(points):
                    ax.annotate(str(i), xy=(p[0], p[1]),
                                horizontalalignment=halign,
                                verticalalignment=valign, size=size)
                ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

    def _build_sub_pointclouds(self):
        sub_pointclouds = []
        for label, indices in self.labels_to_masks.iteritems():
            mask = self.labels_to_masks[label]
            sub_pointclouds.append((label, self.pointcloud.from_mask(mask)))
        return sub_pointclouds

    def _render(self, include_labels=True, **kwargs):
        self._plot_landmarks(include_labels, False, **kwargs)
        return self


class MatplotlibLandmarkViewer2dImage(MatplotlibLandmarkViewer2d):
    def __init__(self, figure_id, new_figure, group_label, pointcloud,
                 labels_to_masks):
        super(MatplotlibLandmarkViewer2dImage, self).__init__(
            figure_id, new_figure, group_label, pointcloud, labels_to_masks)

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
        import matplotlib.pyplot as plt

        source = self.alignment_transform.source.points
        target = self.alignment_transform.target.points
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
        plt.quiver(sample_points[:, x], sample_points[:, y], delta[:, x],
                   delta[:, y])
        delta = target - source
        # plot how the landmarks move from source to target
        plt.quiver(source[:, x], source[:, y], delta[:, x],
                   delta[:, y], angles='xy', scale_units='xy', scale=1)
        # rescale to the bounds
        plt.xlim((x_min_m, x_max_m))
        plt.ylim((y_min_m, y_max_m))
        if image:
            # if we are overlaying points on an image, axis0 (the 'y' axis)
            # is flipped.
            plt.gca().invert_yaxis()


class MatplotlibGraphPlotter(MatplotlibRenderer):

    def __init__(self, figure_id, new_figure, x_axis, y_axis,
                 title=None, legend=None, x_label=None, y_label=None,
                 axis_limits=None):
        super(MatplotlibGraphPlotter, self).__init__(figure_id, new_figure)
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.title = title
        self.legend = legend
        self.x_label = x_label
        self.y_label = y_label
        self.axis_limits = axis_limits

    def _render(self, color_list=None, marker_list=None, **kwargs):
        import matplotlib.pyplot as plt

        ax = plt.gca()
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        for y, c, m in zip(self.y_axis, color_list, marker_list):
            plt.plot(self.x_axis, y, color=c, marker=m, **kwargs)
        if self.axis_limits is not None:
            plt.axis(self.axis_limits)

        plt.grid(True)
        plt.title(self.title)
        plt.legend(self.legend, bbox_to_anchor=(1.05, 1), loc=2,
                   borderaxespad=0.)


class MatplotlibMultiImageViewer2d(MatplotlibRenderer):
    def __init__(self, figure_id, new_figure, image_list):
        super(MatplotlibMultiImageViewer2d, self).__init__(figure_id,
                                                           new_figure)
        self.image_list = image_list

    def _render(self, interval=50, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.animation as animation

        if len(self.image_list[0].shape) == 2:
            # Single channels are viewed in Gray
            _ax = plt.imshow(self.image_list[0], cmap=cm.Greys_r, **kwargs)
        else:
            _ax = plt.imshow(self.image_list[0], **kwargs)

        def init():
            return _ax,

        def animate(j):
            _ax.set_data(self.image_list[j])
            return _ax,

        self._ani = animation.FuncAnimation(self.figure, animate,
                                            init_func=init,
                                            frames=len(self.image_list),
                                            interval=interval, blit=True)
        return self


class MatplotlibMultiImageSubplotsViewer2d(MatplotlibRenderer,
                                           MatplotlibSubplots):
    def __init__(self, figure_id, new_figure, image_list):
        super(MatplotlibMultiImageSubplotsViewer2d, self).__init__(figure_id,
                                                                   new_figure)
        self.image_list = image_list
        self.num_subplots = self.image_list[0].shape[2]
        self.plot_layout = self._subplot_layout(self.num_subplots)

    def _render(self, interval=50, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.animation as animation

        p = self.plot_layout
        _axs = []
        for i in range(self.image_list[0].shape[2]):
            plt.subplot(p[0], p[1], 1 + i)
            # Hide the x and y labels
            plt.axis('off')
            _ax = plt.imshow(self.image_list[0][:, :, i], cmap=cm.Greys_r,
                             **kwargs)
            _axs.append(_ax)

        def init():
            return _axs

        def animate(j):
            for k, _ax in enumerate(_axs):
                _ax.set_data(self.image_list[j][:, :, k])
            return _axs

        self._ani = animation.FuncAnimation(self.figure, animate,
                                            init_func=init,
                                            frames=len(self.image_list),
                                            interval=interval, blit=True)
        return self


class MatplotlibFittingViewer2d(MatplotlibImageViewer2d):
    def __init__(self, figure_id, new_figure, image, target_list):
        super(MatplotlibFittingViewer2d, self).__init__(figure_id,
                                                        new_figure, image)
        self.target_list = target_list

    def _render(self, interval=50,  marker='s', color='r',
                markersize=3, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.animation as animation

        _ax = plt.axes()
        _ax.axis('off')

        if len(self.image.shape) == 2:
            # Single channels are viewed in Gray
            _ax.imshow(self.image, cmap=cm.Greys_r)
        else:
            _ax.imshow(self.image)

        _line, = _ax.plot([], [], linestyle=' ', marker=marker, color=color,
                          markersize=markersize, **kwargs)

        def init():
            return _line,

        def animate(j):
            _line.set_data(self.target_list[j][:, 1],
                           self.target_list[j][:, 0])
            return _line,

        self._ani = animation.FuncAnimation(self.figure, animate,
                                            init_func=init,
                                            frames=len(self.target_list),
                                            interval=interval, blit=True)
        return self


class MatplotlibFittingSubplotsViewer2d(MatplotlibImageSubplotsViewer2d):
    def __init__(self, figure_id, new_figure, image, target_list):
        super(MatplotlibFittingSubplotsViewer2d, self).__init__(
            figure_id, new_figure, image)
        self.target_list = target_list

    def _render(self, interval=50, marker='s', color='r',
                markersize=3, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.animation as animation

        p = self.plot_layout
        _lines = []
        for j in range(self.image.shape[2]):
            plt.subplot(p[0], p[1], 1 + j)
            _ax = plt.axes()
            # Hide the x and y labels
            _ax.axis('off')
            _ax.imshow(self.image[:, :, j], cmap=cm.Greys_r)
            _line, = _ax.plot([], [], linestyle=' ', marker=marker,
                              color=color, markersize=markersize, **kwargs)
            _lines.append(_line)

        def init():
            return _lines

        def animate(j):
            for _line in enumerate(_lines):
                _line.set_data(self.target_list[j][:, 1],
                               self.target_list[j][:, 0])
            return _lines

        self._ani = animation.FuncAnimation(self.figure, animate,
                                            init_func=init,
                                            frames=len(self.target_list),
                                            interval=interval, blit=True)
        return self
