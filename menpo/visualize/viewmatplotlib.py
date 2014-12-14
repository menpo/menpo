import abc

import numpy as np
import matplotlib.pyplot as plt

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
        if self.new_figure or self.figure_id is not None:
            self.figure = plt.figure(self.figure_id)
        else:
            self.figure = plt.gcf()

        self.figure_id = self.figure.number

        return self.figure

    def save_figure(self, filename, format='png', dpi=None, face_colour='w',
                    edge_colour='w', orientation='portrait',
                    paper_type='letter', transparent=False, pad_inches=0.1):
        self.figure.savefig(filename, dpi=dpi, facecolour=face_colour,
                            edgecolour=edge_colour, orientation=orientation,
                            papertype=paper_type, format=format,
                            transparent=transparent, pad_inches=pad_inches,
                            bbox_inches='tight', frameon=None)

    def save_figure_widget(self, popup=True):
        r"""
        Method for saving the figure of the current `figure_id` to file using
        :map:`menpo.visualize.widgets.save_matplotlib_figure` widget.

        Parameters
        ----------
        popup : `bool`, optional
            If ``True``, the widget will appear as a popup window.
        """
        from menpo.visualize.widgets import save_matplotlib_figure
        save_matplotlib_figure(self, popup=popup)


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

    def _render(self, render_axes=False, axes_font_name='sans-serif',
                axes_font_size=10, axes_font_style='normal',
                axes_font_weight='normal', axes_x_limits=None,
                axes_y_limits=None, figure_size=(6, 4)):
        import matplotlib.cm as cm

        if len(self.image.shape) == 2:  # Single channels are viewed in Gray
            plt.imshow(self.image, cmap=cm.Greys_r)
        else:
            plt.imshow(self.image)

        # render axes options
        if render_axes:
            plt.axis('on')
            # set font options
            for l in (plt.gca().get_xticklabels() +
                      plt.gca().get_yticklabels()):
                l.set_fontsize(axes_font_size)
                l.set_fontname(axes_font_name)
                l.set_fontstyle(axes_font_style)
                l.set_fontweight(axes_font_weight)
        else:
            plt.axis('off')

        # Set axes limits
        if axes_x_limits is not None:
            plt.xlim(axes_x_limits)
        if axes_y_limits is not None:
            plt.ylim(axes_y_limits[::-1])

        # Set figure size
        if figure_size is not None:
            plt.gcf().set_size_inches(np.asarray(figure_size))

        return self


class MatplotlibImageSubplotsViewer2d(MatplotlibRenderer, MatplotlibSubplots):
    def __init__(self, figure_id, new_figure, image):
        super(MatplotlibImageSubplotsViewer2d, self).__init__(figure_id,
                                                              new_figure)
        self.image = image
        self.num_subplots = self.image.shape[2]
        self.plot_layout = self._subplot_layout(self.num_subplots)

    def _render(self, render_axes=False, axes_font_name='sans-serif',
                axes_font_size=10, axes_font_style='normal',
                axes_font_weight='normal', axes_x_limits=None,
                axes_y_limits=None, figure_size=(6, 4)):
        import matplotlib.cm as cm

        p = self.plot_layout
        for i in range(self.image.shape[2]):
            plt.subplot(p[0], p[1], 1 + i)

            # render axes options
            if render_axes:
                plt.axis('on')
                # set font options
                for l in (plt.gca().get_xticklabels() +
                          plt.gca().get_yticklabels()):
                    l.set_fontsize(axes_font_size)
                    l.set_fontname(axes_font_name)
                    l.set_fontstyle(axes_font_style)
                    l.set_fontweight(axes_font_weight)
            else:
                plt.axis('off')

            # Set axes limits
            if axes_x_limits is not None:
                plt.xlim(axes_x_limits)
            if axes_y_limits is not None:
                plt.ylim(axes_y_limits[::-1])

            # show image
            plt.imshow(self.image[:, :, i], cmap=cm.Greys_r)

            # Set figure size
            if figure_size is not None:
                plt.gcf().set_size_inches(np.asarray(figure_size))
        return self


class MatplotlibPointGraphViewer2d(MatplotlibRenderer):
    def __init__(self, figure_id, new_figure, points, adjacency_array):
        super(MatplotlibPointGraphViewer2d, self).__init__(figure_id,
                                                           new_figure)
        self.points = points
        self.adjacency_array = adjacency_array

    def _render(self, image_view=False, render_lines=True, line_colour='r',
                line_style='-', line_width=1, render_markers=True,
                marker_style='o', marker_size=20, marker_face_colour='k',
                marker_edge_colour='k', marker_edge_width=1., render_axes=True,
                axes_font_name='sans-serif', axes_font_size=10,
                axes_font_style='normal', axes_font_weight='normal',
                axes_x_limits=None, axes_y_limits=None, figure_size=(6, 4),
                label=None):
        from matplotlib import collections as mc
        import matplotlib.cm as cm

        # Flip x and y for viewing if points are tied to an image
        points = self.points[:, ::-1] if image_view else self.points

        ax = plt.gca()

        # Check if graph has edges to be rendered (for example a PointCLoud
        # won't have any edges)
        if render_lines and np.array(self.adjacency_array).shape[0] > 0:
            # Get edges to be rendered
            lines = zip(points[self.adjacency_array[:, 0], :],
                        points[self.adjacency_array[:, 1], :])

            # Draw line objects
            lc = mc.LineCollection(lines, colors=line_colour,
                                   linestyles=line_style, linewidths=line_width,
                                   cmap=cm.jet, label=label)
            ax.add_collection(lc)

            # If a label is defined, it should only be applied to the lines, of
            # a PointGraph, which represent each one of the labels, unless a
            # PointCLoud is passed in.
            label = None

        # Scatter
        if render_markers:
            plt.scatter(points[:, 0], points[:, 1], cmap=cm.jet,
                        c=marker_face_colour, s=marker_size,
                        marker=marker_style, linewidths=marker_edge_width,
                        edgecolors=marker_edge_colour,
                        facecolors=marker_face_colour, label=label)

        # Apply axes options
        if render_axes:
            plt.axis('on')
            # set font options
            for l in (plt.gca().get_xticklabels() +
                      plt.gca().get_yticklabels()):
                l.set_fontsize(axes_font_size)
                l.set_fontname(axes_font_name)
                l.set_fontstyle(axes_font_style)
                l.set_fontweight(axes_font_weight)
        else:
            plt.axis('off')

        # Plot on image mode
        if image_view:
            plt.gca().set_aspect('equal', adjustable='box')
            plt.gca().invert_yaxis()

        # Set axes limits
        if axes_x_limits is not None:
            plt.xlim(axes_x_limits)
        if axes_y_limits is not None:
            plt.ylim(axes_y_limits[::-1]) if image_view \
                else plt.ylim(axes_y_limits)

        # Set figure size
        if figure_size is not None:
            plt.gcf().set_size_inches(np.asarray(figure_size))

        return self


class MatplotlibLandmarkViewer2d(MatplotlibRenderer):
    def __init__(self, figure_id, new_figure, group, pointcloud,
                 labels_to_masks):
        super(MatplotlibLandmarkViewer2d, self).__init__(figure_id, new_figure)
        self.group = group
        self.pointcloud = pointcloud
        self.labels_to_masks = labels_to_masks

    def _render(self, image_view=False, render_lines=True, line_colour='r',
                line_style='-', line_width=1, render_markers=True,
                marker_style='o', marker_size=20, marker_face_colour='k',
                marker_edge_colour='k', marker_edge_width=1.,
                render_numbering=False, numbers_horizontal_align='center',
                numbers_vertical_align='bottom',
                numbers_font_name='sans-serif', numbers_font_size=10,
                numbers_font_style='normal',
                numbers_font_weight='normal', numbers_font_colour='k',
                render_legend=True, legend_title='',
                legend_font_name='sans-serif',
                legend_font_style='normal', legend_font_size=10,
                legend_font_weight='normal', legend_marker_scale=None,
                legend_location=2, legend_bbox_to_anchor=(1.05, 1.),
                legend_border_axes_pad=None, legend_n_columns=1,
                legend_horizontal_spacing=None,
                legend_vertical_spacing=None, legend_border=True,
                legend_border_padding=None, legend_shadow=False,
                legend_rounded_corners=False, render_axes=True,
                axes_font_name='sans-serif', axes_font_size=10,
                axes_font_style='normal', axes_font_weight='normal',
                axes_x_limits=None, axes_y_limits=None, figure_size=(6, 4)):
        # Regarding the labels colours, we may get passed either no colours (in
        # which case we generate random colours) or a single colour to colour
        # all the labels with
        if render_lines:
            n_labels = len(self.labels_to_masks)
            if line_colour is None:
                # sample colours from jet colour map
                line_colour = sample_colours_from_colourmap(n_labels, 'jet')
            if len(line_colour) == 1:
                line_colour *= n_labels
            elif len(line_colour) != n_labels:
                raise ValueError('Must pass a list of n_labels line colours '
                                 'or a single line colour for all labels.')

        # Get pointcloud of each label
        sub_pointclouds = self._build_sub_pointclouds()

        for i, (label, pc) in enumerate(sub_pointclouds):
            # Set kwargs assuming that the pointclouds are viewed using
            # Matplotlib
            pc.view_on(figure_id=self.figure_id, image_view=image_view,
                       render_lines=render_lines, line_colour=line_colour[i],
                       line_style=line_style, line_width=line_width,
                       render_markers=render_markers, marker_style=marker_style,
                       marker_size=marker_size,
                       marker_face_colour=marker_face_colour,
                       marker_edge_colour=marker_edge_colour,
                       marker_edge_width=marker_edge_width,
                       render_axes=render_axes, axes_font_name=axes_font_name,
                       axes_font_size=axes_font_size,
                       axes_font_style=axes_font_style,
                       axes_font_weight=axes_font_weight, axes_x_limits=None,
                       axes_y_limits=None, figure_size=None,
                       label='{0}: {1}'.format(self.group, label))

            ax = plt.gca()

            if render_numbering:
                points = pc.points[:, ::-1] if image_view else pc.points
                for k, p in enumerate(points):
                    ax.annotate(str(k), xy=(p[0], p[1]),
                                horizontalalignment=numbers_horizontal_align,
                                verticalalignment=numbers_vertical_align,
                                size=numbers_font_size,
                                family=numbers_font_name,
                                fontstyle=numbers_font_style,
                                fontweight=numbers_font_weight,
                                color=numbers_font_colour)

        if render_legend:
            # Options related to legend's font
            prop = {'family': legend_font_name, 'size': legend_font_size,
                    'style': legend_font_style,
                    'weight': legend_font_weight}

            # Render legend
            ax.legend(title=legend_title, prop=prop, loc=legend_location,
                      bbox_to_anchor=legend_bbox_to_anchor,
                      borderaxespad=legend_border_axes_pad,
                      ncol=legend_n_columns,
                      columnspacing=legend_horizontal_spacing,
                      labelspacing=legend_vertical_spacing,
                      frameon=legend_border,
                      borderpad=legend_border_padding, shadow=legend_shadow,
                      fancybox=legend_rounded_corners,
                      markerscale=legend_marker_scale)

        # Apply axes options
        if render_axes:
            plt.axis('on')
            # set font options
            for l in (plt.gca().get_xticklabels() +
                      plt.gca().get_yticklabels()):
                l.set_fontsize(axes_font_size)
                l.set_fontname(axes_font_name)
                l.set_fontstyle(axes_font_style)
                l.set_fontweight(axes_font_weight)
        else:
            plt.axis('off')

        # Set axes limits
        if axes_x_limits is not None:
            plt.xlim(axes_x_limits)
        if axes_y_limits is not None:
            plt.ylim(axes_y_limits[::-1]) if image_view \
                else plt.ylim(axes_y_limits)

        # Set figure size
        if figure_size is not None:
            plt.gcf().set_size_inches(np.asarray(figure_size))

        return self

    def _build_sub_pointclouds(self):
        sub_pointclouds = []
        for label, indices in self.labels_to_masks.iteritems():
            mask = self.labels_to_masks[label]
            sub_pointclouds.append((label, self.pointcloud.from_mask(mask)))
        return sub_pointclouds


class MatplotlibAlignmentViewer2d(MatplotlibRenderer):
    def __init__(self, figure_id, new_figure, alignment_transform):
        super(MatplotlibAlignmentViewer2d, self).__init__(figure_id,
                                                          new_figure)
        self.alignment_transform = alignment_transform

    def _render(self, image=False, **kwargs):
        r"""
        Visualize how points are affected by the warp in 2 dimensions.
        """
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
        return self


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

    def _render(self, colour_list=None, marker_list=None, **kwargs):
        ax = plt.gca()
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        for y, c, m in zip(self.y_axis, colour_list, marker_list):
            plt.plot(self.x_axis, y, color=c, marker=m, **kwargs)
        if self.axis_limits is not None:
            plt.axis(self.axis_limits)

        plt.grid(True)
        plt.title(self.title)
        plt.legend(self.legend, bbox_to_anchor=(1.05, 1), loc=2,
                   borderaxespad=0.)
        return self


class MatplotlibMultiImageViewer2d(MatplotlibRenderer):
    def __init__(self, figure_id, new_figure, image_list):
        super(MatplotlibMultiImageViewer2d, self).__init__(figure_id,
                                                           new_figure)
        self.image_list = image_list

    def _render(self, interval=50, **kwargs):
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


def sample_colours_from_colourmap(n_colours, colour_map):
    cm = plt.get_cmap(colour_map)
    return [cm(1.*i/n_colours)[:3] for i in range(n_colours)]
