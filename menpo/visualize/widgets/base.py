from IPython.html.widgets import (interact, fixed, IntSliderWidget,
                                  PopupWidget, ContainerWidget)
from IPython.display import display, clear_output

from .helpers import (figure_options, format_figure_options, channel_options,
                      format_channel_options, landmark_options,
                      format_landmark_options, info_print, format_info_print)

import numpy as np
from numpy import asarray


def visualize_images(images, with_labels=None, without_labels=None,
                     figure_size=(7, 7), figure_scales=(0.5, 1.5), popup=False,
                     **kwargs):
    r"""
    Allows browsing through a list of images.

    Parameters
    -----------
    images : `list` of :map:`Images` or subclass
        The list of images to be displayed.

        .. note::
        This function assumes that all images have the same number of
        channels and that they all have the same landmark groups.

    with_labels : ``None`` or `str` or `list` of `str`, optional
        If not ``None``, only show the given label(s). Should **not** be
        used with the ``without_labels`` kwarg. If ``render_labels`` is
        ``False`` this kwarg is ignored.

    without_labels : ``None`` or `str` or `list` of `str`, optional
        If not ``None``, show all except the given label(s). Should **not**
        be used with the ``with_labels`` kwarg. If ``render_labels`` is
        ``False`` this kwarg is ignored.

    figure_size : (`int`, `int`), optional
        The size of the plotted figures.

    figure_scales : (`float`, `float`), optional
        The range of scales that can be optionally applied to the figure.

    popup : `boolean`, optional
        If enabled, the widget will appear as a popup window.

    kwargs : `dict`, optional
        Passed through to the viewer.
    """
    import matplotlib.pylab as plt

    # Create options widgets
    image_number_wid = IntSliderWidget(min=0, max=len(images)-1, step=1,
                                       value=1, description='Image Number')
    channel_options_wid = channel_options(images[0].n_channels,
                                          toggle_show_default=False)
    all_groups_keys = images[0].landmarks.keys()
    all_subgroups_keys = [images[0].landmarks[g].keys()
                          for g in all_groups_keys]
    landmark_options_wid = landmark_options(all_groups_keys, all_subgroups_keys,
                                            toggle_show_default=False,
                                            landmarks_default=True,
                                            labels_default=False)
    figure_options_wid = figure_options(x_scale_default=figure_scales[0],
                                        y_scale_default=figure_scales[1],
                                        toggle_show_default=False)
    info_wid = info_print(toggle_show_default=False)

    # Define function
    def show_img(name, value):
        # clear current figure
        clear_output()

        # get channels
        if channel_options_wid.children[1].children[0].value == "Single":
            channels = channel_options_wid.children[1].children[1].children[0].children[0].value
        else:
            channels = range(
                channel_options_wid.children[1].children[1].children[0].children[0].value,
                channel_options_wid.children[1].children[1].children[0].children[1].value + 1)

        # get flag values
        glyph_enabled = channel_options_wid.children[1].children[1].children[1].children[1].children[0].value
        sum_enabled = channel_options_wid.children[1].children[1].children[1].children[0].value
        landmarks_enabled = landmark_options_wid.children[1].children[0].value
        labels_enabled = landmark_options_wid.children[1].children[1].value
        group = landmark_options_wid.children[2].children[0].value
        with_labels = []
        for w in landmark_options_wid.children[2].children[1].children:
            if w.value:
                with_labels.append(str(w.description))

        # plot
        if glyph_enabled:
            s1 = channel_options_wid.children[1].children[1].children[1].children[1].children[1].children[0].value
            s2 = channel_options_wid.children[1].children[1].children[1].children[1].children[1].children[1].value
            if landmarks_enabled:
                images[image_number_wid.value].glyph(vectors_block_size=s1,
                                                     use_negative=s2,
                                                     channels=channels).\
                    view_landmarks(group_label=group, with_labels=with_labels,
                                   without_labels=without_labels,
                                   render_labels=labels_enabled, **kwargs)
            else:
                images[image_number_wid.value].glyph(vectors_block_size=s1,
                                                     use_negative=s2,
                                                     channels=channels).view()
        elif sum_enabled:
            s2 = channel_options_wid.children[1].children[1].children[1].children[1].children[1].children[1].value
            if landmarks_enabled:
                images[image_number_wid.value].glyph(vectors_block_size=1,
                                                     use_negative=s2,
                                                     channels=channels).\
                    view_landmarks(group_label=group, with_labels=with_labels,
                                   without_labels=without_labels,
                                   render_labels=labels_enabled, **kwargs)
            else:
                images[image_number_wid.value].glyph(vectors_block_size=1,
                                                     use_negative=s2,
                                                     channels=channels).view()
        else:
            if landmarks_enabled:
                images[image_number_wid.value].view_landmarks(
                    group_label=group, with_labels=with_labels,
                    without_labels=without_labels, render_labels=labels_enabled,
                    channels=channels, **kwargs)
            else:
                images[image_number_wid.value].view(channels=channels)

        # set figure size
        x_scale = figure_options_wid.children[1].children[0].value
        y_scale = figure_options_wid.children[1].children[1].value
        plt.gcf().set_size_inches([x_scale, y_scale] * asarray(figure_size))
        # turn axis on/off
        if not figure_options_wid.children[2].value:
            plt.axis('off')

        # change info_wid info
        txt = "$\\bullet~\\texttt{Image of size " + \
              "{}".format(images[image_number_wid.value]._str_shape) + \
              " with " + \
              "{}".format(images[image_number_wid.value].n_channels) + \
              " channels.}\\\\ \\bullet~\\texttt{" + \
              "{}".format(images[image_number_wid.value].landmarks['PTS'].lms.n_points) + \
              " landmark points.}\\\\ \\bullet~\\texttt{min=" + \
              "{0:.3f}".format(images[image_number_wid.value].pixels.min()) + \
              ", max=" + \
              "{0:.3f}".format(images[image_number_wid.value].pixels.max()) + \
              "}$"
        info_wid.children[1].value = txt

    # Define traits
    image_number_wid.on_trait_change(show_img, 'value')
    figure_options_wid.children[1].children[0].on_trait_change(show_img,
                                                               'value')
    figure_options_wid.children[1].children[1].on_trait_change(show_img,
                                                               'value')
    figure_options_wid.children[2].on_trait_change(show_img, 'value')
    channel_options_wid.children[1].children[0].on_trait_change(show_img,
                                                                'value')
    channel_options_wid.children[1].children[1].children[0].children[0].on_trait_change(show_img, 'value')
    channel_options_wid.children[1].children[1].children[0].children[1].on_trait_change(show_img, 'value')
    channel_options_wid.children[1].children[1].children[1].children[0].on_trait_change(show_img, 'value')
    channel_options_wid.children[1].children[1].children[1].children[1].children[0].on_trait_change(show_img, 'value')
    channel_options_wid.children[1].children[1].children[1].children[1].children[1].children[0].on_trait_change(show_img, 'value')
    channel_options_wid.children[1].children[1].children[1].children[1].children[1].children[1].on_trait_change(show_img, 'value')
    landmark_options_wid.children[1].children[0].on_trait_change(show_img,
                                                                 'value')
    landmark_options_wid.children[1].children[1].on_trait_change(show_img,
                                                                 'value')
    landmark_options_wid.children[2].children[0].on_trait_change(show_img,
                                                                 'value')
    for w in landmark_options_wid.children[2].children[1].children:
        w.on_trait_change(show_img, 'value')

    # Display widget
    cont1 = ContainerWidget(children=[channel_options_wid,
                                      landmark_options_wid])
    cont2 = ContainerWidget(children=[figure_options_wid,
                                      info_wid])
    if popup:
        wid = PopupWidget(children=[image_number_wid, cont1, cont2],
                          button_text='View Images')
    else:
        wid = ContainerWidget(children=[image_number_wid, cont1, cont2])
    display(wid)

    # Format widget
    format_channel_options(channel_options_wid, container_padding='6px',
                           container_margin='6px',
                           container_border='1px solid black',
                           toggle_button_font_weight='bold')
    format_landmark_options(landmark_options_wid, container_padding='6px',
                            container_margin='6px',
                            container_border='1px solid black',
                            toggle_button_font_weight='bold')
    format_figure_options(figure_options_wid, container_padding='6px',
                          container_margin='6px',
                          container_border='1px solid black',
                          toggle_button_font_weight='bold')
    format_info_print(info_wid, font_size_in_pt='9pt', container_padding='6px',
                      container_margin='6px',
                      container_border='1px solid black',
                      toggle_button_font_weight='bold')
    #cont1.remove_class('vbox')
    #cont1.add_class('hbox')
    #cont2.remove_class('vbox')
    #cont2.add_class('hbox')

    # Reset value to enable initial visualization
    image_number_wid.value = 0


def browse_images(images, with_labels=None, without_labels=None,
                  figure_size=(7, 7), figure_scales=(0.5, 1.5), **kwargs):
    r"""
    Allows browsing through a list of images using a simple slider.

    Parameters
    -----------
    images : `list` of :map:`Images` or subclass
        The list of images to be displayed.

        .. note::
        This function assumes that all images have the same number of
        channels and that they all have the same landmark groups.

    with_labels : ``None`` or `str` or `list` of `str`, optional
        If not ``None``, only show the given label(s). Should **not** be
        used with the ``without_labels`` kwarg. If ``render_labels`` is
        ``False`` this kwarg is ignored.

    without_labels : ``None`` or `str` or `list` of `str`, optional
        If not ``None``, show all except the given label(s). Should **not**
        be used with the ``with_labels`` kwarg. If ``render_labels`` is
        ``False`` this kwarg is ignored.

    figure_size : (`int`, `int`), optional
        The size of the plotted figures.

    figure_scales : (`float`, `float`), optional
        The range of scales that can be optionally applied to the figure.

    kwargs : `dict`, optional
        Passed through to the viewer.
    """
    import matplotlib.pylab as plt

    # define relevant visualization options
    image_indices = (0, len(images)-1)
    groups = images[0].landmarks.keys()
    n_channels = images[0].n_channels
    if n_channels == 1:
        channels = [0]
    elif n_channels == 3:
        channels = range(n_channels) + [None, 'all']
        channel_indices = (0, n_channels + 1)
    else:
        channels = range(n_channels) + ['all']
        channel_indices = (0, n_channels)

    # define the visualization function
    def view_image(landmarks, group, labels, channel_index, figure_scale,
                   axis, image_index):
        if landmarks:
            # view image with landmarks
            images[image_index].view_landmarks(
                group_label=group, with_labels=with_labels,
                without_labels=without_labels, render_labels=labels,
                channels=channels[channel_index], **kwargs)
        else:
            # view image without landmarks
            images[image_index].view(channels=channels[channel_index],
                                     **kwargs)
        # set figure size
        plt.gcf().set_size_inches(figure_scale*np.asarray(figure_size))
        if not axis:
            # turn axis on/off
            plt.axis('off')

    # set the appropriate options in the image visualization function
    if groups:
        # image with landmarks
        if len(channels) == 1:
            # uni-channel image
            interact(view_image, landmarks=True, group=groups, labels=True,
                     figure_scale=figure_scales, axis=True,
                     channel_index=fixed(0),
                     image_index=image_indices)
        else:
            # multi-channel image
            interact(view_image, landmarks=True, group=groups, labels=True,
                     figure_scale=figure_scales, axis=True,
                     channel_index=channel_indices, image_index=image_indices)
    else:
        #image without landmarks
        if len(channels) == 1:
            # multi-channel image
            interact(view_image, landmarks=fixed(False), group=fixed(None),
                     labels=fixed(False), figure_scale=figure_scales,
                     axis=True, channel_index=fixed(0),
                     image_index=image_indices)
        else:
            # multi-channel image
            interact(view_image, landmarks=fixed(False), group=fixed(None),
                     labels=fixed(False), figure_scale=figure_scales,
                     axis=True, channel_index=channel_indices,
                     image_index=image_indices)


def visualize_aam(aam, bounds=(-3.0, 3.0), with_labels=None,
                  without_labels=None, figure_size=(7, 7),
                  figure_scales=(0.5, 1.5), **kwargs):
    r"""
    Allows the dynamic visualization of an AAM by means of six simple sliders
    that control three weights associated to the first three shape and
    appearance components.

    Parameters
    -----------
    aam : :map:`AAM` or subclass
        The AAM to be displayed.

    bounds : (`float`, `float`), optional
        The minimum and maximum bounds, in std units, for the sliders
        controlling the shape and appearance components.

    with_labels : ``None`` or `str` or `list` of `str`, optional
        If not ``None``, only show the given label(s). Should **not** be
        used with the ``without_labels`` kwarg. If ``render_labels`` is
        ``False`` this kwarg is ignored.

    without_labels : ``None`` or `str` or `list` of `str`, optional
        If not ``None``, show all except the given label(s). Should **not**
        be used with the ``with_labels`` kwarg. If ``render_labels`` is
        ``False`` this kwarg is ignored.

    figure_size : (`int`, `int`), optional
        The size of the plotted figures.

    figure_scales : (`float`, `float`), optional
        The range of scales that can be optionally applied to the figure.

    kwargs : `dict`, optional
        Passed through to the viewer.
    """
    import matplotlib.pylab as plt

    # define relevant visualization options
    instance = aam.instance()
    groups = instance.landmarks.keys()
    levels = (0, aam.n_levels-1)
    n_channels = instance.n_channels
    if n_channels == 1:
        channels = [0]
    elif n_channels == 3:
        channels = range(n_channels) + [None, 'all']
        channel_indices = (0, n_channels + 1)
    else:
        channels = range(n_channels) + ['all']
        channel_indices = (0, n_channels)

    # define the aam visualization function
    def view_aam(landmarks, group, labels, figure_scale, axis, channel_index,
                 level, shape_pc1, shape_pc2, shape_pc3, tex_pc1, tex_pc2,
                 tex_pc3):
        # generate shape and texture weights
        shape_weights = [shape_pc1, shape_pc2, shape_pc3]
        appearance_weights = [tex_pc1, tex_pc2, tex_pc3]
        # generate aam instance
        aam_instance = aam.instance(level=level, shape_weights=shape_weights,
                                    appearance_weights=appearance_weights)
        if landmarks:
            # view instance with landmarks
            aam_instance.view_landmarks(
                group_label=group, render_labels=labels,
                with_labels=with_labels, without_labels=without_labels,
                channels=channels[channel_index], **kwargs)
        else:
            # view instance without landmarks
            aam_instance.view(channels=channels[channel_index], **kwargs)
        # set figure size
        plt.gcf().set_size_inches(figure_scale*np.asarray(figure_size))
        if not axis:
            # turn axis on/off
            plt.axis('off')

    # set the appropriate options in the aam visualization function
    if len(channels) == 1:
        # uni-channel aam
        interact(view_aam, landmarks=True, group=groups, labels=True,
                 figure_scale=figure_scales, axis=True, channel_index=fixed(0),
                 level=levels, shape_pc1=bounds, shape_pc2=bounds,
                 shape_pc3=bounds, tex_pc1=bounds, tex_pc2=bounds,
                 tex_pc3=bounds)
    else:
        # multi-channel aam
        interact(view_aam, landmarks=True, group=groups, labels=True,
                 figure_scale=figure_scales, axis=True,
                 channel_index=channel_indices, level=levels,
                 shape_pc1=bounds, shape_pc2=bounds, shape_pc3=bounds,
                 tex_pc1=bounds, tex_pc2=bounds, tex_pc3=bounds)


def browse_fitted_images(fitted_images, figure_size=(21, 21),
                         figure_scales=(0.5, 1.5), **kwargs):
    r"""
    Allows browsing through a list of fitted images using a simple slider.

    Parameters
    -----------
    fitted_images : `list` of :map:`Images` or subclass
        The list of fitted images to be displayed.

        .. note::
        Using Menpo's fitting framework, a list of fitted images is
        obtained by appending calls to the property ``fitted_image``
        on several fitting result objects:

            ``fitted_images = [fr.fitted_image for fr in fitting_results]``

        where ``fitting_results`` is a `list` of :map:`FittingResult`
        objects.

    figure_size : (`int`, `int`), optional
        The size of the plotted figures.

    figure_scales : (`float`, `float`), optional
        The range of scales that can be optionally applied to the figure.

    kwargs : `dict`, optional
        Passed through to the viewer.
    """
    import matplotlib.pylab as plt

    # define relevant visualization options
    image_indices = (0, len(fitted_images)-1)
    n_channels = fitted_images[0].n_channels
    if n_channels == 1:
        channels = [0]
    elif n_channels == 3:
        channels = range(n_channels) + [None, 'all']
        channel_indices = (0, n_channels + 1)
    else:
        channels = range(n_channels) + ['all']
        channel_indices = (0, n_channels)

    # define the visualization function
    def view_fitted_image(labels, figure_scale,  axis, channel_index,
                          image_index):
        image = fitted_images[image_index]
        if 'ground' in fitted_images[image_index].landmarks.keys():
            # visualize with ground truth
            plt.subplot(3, 1, 1)
            image.view_landmarks(group_label='initial', render_labels=labels,
                                 channels=channels[channel_index], **kwargs)
            plt.subplot(3, 1, 2)
            image.view_landmarks(group_label='final', render_labels=labels,
                                 channels=channels[channel_index], **kwargs)
            plt.subplot(3, 1, 3)
            image.view_landmarks(group_label='ground', render_labels=labels,
                                 channels=channels[channel_index], **kwargs)
        else:
            # visualize without ground truth
            plt.subplot(3, 1, 1)
            image.view_landmarks(group_label='initial', render_labels=labels,
                                 channels=channels[channel_index], **kwargs)
            plt.subplot(3, 1, 2)
            image.view_landmarks(group_label='final', render_labels=labels,
                                 channels=channels[channel_index], **kwargs)
        # set figure size
        plt.gcf().set_size_inches(figure_scale*np.asarray(figure_size))
        if not axis:
            # turn axis on/off
            plt.axis('off')

    # set the appropriate options in the visualization function
    if len(channels) == 1:
        # uni-channel image
        interact(view_fitted_image, labels=True, figure_scale=figure_scales,
                 axis=True, channel_index=fixed(0), image_index=image_indices)
    else:
        # multi-channel image
        interact(view_fitted_image, labels=True, figure_scale=figure_scales,
                 axis=True, channel_index=channel_indices,
                 image_index=image_indices)


def browse_iter_images(iter_images, figure_size=(7, 7),
                       figure_scales=(0.5, 1.5), **kwargs):
    r"""
    Allows browsing through the intermediate fitted shapes obtained
    at each iteration of a particular fitting procedure.

    Parameters
    -----------
    iter_images : `list` of :map:`Images` or subclass
        The list of iter images to be displayed.

        .. note::
        Using Menpo's fitting framework, a list of iter images is
        obtained by appending calls to the property ``iter_image``
        on several fitting result objects:

            ``iter_images = [fr.iter_image for fr in fitting_results]``

        where ``fitting_results`` is a `list` of :map:`FittingResult`
        objects.

    figure_size : (`int`, `int`), optional
        The size of the plotted figures.

    figure_scales : (`float`, `float`), optional
        The range of scales that can be optionally applied to the figure.

    kwargs : `dict`, optional
        Passed through to the viewer.
    """
    import matplotlib.pylab as plt

    # define relevant visualization options
    image_indices = (0, len(iter_images)-1)
    iter_indices = (0, len(iter_images[0].landmarks.keys())-1)
    n_channels = iter_images[0].n_channels
    if n_channels == 1:
        channels = [0]
    elif n_channels == 3:
        channels = range(n_channels) + [None, 'all']
        channel_indices = (0, n_channels + 1)
    else:
        channels = range(n_channels) + ['all']
        channel_indices = (0, n_channels)

    # define the visualization function
    def view_iter_image(labels, figure_scale,  axis, channel_index,
                        image_index, iteration):
        iter_images[image_index].view_landmarks(
            group_label='iter_'+str(iteration), render_labels=labels,
            channels=channels[channel_index], **kwargs)
        # set figure size
        plt.gcf().set_size_inches(figure_scale*np.asarray(figure_size))
        if not axis:
            # turn axis on/off
            plt.axis('off')

    # set the appropriate options in the visualization function
    if len(channels) == 1:
        # uni-channel image
        interact(view_iter_image, labels=True, figure_scale=figure_scales,
                 axis=True, channel_index=fixed(0), iteration=iter_indices,
                 image_index=image_indices)
    else:
        # multi-channel image
        interact(view_iter_image, labels=True, figure_scale=figure_scales,
                 axis=True, channel_index=channel_indices,
                 iteration=iter_indices, image_index=image_indices)


def plot_ced(final_errors, x_axis=None, initial_errors=None, title=None,
             x_label=None, y_label=None, legend=None, colors=None,
             markers=None, plot_size=(14, 7)):
    r"""
    Plots the Cumulative Error Distribution (CED) graph given a list of
    final fitting errors, or a list of lists containing final fitting errors.

    Parameters
    -----------
    final_errors : `list` of `floats` or `list` of `list` of `floats`
        The list of final errors or a list containing a list of
        final fitting errors.

        .. note::
        Using Menpo's fitting framework, the typical way to obtain a
        list of final errors is to append calls to the method
        ``final_error()`` on several fitting result objects:

            ``final_errors = [fr.final_error() for fr in fitting_results]``

        where ``fitting_results`` is a `list` of :map:`FittingResult`
        objects.

    x_axis : `list` of `float`, optional
        The x axis to be used.

    initial_errors : `list` of `floats`, optional.
        The list of initial fitting errors.

        .. note::
        Using Menpo's fitting framework, the typical way to obtain a
        list of initial errors is to append calls to the method
        ``initial_error()`` on several fitting result objects:

            ``initial_errors = [fr.initial_error() for fr in fitting_results]``

        where ``fitting_results`` is a `list` of :map:`FittingResult`
        objects.

    title : `str`, optional
        The figure title.

    x_label : `str`, optional
        The label associated to the x axis.

    y_label : `str`, optional
        The label associated to the y axis.

    legend : `str` or `list` of `str`, optional

    colors : `matplotlib color` or `list` of `matplotlib color`, optional
        The color of the line to be plotted.

    markers : `matplotlib marker` or `list` of `matplotlib marker`, optional
        The marker of the line to be plotted.

    figure_size : (`int`, `int`), optional
        The size of the plotted figures.

    figure_scales : (`float`, `float`), optional
        The range of scales that can be optionally applied to the figure.

    kwargs : `dict`, optional
        Passed through to the viewer.
    """
    import matplotlib.pylab as plt
    from menpo.fitmultilevel.functions import compute_cumulative_error

    if type(final_errors[0]) != list:
        # if final_errors is not a list of lists, turn it into list of lists
        final_errors = [final_errors]

    if title is None:
        title = 'Cumulative error distribution'
    if x_label is None:
        x_label = 'Error'
    if y_label is None:
        y_label = 'Proportion of images'

    if colors is None:
        # color are chosen at random
        colors = [np.random.random((3,)) for _ in range(len(final_errors))]
    elif len(colors) == 1 and len(final_errors) > 1:
        colors = [colors[0] for _ in range(len(final_errors))]
    elif len(colors) != len(final_errors):
        raise ValueError('colors must be...'.format())

    if markers is None:
        # markers default to square
        markers = ['s' for _ in range(len(final_errors))]
    elif len(markers) == 1 and len(final_errors) > 1:
        markers = [markers[0] for _ in range(len(final_errors))]
    elif len(markers) != len(final_errors):
        raise ValueError('markers must be...'.format())

    if legend is None:
        length = len(final_errors)
        if initial_errors:
            length += 1
        # number based legend
        legend = [str(j) for j in range(length)]
    else:
        if initial_errors:
            if len(legend) != len(final_errors)+1:
                raise ValueError('legend must be...'.format())
        else:
            if len(legend) != len(final_errors):
                raise ValueError('legend must be...'.format())

    if x_axis is None:
        # assume final_errors are computed using norm_me
        x_axis = np.arange(0, 0.101, 0.005)

    if initial_errors:
        # compute cumulative error for the initial errors
        initial_cumulative_error = compute_cumulative_error(initial_errors,
                                                            x_axis)

    # compute cumulative errors
    final_cumulative_errors = [compute_cumulative_error(e, x_axis)
                               for e in final_errors]

    def plot_graph(x_limit):

        if initial_errors:
            plt.plot(x_axis, initial_cumulative_error,
                     color='black',  marker='*')

        for fce, c, m in zip(final_cumulative_errors, colors, markers):
            plt.plot(x_axis, fce, color=c,  marker=m)

        plt.grid(True)
        ax = plt.gca()

        plt.title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.legend(legend, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.xlim([0, x_limit])

        plt.gcf().set_size_inches(plot_size)

    interact(plot_graph, x_limit=(0.0, x_axis[-1], 0.001))
