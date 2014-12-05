from collections import Sized
from .helpers import (figure_options, format_figure_options, channel_options,
                      format_channel_options, update_channel_options,
                      landmark_options, format_landmark_options,
                      update_landmark_options, info_print, format_info_print,
                      animation_options, format_animation_options,
                      save_figure_options, format_save_figure_options)
from IPython.html.widgets import (PopupWidget, ContainerWidget, TabWidget,
                                  RadioButtonsWidget)
from IPython.display import display, clear_output
from menpo.visualize.viewmatplotlib import MatplotlibSubplots
import numpy as np

# This glyph import is called frequently during visualisation, so we ensure
# that we only import it once
glyph = None


def _bullet_text(line):
    r"""
    Take a single line and wrap it in a latex bullet string.

    Parameters
    ----------
    line : `str`
        Single line of text

    Returns
    -------
    bullet_line : `str`
        Text wrapped in a bullet latex and verbatim.
    """
    return r'\bullet~\texttt{{{}}}'.format(line)


def _split_wall_of_text(wall):
    r""""
    Take a raw string of text and split it in to a list of lines split on the
    new line symbol. Discards any lines that are empty strings (strips white
    space from the left and right before discarding).

    Parameters
    ----------
    wall : `str`
        A multiline raw string.

    Returns
    -------
    lines : `list` of `str`
        List of strings not including empty strings
    """
    return filter(lambda x: x.strip(), wall.split('\n'))


def _join_bullets_as_latex_math(bullets):
    r""""
    Take a list of lines (wrapped in math mode latex commands) and re-join then
    using the latex new line command (wrapped in math mode).

    Parameters
    ----------
    bullets : `list` of `str`
        List of lines containing latex math commands

    Returns
    -------
    latex_string : `str`
        Single string wrapped in latex math mode '$...$'
    """
    return r'${}$'.format(r'\\'.join(bullets))


def _raw_info_string_to_latex(raw):
    r""""
    A raw string of multiple lines converted into a single math latex command
    containing multiple bullet points.

    Parameters
    ----------
    raw : `str`
        Multiline raw string

    Returns
    -------
    info_str : `str`
        Latex math mode string containing multiple bullet points. Each new line
        is converted to a bullet point.
    """
    lines = _split_wall_of_text(raw)
    bullets = map(lambda x: _bullet_text(x), lines)
    return _join_bullets_as_latex_math(bullets)


def visualize_images(images, figure_size=(7, 7), popup=False, **kwargs):
    r"""
    Widget that allows browsing through a list of images.

    Parameters
    -----------
    images : `list` of :map:`Image` or subclass
        The list of images to be displayed. Note that the images can have
        different attributes between them, i.e. different landmark groups and
        labels, different number of channels etc.

    figure_size : (`int`, `int`), optional
        The initial size of the plotted figures.

    popup : `boolean`, optional
        If enabled, the widget will appear as a popup window.

    kwargs : `dict`, optional
        Passed through to the viewer.
    """
    from menpo.image import MaskedImage
    import matplotlib.pyplot as plt

    # make sure that images is a list even with one image member
    if not isinstance(images, Sized):
        images = [images]

    # find number of images
    n_images = len(images)

    # find initial groups and labels that will be passed to the landmark options
    # widget creation
    first_has_landmarks = images[0].landmarks.n_groups != 0
    if first_has_landmarks:
        initial_groups_keys, initial_labels_keys = \
            _extract_groups_labels(images[0])
    else:
        initial_groups_keys = [' ']
        initial_labels_keys = [[' ']]

    # define plot function
    def plot_function(name, value):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        clear_output(wait=True)

        # get selected image number
        im = 0
        if n_images > 1:
            im = image_number_wid.selected_index

        # update info text widget
        image_has_landmarks = images[im].landmarks.n_groups != 0
        image_is_masked = isinstance(images[im], MaskedImage)
        update_info(images[im], image_is_masked, image_has_landmarks,
                    landmark_options_wid.group)

        # get the current figure id
        figure_id = save_figure_wid.figure_id

        # show image with selected options
        new_figure_id = _plot_figure(
            image=images[im], figure_id=figure_id, image_enabled=True,
            landmarks_enabled=landmark_options_wid.landmarks_enabled,
            image_is_masked=channel_options_wid.image_is_masked,
            masked_enabled=(channel_options_wid.masked_enabled and
                            image_is_masked),
            channels=channel_options_wid.channels,
            glyph_enabled=channel_options_wid.glyph_enabled,
            glyph_block_size=channel_options_wid.glyph_block_size,
            glyph_use_negative=channel_options_wid.glyph_use_negative,
            sum_enabled=channel_options_wid.sum_enabled,
            groups=[landmark_options_wid.group],
            with_labels=[landmark_options_wid.with_labels],
            groups_colours=dict(), subplots_enabled=False,
            subplots_titles=dict(), image_axes_mode=True,
            legend_enabled=landmark_options_wid.legend_enabled,
            numbering_enabled=landmark_options_wid.numbering_enabled,
            x_scale=figure_options_wid.x_scale,
            y_scale=figure_options_wid.x_scale,
            axes_visible=figure_options_wid.axes_visible,
            figure_size=figure_size, **kwargs)

        # save the current figure id
        save_figure_wid.figure_id = new_figure_id

    # define function that updates info text
    def update_info(image, image_is_masked, image_has_landmarks, group):
        # Prepare masked (or non-masked) string
        masked_str = 'Masked Image' if image_is_masked else 'Image'
        # Display masked pixels if image is masked
        masked_pixels_str = (r'{} masked pixels (attached mask {:.1%} true)'.
                             format(image.n_true_pixels(),
                                    image.mask.proportion_true())
                             if image_is_masked else '')
        # Display number of landmarks if image is landmarked
        landmarks_str = (r'{} landmark points.'.
                         format(image.landmarks[group].lms.n_points)
                         if image_has_landmarks else '')
        path_str = image.path if hasattr(image, 'path') else 'NO PATH'

        # Create info string
        info_txt = r"""
             {} of size {} with {} channel{}
             {}
             {}
             min={:.3f}, max={:.3f}
             {}
        """.format(masked_str, image._str_shape, image.n_channels,
                   's' * (image.n_channels > 1), path_str, masked_pixels_str,
                   image.pixels.min(), image.pixels.max(), landmarks_str)

        # update info widget text
        info_wid.children[1].value = _raw_info_string_to_latex(info_txt)

    # create options widgets
    channel_options_wid = channel_options(images[0].n_channels,
                                          isinstance(images[0], MaskedImage),
                                          plot_function,
                                          masked_default=False,
                                          toggle_show_default=True,
                                          toggle_show_visible=False)
    # The landmarks checkbox default value if the first image doesn't have
    # landmarks
    landmark_options_wid = landmark_options(
        initial_groups_keys, initial_labels_keys, plot_function,
        toggle_show_default=True, landmarks_default=first_has_landmarks,
        legend_default=True, numbering_default=False, toggle_show_visible=False)
    # if only a single image is passed in and it doesn't have landmarks, then
    # landmarks checkbox should be disabled
    landmark_options_wid.children[1].children[0].disabled = \
        not first_has_landmarks
    figure_options_wid = figure_options(plot_function, scale_default=1.,
                                        show_axes_default=False,
                                        toggle_show_default=True,
                                        figure_scale_bounds=(0.1, 2),
                                        figure_scale_step=0.1,
                                        figure_scale_visible=True,
                                        toggle_show_visible=False)
    info_wid = info_print(toggle_show_default=True,
                          toggle_show_visible=False)
    initial_figure_id = plt.figure()
    save_figure_wid = save_figure_options(initial_figure_id,
                                          toggle_show_default=True,
                                          toggle_show_visible=False)

    # define function that updates options' widgets state
    def update_widgets(name, value):
        # get new groups and labels, update landmark options and format them
        group_keys, labels_keys = _extract_groups_labels(images[value])
        update_landmark_options(landmark_options_wid, group_keys,
                                labels_keys, plot_function)
        format_landmark_options(landmark_options_wid, container_padding='6px',
                                container_margin='6px',
                                container_border='1px solid black',
                                toggle_button_font_weight='bold',
                                border_visible=False)

        # update channel options
        update_channel_options(channel_options_wid,
                               n_channels=images[value].n_channels,
                               image_is_masked=isinstance(images[value],
                                                          MaskedImage))

    # create final widget
    if n_images > 1:
        # image selection slider
        image_number_wid = animation_options(
            index_min_val=0, index_max_val=n_images-1,
            plot_function=plot_function, update_function=update_widgets,
            index_step=1, index_default=0,
            index_description='Image Number', index_minus_description='<',
            index_plus_description='>', index_style='buttons',
            index_text_editable=True, loop_default=True, interval_default=0.3,
            toggle_show_title='Image Options', toggle_show_default=True,
            toggle_show_visible=False)

        # final widget
        cont_wid = TabWidget(children=[info_wid, channel_options_wid,
                                       landmark_options_wid,
                                       figure_options_wid, save_figure_wid])
        wid = ContainerWidget(children=[image_number_wid, cont_wid])
        button_title = 'Images Menu'
    else:
        # final widget
        wid = TabWidget(children=[info_wid, channel_options_wid,
                                  landmark_options_wid, figure_options_wid,
                                  save_figure_wid])
        button_title = 'Image Menu'
    # create popup widget if asked
    if popup:
        wid = PopupWidget(children=[wid], button_text=button_title)

    # display final widget
    display(wid)

    # set final tab titles
    tab_titles = ['Image info', 'Channels options', 'Landmarks options',
                  'Figure options', 'Save figure']
    if popup:
        if n_images > 1:
            for (k, tl) in enumerate(tab_titles):
                wid.children[0].children[1].set_title(k, tl)
        else:
            for (k, tl) in enumerate(tab_titles):
                wid.children[0].set_title(k, tl)
    else:
        if n_images > 1:
            for (k, tl) in enumerate(tab_titles):
                wid.children[1].set_title(k, tl)
        else:
            for (k, tl) in enumerate(tab_titles):
                wid.set_title(k, tl)

    # align-start the image number widget and the rest
    if n_images > 1:
        wid.add_class('align-start')

    # format options' widgets
    if n_images > 1:
        format_animation_options(image_number_wid, index_text_width='0.5cm',
                                 container_padding='6px',
                                 container_margin='6px',
                                 container_border='1px solid black',
                                 toggle_button_font_weight='bold',
                                 border_visible=False)
    format_channel_options(channel_options_wid, container_padding='6px',
                           container_margin='6px',
                           container_border='1px solid black',
                           toggle_button_font_weight='bold',
                           border_visible=False)
    format_landmark_options(landmark_options_wid, container_padding='6px',
                            container_margin='6px',
                            container_border='1px solid black',
                            toggle_button_font_weight='bold',
                            border_visible=False)
    format_figure_options(figure_options_wid, container_padding='6px',
                          container_margin='6px',
                          container_border='1px solid black',
                          toggle_button_font_weight='bold',
                          border_visible=False)
    format_info_print(info_wid, font_size_in_pt='9pt', container_padding='6px',
                      container_margin='6px',
                      container_border='1px solid black',
                      toggle_button_font_weight='bold', border_visible=False)
    format_save_figure_options(save_figure_wid, container_padding='6px',
                               container_margin='6px',
                               container_border='1px solid black',
                               toggle_button_font_weight='bold',
                               tab_top_margin='0cm', border_visible=False)

    # update widgets' state for image number 0
    update_widgets('', 0)

    # Reset value to trigger initial visualization
    landmark_options_wid.children[1].children[1].value = False


def visualize_shapes(shapes, figure_size=(7, 7), popup=False, **kwargs):
    r"""
    Widget that allows browsing through a list of shapes.

    Parameters
    -----------
    shapes : `list` of :map:`LandmarkManager` or subclass
        The list of shapes to be displayed. Note that the shapes can have
        different attributes between them, i.e. different landmark groups and
        labels etc.

    figure_size : (`int`, `int`), optional
        The initial size of the plotted figures.

    popup : `boolean`, optional
        If enabled, the widget will appear as a popup window.

    kwargs : `dict`, optional
        Passed through to the viewer.
    """
    import matplotlib.pyplot as plt

    # make sure that shapes is a list even with one shape member
    if not isinstance(shapes, list):
        shapes = [shapes]

    # find number of shapes
    n_shapes = len(shapes)

    # find initial groups and labels that will be passed to the landmark options
    # widget creation
    first_has_landmarks = shapes[0].n_groups != 0
    if first_has_landmarks:
        initial_groups_keys, initial_labels_keys = \
            _exrtact_group_labels_landmarks(shapes[0])
    else:
        initial_groups_keys = [' ']
        initial_labels_keys = [[' ']]

    # Define plot function
    def plot_function(name, value):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        clear_output(wait=True)

        # get params
        s = 0
        if n_shapes > 1:
            s = image_number_wid.selected_index
        axis_mode = axes_mode_wid.value
        x_scale = figure_options_wid.x_scale
        y_scale = figure_options_wid.y_scale
        axes_visible = figure_options_wid.axes_visible

        # get the current figure id
        figure_id = plt.figure(save_figure_wid.figure_id.number)

        # plot
        if (landmark_options_wid.landmarks_enabled and
                    landmark_options_wid.group != ' '):
            # invert axis if image mode is enabled
            if axis_mode == 1:
                plt.gca().invert_yaxis()

            # plot
            shapes[s].view(
                group=landmark_options_wid.group,
                with_labels=landmark_options_wid.with_labels,
                image_view=axis_mode == 1,
                render_legend=landmark_options_wid.legend_enabled,
                render_numbering=landmark_options_wid.numbering_enabled,
                **kwargs)
            plt.hold(False)
            plt.gca().axis('equal')
            # set figure size
            plt.gcf().set_size_inches([x_scale, y_scale] *
                                      np.asarray(figure_size))
            # turn axis on/off
            if not axes_visible:
                plt.axis('off')
            plt.show()

        # save the current figure id
        save_figure_wid.figure_id = figure_id

        # info_wid string
        if landmark_options_wid.group != ' ':
            info_txt = r"""
                {} landmark points.
                Shape range: {:.1f} x {:.1f}.
                Shape centre: {:.1f} x {:.1f}.
                Shape norm is {:.2f}.
            """.format(shapes[s][landmark_options_wid.group][None].n_points,
                       shapes[s][landmark_options_wid.group][None].range()[0],
                       shapes[s][landmark_options_wid.group][None].range()[1],
                       shapes[s][landmark_options_wid.group][None].centre()[0],
                       shapes[s][landmark_options_wid.group][None].centre()[1],
                       shapes[s][landmark_options_wid.group][None].norm())
        else:
            info_txt = "There are no landmarks."

        info_wid.children[1].value = _raw_info_string_to_latex(info_txt)

    # create options widgets
    # The landmarks checkbox default value if the first image doesn't have
    # landmarks
    landmark_options_wid = landmark_options(
        initial_groups_keys, initial_labels_keys, plot_function,
        toggle_show_default=True, landmarks_default=first_has_landmarks,
        legend_default=True, numbering_default=False, toggle_show_visible=False)
    # if only a single image is passed in and it doesn't have landmarks, then
    # landmarks checkbox should be disabled
    landmark_options_wid.children[1].children[0].disabled = \
        not first_has_landmarks
    figure_options_wid = figure_options(plot_function, scale_default=1.,
                                        show_axes_default=False,
                                        toggle_show_default=True,
                                        toggle_show_visible=False)
    axes_mode_wid = RadioButtonsWidget(values={'Image': 1, 'Point cloud': 2},
                                       description='Axes mode:', value=1)
    axes_mode_wid.on_trait_change(plot_function, 'value')
    ch = list(figure_options_wid.children)
    ch.insert(3, axes_mode_wid)
    figure_options_wid.children = ch
    info_wid = info_print(toggle_show_default=True, toggle_show_visible=False)
    initial_figure_id = plt.figure()
    save_figure_wid = save_figure_options(initial_figure_id,
                                          toggle_show_default=True,
                                          toggle_show_visible=False)

    # define function that updates options' widgets state
    def update_widgets(name, value):
        # get new groups and labels, update landmark options and format them
        group_keys, labels_keys = _exrtact_group_labels_landmarks(shapes[value])
        update_landmark_options(landmark_options_wid, group_keys,
                                labels_keys, plot_function)
        format_landmark_options(landmark_options_wid, container_padding='6px',
                                container_margin='6px',
                                container_border='1px solid black',
                                toggle_button_font_weight='bold',
                                border_visible=False)

    # create final widget
    if n_shapes > 1:
        # image selection slider
        image_number_wid = animation_options(
            index_min_val=0, index_max_val=n_shapes-1,
            plot_function=plot_function, update_function=update_widgets,
            index_step=1, index_default=0,
            index_description='Shape Number', index_minus_description='<',
            index_plus_description='>', index_style='buttons',
            index_text_editable=True, loop_default=True, interval_default=0.3,
            toggle_show_title='Shape Options', toggle_show_default=True,
            toggle_show_visible=False)

        # final widget
        cont_wid = TabWidget(children=[info_wid, landmark_options_wid,
                                       figure_options_wid, save_figure_wid])
        wid = ContainerWidget(children=[image_number_wid, cont_wid])
        button_title = 'Shapes Menu'
    else:
        # final widget
        wid = TabWidget(children=[info_wid, landmark_options_wid,
                                  figure_options_wid, save_figure_wid])
        button_title = 'Shape Menu'
    # create popup widget if asked
    if popup:
        wid = PopupWidget(children=[wid], button_text=button_title)

    # display final widget
    display(wid)

    # set final tab titles
    tab_titles = ['Shape info', 'Landmarks options', 'Figure options',
                  'Save figure']
    if popup:
        if n_shapes > 1:
            for (k, tl) in enumerate(tab_titles):
                wid.children[0].children[1].set_title(k, tl)
        else:
            for (k, tl) in enumerate(tab_titles):
                wid.children[0].set_title(k, tl)
    else:
        if n_shapes > 1:
            for (k, tl) in enumerate(tab_titles):
                wid.children[1].set_title(k, tl)
        else:
            for (k, tl) in enumerate(tab_titles):
                wid.set_title(k, tl)

    # align-start the image number widget and the rest
    if n_shapes > 1:
        wid.add_class('align-start')

    # format options' widgets
    if n_shapes > 1:
        format_animation_options(image_number_wid, index_text_width='0.5cm',
                                 container_padding='6px',
                                 container_margin='6px',
                                 container_border='1px solid black',
                                 toggle_button_font_weight='bold',
                                 border_visible=False)
    format_landmark_options(landmark_options_wid, container_padding='6px',
                            container_margin='6px',
                            container_border='1px solid black',
                            toggle_button_font_weight='bold',
                            border_visible=False)
    format_figure_options(figure_options_wid, container_padding='6px',
                          container_margin='6px',
                          container_border='1px solid black',
                          toggle_button_font_weight='bold',
                          border_visible=False)
    format_info_print(info_wid, font_size_in_pt='9pt', container_padding='6px',
                      container_margin='6px',
                      container_border='1px solid black',
                      toggle_button_font_weight='bold', border_visible=False)
    format_save_figure_options(save_figure_wid, container_padding='6px',
                               container_margin='6px',
                               container_border='1px solid black',
                               toggle_button_font_weight='bold',
                               tab_top_margin='0cm', border_visible=False)

    # update widgets' state for image number 0
    update_widgets('', 0)

    # Reset value to trigger initial visualization
    landmark_options_wid.children[1].children[1].value = False


def _plot_figure(image, figure_id, image_enabled, landmarks_enabled,
                 image_is_masked, masked_enabled, channels, glyph_enabled,
                 glyph_block_size, glyph_use_negative, sum_enabled, groups,
                 with_labels, groups_colours, subplots_enabled, subplots_titles,
                 image_axes_mode, legend_enabled, numbering_enabled, x_scale,
                 y_scale, axes_visible, figure_size, **kwargs):
    r"""
    Helper function that plots an object given a set of selected options.

    Parameters
    -----------
    image : :map:`Image` or subclass
       The image to be displayed.

    figure_id : matplotlib.pyplot.Figure instance
        The handle of the figure to be saved.

    image_enabled : `bool`
        Flag that determines whether to display the image.

    landmarks_enabled : `bool`
        Flag that determines whether to display the landmarks.

    image_is_masked : `bool`
        If True, image is an instance of :map:`MaskedImage`.
        If False, image is an instance of :map:`Image`.

    masked_enabled : `bool`
        If True and the image is an instance of :map:`MaskedImage`, then only
        the masked pixels will be displayed.

    channels : `int` or `list` of `int`
        The image channels to be displayed.

    glyph_enabled : `bool`
        Defines whether to display the image as glyph or not.

    glyph_block_size : `int`
        The size of the glyph's blocks.

    glyph_use_negative : `bool`
        Whether to use the negative hist values.

    sum_enabled : `bool`
        If true, the image will be displayed as glyph with glyph_block_size=1,
        thus the sum of the image's selected channels.

    groups : `list` of `str`
        A list of the landmark groups to be displayed.

    with_labels : `list` of `list` of `str`
        The labels to be displayed for each group in groups.

    groups_colours : `dict` of `str`
        A dictionary that defines a colour for each of the groups, e.g.
        subplots_titles[groups[0]] = 'b'
        subplots_titles[groups[1]] = 'r'

    subplots_enabled : `bool`
        Flag that determines whether to plot all selected landmark groups in a
        single axes object or in subplots.

    subplots_titles : `dict` of `str`
        A dictionary that defines a subplot title for each of the groups, e.g.
        subplots_titles[groups[0]] = 'first group'
        subplots_titles[groups[1]] = 'second group'

    image_axes_mode : `bool`
        If True, then the point clouds are plotted with the axes in the image
        mode.

    legend_enabled : `bool`
        Flag that determines whether to show the legend for the landmarks.

    numbering_enabled : `bool`
        Flag that determines whether to show the numbering for the landmarks.

    x_scale : `float`
        The scale of x axis.

    y_scale : `float`
        The scale of y axis.

    axes_visible : `bool`
        If False, the figure's axes will be invisible.

    figure_size : (`int`, `int`)
        The size of the plotted figures.

    kwargs : `dict`, optional
        Passed through to the viewer.
    """
    import matplotlib.pyplot as plt

    global glyph
    if glyph is None:
        from menpo.visualize.image import glyph

    # select figure
    figure_id = plt.figure(figure_id.number)

    # plot
    if image_enabled:
        # image will be displayed
        if landmarks_enabled and len(groups) > 0:
            # there are selected landmark groups and they will be displayed
            if subplots_enabled:
                # calculate subplots structure
                subplots = MatplotlibSubplots()._subplot_layout(len(groups))
            # show image with landmarks
            for k, group in enumerate(groups):
                if subplots_enabled:
                    # create subplot
                    plt.subplot(subplots[0], subplots[1], k + 1)
                    if legend_enabled:
                        # set subplot's title
                        plt.title(subplots_titles[group])
                    if not axes_visible:
                        # turn axes on/off
                        plt.axis('off')
                if image_is_masked:
                    if glyph_enabled or sum_enabled:
                        # image, landmarks, masked, glyph
                        glyph(image, vectors_block_size=glyph_block_size,
                              use_negative=glyph_use_negative,
                              channels=channels).\
                            view_landmarks(group=group,
                                           with_labels=with_labels[k],
                                           render_legend=(legend_enabled
                                                          and not subplots_enabled),
                                           render_numbering=numbering_enabled,
                                           obj_view_kwargs={'masked':masked_enabled},
                                           lmark_view_kwargs=kwargs)
                    else:
                        # image, landmarks, masked, not glyph
                        image.view_landmarks(group=group,
                                             with_labels=with_labels[k],
                                             render_legend=(legend_enabled
                                                            and not subplots_enabled),
                                             render_numbering=numbering_enabled,
                                             obj_view_kwargs={'channels':channels,
                                                              'masked': masked_enabled},
                                             lmark_view_kwargs=kwargs)
                else:
                    if glyph_enabled or sum_enabled:
                        # image, landmarks, not masked, glyph
                        glyph(image, vectors_block_size=glyph_block_size,
                              use_negative=glyph_use_negative,
                              channels=channels).\
                            view_landmarks(group=group,
                                           with_labels=with_labels[k],
                                           render_legend=(legend_enabled
                                                          and not subplots_enabled),
                                           render_numbering=numbering_enabled,
                                           lmark_view_kwargs=kwargs)
                    else:
                        # image, landmarks, not masked, not glyph
                        image.view_landmarks(group=group,
                                             with_labels=with_labels[k],
                                             render_legend=(legend_enabled
                                                            and not subplots_enabled),
                                             render_numbering=numbering_enabled,
                                             obj_view_kwargs={'channels':channels},
                                             lmark_view_kwargs=kwargs)
        else:
            # either there are not any landmark groups selected or they won't
            # be displayed
            if image_is_masked:
                if glyph_enabled or sum_enabled:
                    # image, not landmarks, masked, glyph
                    glyph(image, vectors_block_size=glyph_block_size,
                          use_negative=glyph_use_negative, channels=channels).\
                        view(masked=masked_enabled, **kwargs)
                else:
                    # image, not landmarks, masked, not glyph
                    image.view(masked=masked_enabled, channels=channels,
                               **kwargs)
            else:
                if glyph_enabled or sum_enabled:
                    # image, not landmarks, not masked, glyph
                    glyph(image, vectors_block_size=glyph_block_size,
                          use_negative=glyph_use_negative, channels=channels).\
                        view(**kwargs)
                else:
                    # image, not landmarks, not masked, not glyph
                    image.view(channels=channels, **kwargs)
    else:
        # image won't be displayed
        if landmarks_enabled and len(groups) > 0:
            # there are selected landmark groups and they will be displayed
            if subplots_enabled:
                # calculate subplots structure
                subplots = MatplotlibSubplots()._subplot_layout(len(groups))
            # not image, landmarks
            for k, group in enumerate(groups):
                if subplots_enabled:
                    # create subplot
                    plt.subplot(subplots[0], subplots[1], k + 1)
                    # set axes to equal spacing
                    plt.gca().axis('equal')
                    if legend_enabled:
                        # set subplot's title
                        plt.title(subplots_titles[group])
                    if not axes_visible:
                        # turn axes on/off
                        plt.axis('off')
                    if image_axes_mode:
                        # set axes mode to image
                        plt.gca().invert_yaxis()
                image.landmarks[group].lms.view(image_view=image_axes_mode,
                                                colour_array=groups_colours[group],
                                                **kwargs)
            if not subplots_enabled:
                # set axes to equal spacing
                plt.gca().axis('equal')
                if legend_enabled:
                    # display legend on side
                    plt.legend(groups, loc='best')
                if image_axes_mode:
                    # set axes mode to image
                    plt.gca().invert_yaxis()

    # set figure size
    plt.gcf().set_size_inches([x_scale, y_scale] * np.asarray(figure_size))

    # turn axis on/off
    if not axes_visible:
        plt.axis('off')

    # show plot
    plt.show()

    return figure_id


def _plot_graph(figure_id, horizontal_axis_values, vertical_axis_values,
                plot_options_list, legend_visible, grid_visible, gridlinestyle,
                x_limit, y_limit, title, x_label, y_label, x_scale, y_scale,
                figure_size, axes_fontsize, labels_fontsize):
    r"""
    Helper function that plots a graph given a set of selected options. It
    supports plotting of multiple curves.

    Parameters
    -----------
    figure_id : matplotlib.pyplot.Figure instance
        The handle of the figure to be saved.

    horizontal_axis_values : `list` of `list`
        The horizontal axis values of each curve.

    vertical_axis_values : `list` of `list`
        The horizontal axis values of each curve.

    plot_options_list : `list` of `dict`
        The plot options for each curve. A typical plot options dictionary has
        the following elements:
            {'show_line':True,
             'linewidth':2,
             'linecolor':'r',
             'linestyle':'-',
             'show_marker':True,
             'markersize':20,
             'markerfacecolor':'r',
             'markeredgecolor':'b',
             'markerstyle':'o',
             'markeredgewidth':1,
             'legend_entry':'final errors'}

    legend_visible : `boolean`
        Flag that determines whether to show the legend of the plot.

    grid_visible : `boolean`
        Flag that determines whether to show the grid of the plot.

    gridlinestyle : `str`
        The style of the grid lines.

    x_limit : `float`
        The limit of the horizontal axis.

    y_limit : `float`
        The limit of the vertical axis.

    title : `str`
        The title of the figure.

    x_label : `str`
        The label of the horizontal axis.

    y_label : `str`
        The label of the vertical axis.

    x_scale : `float`
        The scale of horizontal axis.

    y_scale : `float`
        The scale of vertical axis.

    figure_size : (`int`, `int`)
        The size of the plotted figure.

    axes_fontsize : `float`
        The fontsize of the axes' markers.

    labels_fontsize : `float`
        The fontsize of the title, x_label, y_label and legend.
    """
    import matplotlib.pyplot as plt

    # select figure
    figure_id = plt.figure(figure_id.number)

    # plot all curves with the provided plot options
    for x_vals, y_vals, options in zip(horizontal_axis_values,
                                       vertical_axis_values, plot_options_list):
        # check if line is enabled
        linestyle = options['linestyle']
        if not options['show_line']:
            linestyle = 'None'
        # check if markers are enabled
        markerstyle = options['markerstyle']
        if not options['show_marker']:
            markerstyle = 'None'
        # plot
        plt.plot(x_vals, y_vals,
                 linestyle=linestyle,
                 marker=markerstyle,
                 color=options['linecolor'],
                 linewidth=options['linewidth'],
                 markersize=options['markersize'],
                 markerfacecolor=options['markerfacecolor'],
                 markeredgecolor=options['markeredgecolor'],
                 markeredgewidth=options['markeredgewidth'],
                 label=options['legend_entry'])
        plt.hold(True)

    # turn grid on/off
    if grid_visible:
        plt.grid(grid_visible, linestyle=gridlinestyle)

    # set title, x_label, y_label
    plt.title(title, fontsize=labels_fontsize)
    plt.gca().set_xlabel(x_label, fontsize=labels_fontsize)
    plt.gca().set_ylabel(y_label, fontsize=labels_fontsize)

    # set axes fontsize
    for l in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        l.set_fontsize(axes_fontsize)

    # create legend if asked
    if legend_visible:
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                   fontsize=labels_fontsize)

    # set figure size and aspect ratio
    size = [x_scale, y_scale] * np.asarray(figure_size)
    plt.gcf().set_size_inches(size)
    aspect = size[1] / size[0]
    aspect *= x_limit / y_limit
    plt.gca().set(adjustable='box', aspect=float(aspect))

    # set axes limits
    plt.xlim([0., x_limit])
    plt.ylim([0., y_limit])

    # show plot
    plt.show()

    return figure_id


def _plot_eigenvalues(figure_id, model, figure_size, x_scale, y_scale):
    r"""
    Helper function that plots a model's eigenvalues.

    Parameters
    -----------
    figure_id : matplotlib.pyplot.Figure instance
        The handle of the figure to be saved.

    model : :map:`PCAModel` or subclass
       The model to be used.

    figure_size : (`int`, `int`)
        The size of the plotted figures.

    x_scale : `float`
        The scale of x axis.

    y_scale : `float`
        The scale of y axis.
    """
    import matplotlib.pyplot as plt

    # select figure
    figure_id = plt.figure(figure_id.number)

    # plot eigenvalues ratio
    plt.subplot(211)
    plt.bar(range(len(model.eigenvalues_ratio())),
            model.eigenvalues_ratio())
    plt.ylabel('Variance Ratio')
    plt.xlabel('Component Number')
    plt.title('Variance Ratio per Eigenvector')
    plt.grid("on")

    # plot eigenvalues cumulative ratio
    plt.subplot(212)
    plt.bar(range(len(model.eigenvalues_cumulative_ratio())),
            model.eigenvalues_cumulative_ratio())
    plt.ylim((0., 1.))
    plt.ylabel('Cumulative Variance Ratio')
    plt.xlabel('Component Number')
    plt.title('Cumulative Variance Ratio')
    plt.grid("on")

    # set figure size
    #plt.gcf().tight_layout()
    plt.gcf().set_size_inches([x_scale, y_scale] * np.asarray(figure_size))

    plt.show()

    return figure_id


def _extract_groups_labels(image):
    r"""
    Function that extracts the groups and labels from an image's landmarks.

    Parameters
    ----------
    image : :map:`Image` or subclass
       The input image object.

    Returns
    -------
    group_keys : `list` of `str`
        The list of landmark groups found.

    labels_keys : `list` of `str`
        The list of lists of each landmark group's labels.
    """
    groups_keys, labels_keys = _exrtact_group_labels_landmarks(image.landmarks)
    return groups_keys, labels_keys


def _exrtact_group_labels_landmarks(landmark_manager):
    r"""
    Function that extracts the groups and labels from a landmark manager object.

    Parameters
    ----------
    landmark_manager : :map:`LandmarkManager` or subclass
       The input landmark manager object.

    Returns
    -------
    group_keys : `list` of `str`
        The list of landmark groups found.

    labels_keys : `list` of `str`
        The list of lists of each landmark group's labels.
    """
    groups_keys = landmark_manager.keys()
    if len(groups_keys) == 0:
        groups_keys = [' ']
        labels_keys = [[' ']]
    else:
        labels_keys = [landmark_manager[g].keys() for g in groups_keys]
    return groups_keys, labels_keys


def _check_n_parameters(n_params, n_levels, max_n_params):
    r"""
    Checks the maximum number of components per level either of the shape
    or the appearance model. It must be None or int or float or a list of
    those containing 1 or {n_levels} elements.
    """
    str_error = ("n_params must be None or 1 <= int <= max_n_params or "
                 "a list of those containing 1 or {} elements").format(n_levels)
    if not isinstance(n_params, list):
        n_params_list = [n_params] * n_levels
    elif len(n_params) == 1:
        n_params_list = [n_params[0]] * n_levels
    elif len(n_params) == n_levels:
        n_params_list = n_params
    else:
        raise ValueError(str_error)
    for i, comp in enumerate(n_params_list):
        if comp is None:
            n_params_list[i] = max_n_params[i]
        else:
            if isinstance(comp, int):
                if comp > max_n_params[i]:
                    n_params_list[i] = max_n_params[i]
            else:
                raise ValueError(str_error)
    return n_params_list
