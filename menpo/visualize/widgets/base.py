from .helpers import (figure_options, format_figure_options, channel_options,
                      format_channel_options, update_channel_options,
                      landmark_options, format_landmark_options,
                      update_landmark_options, info_print, format_info_print,
                      model_parameters, format_model_parameters,
                      update_model_parameters, final_result_options,
                      format_final_result_options, iterations_result_options,
                      format_iterations_result_options, _extract_groups_labels)
from IPython.html.widgets import (interact, IntSliderWidget, PopupWidget,
                                  ContainerWidget, TabWidget,
                                  RadioButtonsWidget, CheckboxWidget)
from IPython.display import display, clear_output
import matplotlib.pylab as plt
import numpy as np

# This glyph import is called frequently during visualisation, so we ensure
# that we only import it once
glyph = None


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

    # make sure that images is a list even with one image member
    if not isinstance(images, list):
        images = [images]

    # find number of images
    n_images = len(images)

    # define plot function
    def plot_function(name, value):
        # get selected image number
        im = 0
        if n_images > 1:
            im = image_number_wid.value

        # show image with selected options
        _show_image(images[im],
                    channel_options_wid.channels,
                    channel_options_wid.glyph_enabled,
                    channel_options_wid.glyph_block_size,
                    channel_options_wid.glyph_use_negative,
                    channel_options_wid.sum_enabled,
                    landmark_options_wid.landmarks_enabled,
                    landmark_options_wid.legend_enabled,
                    landmark_options_wid.group,
                    landmark_options_wid.with_labels,
                    figure_options_wid.x_scale, figure_options_wid.y_scale,
                    figure_options_wid.axes_visible,
                    channel_options_wid.image_is_masked,
                    channel_options_wid.masked_enabled,
                    figure_size,
                    **kwargs)

        # update info text widget
        update_info(images[im],
                    channel_options_wid.image_is_masked,
                    landmark_options_wid.group)

    # define function that updates info text
    def update_info(image, image_is_masked, group):
        # prepare masked (or non-masked) str
        masked_str = "Image"
        if image_is_masked:
            masked_str = "Masked image"

        # prepare channels str
        ch_str = 'channels'
        if image.n_channels == 1:
            ch_str = 'channel'

        # create info str
        txt = "$\\bullet~\\texttt{" + \
              "{} of size {} with {} {}".format(masked_str, image._str_shape,
                                                image.n_channels, ch_str) + \
              ".}\\\\ \\bullet~\\texttt{" + \
              "{}".format(image.landmarks[group].lms.n_points) + \
              " landmark points.}\\\\ "
        if image_is_masked:
            txt += "\\bullet~\\texttt{" + \
                   "{} masked pixels.".format(image.n_true_pixels) + "}\\\\ "
        txt += "\\bullet~\\texttt{min=" + \
               "{0:.3f}".format(image.pixels.min()) + ", max=" + \
               "{0:.3f}".format(image.pixels.max()) + "}$"

        # update info widget text
        info_wid.children[1].value = txt

    # create options widgets
    channel_options_wid = channel_options(images[0].n_channels,
                                          isinstance(images[0], MaskedImage),
                                          plot_function,
                                          masked_default=False,
                                          toggle_show_default=True,
                                          toggle_show_visible=False)
    all_groups_keys, all_labels_keys = _extract_groups_labels(images[0])
    landmark_options_wid = landmark_options(all_groups_keys, all_labels_keys,
                                            plot_function,
                                            toggle_show_default=True,
                                            landmarks_default=True,
                                            legend_default=True,
                                            toggle_show_visible=False)
    figure_options_wid = figure_options(plot_function, scale_default=1.,
                                        show_axes_default=False,
                                        toggle_show_default=True,
                                        figure_scale_bounds=(0.1, 2),
                                        figure_scale_step=0.1,
                                        figure_scale_visible=True,
                                        toggle_show_visible=False)
    info_wid = info_print(toggle_show_default=True,
                          toggle_show_visible=False)

    # define function that updates options' widgets state
    def update_widgets(name, value):
        # get new groups and labels, update landmark options and format them
        group_keys, labels_keys = _extract_groups_labels(images[value])
        update_landmark_options(landmark_options_wid, group_keys, labels_keys,
                                plot_function)
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
        image_number_wid = IntSliderWidget(min=0, max=n_images-1, step=1,
                                           value=0, description='Image Number')
        image_number_wid.on_trait_change(update_widgets, 'value')
        image_number_wid.on_trait_change(plot_function, 'value')

        # final widget
        cont_wid = TabWidget(children=[info_wid, channel_options_wid,
                                       landmark_options_wid,
                                       figure_options_wid])
        wid = ContainerWidget(children=[image_number_wid, cont_wid])
        button_title = 'Images Menu'
    else:
        # final widget
        wid = TabWidget(children=[info_wid, channel_options_wid,
                                  landmark_options_wid, figure_options_wid])
        button_title = 'Image Menu'
    # create popup widget if asked
    if popup:
        wid = PopupWidget(children=[wid], button_text=button_title)

    # display final widget
    display(wid)

    # set final tab titles
    tab_titles = ['Image info', 'Channels options', 'Landmarks options',
                  'Figure options']
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

    # update widgets' state for image number 0
    update_widgets('', 0)

    # Reset value to trigger initial visualization
    landmark_options_wid.children[1].children[1].value = False


def visualize_shape_model(shape_models, n_parameters=5,
                          parameters_bounds=(-3.0, 3.0), figure_size=(7, 7),
                          mode='multiple', popup=False, tab=True, **kwargs):
    r"""
    Allows the dynamic visualization of a multilevel shape model.

    Parameters
    -----------
    shape_models : `list` of :map:`PCAModel` or subclass
        The multilevel shape model to be displayed.

    n_parameters : `int` or None, optional
        The number of principal components to be used for the parameters
        sliders.  If None, all the components will be employed.

    parameters_bounds : (`float`, `float`), optional
        The minimum and maximum bounds, in std units, for the sliders.

    figure_size : (`int`, `int`), optional
        The size of the plotted figures.

    mode : 'single' or 'multiple', optional
        If single, only a single slider is constructed along with a drop down
        menu.
        If multiple, a slider is constructed for each parameter.

    popup : `boolean`, optional
        If enabled, the widget will appear as a popup window.

    tab : `boolean`, optional
        If enabled, the widget will appear as a tab window.

    kwargs : `dict`, optional
        Passed through to the viewer.
    """
    from collections import OrderedDict

    n_levels = len(shape_models)

    # Check n_parameters
    max_n_params = max([sp.n_active_components for sp in shape_models])
    n_parameters = _check_parameters(n_parameters, max_n_params)

    # Define plot function
    def show_instance(name, value):
        # get params
        level = 0
        if n_levels > 1:
            level = level_wid.value
        def_mode = mode_wid.value
        axis_mode = axes_mode_wid.value
        parameters_values = model_parameters_wid.parameters_values
        x_scale = figure_options_wid.x_scale
        y_scale = figure_options_wid.y_scale
        axes_visible = figure_options_wid.axes_visible

        # compute weights
        weights = parameters_values * shape_models[level].eigenvalues[:len(parameters_values)] ** 0.5

        # clear current figure
        clear_output()

        # invert axis if image mode is enabled
        if axis_mode == 1:
            plt.gca().invert_yaxis()

        # compute and show instance
        if def_mode == 1:
            # Deformation mode
            # compute instance
            instance = shape_models[level].instance(weights)

            # plot
            if mean_wid.value:
                shape_models[level].mean.view(image_view=axis_mode == 1,
                                              colour_array='y', **kwargs)
                plt.hold = True
            instance.view(image_view=axis_mode == 1, **kwargs)

            # instance range
            tmp_range = instance.range()
        else:
            # Vectors mode
            # compute instance
            instance_lower = shape_models[level].instance([-p for p in weights])
            instance_upper = shape_models[level].instance(weights)

            # plot
            shape_models[level].mean.view(image_view=axis_mode == 1, **kwargs)
            plt.hold = True
            for p in range(shape_models[level].mean.n_points):
                xm = shape_models[level].mean.points[p, 0]
                ym = shape_models[level].mean.points[p, 1]
                xl = instance_lower.points[p, 0]
                yl = instance_lower.points[p, 1]
                xu = instance_upper.points[p, 0]
                yu = instance_upper.points[p, 1]
                if axis_mode == 1:
                    # image mode
                    plt.plot([ym, yl], [xm, xl], 'r-', lw=2)
                    plt.plot([ym, yu], [xm, xu], 'g-', lw=2)
                else:
                    # point cloud mode
                    plt.plot([xm, xl], [ym, yl], 'r-', lw=2)
                    plt.plot([xm, xu], [ym, yu], 'g-', lw=2)

            # instance range
            tmp_range = shape_models[level].mean.range()

        plt.hold = False
        plt.gca().axis('equal')
        # set figure size
        plt.gcf().set_size_inches([x_scale, y_scale] * np.asarray(figure_size))
        # turn axis on/off
        if not axes_visible:
            plt.axis('off')
        plt.show()

        # change info_wid info
        txt = "$\\bullet~\\texttt{Level: " + \
              "{}".format(level+1) + \
              " out of " + \
              "{}".format(n_levels) + \
              ".}\\\\ \\bullet~\\texttt{" + \
              "{}".format(shape_models[level].n_components) + \
              " components in total.}\\\\ \\bullet~\\texttt{" + \
              "{}".format(shape_models[level].n_active_components) + \
              " active components.}\\\\ \\bullet~\\texttt{" + \
              "{0:.1f}".format(shape_models[level].variance_ratio*100) + \
              "% variance kept.}\\\\ " \
              "\\bullet~\\texttt{Instance range: " + \
              "{0:.1f} x {1:.1f}".format(tmp_range[0], tmp_range[1]) + \
              ".}\\\\ \\bullet~\\texttt{" + \
              "{}".format(shape_models[level].mean.n_points) + \
              " landmark points, " + \
              "{}".format(shape_models[level].n_features) + \
              " features.}$"
        info_wid.children[1].value = txt

    # Plot eigenvalues function
    def plot_eigenvalues(name):
        # clear current figure
        clear_output()

        # plot eigenvalues ratio
        level = 0
        if n_levels > 1:
            level = level_wid.value
        plt.subplot(211)
        plt.bar(range(len(shape_models[level].eigenvalues_ratio)),
                shape_models[level].eigenvalues_ratio)
        plt.ylabel('Variance Ratio')
        plt.xlabel('Component Number')
        plt.title('Variance Ratio per Eigenvector')
        plt.grid("on")
        # plot eigenvalues cumulative ratio
        plt.subplot(212)
        plt.bar(range(len(shape_models[level].eigenvalues_cumulative_ratio)),
                shape_models[level].eigenvalues_cumulative_ratio)
        plt.ylim((0., 1.))
        plt.ylabel('Cumulative Variance Ratio')
        plt.xlabel('Component Number')
        plt.title('Cumulative Variance Ratio')
        plt.grid("on")
        # set figure size
        plt.gcf().tight_layout()
        x_scale = figure_options_wid.x_scale
        y_scale = figure_options_wid.y_scale
        plt.gcf().set_size_inches([x_scale, y_scale] * np.asarray(figure_size))

    # Create options widgets
    mode_dict = OrderedDict()
    mode_dict['Deformation'] = 1
    mode_dict['Vectors'] = 2
    mode_wid = RadioButtonsWidget(values=mode_dict, description='Mode:',
                                  value=1)
    mode_wid.on_trait_change(show_instance, 'value')
    mean_wid = CheckboxWidget(value=False, description='Show mean shape')
    mean_wid.on_trait_change(show_instance, 'value')

    def mean_visible(name, value):
        if value == 1:
            mean_wid.disabled = False
        else:
            mean_wid.disabled = True
            mean_wid.value = False
    mode_wid.on_trait_change(mean_visible, 'value')
    model_parameters_wid = model_parameters(
        n_parameters, plot_function=show_instance, params_str='param ',
        mode=mode, params_bounds=parameters_bounds, toggle_show_default=True,
        toggle_show_visible=False, plot_eig_visible=True,
        plot_eig_function=plot_eigenvalues)
    figure_options_wid = figure_options(show_instance, scale_default=1.,
                                        show_axes_default=True,
                                        toggle_show_default=tab,
                                        toggle_show_visible=not tab)
    axes_mode_wid = RadioButtonsWidget(values={'Image': 1, 'Point cloud': 2},
                                       description='Axes mode:', value=1)
    axes_mode_wid.on_trait_change(show_instance, 'value')
    ch = list(figure_options_wid.children)
    ch.insert(3, axes_mode_wid)
    figure_options_wid.children = ch
    info_wid = info_print(toggle_show_default=tab, toggle_show_visible=not tab)

    # Create final widget
    def update_wrt_level(name, value):
        n_params = min([n_parameters, shape_models[value].n_active_components])
        model_parameters_wid.parameters_values = model_parameters_wid.parameters_values[:n_params]
        update_model_parameters(model_parameters_wid, n_params,
                                params_str='param ')

    if n_levels > 1:
        radio_str = OrderedDict()
        for l in range(n_levels):
            if l == 0:
                radio_str["Level {} (low)".format(l)] = l
            elif l == n_levels - 1:
                radio_str["Level {} (high)".format(l)] = l
            else:
                radio_str["Level {}".format(l)] = l
        level_wid = RadioButtonsWidget(values=radio_str,
                                       description='Pyramid:', value=0)
        level_wid.on_trait_change(update_wrt_level, 'value')
        level_wid.on_trait_change(show_instance, 'value')
        radio_children = [level_wid, mode_wid, mean_wid]
    else:
        radio_children = [mode_wid, mean_wid]
    radio_wids = ContainerWidget(children=radio_children)
    tmp_wid = ContainerWidget(children=[radio_wids, model_parameters_wid])
    if tab:
        wid = TabWidget(children=[tmp_wid, figure_options_wid, info_wid])
    else:
        wid = ContainerWidget(children=[tmp_wid, figure_options_wid, info_wid])
    if popup:
        wid = PopupWidget(children=[wid], button_text='Shape Model Menu')

    # Display and format widget
    display(wid)
    update_wrt_level('', 0)
    if tab and popup:
        wid.children[0].set_title(0, 'Shape parameters')
        wid.children[0].set_title(1, 'Figure options')
        wid.children[0].set_title(2, 'Model info')
    elif tab and not popup:
        wid.set_title(0, 'Shape parameters')
        wid.set_title(1, 'Figure options')
        wid.set_title(2, 'Model info')
    tmp_wid.remove_class('vbox')
    tmp_wid.add_class('hbox')
    format_model_parameters(model_parameters_wid, container_padding='6px',
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

    # Reset value to enable initial visualization
    figure_options_wid.children[2].value = False


def visualize_appearance_model(appearance_models, n_parameters=5,
                               parameters_bounds=(-3.0, 3.0),
                               figure_size=(7, 7), mode='multiple',
                               popup=False, tab=True, **kwargs):
    r"""
    Allows the dynamic visualization of a multilevel appearance model.

    Parameters
    -----------
    appearance_models : `list` of :map:`PCAModel` or subclass
        The multilevel appearance model to be displayed.

    n_parameters : `int` or None, optional
        The number of principal components to be used for the parameters
        sliders. If None, all the components will be employed.

    parameters_bounds : (`float`, `float`), optional
        The minimum and maximum bounds, in std units, for the sliders.

    figure_size : (`int`, `int`), optional
        The size of the plotted figures.

    mode : 'single' or 'multiple', optional
        If single, only a single slider is constructed along with a drop down
        menu.
        If multiple, a slider is constructed for each parameter.

    popup : `boolean`, optional
        If enabled, the widget will appear as a popup window.

    tab : `boolean`, optional
        If enabled, the widget will appear as a tab window.

    kwargs : `dict`, optional
        Passed through to the viewer.
    """
    from collections import OrderedDict
    from menpo.visualize.image import glyph
    from menpo.image import MaskedImage

    n_levels = len(appearance_models)
    images_are_masked = isinstance(appearance_models[0].mean, MaskedImage)

    # Check n_parameters
    max_n_params = max([ap.n_active_components for ap in appearance_models])
    n_parameters = _check_parameters(n_parameters, max_n_params)

    # Define plot function
    def show_instance(name, value):
        # get params
        level = 0
        if n_levels > 1:
            level = level_wid.value
        parameters_values = model_parameters_wid.parameters_values
        channels = channel_options_wid.channels
        glyph_enabled = channel_options_wid.glyph_enabled
        glyph_block_size = channel_options_wid.glyph_block_size
        glyph_use_negative = channel_options_wid.glyph_use_negative
        sum_enabled = channel_options_wid.sum_enabled
        masked = channel_options_wid.masked
        landmarks_enabled = landmark_options_wid.landmarks_enabled
        legend_enabled = landmark_options_wid.legend_enabled
        group = landmark_options_wid.group
        with_labels = landmark_options_wid.with_labels
        x_scale = figure_options_wid.x_scale
        y_scale = figure_options_wid.y_scale
        axes_visible = figure_options_wid.axes_visible

        # compute instance
        weights = parameters_values * appearance_models[level].eigenvalues[:len(parameters_values)] ** 0.5
        instance = appearance_models[level].instance(weights)

        # clear current figure
        clear_output()

        # plot
        if images_are_masked:
            if glyph_enabled or sum_enabled:
                if landmarks_enabled:
                    glyph(instance, vectors_block_size=glyph_block_size,
                          use_negative=glyph_use_negative, channels=channels).\
                        view_landmarks(masked=masked, group_label=group,
                                       with_labels=with_labels,
                                       render_labels=legend_enabled, **kwargs)
                else:
                    glyph(instance, vectors_block_size=glyph_block_size,
                          use_negative=glyph_use_negative,
                          channels=channels).view(masked=masked, **kwargs)
            else:
                if landmarks_enabled:
                    instance.view_landmarks(masked=masked, group_label=group,
                                            with_labels=with_labels,
                                            render_labels=legend_enabled,
                                            channels=channels, **kwargs)
                else:
                    instance.view(masked=masked, channels=channels, **kwargs)
        else:
            if glyph_enabled or sum_enabled:
                if landmarks_enabled:
                    glyph(instance, vectors_block_size=glyph_block_size,
                          use_negative=glyph_use_negative, channels=channels).\
                        view_landmarks(group_label=group,
                                       with_labels=with_labels,
                                       render_labels=legend_enabled, **kwargs)
                else:
                    glyph(instance, vectors_block_size=glyph_block_size,
                          use_negative=glyph_use_negative,
                          channels=channels).view(**kwargs)
            else:
                if landmarks_enabled:
                    instance.view_landmarks(group_label=group,
                                            with_labels=with_labels,
                                            render_labels=legend_enabled,
                                            channels=channels, **kwargs)
                else:
                    instance.view(channels=channels, **kwargs)

        # set figure size
        plt.gcf().set_size_inches([x_scale, y_scale] * np.asarray(figure_size))
        # turn axis on/off
        if not axes_visible:
            plt.axis('off')
        plt.show()

        # change info_wid info
        ch_str = 'channels'
        if instance.n_channels == 1:
            ch_str = 'channel'
        txt = "$\\bullet~\\texttt{Level: " + \
              "{}".format(level+1) + \
              " out of " + \
              "{}".format(n_levels) + \
              ".}\\\\ \\bullet~\\texttt{" + \
              "{}".format(appearance_models[level].n_components) + \
              " components in total.}\\\\ \\bullet~\\texttt{" + \
              "{}".format(appearance_models[level].n_active_components) + \
              " active components.}\\\\ \\bullet~\\texttt{" + \
              "{0:.1f}".format(appearance_models[level].variance_ratio*100) + \
              "% variance kept.}\\\\ " \
              "\\bullet~\\texttt{Reference shape of size " + \
              instance._str_shape + \
              " with " + \
              "{} {}".format(instance.n_channels, ch_str) + \
              ".}\\\\ \\bullet~\\texttt{" + \
              "{}".format(appearance_models[level].n_features) + \
              " features.}\\\\ \\bullet~\\texttt{" + \
              "{}".format(instance.landmarks[group].lms.n_points) + \
              " landmark points.}\\\\ \\bullet~\\texttt{Instance: min=" + \
              "{0:.3f}".format(instance.pixels.min()) + \
              ", max=" + \
              "{0:.3f}".format(instance.pixels.max()) + \
              "}$"
        info_wid.children[1].value = txt

    # Plot eigenvalues function
    def plot_eigenvalues(name):
        # clear current figure
        clear_output()

        # plot eigenvalues ratio
        level = 0
        if n_levels > 1:
            level = level_wid.value
        plt.subplot(211)
        plt.bar(range(len(appearance_models[level].eigenvalues_ratio)),
                appearance_models[level].eigenvalues_ratio)
        plt.ylabel('Variance Ratio')
        plt.xlabel('Component Number')
        plt.title('Variance Ratio per Eigenvector')
        plt.grid("on")
        # plot eigenvalues cumulative ratio
        plt.subplot(212)
        plt.bar(range(len(appearance_models[level].eigenvalues_cumulative_ratio)),
                appearance_models[level].eigenvalues_cumulative_ratio)
        plt.ylim((0., 1.))
        plt.ylabel('Cumulative Variance Ratio')
        plt.xlabel('Component Number')
        plt.title('Cumulative Variance Ratio')
        plt.grid("on")
        # set figure size
        plt.gcf().tight_layout()
        x_scale = figure_options_wid.x_scale
        y_scale = figure_options_wid.y_scale
        plt.gcf().set_size_inches([x_scale, y_scale] * np.asarray(figure_size))

    # Create options widgets
    model_parameters_wid = model_parameters(
        n_parameters, plot_function=show_instance, params_str='param ',
        mode=mode, params_bounds=parameters_bounds, toggle_show_default=True,
        toggle_show_visible=False, plot_eig_visible=True,
        plot_eig_function=plot_eigenvalues)
    channel_options_wid = channel_options(appearance_models[0].mean.n_channels,
                                          show_instance, masked_default=True,
                                          masked_visible=images_are_masked,
                                          toggle_show_default=tab,
                                          toggle_show_visible=not tab)
    all_groups_keys = appearance_models[0].mean.landmarks.keys()
    all_labels_keys = [appearance_models[0].mean.landmarks[g].keys()
                       for g in all_groups_keys]
    landmark_options_wid = landmark_options(all_groups_keys, all_labels_keys,
                                            show_instance,
                                            toggle_show_default=tab,
                                            landmarks_default=True,
                                            legend_default=False,
                                            toggle_show_visible=not tab)
    figure_options_wid = figure_options(show_instance, scale_default=1.,
                                        show_axes_default=True,
                                        toggle_show_default=tab,
                                        toggle_show_visible=not tab)
    info_wid = info_print(toggle_show_default=tab, toggle_show_visible=not tab)

    # Create final widget
    # Create final widget
    def update_wrt_level(name, value):
        n_params = min([n_parameters, appearance_models[value].n_active_components])
        model_parameters_wid.parameters_values = model_parameters_wid.parameters_values[:n_params]
        update_model_parameters(model_parameters_wid, n_params,
                                params_str='param ')
    tmp_children = [model_parameters_wid]
    if n_levels > 1:
        radio_str = OrderedDict()
        for l in range(n_levels):
            if l == 0:
                radio_str["Level {} (low)".format(l)] = l
            elif l == n_levels - 1:
                radio_str["Level {} (high)".format(l)] = l
            else:
                radio_str["Level {}".format(l)] = l
        level_wid = RadioButtonsWidget(values=radio_str,
                                       description='Pyramid:', value=0)
        level_wid.on_trait_change(update_wrt_level, 'value')
        level_wid.on_trait_change(show_instance, 'value')
        tmp_children.insert(0, level_wid)
    tmp_wid = ContainerWidget(children=tmp_children)
    if tab:
        wid = TabWidget(children=[tmp_wid, channel_options_wid,
                                  landmark_options_wid, figure_options_wid,
                                  info_wid])
    else:
        wid = ContainerWidget(children=[tmp_wid, channel_options_wid,
                                        landmark_options_wid,
                                        figure_options_wid, info_wid])
    if popup:
        wid = PopupWidget(children=[wid], button_text='Appearance Model Menu')

    # Display and format widget
    display(wid)
    update_wrt_level('', 0)
    if tab and popup:
        wid.children[0].set_title(0, 'Appearance parameters')
        wid.children[0].set_title(1, 'Channels options')
        wid.children[0].set_title(2, 'Landmarks options')
        wid.children[0].set_title(3, 'Figure options')
        wid.children[0].set_title(4, 'Model info')
    elif tab and not popup:
        wid.set_title(0, 'Appearance parameters')
        wid.set_title(1, 'Channels options')
        wid.set_title(2, 'Landmarks options')
        wid.set_title(3, 'Figure options')
        wid.set_title(4, 'Model info')
    tmp_wid.remove_class('vbox')
    tmp_wid.add_class('hbox')
    format_model_parameters(model_parameters_wid, container_padding='6px',
                            container_margin='6px',
                            container_border='1px solid black',
                            toggle_button_font_weight='bold')
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

    # Reset value to enable initial visualization
    figure_options_wid.children[2].value = False


def visualize_aam(aam, n_shape_parameters=5, n_appearance_parameters=5,
                  parameters_bounds=(-3.0, 3.0), figure_size=(7, 7),
                  mode='multiple', popup=False, tab=True, **kwargs):
    r"""
    Allows the dynamic visualization of a multilevel AAM.

    Parameters
    -----------
    aam : :map:`AAM` or subclass
        The multilevel AAM to be displayed.

    n_shape_parameters : `int` or None, optional
        The number of shape principal components to be used for the parameters
        sliders.  If None, all the components will be employed.

    n_appearance_parameters : `int` or None, optional
        The number of appearance principal components to be used for the
        parameters sliders.  If None, all the components will be employed.

    parameters_bounds : (`float`, `float`), optional
        The minimum and maximum bounds, in std units, for the sliders.

    figure_size : (`int`, `int`), optional
        The size of the plotted figures.

    mode : 'single' or 'multiple', optional
        If single, only a single slider is constructed along with a drop down
        menu.
        If multiple, a slider is constructed for each parameter.

    popup : `boolean`, optional
        If enabled, the widget will appear as a popup window.

    tab : `boolean`, optional
        If enabled, the widget will appear as a tab window.

    kwargs : `dict`, optional
        Passed through to the viewer.
    """
    from collections import OrderedDict
    from menpo.visualize.image import glyph
    from menpo.image import MaskedImage

    n_levels = aam.n_levels
    images_are_masked = isinstance(aam.appearance_models[0].mean, MaskedImage)

    # Check n_shape_parameters and n_appearance_parameters
    max_n_shape = max([sp.n_active_components for sp in aam.shape_models])
    max_n_appearance = max([ap.n_active_components
                            for ap in aam.appearance_models])
    n_shape_parameters = _check_parameters(n_shape_parameters, max_n_shape)
    n_appearance_parameters = _check_parameters(n_appearance_parameters,
                                                max_n_appearance)

    # Define plot function
    def show_instance(name, value):
        # clear current figure
        clear_output()

        # get params
        level = 0
        if n_levels > 1:
            level = level_wid.value
        shape_weights = shape_model_parameters_wid.parameters_values
        appearance_weights = appearance_model_parameters_wid.parameters_values
        channels = channel_options_wid.channels
        glyph_enabled = channel_options_wid.glyph_enabled
        glyph_block_size = channel_options_wid.glyph_block_size
        glyph_use_negative = channel_options_wid.glyph_use_negative
        sum_enabled = channel_options_wid.sum_enabled
        masked = channel_options_wid.masked
        landmarks_enabled = landmark_options_wid.landmarks_enabled
        legend_enabled = landmark_options_wid.legend_enabled
        group = landmark_options_wid.group
        with_labels = landmark_options_wid.with_labels
        x_scale = figure_options_wid.x_scale
        y_scale = figure_options_wid.y_scale
        axes_visible = figure_options_wid.axes_visible

        # compute instance
        instance = aam.instance(level=level, shape_weights=shape_weights,
                                appearance_weights=appearance_weights)

        # plot
        if images_are_masked:
            if glyph_enabled or sum_enabled:
                if landmarks_enabled:
                    glyph(instance, vectors_block_size=glyph_block_size,
                          use_negative=glyph_use_negative, channels=channels).\
                        view_landmarks(masked=masked, group_label=group,
                                       with_labels=with_labels,
                                       render_labels=legend_enabled, **kwargs)
                else:
                    glyph(instance, vectors_block_size=glyph_block_size,
                          use_negative=glyph_use_negative,
                          channels=channels).view(masked=masked, **kwargs)
            else:
                if landmarks_enabled:
                    instance.view_landmarks(masked=masked, group_label='source',
                                            with_labels=['all'],
                                            render_labels=legend_enabled,
                                            channels=channels, **kwargs)
                else:
                    instance.view(masked=masked, channels=channels, **kwargs)
        else:
            if glyph_enabled or sum_enabled:
                if landmarks_enabled:
                    glyph(instance, vectors_block_size=glyph_block_size,
                          use_negative=glyph_use_negative, channels=channels).\
                        view_landmarks(group_label=group,
                                       with_labels=with_labels,
                                       render_labels=legend_enabled, **kwargs)
                else:
                    glyph(instance, vectors_block_size=glyph_block_size,
                          use_negative=glyph_use_negative,
                          channels=channels).view(**kwargs)
            else:
                if landmarks_enabled:
                    instance.view_landmarks(group_label=group,
                                            with_labels=with_labels,
                                            render_labels=legend_enabled,
                                            channels=channels, **kwargs)
                else:
                    instance.view(channels=channels, **kwargs)

        # set figure size
        plt.gcf().set_size_inches([x_scale, y_scale] * np.asarray(figure_size))
        # turn axis on/off
        if not axes_visible:
            plt.axis('off')
        plt.show()

        # Change info_wid info
        # features info
        from menpo.fitmultilevel.base import name_of_callable
        if aam.appearance_models[level].mean.n_channels == 1:
            if aam.pyramid_on_features:
                tmp_feat = "Feature is {} with 1 channel.".\
                    format(name_of_callable(aam.features))
            else:
                tmp_feat = "Feature is {} with 1 channel.".\
                    format(name_of_callable(aam.features[level]))
        else:
            if aam.pyramid_on_features:
                tmp_feat = "Feature is {} with {} channel.".\
                    format(name_of_callable(aam.features),
                           aam.appearance_models[level].mean.n_channels)
            else:
                tmp_feat = "Feature is {} with {} channel.".\
                    format(name_of_callable(aam.features[level]),
                           aam.appearance_models[level].mean.n_channels)
        # create final str
        if n_levels > 1:
            # shape models info
            if aam.scaled_shape_models:
                tmp_shape_models = "Each level has a scaled shape model " \
                                   "(reference frame)."
            else:
                tmp_shape_models = "Shape models (reference frames) are " \
                                   "not scaled."

            # pyramid info
            if aam.pyramid_on_features:
                tmp_pyramid = "Pyramid was applied on feature space."
            else:
                tmp_pyramid = "Features were extracted at each pyramid level."

            txt = "$\\bullet~\\texttt{" + \
                  "{}".format(aam.n_training_images) + \
                  " training images.}" + \
                  "\\\\ \\bullet~\\texttt{Warp using " + \
                  aam.transform.__name__ + \
                  " transform.} \\\\ \\bullet~\\texttt{Level " + \
                  "{}/{}".format(level+1, aam.n_levels) + \
                  " (downscale=" + \
                  "{0:.1f}".format(aam.downscale) + \
                  ").}" + \
                  "\\\\ \\bullet~\\texttt{" + \
                  tmp_shape_models + \
                  "}" + \
                  "\\\\ \\bullet~\\texttt{" + \
                  tmp_pyramid + \
                  "}\\\\ \\bullet~\\texttt{" + \
                  tmp_feat + \
                  "}\\\\ \\bullet~\\texttt{Reference frame of length " + \
                  "{} ({} x {}C, {} x {}C).".format(
                      aam.appearance_models[level].n_features,
                      aam.appearance_models[level].template_instance.n_true_pixels,
                      aam.appearance_models[level].mean.n_channels,
                      aam.appearance_models[level].template_instance._str_shape,
                      aam.appearance_models[level].mean.n_channels) + \
                  "}\\\\ \\bullet~\\texttt{" + \
                  "{0} shape components ({1:.2f}% of variance).".format(
                      aam.shape_models[level].n_components,
                      aam.shape_models[level].variance_ratio * 100) + \
                  "}\\\\ \\bullet~\\texttt{" + \
                  "{0} appearance components ({1:.2f}% of variance).".format(
                      aam.appearance_models[level].n_components,
                      aam.appearance_models[level].variance_ratio * 100) + \
                  "}\\\\ \\bullet~\\texttt{" + \
                  "{}".format(instance.landmarks[group].lms.n_points) + \
                  " landmark points.}\\\\ \\bullet~\\texttt{Instance: min=" + \
                  "{0:.3f}".format(instance.pixels.min()) + \
                  ", max=" + \
                  "{0:.3f}".format(instance.pixels.max()) + \
                  "}$"
        else:
            txt = "$\\bullet~\\texttt{" + \
                  "{}".format(aam.n_training_images) + \
                  " training images.}" + \
                  "\\\\ \\bullet~\\texttt{Warp using " + \
                  aam.transform.__name__ + \
                  " transform with '" + \
                  aam.interpolator + \
                  "' interpolation.}" + \
                  "\\\\ \\bullet~\\texttt{" + \
                  tmp_feat + \
                  "}\\\\ \\bullet~\\texttt{Reference frame of length " + \
                  "{} ({} x {}C, {} x {}C).".format(
                      aam.appearance_models[level].n_features,
                      aam.appearance_models[level].template_instance.n_true_pixels,
                      aam.appearance_models[level].mean.n_channels,
                      aam.appearance_models[level].template_instance._str_shape,
                      aam.appearance_models[level].mean.n_channels) + \
                  "}\\\\ \\bullet~\\texttt{" + \
                  "{0} shape components ({1:.2f}% of variance).".format(
                      aam.shape_models[level].n_components,
                      aam.shape_models[level].variance_ratio * 100) + \
                  "}\\\\ \\bullet~\\texttt{" + \
                  "{0} appearance components ({1:.2f}% of variance).".format(
                      aam.appearance_models[level].n_components,
                      aam.appearance_models[level].variance_ratio * 100) + \
                  "}\\\\ \\bullet~\\texttt{" + \
                  "{}".format(instance.landmarks[group].lms.n_points) + \
                  " landmark points.}\\\\ \\bullet~\\texttt{Instance: min=" + \
                  "{0:.3f}".format(instance.pixels.min()) + \
                  ", max=" + \
                  "{0:.3f}".format(instance.pixels.max()) + \
                  "}$"
        info_wid.children[1].value = txt

    # Plot eigenvalues function
    def plot_shape_eigenvalues(name):
        # clear current figure
        clear_output()

        # plot eigenvalues ratio
        level = 0
        if n_levels > 1:
            level = level_wid.value
        plt.subplot(211)
        plt.bar(range(len(aam.shape_models[level].eigenvalues_ratio)),
                aam.shape_models[level].eigenvalues_ratio)
        plt.ylabel('Variance Ratio')
        plt.xlabel('Component Number')
        plt.grid("on")
        # plot eigenvalues cumulative ratio
        plt.subplot(212)
        plt.bar(range(len(aam.shape_models[level].eigenvalues_cumulative_ratio)),
                aam.shape_models[level].eigenvalues_cumulative_ratio)
        plt.ylabel('Cumulative Variance Ratio')
        plt.xlabel('Component Number')
        plt.grid("on")
        # set figure size
        x_scale = figure_options_wid.x_scale
        y_scale = figure_options_wid.y_scale
        plt.gcf().set_size_inches([x_scale, y_scale] * np.asarray(figure_size))

    def plot_appearance_eigenvalues(name):
        # clear current figure
        clear_output()

        # plot eigenvalues ratio
        level = 0
        if n_levels > 1:
            level = level_wid.value
        plt.subplot(211)
        plt.bar(range(len(aam.appearance_models[level].eigenvalues_ratio)),
                aam.appearance_models[level].eigenvalues_ratio)
        plt.ylabel('Variance Ratio')
        plt.xlabel('Component Number')
        plt.grid("on")
        # plot eigenvalues cumulative ratio
        plt.subplot(212)
        plt.bar(range(len(aam.appearance_models[level].eigenvalues_cumulative_ratio)),
                aam.appearance_models[level].eigenvalues_cumulative_ratio)
        plt.ylabel('Cumulative Variance Ratio')
        plt.xlabel('Component Number')
        plt.grid("on")
        # set figure size
        x_scale = figure_options_wid.x_scale
        y_scale = figure_options_wid.y_scale
        plt.gcf().set_size_inches([x_scale, y_scale] * np.asarray(figure_size))

    # Create options widgets
    shape_model_parameters_wid = model_parameters(
        n_shape_parameters, plot_function=show_instance,
        params_str='param ', mode=mode, params_bounds=parameters_bounds,
        toggle_show_default=False, toggle_show_visible=True,
        toggle_show_name='Shape Parameters', plot_eig_visible=True,
        plot_eig_function=plot_shape_eigenvalues)
    appearance_model_parameters_wid = model_parameters(
        n_appearance_parameters, plot_function=show_instance,
        params_str='param ', mode=mode,
        params_bounds=parameters_bounds, toggle_show_default=False,
        toggle_show_visible=True, toggle_show_name='Appearance Parameters',
        plot_eig_visible=True, plot_eig_function=plot_appearance_eigenvalues)
    channel_options_wid = channel_options(
        aam.appearance_models[0].mean.n_channels, show_instance,
        masked_default=True, masked_visible=images_are_masked,
        toggle_show_default=tab, toggle_show_visible=not tab)
    all_groups_keys = aam.appearance_models[0].mean.landmarks.keys()
    all_labels_keys = [aam.appearance_models[0].mean.landmarks[g].keys()
                       for g in all_groups_keys]
    landmark_options_wid = landmark_options(all_groups_keys, all_labels_keys,
                                            show_instance,
                                            toggle_show_default=tab,
                                            landmarks_default=True,
                                            legend_default=False,
                                            toggle_show_visible=not tab)
    figure_options_wid = figure_options(show_instance, scale_default=1.,
                                        show_axes_default=True,
                                        toggle_show_default=tab,
                                        toggle_show_visible=not tab)
    info_wid = info_print(toggle_show_default=tab, toggle_show_visible=not tab)

    # Create final widget
    def update_wrt_level(name, value):
        n_shape_params = min([n_shape_parameters, aam.shape_models[value].n_active_components])
        shape_model_parameters_wid.parameters_values = shape_model_parameters_wid.parameters_values[:n_shape_params]
        n_appearance_params = min([n_appearance_parameters, aam.appearance_models[value].n_active_components])
        appearance_model_parameters_wid.parameters_values = appearance_model_parameters_wid.parameters_values[:n_appearance_params]
        update_model_parameters(shape_model_parameters_wid, n_shape_params,
                                params_str='param ')
        update_model_parameters(appearance_model_parameters_wid,
                                n_appearance_params, params_str='param ')

    model_parameters_wid = ContainerWidget(
        children=[shape_model_parameters_wid,
                  appearance_model_parameters_wid])
    tmp_children = [model_parameters_wid]
    if n_levels > 1:
        radio_str = OrderedDict()
        for l in range(n_levels):
            if l == 0:
                radio_str["Level {} (low)".format(l)] = l
            elif l == n_levels - 1:
                radio_str["Level {} (high)".format(l)] = l
            else:
                radio_str["Level {}".format(l)] = l
        level_wid = RadioButtonsWidget(values=radio_str,
                                       description='Pyramid:', value=0)
        level_wid.on_trait_change(update_wrt_level, 'value')
        level_wid.on_trait_change(show_instance, 'value')
        tmp_children.insert(0, level_wid)
    tmp_wid = ContainerWidget(children=tmp_children)
    if tab:
        wid = TabWidget(children=[tmp_wid, channel_options_wid,
                                  landmark_options_wid, figure_options_wid,
                                  info_wid])
    else:
        wid = ContainerWidget(children=[tmp_wid, channel_options_wid,
                                        landmark_options_wid,
                                        figure_options_wid, info_wid])
    if popup:
        wid = PopupWidget(children=[wid], button_text='AAM Menu')

    # Display and format widget
    display(wid)
    update_wrt_level('', 0)
    if tab and popup:
        wid.children[0].set_title(0, 'AAM parameters')
        wid.children[0].set_title(1, 'Channels options')
        wid.children[0].set_title(2, 'Landmarks options')
        wid.children[0].set_title(3, 'Figure options')
        wid.children[0].set_title(4, 'Model info')
    elif tab and not popup:
        wid.set_title(0, 'AAM parameters')
        wid.set_title(1, 'Channels options')
        wid.set_title(2, 'Landmarks options')
        wid.set_title(3, 'Figure options')
        wid.set_title(4, 'Model info')
    if n_levels > 1:
        tmp_wid.remove_class('vbox')
        tmp_wid.add_class('hbox')
    format_model_parameters(shape_model_parameters_wid,
                            container_padding='6px', container_margin='6px',
                            container_border='1px solid black',
                            toggle_button_font_weight='bold')
    format_model_parameters(appearance_model_parameters_wid,
                            container_padding='6px', container_margin='6px',
                            container_border='1px solid black',
                            toggle_button_font_weight='bold')
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

    # Reset value to enable initial visualization
    figure_options_wid.children[2].value = False


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


def _show_image(image, channels, glyph_enabled, glyph_block_size,
                glyph_use_negative, sum_enabled, landmarks_enabled,
                legend_enabled, group, with_labels, x_scale, y_scale,
                axes_visible, image_is_masked, masked_enabled, figure_size,
                **kwargs):
    r"""
    Helper function that plots and image given a set of selected options.

    Parameters
    -----------
    image : :map:`Image` or subclass
       The image to be displayed.

    channels : `int` or `list` of `int`
        The image channels to be displayed.

    glyph_enabled : `boolean`
        Defines whether to display the image as glyph or not.

    glyph_block_size : `int`
        The size of the glyph's blocks.

    glyph_use_negative : `boolean`
        Whether to use the negative hist values.

    sum_enabled : `boolean`
        If true, the image will be displayed as glyph with glyph_block_size=1,
        thus the sum of the image's selected channels.

    landmarks_enabled : `boolean`
        Whether to also display the landmarks on top of the image.

    legend_enabled : `boolean`
        Whether to show the legend for the landmarks. If True, it also prints
        the landmark points' numbers.

    group : `str`
        The landmark group to be displayed.

    with_labels : `list` of `str`
        The group's labels to be displayed.

    x_scale : `float`
        The scale of x axis.

    y_scale : `float`
        The scale of y axis.

    axes_visible : `boolean`
        If False, the figure's axes will be invisible.

    image_is_masked : `boolean`
        If True, image is an instance of :map:`MaskedImage`.
        If False, image is an instance of :map:`Image`.

    masked_enabled : `boolean`
        If True and the image is an instance of :map:`MaskedImage`, then only
        the masked pixels will be displayed.

    figure_size : (`int`, `int`)
        The size of the plotted figures.

    kwargs : `dict`, optional
        Passed through to the viewer.
    """
    global glyph
    if glyph is None:
        from menpo.visualize.image import glyph
    # clear current figure
    clear_output()

    # plot
    if image_is_masked:
        if glyph_enabled or sum_enabled:
            if landmarks_enabled:
                # image is masked, glyph and has landmarks
                glyph(image, vectors_block_size=glyph_block_size,
                      use_negative=glyph_use_negative, channels=channels).\
                    view_landmarks(masked=masked_enabled, group_label=group,
                                   with_labels=with_labels,
                                   render_labels=legend_enabled, **kwargs)
            else:
                # image is masked, glyph and doesn't have landmarks
                glyph(image, vectors_block_size=glyph_block_size,
                      use_negative=glyph_use_negative, channels=channels).\
                    view(masked=masked_enabled, **kwargs)
        else:
            if landmarks_enabled:
                # image is masked, non-glyph and has landmarks
                image.view_landmarks(masked=masked_enabled, group_label=group,
                                     with_labels=with_labels,
                                     render_labels=legend_enabled,
                                     channels=channels, **kwargs)
            else:
                # image is masked, non-glyph and doesn't have landmarks
                image.view(masked=masked_enabled, channels=channels, **kwargs)
    else:
        if glyph_enabled or sum_enabled:
            if landmarks_enabled:
                # image is non-masked, glyph and has landmarks
                glyph(image, vectors_block_size=glyph_block_size,
                      use_negative=glyph_use_negative, channels=channels).\
                    view_landmarks(group_label=group,
                                   with_labels=with_labels,
                                   render_labels=legend_enabled, **kwargs)
            else:
                # image is non-masked, glyph and doesn't have landmarks
                glyph(image, vectors_block_size=glyph_block_size,
                      use_negative=glyph_use_negative, channels=channels).\
                    view(**kwargs)
        else:
            if landmarks_enabled:
                # image is non-masked, non-glyph and has landmarks
                image.view_landmarks(group_label=group, with_labels=with_labels,
                                     render_labels=legend_enabled,
                                     channels=channels, **kwargs)
            else:
                # image is non-masked, non-glyph and doesn't have landmarks
                image.view(channels=channels, **kwargs)

    # set figure size
    plt.gcf().set_size_inches([x_scale, y_scale] * np.asarray(figure_size))

    # turn axis on/off
    if not axes_visible:
        plt.axis('off')

    # show plot
    plt.show()


def _check_parameters(n_params, max_n_params):
    if n_params is None:
        n_params = max_n_params
    elif n_params > max_n_params:
        raise ValueError("too many parameters asked "
                         "(maximum {})".format(max_n_params))
    return n_params
