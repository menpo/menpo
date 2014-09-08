from IPython.html.widgets import (FloatSliderWidget, ContainerWidget,
                                  IntSliderWidget, CheckboxWidget,
                                  ToggleButtonWidget, RadioButtonsWidget,
                                  IntTextWidget, DropdownWidget)


def figure_options(x_scale_default=1.5, y_scale_default=0.5,
                   coupled_default=False, show_axes_default=True,
                   toggle_show_default=True,
                   figure_scales_bounds=(0.1, 2), figure_scales_step=0.1,
                   figure_scales_visible=True, show_axes_visible=True):
    r"""
    Creates a small widget with Figure Options. Specifically, it has:
        1) Two sliders that control the horizontal and vertical scaling of the
           figure.
        2) A checkbox that couples/decouples the above sliders.
        3) A checkbox that controls the visibility of the figure's axes.
        4) A toggle button that controls the visibility of all the above, i.e.
           the figure options.
    The structure of the widgets is the following:
        figure_options_wid.children = [toggle_button, figure_scale,
                                       show_axes_checkbox]
        figure_scale.children = [X_scale_slider, Y_scale_slider,
                                 coupled_checkbox]
    To fix the alignment within this widget please refer to
    `format_figure_options()` function.

    Parameters
    ----------
    x_scale_default : `float`, optional
        The initial value of the horizontal axis scale.

    y_scale_default : `float`, optional
        The initial value of the vertical axis scale.

    coupled_default : `boolean`, optional
        The initial value of the coupled checkbox.

    show_axes_default : `boolean`, optional
        The initial value of the axes visibility checkbox.

    toggle_show_default : `boolean`, optional
        Defines whether the options will be visible upon construction.

    figure_scales_bounds : (`float`, `float`), optional
        The range of scales that can be optionally applied to the figure.

    figure_scales_step : `float`, optional
        The step of the scale sliders.

    figure_scales_visible : `boolean`, optional
        The visibility of the figure scales sliders.

    show_axes_visible : `boolean`, optional
        The visibility of the axes checkbox.
    """
    # Toggle button that controls options' visibility
    but = ToggleButtonWidget(description='Figure Options',
                             value=toggle_show_default)

    # Figure scale container
    X_scale = FloatSliderWidget(description='Figure size: X scale',
                                value=x_scale_default,
                                min=figure_scales_bounds[0],
                                max=figure_scales_bounds[1],
                                step=figure_scales_step)
    Y_scale = FloatSliderWidget(description='Y scale',
                                value=y_scale_default,
                                min=figure_scales_bounds[0],
                                max=figure_scales_bounds[1],
                                step=figure_scales_step,
                                disabled=coupled_default)
    coupled = CheckboxWidget(description='Coupled', value=coupled_default)
    figure_scale = ContainerWidget(children=[X_scale, Y_scale, coupled],
                                   visible=figure_scales_visible)

    # Show axes
    show_axes = CheckboxWidget(description='Show axes',
                               value=show_axes_default,
                               visible=show_axes_visible)

    # Widget container
    figure_options_wid = ContainerWidget(children=[but, figure_scale,
                                                   show_axes])

    # Toggle button function
    if figure_scales_visible and show_axes_visible:
        def show_options(name, value):
            if value:
                figure_scale.visible = True
                show_axes.visible = True
            else:
                figure_scale.visible = False
                show_axes.visible = False
        if toggle_show_default:
            figure_scale.visible = True
            show_axes.visible = True
        else:
            figure_scale.visible = False
            show_axes.visible = False
    elif figure_scales_visible and not show_axes_visible:
        def show_options(name, value):
            if value:
                figure_scale.visible = True
            else:
                figure_scale.visible = False
        if toggle_show_default:
            figure_scale.visible = True
        else:
            figure_scale.visible = False
    elif not figure_scales_visible and show_axes_visible:
        def show_options(name, value):
            if value:
                show_axes.visible = True
            else:
                show_axes.visible = False
        if toggle_show_default:
            show_axes.visible = True
        else:
            show_axes.visible = False
    else:
        def show_options(name, value):
            figure_scale.visible = False
            show_axes.visible = False
        figure_scale.visible = False
        show_axes.visible = False
    but.on_trait_change(show_options, 'value')

    # Coupled sliders function
    def coupled_sliders(name, value):
        if value:
            Y_scale.disabled = True
        else:
            Y_scale.disabled = False
    coupled.on_trait_change(coupled_sliders, 'value')

    def x_fun(name, old_value, value):
        if coupled.value:
            Y_scale.value += value - old_value
    X_scale.on_trait_change(x_fun, 'value')

    return figure_options_wid


def format_figure_options(figure_options_wid):
    r"""
    Functions that corrects the align (style format) of a given figure_options
    widget. Usage example:
        figure_options_wid = figure_options()
        display(figure_options_wid)
        format_figure_options(figure_options_wid)

    Parameters
    ----------
    figure_options_wid :
        The widget object generated by the `figure_options()` function.
    """
    figure_options_wid.children[1].remove_class('vbox')
    figure_options_wid.children[1].add_class('hbox')
    figure_options_wid.children[1].children[0].set_css('width', '3cm')
    figure_options_wid.children[1].children[1].set_css('width', '3cm')


def channel_options(n_channels, toggle_show_default=True):
    r"""
    Creates a widget with Channel Options. Specifically, it has:
        1) Two radiobuttons that select an options mode, depending on whether
           the user wants to visualize a "Single" or "Multiple" channels.
        2) If mode is "Single", the channel number is selected by one slider.
           If mode is "Multiple", the channel range is selected by two sliders.
        3) If mode is "Multiple", there is a checkbox option to visualize the
           sum of the channels.
        4) If mode is "Multiple", there is a checkbox option to visualize the
           glyph.
        5) The glyph option is accompanied by a block size text field and a
           checkbox that enables negative values visualization.
        6) A toggle button that controls the visibility of all the above, i.e.
           the channel options.
    The structure of the widgets is the following:
        channel_options_wid.children = [toggle_button, all_but_toggle]
        all_but_toggle.children = [mode_radiobuttons, all_but_radiobuttons]
        all_but_radiobuttons.children = [all_sliders, multiple_checkboxes]
        all_sliders.children = [first_slider, second_slider]
        multiple_checkboxes.children = [sum_checkbox, glyph_all]
        glyph_all.children = [glyph_checkbox, glyph_options]
        glyph_options.children = [block_size_text, use_negative_checkbox]

    To fix the alignment within this widget please refer to
    `format_channel_options()` function.

    Parameters
    ----------
    n_channels : `int`
        The number of channels.

    toggle_show_default : `boolean`, optional
        Defines whether the options will be visible upon construction.
    """
    # Create all necessary widgets
    but = ToggleButtonWidget(description='Channels Options',
                             value=toggle_show_default)
    mode = RadioButtonsWidget(values=["Single", "Multiple"], value="Single",
                              description='Mode:')
    mode.visible = toggle_show_default
    first_slider_wid = IntSliderWidget(min=0, max=n_channels-1, step=1,
                                       value=0, description='Channel')
    first_slider_wid.visible = toggle_show_default
    second_slider_wid = IntSliderWidget(min=1, max=n_channels-1, step=1,
                                        value=n_channels-1, description='To',
                                        visible=False)
    sum_wid = CheckboxWidget(value=False, description='Sum', visible=False)
    glyph_wid = CheckboxWidget(value=False, description='Glyph', visible=False)
    glyph_block_size = IntTextWidget(description='Block size', value='3',
                                     visible=False)
    glyph_use_negative = CheckboxWidget(description='Negative values',
                                        value=False, visible=False)

    # if single channel, disable multiple options
    if n_channels == 1:
        mode.value = "Single"
        mode.disabled = True
        first_slider_wid.disabled = True
        second_slider_wid.disabled = True
        sum_wid.disabled = True
        glyph_wid.disabled = True
        glyph_block_size.disabled = True
        glyph_use_negative.disabled = True

    # Group widgets
    glyph_options = ContainerWidget(children=[glyph_block_size,
                                              glyph_use_negative])
    glyph_all = ContainerWidget(children=[glyph_wid, glyph_options])
    multiple_checkboxes = ContainerWidget(children=[sum_wid, glyph_all])
    sliders = ContainerWidget(children=[first_slider_wid, second_slider_wid])
    all_but_radiobuttons = ContainerWidget(children=[sliders,
                                                     multiple_checkboxes])
    all_but_toggle = ContainerWidget(children=[mode, all_but_radiobuttons])
    channel_options_wid = ContainerWidget(children=[but, all_but_toggle])

    # Define mode visibility
    def mode_selection(name, value):
        if value == 'Single':
            first_slider_wid.description = 'Channel'
            first_slider_wid.min = 0
            first_slider_wid.max = n_channels-1
            second_slider_wid.visible = False
            sum_wid.visible = False
            sum_wid.value = False
            glyph_wid.visible = False
            glyph_wid.value = False
            glyph_options.children[0].visible = False
            glyph_options.children[1].visible = False
            glyph_options.children[0].value = '3'
            glyph_options.children[1].value = False
        else:
            first_slider_wid.description = 'From'
            first_slider_wid.min = 0
            first_slider_wid.max = n_channels-1
            second_slider_wid.min = 0
            second_slider_wid.max = n_channels-1
            if first_slider_wid.value < n_channels - 2:
                second_slider_wid.value = first_slider_wid.value + 1
            else:
                second_slider_wid.value = n_channels - 1
            second_slider_wid.visible = True
            sum_wid.visible = True
            sum_wid.value = False
            glyph_wid.visible = True
            glyph_wid.value = False
            glyph_options.children[0].visible = False
            glyph_options.children[1].visible = False
            glyph_options.children[0].value = '3'
            glyph_options.children[1].value = False
    mode.on_trait_change(mode_selection, 'value')

    # Define glyph visibility
    def glyph_options_visibility(name, value):
        if value:
            glyph_options.children[0].visible = True
            glyph_options.children[1].visible = True
            sum_wid.value = False
        else:
            glyph_options.children[0].visible = False
            glyph_options.children[1].visible = False
    glyph_wid.on_trait_change(glyph_options_visibility, 'value')

    # Define sum functionality
    def sum_fun(name, value):
        if value:
            glyph_wid.value = False
    sum_wid.on_trait_change(sum_fun, 'value')

    # Define multiple channels sliders functionality
    def first_slider_val(name, value):
        if mode.value == 'Multiple' and value >= second_slider_wid.value:
            first_slider_wid.value = second_slider_wid.value - 1

    def second_slider_val(name, value):
        if mode.value == 'Multiple' and value <= first_slider_wid.value:
            second_slider_wid.value = first_slider_wid.value + 1
        else:
            first_slider_wid.max = n_channels - 1
    first_slider_wid.on_trait_change(first_slider_val, 'value')
    second_slider_wid.on_trait_change(second_slider_val, 'value')

    # Toggle button function
    def toggle_image_options(name, value):
        if value:
            mode.visible = True
            if mode.value == 'Single':
                first_slider_wid.visible = True
            else:
                first_slider_wid.visible = True
                second_slider_wid.visible = True
                sum_wid.visible = True
                glyph_wid.visible = True
                glyph_options_visibility('', glyph_wid.value)
        else:
            mode.visible = False
            first_slider_wid.visible = False
            second_slider_wid.visible = False
            sum_wid.visible = False
            glyph_wid.visible = False
            glyph_options.children[0].visible = False
            glyph_options.children[1].visible = False
    but.on_trait_change(toggle_image_options, 'value')

    return channel_options_wid


def format_channel_options(channel_options_wid):
    r"""
    Functions that corrects the align (style format) of a given channel_options
    widget. Usage example:
        channel_options_wid = channel_options()
        display(channel_options_wid)
        format_channel_options(channel_options_wid)

    Parameters
    ----------
    channel_options_wid :
        The widget object generated by the `channel_options()` function.
    """
    # align glyph options
    channel_options_wid.children[1].children[1].children[1].children[1].children[1].remove_class('vbox')
    channel_options_wid.children[1].children[1].children[1].children[1].children[1].add_class('hbox')
    channel_options_wid.children[1].children[1].children[1].children[1].children[1].children[0].set_css('width', '0.8cm')

    # align sum and glyph checkboxes
    channel_options_wid.children[1].children[1].children[1].remove_class('vbox')
    channel_options_wid.children[1].children[1].children[1].add_class('hbox')

    # align radiobuttons with the rest
    channel_options_wid.children[1].remove_class('vbox')
    channel_options_wid.children[1].add_class('hbox')


def landmark_options(group_keys, toggle_show_default=True,
                     landmarks_default=True, labels_default=True):
    r"""
    Creates a widget with Landmark Options. Specifically, it has:
        1) A checkbox that controls the landmarks' visibility.
        2) A dropdown menu with the available landmark groups.
        3) A checkbox that controls the labels' visibility.
        4) A toggle button that controls the visibility of all the above, i.e.
           the landmark options.
    The structure of the widgets is the following:
        landmark_options_wid.children = [toggle_button, landmarks_checkbox,
                                         landmark_more]
        landmark_more.children = [group_dropdownmenu, labels_checkbox]
    To fix the alignment within this widget please refer to
    `format_landmark_options()` function.

    Parameters
    ----------
    group_keys : `list` of `str`
        A list of the available landmark groups.

    toggle_show_default : `boolean`, optional
        Defines whether the options will be visible upon construction.

    landmarks_default : `boolean`, optional
        The initial value of the landmarks visibility checkbox.

    labels_default : `boolean`, optional
        The initial value of the labels visibility checkbox.
    """
    # Toggle button that controls options' visibility
    but = ToggleButtonWidget(description='Landmark Options',
                             value=toggle_show_default)

    # Create widgets
    landmarks = CheckboxWidget(description='Show landmarks',
                               value=landmarks_default)
    labels = CheckboxWidget(description='Show labels', value=labels_default)
    group = DropdownWidget(values=group_keys, value=group_keys[0],
                           description='Group')

    # Group widgets
    partial_wid = ContainerWidget(children=[group, labels])
    landmark_options_wid = ContainerWidget(children=[but, landmarks,
                                                     partial_wid])

    # Initialize widgets values
    if not landmarks_default:
        labels.disabled = True
        group.disabled = True

    # Disability control
    def landmarks_fun(name, value):
        labels.disabled = not value
        group.disabled = not value
    landmarks.on_trait_change(landmarks_fun, 'value')

    # Toggle button function
    def show_options(name, value):
        landmarks.visible = value
        labels.visible = value
        group.visible = value
    show_options('', toggle_show_default)
    but.on_trait_change(show_options, 'value')

    return landmark_options_wid


def format_landmark_options(landmark_options_wid):
    r"""
    Functions that corrects the align (style format) of a given landmark_options
    widget. Usage example:
        landmark_options_wid = landmark_options()
        display(landmark_options_wid)
        format_landmark_options(landmark_options_wid)

    Parameters
    ----------
    landmark_options_wid :
        The widget object generated by the `landmark_options()` function.
    """
    landmark_options_wid.children[2].remove_class('vbox')
    landmark_options_wid.children[2].add_class('hbox')
