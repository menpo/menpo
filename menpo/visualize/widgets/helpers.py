from IPython.html.widgets import (FloatSliderWidget, ContainerWidget,
                                  IntSliderWidget, CheckboxWidget,
                                  ToggleButtonWidget, RadioButtonsWidget,
                                  IntTextWidget, DropdownWidget, LatexWidget,
                                  ButtonWidget)


def figure_options(x_scale_default=1., y_scale_default=1.,
                   coupled_default=False, show_axes_default=True,
                   toggle_show_default=True,
                   figure_scales_bounds=(0.1, 2), figure_scales_step=0.1,
                   figure_scales_visible=True, show_axes_visible=True,
                   toggle_show_visible=True):
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

    toggle_show_visible : `boolean`, optional
        The visibility of the toggle button.
    """
    # Toggle button that controls options' visibility
    but = ToggleButtonWidget(description='Figure Options',
                             value=toggle_show_default,
                             visible=toggle_show_visible)

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


def format_figure_options(figure_options_wid, container_padding='6px',
                          container_margin='6px',
                          container_border='1px solid black',
                          toggle_button_font_weight='bold'):
    r"""
    Function that corrects the align (style format) of a given figure_options
    widget. Usage example:
        figure_options_wid = figure_options()
        display(figure_options_wid)
        format_figure_options(figure_options_wid)

    Parameters
    ----------
    figure_options_wid :
        The widget object generated by the `figure_options()` function.

    container_padding : `str`, optional
        The padding around the widget, e.g. '6px'

    container_margin : `str`, optional
        The margin around the widget, e.g. '6px'

    container_border : `str`, optional
        The border around the widget, e.g. '1px solid black'

    toggle_button_font_weight : `str`
        The font weight of the toggle button, e.g. 'bold'
    """
    # align figure scale sliders and checkbox
    figure_options_wid.children[1].remove_class('vbox')
    figure_options_wid.children[1].add_class('hbox')

    # fix figure scale sliders width
    figure_options_wid.children[1].children[0].set_css('width', '3cm')
    figure_options_wid.children[1].children[1].set_css('width', '3cm')

    # set toggle button font bold
    figure_options_wid.children[0].set_css('font-weight',
                                           toggle_button_font_weight)

    # margin and border around container widget
    figure_options_wid.set_css('padding', container_padding)
    figure_options_wid.set_css('margin', container_margin)
    figure_options_wid.set_css('border', container_border)


def channel_options(n_channels, toggle_show_default=True,
                    toggle_show_visible=True):
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

    toggle_show_visible : `boolean`, optional
        The visibility of the toggle button.
    """
    # Create all necessary widgets
    but = ToggleButtonWidget(description='Channels Options',
                             value=toggle_show_default,
                             visible=toggle_show_visible)
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


def format_channel_options(channel_options_wid, container_padding='6px',
                           container_margin='6px',
                           container_border='1px solid black',
                           toggle_button_font_weight='bold'):
    r"""
    Function that corrects the align (style format) of a given channel_options
    widget. Usage example:
        channel_options_wid = channel_options()
        display(channel_options_wid)
        format_channel_options(channel_options_wid)

    Parameters
    ----------
    channel_options_wid :
        The widget object generated by the `channel_options()` function.

    container_padding : `str`, optional
        The padding around the widget, e.g. '6px'

    container_margin : `str`, optional
        The margin around the widget, e.g. '6px'

    container_border : `str`, optional
        The border around the widget, e.g. '1px solid black'

    toggle_button_font_weight : `str`
        The font weight of the toggle button, e.g. 'bold'
    """
    # align glyph options
    channel_options_wid.children[1].children[1].children[1].children[1].children[1].remove_class('vbox')
    channel_options_wid.children[1].children[1].children[1].children[1].children[1].add_class('hbox')
    channel_options_wid.children[1].children[1].children[1].children[1].children[1].children[0].set_css('width', '0.8cm')

    # align sum and glyph checkboxes
    channel_options_wid.children[1].children[1].children[1].remove_class('vbox')
    channel_options_wid.children[1].children[1].children[1].add_class('hbox')

    # align radio buttons with the rest
    channel_options_wid.children[1].remove_class('vbox')
    channel_options_wid.children[1].add_class('hbox')
    channel_options_wid.children[1].add_class('align-start')

    # set toggle button font bold
    channel_options_wid.children[0].set_css('font-weight',
                                            toggle_button_font_weight)

    # margin and border around container widget
    channel_options_wid.set_css('padding', container_padding)
    channel_options_wid.set_css('margin', container_margin)
    channel_options_wid.set_css('border', container_border)


def landmark_options(group_keys, subgroup_keys, toggle_show_default=True,
                     landmarks_default=True, labels_default=True,
                     toggle_show_visible=True):
    r"""
    Creates a widget with Landmark Options. Specifically, it has:
        1) A checkbox that controls the landmarks' visibility.
        2) A drop down menu with the available landmark groups.
        3) Several toggle buttons with the available landmark sub-groups.
        4) A checkbox that controls the labels' visibility.
        5) A toggle button that controls the visibility of all the above, i.e.
           the landmark options.
    The structure of the widgets is the following:
        landmark_options_wid.children = [toggle_button, checkboxes, groups]
        checkboxes.children = [landmarks_checkbox, labels_checkbox]
        groups.children = [group_drop_down_menu, subgroup]
        subgroup.children = [subgroup_text, subgroup_toggle_buttons]
    To fix the alignment within this widget please refer to
    `format_landmark_options()` function.

    Parameters
    ----------
    group_keys : `list` of `str`
        A list of the available landmark groups.

    subgroup_keys : `list` of `str`
        A list of lists of each landmark group's subgroups.

    toggle_show_default : `boolean`, optional
        Defines whether the options will be visible upon construction.

    landmarks_default : `boolean`, optional
        The initial value of the landmarks visibility checkbox.

    labels_default : `boolean`, optional
        The initial value of the labels visibility checkbox.

    toggle_show_visible : `boolean`, optional
        The visibility of the toggle button.
    """
    # Toggle button that controls options' visibility
    but = ToggleButtonWidget(description='Landmark Options',
                             value=toggle_show_default,
                             visible=toggle_show_visible)

    # Create widgets
    landmarks = CheckboxWidget(description='Show landmarks',
                               value=landmarks_default)
    labels = CheckboxWidget(description='Show legend', value=labels_default)
    group = DropdownWidget(values=group_keys, description='Group')
    subgroup_toggles = [[ToggleButtonWidget(description=k, value=True)
                         for k in s_keys]
                        for s_keys in subgroup_keys]
    subgroup_text = LatexWidget(value='Labels')
    subgroup = ContainerWidget(children=subgroup_toggles[0])

    # Group widgets
    checkboxes_wid = ContainerWidget(children=[landmarks, labels])
    subgroup_and_text = ContainerWidget(children=[subgroup_text, subgroup])
    group_wid = ContainerWidget(children=[group, subgroup_and_text])
    landmark_options_wid = ContainerWidget(children=[but, checkboxes_wid,
                                                     group_wid])

    # Disability control
    def landmarks_fun(name, value):
        labels.disabled = not value
        group.disabled = not value
        for s in subgroup_toggles:
            for ww in s:
                ww.disabled = not value
        all_values = [ww.value for ww in subgroup.children]
        if all(item is False for item in all_values):
            for ww in subgroup.children:
                ww.value = True
    landmarks.on_trait_change(landmarks_fun, 'value')
    landmarks_fun('', landmarks_default)

    # Group drop down method
    def group_fun(name, value):
        subgroup.children = subgroup_toggles[group_keys.index(value)]
    group.on_trait_change(group_fun, 'value')

    # Subgroup function
    def subgroup_fun(name, value):
        all_values = [ww.value for ww in subgroup.children]
        if all(item is False for item in all_values):
            landmarks.value = False
    for s_group in subgroup_toggles:
        for w in s_group:
            w.on_trait_change(subgroup_fun, 'value')

    # Toggle button function
    def show_options(name, value):
        group_wid.visible = value
        checkboxes_wid.visible = value
    show_options('', toggle_show_default)
    but.on_trait_change(show_options, 'value')

    return landmark_options_wid


def format_landmark_options(landmark_options_wid, container_padding='6px',
                            container_margin='6px',
                            container_border='1px solid black',
                            toggle_button_font_weight='bold'):
    r"""
    Function that corrects the align (style format) of a given landmark_options
    widget. Usage example:
        landmark_options_wid = landmark_options()
        display(landmark_options_wid)
        format_landmark_options(landmark_options_wid)

    Parameters
    ----------
    landmark_options_wid :
        The widget object generated by the `landmark_options()` function.

    container_padding : `str`, optional
        The padding around the widget, e.g. '6px'

    container_margin : `str`, optional
        The margin around the widget, e.g. '6px'

    container_border : `str`, optional
        The border around the widget, e.g. '1px solid black'

    toggle_button_font_weight : `str`
        The font weight of the toggle button, e.g. 'bold'
    """
    # align subgroup toggle buttons
    landmark_options_wid.children[2].children[1].children[1].remove_class('vbox')
    landmark_options_wid.children[2].children[1].children[1].add_class('hbox')

    # align subgroup buttons with text
    landmark_options_wid.children[2].children[1].children[0].set_css(
        'margin-right', '5px')
    landmark_options_wid.children[2].children[1].remove_class('vbox')
    landmark_options_wid.children[2].children[1].add_class('hbox')

    # align checkboxes
    landmark_options_wid.children[1].remove_class('vbox')
    landmark_options_wid.children[1].add_class('hbox')

    # set toggle button font bold
    landmark_options_wid.children[0].set_css('font-weight',
                                             toggle_button_font_weight)

    # margin and border around container widget
    landmark_options_wid.set_css('padding', container_padding)
    landmark_options_wid.set_css('margin', container_margin)
    landmark_options_wid.set_css('border', container_border)


def info_print(toggle_show_default=True, toggle_show_visible=True):
    r"""
    Creates a widget that can print information. Specifically, it has:
        1) A latex widget where user can write the info text in latex format.
        2) A toggle button that controls the visibility of all the above, i.e.
           the info printing.
    The structure of the widgets is the following:
        info_wid.children = [toggle_button, text_widget]

    Parameters
    ----------
    toggle_show_default : `boolean`, optional
        Defines whether the info will be visible upon construction.

    toggle_show_visible : `boolean`, optional
        The visibility of the toggle button.
    """
    # Create toggle button
    but = ToggleButtonWidget(description='Info', value=toggle_show_default,
                             visible=toggle_show_visible)

    # Create text widget
    text_wid = LatexWidget(value="$\\bullet~$")

    # Toggle button function
    def show_options(name, value):
        text_wid.visible = value
    show_options('', toggle_show_default)
    but.on_trait_change(show_options, 'value')

    # Group widgets
    info_wid = ContainerWidget(children=[but, text_wid])

    return info_wid


def format_info_print(info_wid, font_size_in_pt='9pt', container_padding='6px',
                      container_margin='6px',
                      container_border='1px solid black',
                      toggle_button_font_weight='bold'):
    r"""
    Function that corrects the align (style format) of a given info widget.
    Usage example:
        info_wid = info_print()
        display(info_wid)
        format_info_print(info_wid)

    Parameters
    ----------
    info_wid :
        The widget object generated by the `info_print()` function.

    font_size_in_pt : `str`, optional
        The font size of the latex text, e.g. '9pt'

    container_padding : `str`, optional
        The padding around the widget, e.g. '6px'

    container_margin : `str`, optional
        The margin around the widget, e.g. '6px'

    container_border : `str`, optional
        The border around the widget, e.g. '1px solid black'

    toggle_button_font_weight : `str`
        The font weight of the toggle button, e.g. 'bold'
    """
    # latex widget formatting
    info_wid.children[1].set_css({'border': '1px dashed gray',
                                  'padding': '4px',
                                  'margin-top': '5px',
                                  'font-size': font_size_in_pt})

    # set toggle button font bold
    info_wid.children[0].set_css('font-weight', toggle_button_font_weight)

    # margin and border around container widget
    info_wid.set_css('padding', container_padding)
    info_wid.set_css('margin', container_margin)
    info_wid.set_css('border', container_border)


def model_parameters(n_params, params_str, params_bounds=(-3., 3.),
                     mode='multiple', toggle_show_default=True,
                     toggle_show_visible=True):
    r"""
    Creates a widget with Model Parameters. Specifically, it has:
        1) A slider for each parameter if mode is 'multiple'.
        2) A single slider and a drop down menu selection if mode is 'single'.
        3) A reset button.
    The structure of the widgets is the following:
        model_parameters_wid.children = [toggle_button, parameters_and_reset]
        parameters_and_reset.children = [parameters_widgets, reset_button]
        If mode is single:
        parameters_widgets.children = [drop_down_menu, slider]
        If mode is multiple:
        parameters_widgets.children = [all_sliders]
    The model_parameters_wid object also holds the mode and parameters_values
    information.
    To fix the alignment within this widget please refer to
    `format_model_parameters()` function.

    Parameters
    ----------
    n_params : `int`
        The number of principal components to use for the sliders.

    params_str : `str`
        The string that will be used for each parameters name.

    bounds : (`float`, `float`), optional
        The minimum and maximum bounds, in std units, for the sliders.

    mode : 'single' or 'multiple', optional
        If single, only a single slider is constructed along with a drop down
        menu.
        If multiple, a slider is constructed for each parameter.

    toggle_show_default : `boolean`, optional
        Defines whether the options will be visible upon construction.

    toggle_show_visible : `boolean`, optional
        The visibility of the toggle button.
    """
    # Initialize values list
    parameters_values = [0] * n_params
    # Toggle button that controls visibility
    but = ToggleButtonWidget(description='Parameters',
                             value=toggle_show_default,
                             visible=toggle_show_visible)

    # Create widgets
    reset_button = ButtonWidget(description='Reset')
    if mode == 'multiple':
        sliders = [FloatSliderWidget(description="{}{}".format(params_str, p),
                                     min=params_bounds[0],
                                     max=params_bounds[1],
                                     value=0.) for p in range(n_params)]
        parameters_wid = ContainerWidget(children=sliders)
    else:
        vals = ["{}{}".format(params_str, p) for p in range(n_params)]
        slider = FloatSliderWidget(description='',
                                   min=params_bounds[0],
                                   max=params_bounds[1],
                                   value=0.)
        text = DropdownWidget(values=vals)
        parameters_wid = ContainerWidget(children=[text, slider])
    params_and_reset = ContainerWidget(children=[parameters_wid, reset_button])
    model_parameters_wid = ContainerWidget(children=[but, params_and_reset])

    # set fields
    model_parameters_wid.mode = mode
    model_parameters_wid.parameters_values = parameters_values

    # reset function
    def reset_params(name):
        if mode == 'multiple':
            for ww in parameters_wid.children:
                ww.value = 0.
        else:
            parameters_wid.children[0].value = \
                parameters_wid.children[0].values[0]
            parameters_wid.children[1].value = 0.
        model_parameters_wid.parameters_values = [0] * n_params
    reset_button.on_click(reset_params)

    # parameters sliders function
    def get_params_values(name, value):
        if mode == 'multiple':
            model_parameters_wid.parameters_values = [
                p_wid.value for p_wid in parameters_wid.children]
        else:
            i_param = text.value
            print i_param
    for w in parameters_wid.children:
        w.on_trait_change(get_params_values, 'value')

    # Toggle button function
    def show_options(name, value):
        parameters_wid.visible = value
    show_options('', toggle_show_default)
    but.on_trait_change(show_options, 'value')

    return model_parameters_wid


def format_model_parameters(model_parameters_wid, container_padding='6px',
                            container_margin='6px',
                            container_border='1px solid black',
                            toggle_button_font_weight='bold'):
    r"""
    Function that corrects the align (style format) of a given model_parameters
    widget. Usage example:
        model_parameters_wid = model_parameters()
        display(model_parameters_wid)
        format_model_parameters(model_parameters_wid)

    Parameters
    ----------
    model_parameters_wid :
        The widget object generated by the `model_parameters()` function.

    container_padding : `str`, optional
        The padding around the widget, e.g. '6px'

    container_margin : `str`, optional
        The margin around the widget, e.g. '6px'

    container_border : `str`, optional
        The border around the widget, e.g. '1px solid black'

    toggle_button_font_weight : `str`
        The font weight of the toggle button, e.g. 'bold'
    """
    if model_parameters_wid.mode == 'single':
        # align drop down menu and slider
        model_parameters_wid.children[1].children[0].remove_class('vbox')
        model_parameters_wid.children[1].children[0].add_class('hbox')

    # align reset button to right
    model_parameters_wid.children[1].add_class('align-end')

    # set toggle button font bold
    model_parameters_wid.children[0].set_css('font-weight',
                                             toggle_button_font_weight)

    # margin and border around container widget
    model_parameters_wid.set_css('padding', container_padding)
    model_parameters_wid.set_css('margin', container_margin)
    model_parameters_wid.set_css('border', container_border)
