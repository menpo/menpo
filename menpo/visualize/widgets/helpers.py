from IPython.html.widgets import (FloatSliderWidget, ContainerWidget,
                                  IntSliderWidget, CheckboxWidget,
                                  ToggleButtonWidget, RadioButtonsWidget,
                                  IntTextWidget, DropdownWidget, LatexWidget,
                                  ButtonWidget)
import numpy as np


def figure_options(plot_function, scale_default=1., show_axes_default=True,
                   toggle_show_default=True, figure_scale_bounds=(0.1, 2),
                   figure_scale_step=0.1, figure_scale_visible=True,
                   show_axes_visible=True, toggle_show_visible=True):
    r"""
    Creates a widget with Figure Options. Specifically, it has:
        1) A slider that controls the scaling of the figure.
        2) A checkbox that controls the visibility of the figure's axes.
        3) A toggle button that controls the visibility of all the above, i.e.
           the figure options.

    The structure of the widgets is the following:
        figure_options_wid.children = [toggle_button, figure_scale_slider,
                                       show_axes_checkbox]

    The returned widget saves the selected values in the following fields:
        figure_options_wid.x_scale
        figure_options_wid.y_scale
        figure_options_wid.axes_visible

    To fix the alignment within this widget please refer to
    `format_figure_options()` function.

    Parameters
    ----------
    plot_function : `function` or None, optional
        The plot function that is executed when a widgets' value changes.
        If None, then nothing is assigned.

    scale_default : `float`, optional
        The initial value of the scale.

    show_axes_default : `boolean`, optional
        The initial value of the axes visibility checkbox.

    toggle_show_default : `boolean`, optional
        Defines whether the options will be visible upon construction.

    figure_scale_bounds : (`float`, `float`), optional
        The range of scales that can be optionally applied to the figure.

    figure_scale_step : `float`, optional
        The step of the scale sliders.

    figure_scale_visible : `boolean`, optional
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
    figure_scale = FloatSliderWidget(description='Figure scale:',
                                     value=scale_default,
                                     min=figure_scale_bounds[0],
                                     max=figure_scale_bounds[1],
                                     step=figure_scale_step)

    # Show axes
    show_axes = CheckboxWidget(description='Show axes', value=show_axes_default,
                               visible=show_axes_visible)

    # Widget container
    figure_options_wid = ContainerWidget(children=[but, figure_scale,
                                                   show_axes])

    # Initialize variables
    figure_options_wid.x_scale = scale_default
    figure_options_wid.y_scale = scale_default
    figure_options_wid.axes_visible = show_axes_default

    # Toggle button function
    if figure_scale_visible and show_axes_visible:
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
    elif figure_scale_visible and not show_axes_visible:
        def show_options(name, value):
            if value:
                figure_scale.visible = True
            else:
                figure_scale.visible = False
        if toggle_show_default:
            figure_scale.visible = True
        else:
            figure_scale.visible = False
    elif not figure_scale_visible and show_axes_visible:
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

    # scale slider function
    def scale_fun(name, old_value, value):
        figure_options_wid.x_scale = value
        figure_options_wid.y_scale = value
    figure_scale.on_trait_change(scale_fun, 'value')

    # show axes checkbox function
    def show_axes_fun(name, value):
        figure_options_wid.axes_visible = value
    show_axes.on_trait_change(show_axes_fun, 'value')

    # assign plot_function
    if plot_function is not None:
        figure_scale.on_trait_change(plot_function, 'value')
        show_axes.on_trait_change(plot_function, 'value')

    return figure_options_wid


def format_figure_options(figure_options_wid, container_padding='6px',
                          container_margin='6px',
                          container_border='1px solid black',
                          toggle_button_font_weight='bold',
                          border_visible=True):
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

    border_visible : `boolean`, optional
        Defines whether to draw the border line around the widget.
    """
    # set toggle button font bold
    figure_options_wid.children[0].set_css('font-weight',
                                           toggle_button_font_weight)

    # margin and border around container widget
    figure_options_wid.set_css('padding', container_padding)
    figure_options_wid.set_css('margin', container_margin)
    if border_visible:
        figure_options_wid.set_css('border', container_border)


def figure_options_two_scales(plot_function, x_scale_default=1.,
                              y_scale_default=1., coupled_default=False,
                              show_axes_default=True, toggle_show_default=True,
                              figure_scales_bounds=(0.1, 2),
                              figure_scales_step=0.1,
                              figure_scales_visible=True,
                              show_axes_visible=True, toggle_show_visible=True):
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

    The returned widget saves the selected values in the following fields:
        figure_options_wid.x_scale
        figure_options_wid.y_scale
        figure_options_wid.axes_visible

    To fix the alignment within this widget please refer to
    `format_figure_options_two_scales()` function.

    Parameters
    ----------
    plot_function : `function` or None, optional
        The plot function that is executed when a widgets' value changes.
        If None, then nothing is assigned.

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

    # Initialize variables
    figure_options_wid.x_scale = x_scale_default
    figure_options_wid.y_scale = y_scale_default
    figure_options_wid.axes_visible = show_axes_default

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

    # X scale slider function
    def x_fun(name, old_value, value):
        figure_options_wid.x_scale = value
        if coupled.value:
            Y_scale.value += value - old_value
    X_scale.on_trait_change(x_fun, 'value')

    # Y scale slider function
    def y_fun(name, value):
        figure_options_wid.y_scale = value
    Y_scale.on_trait_change(y_fun, 'value')

    # show axes checkbox function
    def show_axes_fun(name, value):
        figure_options_wid.axes_visible = value
    show_axes.on_trait_change(show_axes_fun, 'value')

    # assign plot_function
    if plot_function is not None:
        X_scale.on_trait_change(plot_function, 'value')
        Y_scale.on_trait_change(plot_function, 'value')
        show_axes.on_trait_change(plot_function, 'value')

    return figure_options_wid


def format_figure_options_two_scales(figure_options_wid,
                                     container_padding='6px',
                                     container_margin='6px',
                                     container_border='1px solid black',
                                     toggle_button_font_weight='bold',
                                     border_visible=True):
    r"""
    Function that corrects the align (style format) of a given figure_options
    widget. Usage example:
        figure_options_wid = figure_options_two_scales()
        display(figure_options_wid)
        format_figure_options(figure_options_wid)

    Parameters
    ----------
    figure_options_wid :
        The widget object generated by the `figure_options_two_scales()`
        function.

    container_padding : `str`, optional
        The padding around the widget, e.g. '6px'

    container_margin : `str`, optional
        The margin around the widget, e.g. '6px'

    container_border : `str`, optional
        The border around the widget, e.g. '1px solid black'

    toggle_button_font_weight : `str`
        The font weight of the toggle button, e.g. 'bold'

    border_visible : `boolean`, optional
        Defines whether to draw the border line around the widget.
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
    if border_visible:
        figure_options_wid.set_css('border', container_border)


def channel_options(n_channels, image_is_masked, plot_function=None,
                    masked_default=False, toggle_show_default=True,
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
        6) A checkbox that defines whether the masked image will be displayed.
        7) A toggle button that controls the visibility of all the above, i.e.
           the channel options.

    The structure of the widgets is the following:
        channel_options_wid.children = [toggle_button, all_but_toggle]
        all_but_toggle.children = [mode_and_masked, all_but_radiobuttons]
        mode_and_masked.children = [mode_radiobuttons, masked_checkbox]
        all_but_radiobuttons.children = [all_sliders, multiple_checkboxes]
        all_sliders.children = [first_slider, second_slider]
        multiple_checkboxes.children = [sum_checkbox, glyph_all]
        glyph_all.children = [glyph_checkbox, glyph_options]
        glyph_options.children = [block_size_text, use_negative_checkbox]

    The returned widget saves the selected values in the following fields:
        channel_options_wid.n_channels
        channel_options_wid.image_is_masked
        channel_options_wid.channels
        channel_options_wid.glyph_enabled
        channel_options_wid.glyph_block_size
        channel_options_wid.glyph_use_negative
        channel_options_wid.sum_enabled
        channel_options_wid.masked_enabled

    To fix the alignment within this widget please refer to
    `format_channel_options()` function.

    To update the state of this widget, please refer to
    `update_channel_options()` function.

    Parameters
    ----------
    n_channels : `int`
        The number of channels.

    image_is_masked : `boolean`
        Flag that defines whether the image is an instance of :map:`MaskedImage`
        or subclass.

    plot_function : `function` or None, optional
        The plot function that is executed when a widgets' value changes.
        If None, then nothing is assigned.

    masked_default : `boolean`, optional
        The initial value of the masked checkbox.

    toggle_show_default : `boolean`, optional
        Defines whether the options will be visible upon construction.

    toggle_show_visible : `boolean`, optional
        The visibility of the toggle button.
    """
    # if image is not masked, then masked flag should be disabled
    if not image_is_masked:
        masked_default = False

    # Create all necessary widgets
    but = ToggleButtonWidget(description='Channels Options',
                             value=toggle_show_default,
                             visible=toggle_show_visible)
    mode = RadioButtonsWidget(values=["Single", "Multiple"], value="Single",
                              description='Mode:', visible=toggle_show_default)
    masked = CheckboxWidget(value=masked_default, description='Masked',
                            visible=toggle_show_default and image_is_masked)
    first_slider_wid = IntSliderWidget(min=0, max=n_channels-1, step=1,
                                       value=0, description='Channel',
                                       visible=toggle_show_default)
    second_slider_wid = IntSliderWidget(min=1, max=n_channels-1, step=1,
                                        value=n_channels-1, description='To',
                                        visible=False)
    sum_wid = CheckboxWidget(value=False, description='Sum', visible=False)
    glyph_wid = CheckboxWidget(value=False, description='Glyph', visible=False)
    glyph_block_size = IntTextWidget(description='Block size', value='3',
                                     visible=False)
    glyph_use_negative = CheckboxWidget(description='Negative values',
                                        value=False, visible=False)

    # if single channel, disable all options
    # apart from masked
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
    mode_and_masked = ContainerWidget(children=[mode, masked])
    all_but_toggle = ContainerWidget(children=[mode_and_masked,
                                               all_but_radiobuttons])

    # Widget container
    channel_options_wid = ContainerWidget(children=[but, all_but_toggle])

    # Initialize output variables
    channel_options_wid.n_channels = n_channels
    channel_options_wid.image_is_masked = image_is_masked
    channel_options_wid.channels = 0
    channel_options_wid.glyph_enabled = False
    channel_options_wid.glyph_block_size = 3
    channel_options_wid.glyph_use_negative = False
    channel_options_wid.sum_enabled = False
    channel_options_wid.masked_enabled = masked_default

    # Define mode visibility
    def mode_selection_fun(name, value):
        if value == 'Single':
            first_slider_wid.description = 'Channel'
            first_slider_wid.min = 0
            first_slider_wid.max = channel_options_wid.n_channels - 1
            second_slider_wid.visible = False
            sum_wid.visible = False
            sum_wid.value = False
            glyph_wid.visible = False
            glyph_wid.value = False
            glyph_block_size.visible = False
            glyph_block_size.value = '3'
            glyph_use_negative.visible = False
            glyph_use_negative.value = False
        else:
            first_slider_wid.description = 'From'
            first_slider_wid.min = 0
            first_slider_wid.max = channel_options_wid.n_channels - 1
            second_slider_wid.min = 0
            second_slider_wid.max = channel_options_wid.n_channels - 1
            second_slider_wid.value = first_slider_wid.value
            second_slider_wid.visible = True
            sum_wid.visible = True
            sum_wid.value = False
            glyph_wid.visible = True
            glyph_wid.value = False
            glyph_block_size.visible = False
            glyph_block_size.value = '3'
            glyph_use_negative.visible = False
            glyph_use_negative.value = False
    mode.on_trait_change(mode_selection_fun, 'value')

    # Define glyph visibility
    def glyph_options_visibility_fun(name, value):
        glyph_block_size.visible = glyph_wid.value
        glyph_use_negative.visible = glyph_wid.value
        if glyph_wid.value:
            sum_wid.value = False
    glyph_wid.on_trait_change(glyph_options_visibility_fun, 'value')
    glyph_wid.on_trait_change(glyph_options_visibility_fun, 'visible')

    # Define sum functionality
    def sum_fun(name, value):
        if value:
            glyph_wid.value = False
    sum_wid.on_trait_change(sum_fun, 'value')

    # Define masked functionality
    def masked_fun(name, value):
        channel_options_wid.masked_enabled = value
    masked.on_trait_change(masked_fun, 'value')

    # Check block size value
    def block_size_fun(name, value):
        if value <= 0:
            glyph_block_size.value = 1
        if value >25:
            glyph_block_size.value = 25
    glyph_block_size.on_trait_change(block_size_fun, 'value')

    # Function that gets glyph/sum options
    def get_glyph_options(name, value):
        channel_options_wid.glyph_enabled = glyph_wid.value
        channel_options_wid.sum_enabled = sum_wid.value
        channel_options_wid.glyph_use_negative = glyph_use_negative.value
        channel_options_wid.glyph_block_size = glyph_block_size.value
        if channel_options_wid.sum_enabled:
            channel_options_wid.glyph_block_size = 1
    glyph_wid.on_trait_change(get_glyph_options, 'value')
    sum_wid.on_trait_change(get_glyph_options, 'value')
    glyph_use_negative.on_trait_change(get_glyph_options, 'value')
    glyph_block_size.on_trait_change(get_glyph_options, 'value')
    mode.on_trait_change(get_glyph_options, 'value')

    # Define multiple channels sliders functionality
    def first_slider_val(name, value):
        if mode.value == 'Multiple' and value > second_slider_wid.value:
            first_slider_wid.value = second_slider_wid.value

    def second_slider_val(name, value):
        if mode.value == 'Multiple' and value < first_slider_wid.value:
            second_slider_wid.value = first_slider_wid.value

    def get_channels(name, value):
        if mode.value == "Single":
            channel_options_wid.channels = first_slider_wid.value
        else:
            channel_options_wid.channels = range(first_slider_wid.value,
                                                 second_slider_wid.value + 1)
    first_slider_wid.on_trait_change(first_slider_val, 'value')
    second_slider_wid.on_trait_change(second_slider_val, 'value')
    first_slider_wid.on_trait_change(get_channels, 'value')
    second_slider_wid.on_trait_change(get_channels, 'value')
    mode.on_trait_change(get_channels, 'value')

    # Toggle button function
    def toggle_image_options(name, value):
        mode.visible = value
        if value:
            masked.visible = channel_options_wid.image_is_masked
            if mode.value == 'Single':
                first_slider_wid.visible = True
            else:
                first_slider_wid.visible = True
                second_slider_wid.visible = True
                sum_wid.visible = True
                glyph_wid.visible = True
        else:
            masked.visible = False
            first_slider_wid.visible = False
            second_slider_wid.visible = False
            sum_wid.visible = False
            glyph_wid.visible = False
            glyph_options.children[0].visible = False
            glyph_options.children[1].visible = False
    but.on_trait_change(toggle_image_options, 'value')

    # assign plot_function
    if plot_function is not None:
        #mode.on_trait_change(plot_function, 'value')
        masked.on_trait_change(plot_function, 'value')
        first_slider_wid.on_trait_change(plot_function, 'value')
        second_slider_wid.on_trait_change(plot_function, 'value')
        sum_wid.on_trait_change(plot_function, 'value')
        glyph_wid.on_trait_change(plot_function, 'value')
        glyph_block_size.on_trait_change(plot_function, 'value')
        glyph_use_negative.on_trait_change(plot_function, 'value')

    return channel_options_wid


def format_channel_options(channel_options_wid, container_padding='6px',
                           container_margin='6px',
                           container_border='1px solid black',
                           toggle_button_font_weight='bold',
                           border_visible=True):
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

    border_visible : `boolean`, optional
        Defines whether to draw the border line around the widget.
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
    if border_visible:
        channel_options_wid.set_css('border', container_border)


def update_channel_options(channel_options_wid, n_channels, image_is_masked, masked_default=False):
    r"""
    Function that updates the state of a given channel_options widget if the
    image's number of channels or its masked flag has changed. Usage example:
        channel_options_wid = channel_options(n_channels=2, image_is_masked=True)
        display(channel_options_wid)
        format_channel_options(channel_options_wid)
        update_channel_options(channel_options_wid, n_channels=36,
                               image_is_masked=False)

    Parameters
    ----------
    channel_options_wid :
        The widget object generated by the `channel_options()` function.

    n_channels : `int`
        The number of channels.

    image_is_masked : `boolean`
        Flag that defines whether the image is an instance of :map:`MaskedImage` or subclass.

    masked_default : `boolean`, optional
        The value to be assigned at the masked checkbox.
    """
    # if image_is_masked flag has actually changed from the previous value
    if image_is_masked != channel_options_wid.image_is_masked:
        # change the channel_options output
        channel_options_wid.image_is_masked = image_is_masked
        channel_options_wid.masked_enabled = masked_default
        # set the masked checkbox state
        channel_options_wid.children[1].children[0].children[1].visible = channel_options_wid.children[0].value and image_is_masked
        channel_options_wid.children[1].children[0].children[1].value = False

    # if n_channels are actually different from the previous value
    if n_channels != channel_options_wid.n_channels:
        # change the channel_options output
        channel_options_wid.n_channels = n_channels
        channel_options_wid.channels = 0
        # set the channel options state (apart from the masked checkbox)
        if n_channels == 1:
            # set sliders max and min values
            channel_options_wid.children[1].children[1].children[0].children[0].max = 1
            channel_options_wid.children[1].children[1].children[0].children[1].max = 1
            channel_options_wid.children[1].children[1].children[0].children[0].min = 0
            channel_options_wid.children[1].children[1].children[0].children[1].min = 0
            # set sliders state
            channel_options_wid.children[1].children[1].children[0].children[0].disabled = True
            channel_options_wid.children[1].children[1].children[0].children[1].visible = False
            # set glyph/sum state
            channel_options_wid.children[1].children[1].children[1].children[0].disabled = True
            channel_options_wid.children[1].children[1].children[1].children[1].children[0].disabled = True
            channel_options_wid.children[1].children[1].children[1].children[1].children[1].children[0].disabled = True
            channel_options_wid.children[1].children[1].children[1].children[1].children[1].children[1].disabled = True
            # set mode state
            channel_options_wid.children[1].children[0].children[0].disabled = True
            # set mode and sliders values
            for k in range(3):
                if k == 0:
                    channel_options_wid.children[1].children[0].children[0].value = "Single"
                elif k == 1:
                    channel_options_wid.children[1].children[1].children[0].children[0].value = 0
                else:
                    channel_options_wid.children[1].children[1].children[0].children[1].value = 0
        else:
            # set sliders max and min values
            channel_options_wid.children[1].children[1].children[0].children[0].max = n_channels - 1
            channel_options_wid.children[1].children[1].children[0].children[1].max = n_channels - 1
            channel_options_wid.children[1].children[1].children[0].children[0].min = 0
            channel_options_wid.children[1].children[1].children[0].children[1].min = 0
            # set sliders state
            channel_options_wid.children[1].children[1].children[0].children[0].disabled = False
            channel_options_wid.children[1].children[1].children[0].children[1].visible = False
            # set glyph/sum state
            channel_options_wid.children[1].children[1].children[1].children[0].disabled = False
            channel_options_wid.children[1].children[1].children[1].children[1].children[0].disabled = False
            channel_options_wid.children[1].children[1].children[1].children[1].children[1].children[0].disabled = False
            channel_options_wid.children[1].children[1].children[1].children[1].children[1].children[1].disabled = False
            # set mode state
            channel_options_wid.children[1].children[0].children[0].disabled = False
            # set mode and sliders values
            for k in range(3):
                if k == 0:
                    channel_options_wid.children[1].children[0].children[0].value = "Single"
                elif k == 1:
                    channel_options_wid.children[1].children[1].children[0].children[0].value = 0
                else:
                    channel_options_wid.children[1].children[1].children[0].children[1].value = 0


def landmark_options(group_keys, labels_keys, plot_function=None,
                     landmarks_default=True, legend_default=True,
                     toggle_show_default=True, toggle_show_visible=True):
    r"""
    Creates a widget with Landmark Options. Specifically, it has:
        1) A checkbox that controls the landmarks' visibility.
        2) A drop down menu with the available landmark groups.
        3) Several toggle buttons with the group's available labels.
        4) A checkbox that controls the legend's visibility.
        5) A toggle button that controls the visibility of all the above, i.e.
           the landmark options.

    The structure of the widgets is the following:
        landmark_options_wid.children = [toggle_button, checkboxes, groups]
        checkboxes.children = [landmarks_checkbox, legend_checkbox]
        groups.children = [group_drop_down_menu, labels]
        labels.children = [labels_text, labels_toggle_buttons]

    The returned widget saves the selected values in the following fields:
        landmark_options_wid.group_keys
        landmark_options_wid.labels_keys
        landmark_options_wid.labels_toggles
        landmark_options_wid.landmarks_enabled
        landmark_options_wid.legend_enabled
        landmark_options_wid.group
        landmark_options_wid.with_labels

    To fix the alignment within this widget, please refer to
    `format_landmark_options()` function.

    To update the state of this widget, please refer to
    `update_landmark_options()` function.

    Parameters
    ----------
    group_keys : `list` of `str`
        A list of the available landmark groups.

    labels_keys : `list` of `list` of `str`
        A list of lists of each landmark group's labels.

    plot_function : `function` or None, optional
        The plot function that is executed when a widgets' value changes.
        If None, then nothing is assigned.

    landmarks_default : `boolean`, optional
        The initial value of the landmarks visibility checkbox.

    legend_default : `boolean`, optional
        The initial value of the legend's visibility checkbox.

    toggle_show_default : `boolean`, optional
        Defines whether the options will be visible upon construction.

    toggle_show_visible : `boolean`, optional
        The visibility of the toggle button.
    """
    # Create all necessary widgets
    but = ToggleButtonWidget(description='Landmarks Options',
                             value=toggle_show_default,
                             visible=toggle_show_visible)
    landmarks = CheckboxWidget(description='Show landmarks',
                               value=landmarks_default)
    legend = CheckboxWidget(description='Show legend', value=legend_default)
    group = DropdownWidget(values=group_keys, description='Group')
    labels_toggles = [[ToggleButtonWidget(description=k, value=True)
                       for k in s_keys] for s_keys in labels_keys]
    labels_text = LatexWidget(value='Labels')
    labels = ContainerWidget(children=labels_toggles[0])

    # Group widgets
    checkboxes_wid = ContainerWidget(children=[landmarks, legend])
    labels_and_text = ContainerWidget(children=[labels_text, labels])
    group_wid = ContainerWidget(children=[group, labels_and_text])

    # Widget container
    landmark_options_wid = ContainerWidget(children=[but, checkboxes_wid,
                                                     group_wid])

    # Initialize output variables
    landmark_options_wid.group_keys = group_keys
    landmark_options_wid.labels_keys = labels_keys
    landmark_options_wid.labels_toggles = labels_toggles
    landmark_options_wid.landmarks_enabled = landmarks_default
    landmark_options_wid.legend_enabled = legend_default
    landmark_options_wid.group = group_keys[0]
    landmark_options_wid.with_labels = labels_keys[0]

    # Disability control
    def landmarks_fun(name, value):
        # get landmarks_enabled value
        landmark_options_wid.landmarks_enabled = value
        # disable legend checkbox and group drop down menu
        legend.disabled = not value
        group.disabled = not value
        # disable all labels toggles
        for s_keys in landmark_options_wid.labels_toggles:
            for k in s_keys:
                k.disabled = not value
        # if all currently selected labels toggles are False, set them all
        # to True
        all_values = [ww.value for ww in labels.children]
        if all(item is False for item in all_values):
            for ww in labels.children:
                ww.value = True
    landmarks.on_trait_change(landmarks_fun, 'value')
    landmarks_fun('', landmarks_default)

    # Group drop down method
    def group_fun(name, value):
        # get group value
        landmark_options_wid.group = value
        # assign the correct children to the labels toggles
        labels.children = landmark_options_wid.labels_toggles[landmark_options_wid.group_keys.index(value)]
        # get with_labels value
        landmark_options_wid.with_labels = []
        for ww in labels.children:
            if ww.value:
                landmark_options_wid.with_labels.append(str(ww.description))
        # assign plot_function to all enabled labels
        if plot_function is not None:
            for w in labels.children:
                w.on_trait_change(plot_function, 'value')

    group.on_trait_change(group_fun, 'value')

    # Labels function
    def labels_fun(name, value):
        # if all labels toggles are False, set landmarks checkbox to False
        all_values = [ww.value for ww in labels.children]
        if all(item is False for item in all_values):
            landmarks.value = False
        # get with_labels value
        landmark_options_wid.with_labels = []
        for ww in labels.children:
            if ww.value:
                landmark_options_wid.with_labels.append(str(ww.description))
    # assign labels_fun to all labels toggles (even hidden ones)
    for s_group in landmark_options_wid.labels_toggles:
        for w in s_group:
            w.on_trait_change(labels_fun, 'value')

    # Legend function
    def legend_fun(name, value):
        landmark_options_wid.legend_enabled = value
    legend.on_trait_change(legend_fun, 'value')

    # Toggle button function
    def show_options(name, value):
        group_wid.visible = value
        checkboxes_wid.visible = value
    show_options('', toggle_show_default)
    but.on_trait_change(show_options, 'value')

    # assign plot_function
    if plot_function is not None:
        # assign plot_function to landmarks checkbox, legend
        # checkbox and group drop down menu
        landmarks.on_trait_change(plot_function, 'value')
        legend.on_trait_change(plot_function, 'value')
        group.on_trait_change(plot_function, 'value')
        # assign plot_function to all currently active labels toggles
        for w in labels.children:
            w.on_trait_change(plot_function, 'value')

    return landmark_options_wid


def format_landmark_options(landmark_options_wid, container_padding='6px',
                            container_margin='6px',
                            container_border='1px solid black',
                            toggle_button_font_weight='bold',
                            border_visible=True):
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

    border_visible : `boolean`, optional
        Defines whether to draw the border line around the widget.
    """
    # align labels toggle buttons
    landmark_options_wid.children[2].children[1].children[1].remove_class('vbox')
    landmark_options_wid.children[2].children[1].children[1].add_class('hbox')

    # align labels buttons with text
    landmark_options_wid.children[2].children[1].children[0].set_css(
        'margin-right', '5px')
    landmark_options_wid.children[2].children[1].remove_class('vbox')
    landmark_options_wid.children[2].children[1].add_class('hbox')
    landmark_options_wid.children[2].children[1].add_class('align-center')

    # align group drop down menu with labels toggle buttons
    landmark_options_wid.children[2].children[1].set_css('margin-top', '10px')
    landmark_options_wid.children[2].add_class('start')

    # align checkboxes
    landmark_options_wid.children[1].children[0].set_css('margin-right', '25px')
    landmark_options_wid.children[1].remove_class('vbox')
    landmark_options_wid.children[1].add_class('hbox')

    # set toggle button font bold
    landmark_options_wid.children[0].set_css('font-weight',
                                             toggle_button_font_weight)

    # margin and border around container widget
    landmark_options_wid.set_css('padding', container_padding)
    landmark_options_wid.set_css('margin', container_margin)
    if border_visible:
        landmark_options_wid.set_css('border', container_border)


def update_landmark_options(landmark_options_wid, group_keys, labels_keys, plot_function):
    r"""
    Function that updates the state of a given landmark_options widget if the
    group or label keys of an image has changed. Usage example:
        group_keys = ['group1', 'group2']
        labels_keys = [['label11'], ['label21', 'label22']]
        landmark_options_wid = landmark_options(group_keys=group_keys,
                                                labels_keys=labels_keys)
        display(landmark_options_wid)
        format_landmark_options(landmark_options_wid)
        update_landmark_options(landmark_options_wid,
                                group_keys=['group3'],
                                labels_keys=['label31', 'label32', 'label33'])
        format_landmark_options(landmark_options_wid)

    Note that the `format_landmark_options()` function needs to be called again
    after the `update_landmark_options()` function.

    Parameters
    ----------
    landmark_options_wid :
        The widget object generated by the `landmark_options()` function.

    group_keys : `list` of `str`
        A list of the available landmark groups.

    labels_keys : `list` of `list` of `str`
        A list of lists of each landmark group's labels.

    plot_function : `function` or None
        The plot function that is executed when a widgets' value changes.
        If None, then nothing is assigned.
    """
    # check if the new group_keys and labels_keys are the same as the old ones
    if not _compare_groups_and_labels(group_keys, labels_keys,
                                      landmark_options_wid.group_keys,
                                      landmark_options_wid.labels_keys):
        # Create all necessary widgets
        group = DropdownWidget(values=group_keys, description='Group')
        labels_toggles = [[ToggleButtonWidget(description=k, value=True)
                           for k in s_keys] for s_keys in labels_keys]

        # Group widgets
        landmark_options_wid.children[2].children[1].children[1].children = \
            labels_toggles[0]
        labels = landmark_options_wid.children[2].children[1]
        cont = ContainerWidget(children=[group, labels])
        landmark_options_wid.children = [landmark_options_wid.children[0],
                                         landmark_options_wid.children[1], cont]

        # Initialize output variables
        landmark_options_wid.group_keys = group_keys
        landmark_options_wid.labels_keys = labels_keys
        landmark_options_wid.labels_toggles = labels_toggles
        landmark_options_wid.group = group_keys[0]
        landmark_options_wid.with_labels = labels_keys[0]

        # Disability control
        group.disabled = not landmark_options_wid.landmarks_enabled
        # disable all labels toggles
        for s_keys in labels_toggles:
            for k in s_keys:
                k.disabled = not landmark_options_wid.landmarks_enabled

        # Group drop down method
        def group_fun(name, value):
            # get group value
            landmark_options_wid.group = value
            # assign the correct children to the labels toggles
            landmark_options_wid.children[2].children[1].children[1].children = landmark_options_wid.labels_toggles[landmark_options_wid.group_keys.index(value)]
            # get with_labels value
            landmark_options_wid.with_labels = []
            for ww in landmark_options_wid.children[2].children[1].children[1].children:
                if ww.value:
                    landmark_options_wid.with_labels.append(str(ww.description))
            # assign plot_function to all enabled labels
            if plot_function is not None:
                for w in landmark_options_wid.children[2].children[1].children[1].children:
                    w.on_trait_change(plot_function, 'value')

        group.on_trait_change(group_fun, 'value')

        # Labels function
        def labels_fun(name, value):
            # if all labels toggles are False, set landmarks checkbox to False
            all_values = [ww.value for ww in landmark_options_wid.children[2].children[1].children[1].children]
            if all(item is False for item in all_values):
                landmark_options_wid.children[1].children[0].value = False
            # get with_labels value
            landmark_options_wid.with_labels = []
            for ww in landmark_options_wid.children[2].children[1].children[1].children:
                if ww.value:
                    landmark_options_wid.with_labels.append(str(ww.description))
        # assign labels_fun to all labels toggles (even hidden ones)
        for s_group in labels_toggles:
            for w in s_group:
                w.on_trait_change(labels_fun, 'value')

        # assign plot_function
        if plot_function is not None:
            # assign plot_function to landmarks checkbox, legend
            # checkbox and group drop down menu
            group.on_trait_change(plot_function, 'value')
            # assign plot_function to all currently active labels toggles
            for w in labels_toggles[0]:
                w.on_trait_change(plot_function, 'value')

        # Toggle button function
        def show_options(name, value):
            landmark_options_wid.children[1].visible = value
            landmark_options_wid.children[2].visible = value
        show_options('', landmark_options_wid.children[0].value)
        landmark_options_wid.children[0].on_trait_change(show_options, 'value')


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
                      toggle_button_font_weight='bold',
                      border_visible=True):
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

    border_visible : `boolean`, optional
        Defines whether to draw the border line around the widget.
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
    if border_visible:
        info_wid.set_css('border', container_border)


def model_parameters(n_params, plot_function=None, params_str='',
                     mode='multiple', params_bounds=(-3., 3.),
                     plot_eig_visible=True, plot_eig_function=None,
                     toggle_show_default=True, toggle_show_visible=True,
                     toggle_show_name='Parameters'):
    r"""
    Creates a widget with Model Parameters. Specifically, it has:
        1) A slider for each parameter if mode is 'multiple'.
        2) A single slider and a drop down menu selection if mode is 'single'.
        3) A reset button.
        4) A button and two radio buttons for plotting the eigenvalues variance
           ratio.

    The structure of the widgets is the following:
        model_parameters_wid.children = [toggle_button, parameters_and_reset]
        parameters_and_reset.children = [parameters_widgets, reset]
        If plot_eig_visible is True:
        reset = [plot_eigenvalues, reset_button]
        Else:
        reset = reset_button
        If mode is single:
        parameters_widgets.children = [drop_down_menu, slider]
        If mode is multiple:
        parameters_widgets.children = [all_sliders]

    The returned widget saves the selected values in the following fields:
        model_parameters_wid.parameters_values
        model_parameters_wid.mode
        model_parameters_wid.plot_eig_visible

    To fix the alignment within this widget please refer to
    `format_model_parameters()` function.

    Parameters
    ----------
    n_params : `int`
        The number of principal components to use for the sliders.

    plot_function : `function` or None, optional
        The plot function that is executed when a widgets' value changes.
        If None, then nothing is assigned.

    params_str : `str`, optional
        The string that will be used for each parameters name.

    mode : 'single' or 'multiple', optional
        If single, only a single slider is constructed along with a drop down
        menu.
        If multiple, a slider is constructed for each parameter.

    params_bounds : (`float`, `float`), optional
        The minimum and maximum bounds, in std units, for the sliders.

    plot_eig_visible : `boolean`, optional
        Defines whether the options for plotting the eigenvalues variance ratio
        will be visible upon construction.

    plot_eig_function : `function` or None, optional
        The plot function that is executed when the plot eigenvalues button is
        clicked. If None, then nothing is assigned.

    toggle_show_default : `boolean`, optional
        Defines whether the options will be visible upon construction.

    toggle_show_visible : `boolean`, optional
        The visibility of the toggle button.

    toggle_show_name : `str`, optional
        The name of the toggle button.
    """
    from collections import OrderedDict

    # If only one slider, set mode to multiple
    if n_params == 1:
        mode = 'multiple'

    # Toggle button that controls visibility
    but = ToggleButtonWidget(description=toggle_show_name,
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
        vals = OrderedDict()
        for p in range(n_params):
            vals["{}{}".format(params_str, p)] = p
        slider = FloatSliderWidget(description='',
                                   min=params_bounds[0],
                                   max=params_bounds[1],
                                   value=0.)
        dropdown_params = DropdownWidget(values=vals)
        parameters_wid = ContainerWidget(children=[dropdown_params, slider])

    # Group widgets
    if plot_eig_visible:
        plot_button = ButtonWidget(description='Plot eigenvalues')
        if plot_eig_function is not None:
            plot_button.on_click(plot_eig_function)
        plot_and_reset = ContainerWidget(children=[plot_button, reset_button])
        params_and_reset = ContainerWidget(children=[parameters_wid,
                                                     plot_and_reset])
    else:
        params_and_reset = ContainerWidget(children=[parameters_wid,
                                                     reset_button])

    # Widget container
    model_parameters_wid = ContainerWidget(children=[but, params_and_reset])

    # Save mode and parameters values
    model_parameters_wid.parameters_values = [0.0] * n_params
    model_parameters_wid.mode = mode
    model_parameters_wid.plot_eig_visible = plot_eig_visible

    # set up functions
    if mode == 'single':
        # save slider value to parameters values list
        def save_slider_value(name, value):
            model_parameters_wid.parameters_values[dropdown_params.value] = value
        slider.on_trait_change(save_slider_value, 'value')

        # set correct value to slider when drop down menu value changes
        def set_slider_value(name, value):
            slider.value = model_parameters_wid.parameters_values[value]
        dropdown_params.on_trait_change(set_slider_value, 'value')

        # assign main plotting function when slider value changes
        if plot_function is not None:
            slider.on_trait_change(plot_function, 'value')
    else:
        # save all sliders value to parameters values list
        def save_sliders_values(name, value):
            model_parameters_wid.parameters_values = []
            for p_wid in parameters_wid.children:
                if not p_wid.disabled:
                    model_parameters_wid.parameters_values.append(p_wid.value)

        # assign saving values and main plotting function to all sliders
        for w in parameters_wid.children:
            w.on_trait_change(save_sliders_values, 'value')
            if plot_function is not None:
                w.on_trait_change(plot_function, 'value')

    # reset function
    def reset_params(name):
        model_parameters_wid.parameters_values = \
            [0.0] * len(model_parameters_wid.parameters_values)
        if mode == 'multiple':
            for ww in parameters_wid.children:
                ww.value = 0.
        else:
            parameters_wid.children[0].value = 0
            parameters_wid.children[1].value = 0.
    reset_button.on_click(reset_params)

    # Toggle button function
    def show_options(name, value):
        params_and_reset.visible = value
    show_options('', toggle_show_default)
    but.on_trait_change(show_options, 'value')

    return model_parameters_wid


def format_model_parameters(model_parameters_wid, container_padding='6px',
                            container_margin='6px',
                            container_border='1px solid black',
                            toggle_button_font_weight='bold',
                            border_visible=True):
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

    border_visible : `boolean`, optional
        Defines whether to draw the border line around the widget.
    """
    if model_parameters_wid.mode == 'single':
        # align drop down menu and slider
        model_parameters_wid.children[1].children[0].remove_class('vbox')
        model_parameters_wid.children[1].children[0].add_class('hbox')

    # align reset button to right
    if model_parameters_wid.plot_eig_visible:
        model_parameters_wid.children[1].children[1].remove_class('vbox')
        model_parameters_wid.children[1].children[1].add_class('hbox')
    model_parameters_wid.children[1].add_class('align-end')

    # set toggle button font bold
    model_parameters_wid.children[0].set_css('font-weight',
                                             toggle_button_font_weight)

    # margin and border around plot_eigenvalues widget
    if model_parameters_wid.plot_eig_visible:
        model_parameters_wid.children[1].children[1].children[0].set_css(
            'margin-right', container_margin)

    # margin and border around container widget
    model_parameters_wid.set_css('padding', container_padding)
    model_parameters_wid.set_css('margin', container_margin)
    if border_visible:
        model_parameters_wid.set_css('border', container_border)


def update_model_parameters(model_parameters_wid, n_params, params_str=''):
    r"""
    Function that updates the state of a given model_parameters widget if the
    requested number of parameters has changed. Usage example:
        model_parameters_wid = model_parameters(n_params=5)
        display(model_parameters_wid)
        format_model_parameters(model_parameters_wid)
        update_model_parameters(model_parameters_wid, 3)

    Parameters
    ----------
    model_parameters_wid :
        The widget object generated by the `model_parameters()` function.

    n_params : `int`
        The requested number of parameters.

    params_str : `str`, optional
        The string that will be used for each parameters name.
    """
    from collections import OrderedDict

    if model_parameters_wid.mode == 'multiple':
        enabled_params = sum([not w.disabled
                              for w in model_parameters_wid.children[1].children[0].children])
        if n_params != enabled_params:
            available_n_params = len(model_parameters_wid.children[1].children[0].children)
            if n_params <= available_n_params:
                model_parameters_wid.parameters_values = [0.0] * n_params
                for w in model_parameters_wid.children[1].children[0].children[:n_params]:
                    w.disabled = False
                for w in model_parameters_wid.children[1].children[0].children[n_params::]:
                    w.disabled = True
                for w in model_parameters_wid.children[1].children[0].children:
                    w.value = 0.
    else:
        enabled_params = len(model_parameters_wid.children[1].children[0].children[0].values)
        if n_params != enabled_params:
            model_parameters_wid.parameters_values = [0.0] * n_params
            vals = OrderedDict()
            for p in range(n_params):
                vals["{}{}".format(params_str, p)] = p
            model_parameters_wid.children[1].children[0].children[0].values = vals
            model_parameters_wid.children[1].children[0].children[0].value = vals["{}{}".format(params_str, 0)]
            model_parameters_wid.children[1].children[0].children[1].value = 0.


def final_result_options(group_keys, plot_function=None, title='Final Result',
                         show_image_default=True,
                         subplots_enabled_default=False, legend_default=True,
                         toggle_show_default=True, toggle_show_visible=True):
    r"""
    Creates a widget with Final Result Options. Specifically, it has:
        1) A set of toggle buttons representing usually the initial, final and
           ground truth shapes.
        2) A checkbox that controls the visibility of the image.
        3) A set of radio buttons that define whether subplots are enabled.
        4) A checkbox that controls the legend's visibility.
        5) A toggle button that controls the visibility of all the above, i.e.
           the final result options.

    The structure of the widgets is the following:
        final_result_wid.children = [toggle_button, shapes_toggle_buttons,
                                     show_image_checkbox, options]
        options.children = [plot_mode_radio_buttons, legend_checkbox]

    The returned widget saves the selected values in the following fields:
        final_result_wid.groups
        final_result_wid.show_image
        final_result_wid.subplots_enabled
        final_result_wid.legend_enabled

    To fix the alignment within this widget please refer to
    `format_final_result_options()` function.

    Parameters
    ----------
    group_keys : `list` of `str`
        A list of the available landmark groups.

    plot_function : `function` or None, optional
        The plot function that is executed when a widgets' value changes.
        If None, then nothing is assigned.

    title : `str`, optional
        The title of the widget printed at the toggle button.

    show_image_default : `boolean`, optional
        The initial value of the image's visibility checkbox.

    subplots_enabled_default : `boolean`, optional
        The initial value of the plot options' radio buttons that determine
        whether a single plot or subplots will be used.

    legend_default : `boolean`, optional
        The initial value of the legend's visibility checkbox.

    toggle_show_default : `boolean`, optional
        Defines whether the options will be visible upon construction.

    toggle_show_visible : `boolean`, optional
        The visibility of the toggle button.
    """
    # Toggle button that controls options' visibility
    but = ToggleButtonWidget(description=title,
                             value=toggle_show_default,
                             visible=toggle_show_visible)

    # Create widgets
    shapes_checkboxes = [LatexWidget(value='Select shape:')]
    for group in group_keys:
        t = ToggleButtonWidget(description=group, value=False)
        if group == 'final':
            t.value = True
        shapes_checkboxes.append(t)
    show_image = CheckboxWidget(description='Show image',
                                value=show_image_default)
    mode = RadioButtonsWidget(description='Plot mode:',
                              values={'Single': False, 'Multiple': True})
    mode.value = subplots_enabled_default
    show_legend = CheckboxWidget(description='Show legend',
                                 value=legend_default)

    # Group widgets
    shapes_wid = ContainerWidget(children=shapes_checkboxes)
    opts = ContainerWidget(children=[mode, show_legend])

    # Widget container
    final_result_wid = ContainerWidget(children=[but, shapes_wid, show_image,
                                                 opts])

    # Initialize variables
    final_result_wid.groups = ['final']
    final_result_wid.show_image = show_image_default
    final_result_wid.subplots_enabled = subplots_enabled_default
    final_result_wid.legend_enabled = legend_default

    # Groups function
    def groups_fun(name, value):
        final_result_wid.groups = []
        for i in shapes_wid.children[1::]:
            if i.value:
                final_result_wid.groups.append(str(i.description))
    for w in shapes_wid.children[1::]:
        w.on_trait_change(groups_fun, 'value')

    # Show image function
    def show_image_fun(name, value):
        final_result_wid.show_image = value
    show_image.on_trait_change(show_image_fun, 'value')

    # Plot mode function
    def plot_mode_fun(name, value):
        final_result_wid.subplots_enabled = value
    mode.on_trait_change(plot_mode_fun, 'value')

    # Legend function
    def legend_fun(name, value):
        final_result_wid.legend_enabled = value
    show_legend.on_trait_change(legend_fun, 'value')

    # Toggle button function
    def show_options(name, value):
        shapes_wid.visible = value
        show_image.visible = value
        opts.visible = value
    show_options('', toggle_show_default)
    but.on_trait_change(show_options, 'value')

    # assign plot_function
    if plot_function is not None:
        show_image.on_trait_change(plot_function, 'value')
        mode.on_trait_change(plot_function, 'value')
        show_legend.on_trait_change(plot_function, 'value')
        for w in shapes_wid.children[1::]:
            w.on_trait_change(plot_function, 'value')

    return final_result_wid


def format_final_result_options(final_result_wid, container_padding='6px',
                                container_margin='6px',
                                container_border='1px solid black',
                                toggle_button_font_weight='bold',
                                border_visible=True):
    r"""
    Function that corrects the align (style format) of a given
    final_result_options widget. Usage example:
        final_result_wid = final_result_options()
        display(final_result_wid)
        format_final_result_options(final_result_wid)

    Parameters
    ----------
    final_result_wid :
        The widget object generated by the `final_result_options()` function.

    container_padding : `str`, optional
        The padding around the widget, e.g. '6px'

    container_margin : `str`, optional
        The margin around the widget, e.g. '6px'

    container_border : `str`, optional
        The border around the widget, e.g. '1px solid black'

    toggle_button_font_weight : `str`
        The font weight of the toggle button, e.g. 'bold'

    border_visible : `boolean`, optional
        Defines whether to draw the border line around the widget.
    """
    # align shapes toggle buttons
    final_result_wid.children[1].remove_class('vbox')
    final_result_wid.children[1].add_class('hbox')
    final_result_wid.children[1].add_class('align-center')
    final_result_wid.children[1].children[0].set_css('margin-right',
                                                     container_margin)

    # align mode and legend options
    final_result_wid.children[3].remove_class('vbox')
    final_result_wid.children[3].add_class('hbox')
    final_result_wid.children[3].children[0].set_css('margin-right', '20px')

    # set toggle button font bold
    final_result_wid.children[0].set_css('font-weight',
                                         toggle_button_font_weight)
    final_result_wid.children[1].set_css('margin-top', container_margin)

    # margin and border around container widget
    final_result_wid.set_css('padding', container_padding)
    final_result_wid.set_css('margin', container_margin)
    if border_visible:
        final_result_wid.set_css('border', container_border)


def iterations_result_options(n_iters, plot_function=None, iter_str='iter_',
                              title='Iterations Result',
                              show_image_default=True,
                              subplots_enabled_default=False,
                              legend_default=True, toggle_show_default=True,
                              toggle_show_visible=True):
    r"""
    Creates a widget with Iterations Result Options. Specifically, it has:
        1) Two radio buttons that select an options mode, depending on whether
           the user wants to visualize a "Single" or "Multiple" iterations.
        2) If mode is "Single", an iteration number is selected by one slider.
           If mode is "Multiple", the iterations range is selected by two
           sliders.
        3) A checkbox that controls the visibility of the image.
        4) A set of radio buttons that define whether subplots are enabled.
        5) A checkbox that controls the legend's visibility.
        6) A toggle button that controls the visibility of all the above, i.e.
           the final result options.

    The structure of the widgets is the following:
        iterations_result_wid.children = [toggle_button,
                                          iterations_mode_and_sliders,
                                          show_image_and_plot_errors,
                                          options]
        iterations_mode_and_sliders.children = [iterations_mode_radio_buttons,
                                                all_sliders]
        show_image_and_plot_errors.children = [show_image_checkbox,
                                               plot_errors_button]
        all_sliders.children = [first_slider, second_slider]
        options.children = [plot_mode_radio_buttons, legend_checkbox]

    The returned widget saves the selected values in the following fields:
        iterations_result_wid.groups
        iterations_result_wid.show_image
        iterations_result_wid.subplots_enabled
        iterations_result_wid.legend_enabled

    To fix the alignment within this widget please refer to
    `format_iterations_result_options()` function.

    Parameters
    ----------
    n_iters : `int`
        The number of iterations.

    plot_function : `function` or None, optional
        The plot function that is executed when a widgets' value changes.
        If None, then nothing is assigned.

    iter_str : `str`, optional
        The str that is used in the landmark groups shapes.
        E.g. if iter_str == "iter_" then the group label of iteration i has the
        form "{}{}".format(iter_str, i)

    title : `str`, optional
        The title of the widget printed at the toggle button.

    show_image_default : `boolean`, optional
        The initial value of the image's visibility checkbox.

    subplots_enabled_default : `boolean`, optional
        The initial value of the plot options' radio buttons that determine
        whether a single plot or subplots will be used.

    legend_default : `boolean`, optional
        The initial value of the legend's visibility checkbox.

    toggle_show_default : `boolean`, optional
        Defines whether the options will be visible upon construction.

    toggle_show_visible : `boolean`, optional
        The visibility of the toggle button.
    """
    # Toggle button that controls options' visibility
    but = ToggleButtonWidget(description=title, value=toggle_show_default,
                             visible=toggle_show_visible)

    # Create widgets
    iterations_mode = RadioButtonsWidget(values={'Single': 0, 'Multiple': 1},
                                         value=0,
                                         description='Iterations mode:',
                                         visible=toggle_show_default)
    first_slider_wid = IntSliderWidget(min=0, max=n_iters-1, step=1,
                                       value=0, description='Iteration',
                                       visible=toggle_show_default)
    second_slider_wid = IntSliderWidget(min=1, max=n_iters-1, step=1,
                                        value=n_iters-1, description='To',
                                        visible=False)
    show_image = CheckboxWidget(description='Show image',
                                value=show_image_default)
    plot_errors_button = ButtonWidget(description='Plot Errors')
    plot_mode = RadioButtonsWidget(description='Plot mode:',
                                   values={'Single': False, 'Multiple': True})
    plot_mode.value = subplots_enabled_default
    show_legend = CheckboxWidget(description='Show legend',
                                 value=legend_default)

    # if single iteration, disable multiple options
    if n_iters == 1:
        iterations_mode.value = 0
        iterations_mode.disabled = True
        first_slider_wid.disabled = True
        second_slider_wid.disabled = True

    # Group widgets
    sliders = ContainerWidget(children=[first_slider_wid, second_slider_wid])
    iterations_mode_and_sliders = ContainerWidget(children=[iterations_mode,
                                                            sliders])
    show_image_and_plot_errors = ContainerWidget(children=[show_image,
                                                           plot_errors_button])
    opts = ContainerWidget(children=[plot_mode, show_legend])

    # Widget container
    iterations_result_wid = ContainerWidget(children=[
        but, iterations_mode_and_sliders, show_image_and_plot_errors, opts])

    # convert iterations to groups
    def convert_iterations_to_groups(from_iter, to_iter):
        return ["{}{}".format(iter_str, i) for i in range(from_iter, to_iter+1)]

    # Initialize variables
    iterations_result_wid.groups = convert_iterations_to_groups(0, 0)
    iterations_result_wid.show_image = show_image_default
    iterations_result_wid.subplots_enabled = subplots_enabled_default
    iterations_result_wid.legend_enabled = legend_default

    # Define iterations mode visibility
    def iterations_mode_selection(name, value):
        if value == 0:
            first_slider_wid.description = 'Iteration'
            first_slider_wid.min = 0
            first_slider_wid.max = n_iters-1
            second_slider_wid.visible = False
        else:
            first_slider_wid.description = 'From'
            first_slider_wid.min = 0
            first_slider_wid.max = n_iters-1
            second_slider_wid.min = 0
            second_slider_wid.max = n_iters-1
            if first_slider_wid.value == n_iters - 1:
                second_slider_wid.value = n_iters - 1
            else:
                second_slider_wid.value = first_slider_wid.value + 1
            second_slider_wid.visible = True
    iterations_mode.on_trait_change(iterations_mode_selection, 'value')

    # Define multiple channels sliders functionality
    def first_slider_val(name, value):
        if iterations_mode.value == 1 and value > second_slider_wid.value:
            first_slider_wid.value = second_slider_wid.value

    def second_slider_val(name, value):
        if iterations_mode.value == 1 and value < first_slider_wid.value:
            second_slider_wid.value = first_slider_wid.value

    def get_groups(name, value):
        if iterations_mode.value == 0:
            iterations_result_wid.groups = convert_iterations_to_groups(
                first_slider_wid.value, first_slider_wid.value)
        else:
            iterations_result_wid.groups = convert_iterations_to_groups(
                first_slider_wid.value, second_slider_wid.value)
    first_slider_wid.on_trait_change(first_slider_val, 'value')
    second_slider_wid.on_trait_change(second_slider_val, 'value')
    first_slider_wid.on_trait_change(get_groups, 'value')
    second_slider_wid.on_trait_change(get_groups, 'value')
    iterations_mode.on_trait_change(get_groups, 'value')

    # Show image function
    def show_image_fun(name, value):
        iterations_result_wid.show_image = value
    show_image.on_trait_change(show_image_fun, 'value')

    # Plot mode function
    def plot_mode_fun(name, value):
        iterations_result_wid.subplots_enabled = value
    plot_mode.on_trait_change(plot_mode_fun, 'value')

    # Legend function
    def legend_fun(name, value):
        iterations_result_wid.legend_enabled = value
    show_legend.on_trait_change(legend_fun, 'value')

    # Toggle button function
    def show_options(name, value):
        iterations_mode.visible = value
        show_image.visible = value
        opts.visible = value
        if value:
            if iterations_mode.value == 0:
                first_slider_wid.visible = True
            else:
                first_slider_wid.visible = True
                second_slider_wid.visible = True
        else:
            first_slider_wid.visible = False
            second_slider_wid.visible = False
    show_options('', toggle_show_default)
    but.on_trait_change(show_options, 'value')

    # assign plot_function
    if plot_function is not None:
        first_slider_wid.on_trait_change(plot_function, 'value')
        second_slider_wid.on_trait_change(plot_function, 'value')
        iterations_mode.on_trait_change(plot_function, 'value')
        show_image.on_trait_change(plot_function, 'value')
        plot_mode.on_trait_change(plot_function, 'value')
        show_legend.on_trait_change(plot_function, 'value')

    return iterations_result_wid


def format_iterations_result_options(iterations_result_wid,
                                     container_padding='6px',
                                     container_margin='6px',
                                     container_border='1px solid black',
                                     toggle_button_font_weight='bold',
                                     border_visible=True):
    r"""
    Function that corrects the align (style format) of a given
    iterations_result_options widget. Usage example:
        iterations_result_wid = iterations_result_options()
        display(iterations_result_wid)
        format_iterations_result_options(iterations_result_wid)

    Parameters
    ----------
    iterations_result_wid :
        The widget object generated by the `iterations_result_options()`
        function.

    container_padding : `str`, optional
        The padding around the widget, e.g. '6px'

    container_margin : `str`, optional
        The margin around the widget, e.g. '6px'

    container_border : `str`, optional
        The border around the widget, e.g. '1px solid black'

    toggle_button_font_weight : `str`
        The font weight of the toggle button, e.g. 'bold'

    border_visible : `boolean`, optional
        Defines whether to draw the border line around the widget.
    """
    # align sliders and iterations_mode
    iterations_result_wid.children[1].remove_class('vbox')
    iterations_result_wid.children[1].add_class('hbox')
    iterations_result_wid.children[1].add_class('align-start')

    # align show image and plot errors
    iterations_result_wid.children[2].remove_class('vbox')
    iterations_result_wid.children[2].add_class('hbox')
    iterations_result_wid.children[2].add_class('align-center')
    iterations_result_wid.children[2].children[0].set_css('margin-right',
                                                          '20px')

    # align plot_mode and legend options
    iterations_result_wid.children[3].remove_class('vbox')
    iterations_result_wid.children[3].add_class('hbox')
    iterations_result_wid.children[3].children[0].set_css('margin-right',
                                                          '20px')

    # set toggle button font bold
    iterations_result_wid.children[0].set_css('font-weight',
                                              toggle_button_font_weight)
    iterations_result_wid.children[1].set_css('margin-top', container_margin)

    # margin and border around container widget
    iterations_result_wid.set_css('padding', container_padding)
    iterations_result_wid.set_css('margin', container_margin)
    if border_visible:
        iterations_result_wid.set_css('border', container_border)


def _compare_groups_and_labels(groups1, labels1, groups2, labels2):
    r"""
    Function that compares two sets of landmarks groups and labels and returns
    Trues if they are identical else False.

    Parameters
    ----------
    group1 : `list` of `str`
        The first list of landmark groups.

    labels1 : `list` of `list` of `str`
        The first list of lists of each landmark group's labels.

    group2 : `list` of `str`
        The second list of landmark groups.

    labels2 : `list` of `list` of `str`
        The second list of lists of each landmark group's labels.
    """
    # function that compares two lists without taking into account the order
    def comp_lists(l1, l2):
        return len(l1) == len(l2) and \
               np.all([g1 == g2 for g1, g2 in zip(l1, l2)])

    # comparison of the given groups
    groups_same = comp_lists(groups1, groups2)

    # if groups are the same, compare the labels
    if groups_same:
        return len(labels1) == len(labels2) and \
               np.all([comp_lists(g1, g2) for g1, g2 in zip(labels1, labels2)])
    else:
        return False


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
    groups_keys = image.landmarks.keys()
    labels_keys = [image.landmarks[g].keys() for g in groups_keys]
    return groups_keys, labels_keys
