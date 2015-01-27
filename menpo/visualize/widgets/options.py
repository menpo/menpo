from collections import OrderedDict
from functools import partial

import numpy as np

from .tools import (colour_selection, format_colour_selection,
                    hog_options, format_hog_options, igo_options,
                    format_igo_options, lbp_options,
                    format_lbp_options, daisy_options,
                    format_daisy_options, _convert_image_to_bytes,
                    line_options, format_line_options,
                    update_line_options, marker_options,
                    format_marker_options, update_marker_options,
                    numbering_options, format_numbering_options,
                    update_numbering_options, figure_options,
                    format_figure_options, update_figure_options,
                    figure_options_two_scales,
                    format_figure_options_two_scales,
                    update_figure_options_two_scales, grid_options,
                    format_grid_options, update_grid_options,
                    index_selection_slider, index_selection_buttons,
                    format_index_selection, update_index_selection,
                    legend_options, format_legend_options,
                    update_legend_options, image_options, format_image_options,
                    update_image_options)


def channel_options(channels_options_default, plot_function=None,
                    toggle_show_default=True, toggle_show_visible=True):
    r"""
    Creates a widget with Channel Options. Specifically, it has:
        1) Two radio buttons that select an options mode, depending on whether
           the user wants to visualize a "Single" or "Multiple" channels.
        2) If mode is "Single", the channel number is selected by one slider.
           If mode is "Multiple", the channel range is selected by two sliders.
        3) If mode is "Single" and the image has 3 channels, there is a checkbox
           option to visualize the image in RGB mode.
        4) If mode is "Multiple", there is a checkbox option to visualize the
           sum of the channels.
        5) If mode is "Multiple", there is a checkbox option to visualize the
           glyph.
        6) The glyph option is accompanied by a block size text field and a
           checkbox that enables negative values visualization.
        7) A checkbox that defines whether the masked image will be displayed.
        8) A toggle button that controls the visibility of all the above, i.e.
           the channel options.

    The structure of the widgets is the following:
        channel_options_wid.children = [toggle_button, all_but_toggle]
        all_but_toggle.children = [mode_and_masked, all_but_radio_buttons]
        mode_and_masked.children = [mode_radio_buttons, masked_checkbox]
        all_but_radio_buttons.children = [all_sliders, multiple_checkboxes]
        all_sliders.children = [first_slider, second_slider]
        multiple_checkboxes.children = [sum_checkbox, glyph_all, rgb_checkbox]
        glyph_all.children = [glyph_checkbox, glyph_options]
        glyph_options.children = [block_size_text, use_negative_checkbox]

    The returned widget saves the selected values in the following dictionary:
        channel_options_wid.selected_values

    To fix the alignment within this widget please refer to
    `format_channel_options()` function.

    To update the state of this widget, please refer to
    `update_channel_options()` function.

    Parameters
    ----------
    channels_options_default : `dict`
        The initial options. For example:
            channels_options_default = {'n_channels': 10,
                                        'image_is_masked': True,
                                        'channels': 0,
                                        'glyph_enabled': False,
                                        'glyph_block_size': 3,
                                        'glyph_use_negative': False,
                                        'sum_enabled': False,
                                        'masked_enabled': True}

    plot_function : `function` or None, optional
        The plot function that is executed when a widgets' value changes.
        If None, then nothing is assigned.

    toggle_show_default : `boolean`, optional
        Defines whether the options will be visible upon construction.

    toggle_show_visible : `boolean`, optional
        The visibility of the toggle button.
    """
    import IPython.html.widgets as ipywidgets
    # if image is not masked, then masked flag should be disabled
    if not channels_options_default['image_is_masked']:
        channels_options_default['masked_enabled'] = False

    # parse channels
    if isinstance(channels_options_default['channels'], list):
        if len(channels_options_default['channels']) == 1:
            mode_default = 'Single'
            first_slider_default = channels_options_default['channels'][0]
            second_slider_default = channels_options_default['n_channels'] - 1
        else:
            mode_default = 'Multiple'
            first_slider_default = min(channels_options_default['channels'])
            second_slider_default = max(channels_options_default['channels'])
    elif channels_options_default['channels'] is None:
        mode_default = 'Single'
        first_slider_default = 0
        second_slider_default = channels_options_default['n_channels'] - 1
    else:
        mode_default = 'Single'
        first_slider_default = channels_options_default['channels']
        second_slider_default = channels_options_default['n_channels'] - 1

    # Create all necessary widgets
    # If single channel, disable all options apart from masked
    but = ipywidgets.ToggleButtonWidget(description='Channels Options',
                                        value=toggle_show_default,
                                        visible=toggle_show_visible)
    mode = ipywidgets.RadioButtonsWidget(
        values=["Single", "Multiple"], value=mode_default, description='Mode:',
        visible=toggle_show_default,
        disabled=channels_options_default['n_channels'] == 1)
    masked = ipywidgets.CheckboxWidget(
        value=channels_options_default['masked_enabled'], description='Masked',
        visible=toggle_show_default and channels_options_default[
            'image_is_masked'])
    first_slider_wid = ipywidgets.IntSliderWidget(
        min=0, max=channels_options_default['n_channels'] - 1, step=1,
        value=first_slider_default, description='Channel',
        visible=toggle_show_default,
        disabled=channels_options_default['n_channels'] == 1)
    second_slider_wid = ipywidgets.IntSliderWidget(
        min=1, max=channels_options_default['n_channels'] - 1, step=1,
        value=second_slider_default, description='To',
        visible=mode_default == "Multiple")
    rgb_wid = ipywidgets.CheckboxWidget(
        value=channels_options_default['n_channels'] == 3 and
              channels_options_default['channels'] is None,
        description='RGB',
        visible=toggle_show_default and channels_options_default[
                                            'n_channels'] == 3)
    sum_wid = ipywidgets.CheckboxWidget(
        value=channels_options_default['sum_enabled'], description='Sum',
        visible=False, disabled=channels_options_default['n_channels'] == 1)
    glyph_wid = ipywidgets.CheckboxWidget(
        value=channels_options_default['glyph_enabled'], description='Glyph',
        visible=False, disabled=channels_options_default['n_channels'] == 1)
    glyph_block_size = ipywidgets.BoundedIntTextWidget(
        description='Block size', min=1, max=25,
        value=channels_options_default['glyph_block_size'], visible=False,
        disabled=channels_options_default['n_channels'] == 1)
    glyph_use_negative = ipywidgets.CheckboxWidget(
        description='Negative values',
        value=channels_options_default['glyph_use_negative'], visible=False,
        disabled=channels_options_default['n_channels'] == 1)

    # Group widgets
    glyph_options = ipywidgets.ContainerWidget(children=[glyph_block_size,
                                                         glyph_use_negative])
    glyph_all = ipywidgets.ContainerWidget(children=[glyph_wid, glyph_options])
    multiple_checkboxes = ipywidgets.ContainerWidget(
        children=[sum_wid, glyph_all, rgb_wid])
    sliders = ipywidgets.ContainerWidget(
        children=[first_slider_wid, second_slider_wid])
    all_but_radiobuttons = ipywidgets.ContainerWidget(
        children=[sliders, multiple_checkboxes])
    mode_and_masked = ipywidgets.ContainerWidget(children=[mode, masked])
    all_but_toggle = ipywidgets.ContainerWidget(children=[mode_and_masked,
                                                          all_but_radiobuttons])

    # Widget container
    channel_options_wid = ipywidgets.ContainerWidget(
        children=[but, all_but_toggle])

    # Initialize output variables
    channel_options_wid.selected_values = channels_options_default

    # Define mode visibility
    def mode_selection_fun(name, value):
        if value == 'Single':
            first_slider_wid.description = 'Channel'
            first_slider_wid.min = 0
            first_slider_wid.max = \
                channel_options_wid.selected_values['n_channels'] - 1
            second_slider_wid.visible = False
            sum_wid.visible = False
            sum_wid.value = False
            glyph_wid.visible = False
            glyph_wid.value = False
            glyph_block_size.visible = False
            glyph_block_size.value = '3'
            glyph_use_negative.visible = False
            glyph_use_negative.value = False
            rgb_wid.visible = \
                channel_options_wid.selected_values['n_channels'] == 3
            rgb_wid.value = \
                channel_options_wid.selected_values['n_channels'] == 3
        else:
            first_slider_wid.description = 'From'
            first_slider_wid.min = 0
            first_slider_wid.max = \
                channel_options_wid.selected_values['n_channels'] - 1
            second_slider_wid.min = 0
            second_slider_wid.max = \
                channel_options_wid.selected_values['n_channels'] - 1
            second_slider_wid.value = first_slider_wid.value
            second_slider_wid.visible = True
            rgb_wid.visible = False
            rgb_wid.value = False
            sum_wid.visible = True
            sum_wid.value = False
            glyph_wid.visible = True
            glyph_wid.value = False
            glyph_block_size.visible = False
            glyph_block_size.value = '3'
            glyph_use_negative.visible = False
            glyph_use_negative.value = False

    mode_selection_fun('', mode_default)
    mode.on_trait_change(mode_selection_fun, 'value')

    # Define glyph visibility
    def glyph_options_visibility_fun(name, value):
        glyph_block_size.visible = glyph_wid.value
        glyph_use_negative.visible = glyph_wid.value
        if glyph_wid.value:
            sum_wid.value = False

    glyph_wid.on_trait_change(glyph_options_visibility_fun, 'value')
    glyph_wid.on_trait_change(glyph_options_visibility_fun, 'visible')

    # Define rgb functionality
    def rgb_fun(name, value):
        first_slider_wid.disabled = value

    rgb_wid.on_trait_change(rgb_fun, 'value')

    # Define sum functionality
    def sum_fun(name, value):
        if value:
            glyph_wid.value = False

    sum_wid.on_trait_change(sum_fun, 'value')

    # Define masked functionality
    def masked_fun(name, value):
        channel_options_wid.masked_enabled = value

    masked.on_trait_change(masked_fun, 'value')

    # Function that gets glyph/sum options
    def get_glyph_options(name, value):
        channel_options_wid.selected_values['glyph_enabled'] = glyph_wid.value
        channel_options_wid.selected_values['sum_enabled'] = sum_wid.value
        channel_options_wid.selected_values[
            'glyph_use_negative'] = glyph_use_negative.value
        channel_options_wid.selected_values[
            'glyph_block_size'] = glyph_block_size.value
        if channel_options_wid.selected_values['sum_enabled']:
            channel_options_wid.selected_values['glyph_block_size'] = 1

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
            if rgb_wid.value:
                channel_options_wid.selected_values['channels'] = None
            else:
                channel_options_wid.selected_values[
                    'channels'] = first_slider_wid.value
        else:
            channel_options_wid.selected_values['channels'] = range(
                first_slider_wid.value,
                second_slider_wid.value + 1)

    first_slider_wid.on_trait_change(first_slider_val, 'value')
    second_slider_wid.on_trait_change(second_slider_val, 'value')
    first_slider_wid.on_trait_change(get_channels, 'value')
    second_slider_wid.on_trait_change(get_channels, 'value')
    rgb_wid.on_trait_change(get_channels, 'value')
    mode.on_trait_change(get_channels, 'value')

    def get_masked(name, value):
        channel_options_wid.selected_values['masked_enabled'] = value

    masked.on_trait_change(get_masked, 'value')

    # Toggle button function
    def toggle_image_options(name, value):
        mode.visible = value
        if value:
            masked.visible = channel_options_wid.selected_values[
                'image_is_masked']
            if mode.value == 'Single':
                first_slider_wid.visible = True
                visible = channel_options_wid.selected_values['n_channels'] == 3
                rgb_wid.visible = visible
            else:
                first_slider_wid.visible = True
                second_slider_wid.visible = True
                sum_wid.visible = True
                glyph_wid.visible = True
        else:
            masked.visible = False
            first_slider_wid.visible = False
            second_slider_wid.visible = False
            rgb_wid.visible = False
            sum_wid.visible = False
            glyph_wid.visible = False
            glyph_options.children[0].visible = False
            glyph_options.children[1].visible = False

    but.on_trait_change(toggle_image_options, 'value')

    # assign plot_function
    if plot_function is not None:
        # mode.on_trait_change(plot_function, 'value')
        masked.on_trait_change(plot_function, 'value')
        first_slider_wid.on_trait_change(plot_function, 'value')
        second_slider_wid.on_trait_change(plot_function, 'value')
        rgb_wid.on_trait_change(plot_function, 'value')
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
    channel_options_wid.children[1].children[1].children[1].children[
        1].children[1].remove_class('vbox')
    channel_options_wid.children[1].children[1].children[1].children[
        1].children[1].add_class('hbox')
    channel_options_wid.children[1].children[1].children[1].children[
        1].children[1].children[0].set_css('width', '0.8cm')

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


def update_channel_options(channel_options_wid, n_channels, image_is_masked,
                           masked_default=False):
    r"""
    Function that updates the state of a given channel_options widget if the
    image's number of channels or its masked flag has changed. Usage example:

    ::

        channel_options_wid = channel_options(n_channels=2,
                                              image_is_masked=True)
        display(channel_options_wid)
        format_channel_options(channel_options_wid)
        update_channel_options(channel_options_wid, n_channels=36,
                               image_is_masked=False)

    Parameters
    ----------
    channel_options_wid : widget
        The widget object generated by the `channel_options()` function.
    n_channels : `int`
        The number of channels.
    image_is_masked : `boolean`
        Flag that defines whether the image is an instance of :map:`MaskedImage`
        or subclass.
    masked_default : `boolean`, optional
        The value to be assigned at the masked checkbox.
    """
    # if image_is_masked flag has actually changed from the previous value
    if image_is_masked != channel_options_wid.selected_values['image_is_masked']:
        # change the channel_options output
        channel_options_wid.selected_values['image_is_masked'] = image_is_masked
        channel_options_wid.selected_values['masked_enabled'] = masked_default
        # set the masked checkbox state
        channel_options_wid.children[1].children[0].children[1].visible = \
            channel_options_wid.children[0].value and image_is_masked
        channel_options_wid.children[1].children[0].children[1].value = False

    # if n_channels are actually different from the previous value
    if n_channels != channel_options_wid.selected_values['n_channels']:
        # change the channel_options output
        channel_options_wid.selected_values['n_channels'] = n_channels
        channel_options_wid.selected_values['channels'] = 0
        # set the rgb checkbox state
        channel_options_wid.children[1].children[1].children[1].children[
            2].visible = \
            n_channels == 3 and channel_options_wid.children[0].value
        # set the channel options state (apart from the masked checkbox)
        if n_channels == 1:
            # set sliders max and min values
            channel_options_wid.children[1].children[1].children[0].children[
                0].max = 1
            channel_options_wid.children[1].children[1].children[0].children[
                1].max = 1
            channel_options_wid.children[1].children[1].children[0].children[
                0].min = 0
            channel_options_wid.children[1].children[1].children[0].children[
                1].min = 0
            # set sliders state
            channel_options_wid.children[1].children[1].children[0].children[
                0].disabled = True
            channel_options_wid.children[1].children[1].children[0].children[
                1].visible = False
            # set glyph/sum state
            channel_options_wid.children[1].children[1].children[1].children[
                0].disabled = True
            channel_options_wid.children[1].children[1].children[1].children[
                1].children[0].disabled = True
            channel_options_wid.children[1].children[1].children[1].children[
                1].children[1].children[0].disabled = True
            channel_options_wid.children[1].children[1].children[1].children[
                1].children[1].children[1].disabled = True
            # set mode state
            channel_options_wid.children[1].children[0].children[
                0].disabled = True
            # set mode and sliders values
            for k in range(4):
                if k == 0:
                    channel_options_wid.children[1].children[1].children[
                        1].children[2].value = False
                elif k == 1:
                    channel_options_wid.children[1].children[0].children[
                        0].value = "Single"
                elif k == 2:
                    channel_options_wid.children[1].children[1].children[
                        0].children[0].value = 0
                else:
                    channel_options_wid.children[1].children[1].children[
                        0].children[1].value = 0
        else:
            # set sliders max and min values
            channel_options_wid.children[1].children[1].children[0].children[
                0].max = n_channels - 1
            channel_options_wid.children[1].children[1].children[0].children[
                1].max = n_channels - 1
            channel_options_wid.children[1].children[1].children[0].children[
                0].min = 0
            channel_options_wid.children[1].children[1].children[0].children[
                1].min = 0
            # set sliders state
            channel_options_wid.children[1].children[1].children[0].children[
                0].disabled = False
            channel_options_wid.children[1].children[1].children[0].children[
                1].visible = False
            # set glyph/sum state
            channel_options_wid.children[1].children[1].children[1].children[
                0].disabled = False
            channel_options_wid.children[1].children[1].children[1].children[
                1].children[0].disabled = False
            channel_options_wid.children[1].children[1].children[1].children[
                1].children[1].children[0].disabled = False
            channel_options_wid.children[1].children[1].children[1].children[
                1].children[1].children[1].disabled = False
            # set mode state
            channel_options_wid.children[1].children[0].children[
                0].disabled = False
            # set mode and sliders values
            for k in range(4):
                if k == 0:
                    channel_options_wid.children[1].children[1].children[
                        1].children[2].value = n_channels == 3
                elif k == 1:
                    channel_options_wid.children[1].children[0].children[
                        0].value = "Single"
                elif k == 2:
                    channel_options_wid.children[1].children[1].children[
                        0].children[0].value = 0
                else:
                    channel_options_wid.children[1].children[1].children[
                        0].children[1].value = 0


def landmark_options(landmark_options_default, plot_function=None,
                     toggle_show_default=True, toggle_show_visible=True):
    r"""
    Creates a widget with Landmark Options. Specifically, it has:
        1) A checkbox that controls the landmarks' visibility.
        2) A drop down menu with the available landmark groups.
        3) Several toggle buttons with the group's available labels.
        4) A toggle button that controls the visibility of all the above, i.e.
           the landmark options.

    The structure of the widgets is the following:
        landmark_options_wid.children = [toggle_button, landmarks_checkbox,
                                         groups]
        groups.children = [group_drop_down_menu, labels]
        labels.children = [labels_text, labels_toggle_buttons]

    The returned widget saves the selected values in the following dictionary:
        landmark_options_wid.selected_values

    To fix the alignment within this widget, please refer to
    `format_landmark_options()` function.

    To update the state of this widget, please refer to
    `update_landmark_options()` function.

    Parameters
    ----------
    landmark_options_default : `dict`
        The default options. For example ::

            landmark_options_default = {'render_landmarks': True,
                                        'group_keys': ['PTS', 'ibug_face_68'],
                                        'labels_keys': [['all'], ['jaw', 'eye'],
                                        'group': 'PTS',
                                        'with_labels': ['all']}

    plot_function : `function` or None, optional
        The plot function that is executed when a widgets' value changes.
        If None, then nothing is assigned.
    toggle_show_default : `bool`, optional
        Defines whether the options will be visible upon construction.
    toggle_show_visible : `bool`, optional
        The visibility of the toggle button.
    """
    import IPython.html.widgets as ipywidgets
    # Create all necessary widgets
    but = ipywidgets.ToggleButtonWidget(description='Landmarks Options',
                                        value=toggle_show_default,
                                        visible=toggle_show_visible)
    landmarks = ipywidgets.CheckboxWidget(description='Render landmarks',
                                          value=landmark_options_default[
                                              'render_landmarks'])
    group = ipywidgets.DropdownWidget(
        values=landmark_options_default['group_keys'],
        description='Group')
    labels_toggles = [[ipywidgets.ToggleButtonWidget(description=k, value=True)
                       for k in s_keys]
                      for s_keys in landmark_options_default['labels_keys']]
    labels_text = ipywidgets.LatexWidget(value='Labels')
    labels = ipywidgets.ContainerWidget(children=labels_toggles[0])

    # Group widgets
    labels_and_text = ipywidgets.ContainerWidget(children=[labels_text, labels])
    group_wid = ipywidgets.ContainerWidget(children=[group, labels_and_text])

    # Widget container
    landmark_options_wid = ipywidgets.ContainerWidget(
        children=[but, landmarks, group_wid])

    # Initialize output variables
    landmark_options_wid.selected_values = landmark_options_default
    landmark_options_wid.selected_values['labels_toggles'] = labels_toggles
    landmark_options_wid.selected_values['group'] = \
        landmark_options_wid.selected_values['group_keys'][0]
    landmark_options_wid.selected_values['with_labels'] = \
        landmark_options_wid.selected_values['labels_keys'][0]

    # Disability control
    def landmarks_fun(name, value):
        # get landmarks_enabled value
        landmark_options_wid.selected_values['render_landmarks'] = value
        # disable group drop down menu
        group.disabled = not value
        # disable all labels toggles
        for s_keys in landmark_options_wid.selected_values['labels_toggles']:
            for k in s_keys:
                k.disabled = not value
        # if all currently selected labels toggles are False, set them all
        # to True
        all_values = [ww.value for ww in labels.children]
        if all(item is False for item in all_values):
            for ww in labels.children:
                ww.value = True

    landmarks_fun('', landmark_options_wid.selected_values['render_landmarks'])
    landmarks.on_trait_change(landmarks_fun, 'value')

    # Group drop down method
    def group_fun(name, value):
        # get group value
        landmark_options_wid.selected_values['group'] = value
        # assign the correct children to the labels toggles
        labels_toggles = landmark_options_wid.selected_values['labels_toggles']
        group_keys = landmark_options_wid.selected_values['group_keys']
        labels.children = labels_toggles[group_keys.index(value)]
        # get with_labels value
        landmark_options_wid.selected_values['with_labels'] = []
        for ww in labels.children:
            if ww.value:
                landmark_options_wid.selected_values['with_labels'].append(
                    str(ww.description))
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
        landmark_options_wid.selected_values['with_labels'] = []
        for ww in labels.children:
            if ww.value:
                landmark_options_wid.selected_values['with_labels'].append(
                    str(ww.description))

    # assign labels_fun to all labels toggles (even hidden ones)
    for s_group in landmark_options_wid.selected_values['labels_toggles']:
        for w in s_group:
            w.on_trait_change(labels_fun, 'value')

    # Toggle button function
    def show_options(name, value):
        group_wid.visible = value
        landmarks.visible = value

    show_options('', toggle_show_default)
    but.on_trait_change(show_options, 'value')

    # assign plot_function
    if plot_function is not None:
        # assign plot_function to landmarks checkbox and group drop down menu
        landmarks.on_trait_change(plot_function, 'value')
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
    landmark_options_wid.children[2].add_class('align-start')

    # set toggle button font bold
    landmark_options_wid.children[0].set_css('font-weight',
                                             toggle_button_font_weight)

    # margin and border around container widget
    landmark_options_wid.set_css('padding', container_padding)
    landmark_options_wid.set_css('margin', container_margin)
    if border_visible:
        landmark_options_wid.set_css('border', container_border)


def update_landmark_options(landmark_options_wid, group_keys, labels_keys,
                            plot_function):
    r"""
    Function that updates the state of a given landmark_options widget if the
    group or label keys of an image has changed. Usage example:
        landmark_options_default = {'render_landmarks': True,
                                    'group_keys': ['PTS', 'ibug_face_68'],
                                    'labels_keys': [['all'], ['jaw', 'nose'],
                                    'group': 'PTS',
                                    'with_labels': ['all']}
        landmark_options_wid = landmark_options(landmark_options_default)
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
    import IPython.html.widgets as ipywidgets
    # check if the new group_keys and labels_keys are the same as the old
    # ones
    if not _compare_groups_and_labels(
            group_keys, labels_keys,
            landmark_options_wid.selected_values['group_keys'],
            landmark_options_wid.selected_values['labels_keys']):
        # Create all necessary widgets
        group = ipywidgets.DropdownWidget(values=group_keys,
                                          description='Group')
        labels_toggles = [
            [ipywidgets.ToggleButtonWidget(description=k, value=True)
             for k in s_keys] for s_keys in labels_keys]

        # Group widgets
        landmark_options_wid.children[2].children[1].children[1]. \
            children = labels_toggles[0]
        labels = landmark_options_wid.children[2].children[1]
        cont = ipywidgets.ContainerWidget(children=[group, labels])
        landmark_options_wid.children = [landmark_options_wid.children[0],
                                         landmark_options_wid.children[1],
                                         cont]

        # Initialize output variables
        landmark_options_wid.selected_values['group_keys'] = group_keys
        landmark_options_wid.selected_values['labels_keys'] = labels_keys
        landmark_options_wid.selected_values['labels_toggles'] = labels_toggles
        landmark_options_wid.selected_values['group'] = group_keys[0]
        landmark_options_wid.selected_values['with_labels'] = labels_keys[0]

        # Disability control
        def landmarks_fun(name, value):
            # get landmarks_enabled value
            landmark_options_wid.selected_values['render_landmarks'] = value
            # disable group drop down menu
            group.disabled = not value
            # disable all labels toggles
            for s_key in landmark_options_wid.selected_values['labels_toggles']:
                for k in s_key:
                    k.disabled = not value
            # if all currently selected labels toggles are False, set them all
            # to True
            children = (landmark_options_wid.children[2].children[1]
                                            .children[1].children)
            all_values = [ww.value for ww in children]
            if all(item is False for item in all_values):
                children = (landmark_options_wid.children[2].children[1]
                                                .children[1].children)
                for ww in children:
                    ww.value = True

        landmark_options_wid.children[1].on_trait_change(landmarks_fun, 'value')
        landmarks_fun('',
                      landmark_options_wid.selected_values['render_landmarks'])

        # Group drop down method
        def group_fun(name, value):
            # get group value
            landmark_options_wid.selected_values['group'] = value
            # assign the correct children to the labels toggles
            landmark_options_wid.children[2].children[1].children[1]. \
                children = \
            landmark_options_wid.selected_values['labels_toggles'][
                landmark_options_wid.selected_values['group_keys'].index(value)]
            # get with_labels value
            landmark_options_wid.selected_values['with_labels'] = []
            for ww in landmark_options_wid.children[2].children[1].children[
                1].children:
                if ww.value:
                    landmark_options_wid.selected_values['with_labels'].append(
                        str(ww.description))
            # assign plot_function to all enabled labels
            if plot_function is not None:
                children = (landmark_options_wid.children[2].children[1]
                                                .children[1].children)
                for w in children:
                    w.on_trait_change(plot_function, 'value')

        group.on_trait_change(group_fun, 'value')

        # Labels function
        def labels_fun(name, value):
            # if all labels toggles are False, set landmarks checkbox to False
            all_values = [ww.value
                          for ww in
                          landmark_options_wid.children[2].children[1].children[
                              1].children]
            if all(item is False for item in all_values):
                landmark_options_wid.children[1].value = False
            # get with_labels value
            landmark_options_wid.selected_values['with_labels'] = []
            for ww in landmark_options_wid.children[2].children[1].children[1]. \
                    children:
                if ww.value:
                    landmark_options_wid.selected_values['with_labels'].append(
                        str(ww.description))

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

        # If there is only one group with value ' ', this means that the image
        # didn't have any landmarks. So disable the show_landmarks checkbox.
        if len(group_keys) == 1 and group_keys[0] == ' ':
            # No landmarks are provided. So disable the show landmarks checkbox
            landmark_options_wid.children[1].value = False
            landmark_options_wid.children[1].disabled = True
        else:
            if landmark_options_wid.children[1].disabled:
                landmark_options_wid.children[1].disabled = False
                landmark_options_wid.children[1].value = True


def info_print(n_bullets, toggle_show_default=True, toggle_show_visible=True):
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
    import IPython.html.widgets as ipywidgets
    # Create toggle button
    but = ipywidgets.ToggleButtonWidget(description='Info',
                                        value=toggle_show_default,
                                        visible=toggle_show_visible)

    # Create text widget
    children = [ipywidgets.LatexWidget(value="> menpo")
                for _ in xrange(n_bullets)]
    text_wid = ipywidgets.ContainerWidget(children=children)

    # Toggle button function
    def show_options(name, value):
        text_wid.visible = value

    show_options('', toggle_show_default)
    but.on_trait_change(show_options, 'value')

    # Group widgets
    info_wid = ipywidgets.ContainerWidget(children=[but, text_wid])

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
    # text widget formatting
    info_wid.children[1].set_css({'border': container_border,
                                  'padding': '4px',
                                  'margin-top': '1px'})

    # set font size
    for w in info_wid.children[1].children:
        w.set_css({'font-size': font_size_in_pt})

    # set toggle button font bold
    info_wid.children[0].set_css('font-weight', toggle_button_font_weight)

    # margin and border around container widget
    info_wid.set_css('padding', container_padding)
    info_wid.set_css('margin', container_margin)
    if border_visible:
        info_wid.set_css('border', container_border)


def animation_options(index_selection_default, plot_function=None,
                      update_function=None, index_description='Image Number',
                      index_minus_description='-', index_plus_description='+',
                      index_style='buttons', index_text_editable=True,
                      loop_default=False, interval_default=0.5,
                      toggle_show_title='Image Options',
                      toggle_show_default=True, toggle_show_visible=True):
    r"""
    Creates a widget for selecting an index and creating animations.
    Specifically, it has:
        1) An index selection widget. It can either be a slider or +/- buttons.
        2) A play toggle button.
        3) A stop toggle button.
        4) An options toggle button.
        If the options toggle is pressed, the following appear:
        5) An interval text area.
        6) A loop check box.

    The structure of the widget is the following:
        animation_options_wid.children = [toggle_button, options]
        options.children = [index_selection, animation]
        if index_style == 'buttons':
            index_selection.children = [title, minus_button, index_text,
                                        plus_button] (index_selection_buttons())
        elif index_style == 'slider':
            index_selection = index_slider (index_selection_slider())
        animation.children = [buttons, animation_options]
        buttons.children = [play_button, stop_button, play_options_button]
        animation_options.children = [interval_text, loop_checkbox]

    The returned widget saves the selected values in the following dictionary:
        animation_options_wid.selected_values
        animation_options_wid.index_style

    To fix the alignment within this widget please refer to
    `format_animation_options()` function.

    To update the state of this widget, please refer to
    `update_animation_options()` function.

    Parameters
    ----------
    index_selection_default : `dict`
        The dictionary with the default options. For example:
            index_selection_default = {'min':0,
                                       'max':100,
                                       'step':1,
                                       'index':10}

    plot_function : `function` or None, optional
        The plot function that is executed when the index value changes.
        If None, then nothing is assigned.

    update_function : `function` or None, optional
        The update function that is executed when the index value changes.
        If None, then nothing is assigned.

    index_description : `str`, optional
        The title of the index widget.

    index_minus_description : `str`, optional
        The title of the button that decreases the index.

    index_plus_description : `str`, optional
        The title of the button that increases the index.

    index_style : {``buttons`` or ``slider``}, optional
        If 'buttons', then 'index_selection_buttons()' is called.
        If 'slider', then 'index_selection_slider()' is called.

    index_text_editable : `boolean`, optional
        Flag that determines whether the index text will be editable.

    loop_default : `boolean`, optional
        If True, the animation makes loop.
        If False, the animation stops when reaching the index_max_value.

    interval_default : `float`, optional
        The interval between the animation frames.

    toggle_show_title : `str`, optional
        The title of the toggle button.

    toggle_show_default : `boolean`, optional
        Defines whether the options will be visible upon construction.

    toggle_show_visible : `boolean`, optional
        The visibility of the toggle button.
    """
    from time import sleep
    from IPython import get_ipython
    import IPython.html.widgets as ipywidgets

    # get the kernel to use it later in order to make sure that the widgets'
    # traits changes are passed during a while-loop
    kernel = get_ipython().kernel

    # Create index widget
    if index_style == 'slider':
        index_wid = index_selection_slider(index_selection_default,
                                           plot_function=plot_function,
                                           update_function=update_function,
                                           description=index_description)
    elif index_style == 'buttons':
        index_wid = index_selection_buttons(
            index_selection_default, plot_function=plot_function,
            update_function=update_function, description=index_description,
            minus_description=index_minus_description,
            plus_description=index_plus_description, loop=loop_default,
            text_editable=index_text_editable)

    # Create other widgets
    but = ipywidgets.ToggleButtonWidget(description=toggle_show_title,
                                        value=toggle_show_default,
                                        visible=toggle_show_visible)
    play_but = ipywidgets.ToggleButtonWidget(description='Play >', value=False)
    stop_but = ipywidgets.ToggleButtonWidget(description='Stop', value=True,
                                             disabled=True)
    play_options = ipywidgets.ToggleButtonWidget(description='Options',
                                                 value=False)
    loop = ipywidgets.CheckboxWidget(description='Loop', value=loop_default,
                                     visible=False)
    interval = ipywidgets.FloatTextWidget(description='Interval (sec)',
                                          value=interval_default, visible=False)

    # Widget container
    tmp_options = ipywidgets.ContainerWidget(children=[interval, loop])
    buttons = ipywidgets.ContainerWidget(
        children=[play_but, stop_but, play_options])
    animation = ipywidgets.ContainerWidget(children=[buttons, tmp_options])
    cont = ipywidgets.ContainerWidget(children=[index_wid, animation])
    animation_options_wid = ipywidgets.ContainerWidget(children=[but, cont])

    # Initialize variables
    animation_options_wid.selected_values = index_selection_default
    animation_options_wid.index_style = index_style

    # Play button pressed
    def play_press(name, value):
        stop_but.value = not value
        play_but.disabled = value
        play_options.disabled = value
        if value:
            play_options.value = False
    play_but.on_trait_change(play_press, 'value')

    # Stop button pressed
    def stop_press(name, value):
        play_but.value = not value
        stop_but.disabled = value
        play_options.disabled = not value
    stop_but.on_trait_change(stop_press, 'value')

    # show animation options checkbox function
    def play_options_fun(name, value):
        interval.visible = value
        loop.visible = value
    play_options.on_trait_change(play_options_fun, 'value')

    # animation function
    def play_fun(name, value):
        if loop.value:
            # loop is enabled
            i = animation_options_wid.selected_values['index']
            if i < animation_options_wid.selected_values['max']:
                i += animation_options_wid.selected_values['step']
            else:
                i = animation_options_wid.selected_values['min']

            ani_max_selected = animation_options_wid.selected_values['max']
            while i <= ani_max_selected and not stop_but.value:
                # update index value
                if index_style == 'slider':
                    index_wid.value = i
                else:
                    index_wid.children[2].value = i

                # Run IPython iteration.
                # This is the code that makes this operation non-blocking. This
                # will allow widget messages and callbacks to be processed.
                kernel.do_one_iteration()

                # update counter
                if i < animation_options_wid.selected_values['max']:
                    i += animation_options_wid.selected_values['step']
                else:
                    i = animation_options_wid.selected_values['min']

                # wait
                sleep(interval.value)
        else:
            # loop is disabled
            i = animation_options_wid.selected_values['index']
            i += animation_options_wid.selected_values['step']
            while (i <= animation_options_wid.selected_values['max'] and
                       not stop_but.value):
                # update value
                if index_style == 'slider':
                    index_wid.value = i
                else:
                    index_wid.children[2].value = i

                # Run IPython iteration.
                # This is the code that makes this operation non-blocking. This
                # will allow widget messages and callbacks to be processed.
                kernel.do_one_iteration()

                # update counter
                i += animation_options_wid.selected_values['step']

                # wait
                sleep(interval.value)
            if i > index_selection_default['max']:
                stop_but.value = True
    play_but.on_trait_change(play_fun, 'value')

    # Toggle button function
    def show_options(name, value):
        index_wid.visible = value
        buttons.visible = value
        interval.visible = False
        loop.visible = False
        if value:
            play_options.value = False
    show_options('', toggle_show_default)
    but.on_trait_change(show_options, 'value')

    return animation_options_wid


def format_animation_options(animation_options_wid, index_text_width='0.5cm',
                             container_padding='6px', container_margin='6px',
                             container_border='1px solid black',
                             toggle_button_font_weight='bold',
                             border_visible=True):
    r"""
    Function that corrects the align (style format) of a given animation_options
    widget. Usage example:
        animation_options_wid = animation_options()
        display(animation_options_wid)
        format_animation_options(animation_options_wid)

    Parameters
    ----------
    animation_options_wid :
        The widget object generated by the `animation_options()`
        function.

    index_text_width : `str`, optional
        The width of the index value text area.

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
    # format index widget
    format_index_selection(animation_options_wid.children[1].children[0],
                           text_width=index_text_width)

    # align play/stop button with animation options button
    animation_options_wid.children[1].children[1].children[0].remove_class(
        'vbox')
    animation_options_wid.children[1].children[1].children[0].add_class('hbox')
    animation_options_wid.children[1].children[1].add_class('align-end')

    # add margin on the right of the play button
    animation_options_wid.children[1].children[1].children[0].children[1]. \
        set_css('margin-right', container_margin)

    if animation_options_wid.index_style == 'slider':
        # align animation on the right of slider
        animation_options_wid.children[1].add_class('align-end')
    else:
        # align animation and index buttons
        animation_options_wid.children[1].remove_class('vbox')
        animation_options_wid.children[1].add_class('hbox')
        animation_options_wid.children[1].add_class('align-center')
        animation_options_wid.children[1].children[0].set_css('margin-right',
                                                              '1cm')

    # set interval width
    animation_options_wid.children[1].children[1].children[1].children[0]. \
        set_css('width', '20px')

    # set toggle button font bold
    animation_options_wid.children[0].set_css('font-weight',
                                              toggle_button_font_weight)

    # margin and border around container widget
    animation_options_wid.set_css('padding', container_padding)
    animation_options_wid.set_css('margin', container_margin)
    if border_visible:
        animation_options_wid.set_css('border', container_border)


def update_animation_options(animation_options_wid, index_selection_default,
                             plot_function=None, update_function=None):
    r"""
    Function that updates the state of a given animation_options widget if the
    index bounds have changed. Usage example:
        index_selection_default = {'min':0,
                                   'max':100,
                                   'step':1,
                                   'index':10}
        animation_options_wid = animation_options(index_selection_default)
        display(animation_options_wid)
        format_animation_options(animation_options_wid)
        index_selection_default = {'min':0,
                                   'max':10,
                                   'step':5,
                                   'index':5}
        update_animation_options(animation_options_wid, index_selection_default)

    Parameters
    ----------
    animation_options_wid :
        The widget object generated by either the `animation_options()`
        function.

    index_selection_default : `dict`
        The dictionary with the default options. For example:
            index_selection_default = {'min':0,
                                       'max':100,
                                       'step':1,
                                       'index':10}

    plot_function : `function` or None, optional
        The plot function that is executed when the index value changes.
        If None, then nothing is assigned.

    update_function : `function` or None, optional
        The update function that is executed when the index value changes.
        If None, then nothing is assigned.
    """
    update_index_selection(animation_options_wid.children[1].children[0],
                           index_selection_default,
                           plot_function=plot_function,
                           update_function=update_function)


def viewer_options(viewer_options_default, options_tabs, objects_names=None,
                   labels=None, plot_function=None, toggle_show_visible=True,
                   toggle_show_default=True):
    r"""
    Creates a widget with Viewer Options. Specifically, it has:
        1) A drop down menu for object selection.
        2) A tab widget with any of line, marker, numbers and feature options
        3) A toggle button that controls the visibility of all the above, i.e.
           the viewer options.

    The structure of the widgets is the following:
        viewer_options_wid.children = [toggle_button, options]
        options.children = [selection_menu, tab_options]
        tab_options.children = [line_options, marker_options,
                                numbers_options, figure_options, legend_options]

    The returned widget saves the selected values in the following dictionary:
        viewer_options_wid.selected_values

    To fix the alignment within this widget please refer to
    `format_viewer_options()` function.

    Parameters
    ----------
    viewer_options_default : list of `dict`
        A list of dictionaries with the initial selected viewer options per
        object. Example:

            lines_options = {'render_lines': True,
                             'line_width': 1,
                             'line_colour': ['b'],
                             'line_style': '-'}

            markers_options = {'render_markers':True,
                               'marker_size':20,
                               'marker_face_colour':['r'],
                               'marker_edge_colour':['k'],
                               'marker_style':'o',
                               'marker_edge_width':1}

            numbers_options = {'render_numbering': True,
                               'numbers_font_name': 'serif',
                               'numbers_font_size': 10,
                               'numbers_font_style': 'normal',
                               'numbers_font_weight': 'normal',
                               'numbers_font_colour': ['k'],
                               'numbers_horizontal_align': 'center',
                               'numbers_vertical_align': 'bottom'}

            legend_options = {'render_legend':True,
                              'legend_title':'',
                              'legend_font_name':'serif',
                              'legend_font_style':'normal',
                              'legend_font_size':10,
                              'legend_font_weight':'normal',
                              'legend_marker_scale':1.,
                              'legend_location':2,
                              'legend_bbox_to_anchor':(1.05, 1.),
                              'legend_border_axes_pad':1.,
                              'legend_n_columns':1,
                              'legend_horizontal_spacing':1.,
                              'legend_vertical_spacing':1.,
                              'legend_border':True,
                              'legend_border_padding':0.5,
                              'legend_shadow':False,
                              'legend_rounded_corners':True}

            figure_options = {'x_scale': 1.,
                              'y_scale': 1.,
                              'render_axes': True,
                              'axes_font_name': 'serif',
                              'axes_font_size': 10,
                              'axes_font_style': 'normal',
                              'axes_font_weight': 'normal',
                              'axes_x_limits': None,
                              'axes_y_limits': None}

            grid_options = {'render_grid': True,
                            'grid_line_style': '--',
                            'grid_line_width': 0.5}

            viewer_options_default = {'lines': lines_options,
                                      'markers': markers_options,
                                      'numbering': numbering_options,
                                      'legend': legend_options,
                                      'figure': figure_options,
                                      'grid': grid_options}

    options_tabs : `list` of `str`
        List that defines the ordering of the options tabs. It can take one of
        {``lines``, ``markers``, ``numbering``, ``figure_one``, ``figure_two``,
        ``legend``, ``grid``}

    objects_names : `list` of `str`, optional
        A list with the names of the objects that will be used in the selection
        dropdown menu. If None, then the names will have the form ``%d``.

    plot_function : `function` or None, optional
        The plot function that is executed when a widgets' value changes.
        If None, then nothing is assigned.

    toggle_show_default : `boolean`, optional
        Defines whether the options will be visible upon construction.

    toggle_show_visible : `boolean`, optional
        The visibility of the toggle button.
    """
    import IPython.html.widgets as ipywidgets
    # make sure that viewer_options_default is list even with one member
    if not isinstance(viewer_options_default, list):
        viewer_options_default = [viewer_options_default]

    # find number of objects
    n_objects = len(viewer_options_default)
    selection_visible = n_objects > 1

    # Create widgets
    # toggle button
    but = ipywidgets.ToggleButtonWidget(description='Viewer Options',
                                        value=toggle_show_default,
                                        visible=toggle_show_visible)

    # select object drop down menu
    objects_dict = OrderedDict()
    if objects_names is None:
        for k in range(n_objects):
            objects_dict[str(k)] = k
    else:
        for k, g in enumerate(objects_names):
            objects_dict[g] = k
    selection = ipywidgets.DropdownWidget(values=objects_dict, value=0,
                                          description='Select',
                                          visible=(selection_visible and
                                                   toggle_show_default))

    # options widgets
    options_widgets = []
    tab_titles = []
    if labels is None:
        labels = [str(j) for j in range(len(options_tabs))]
    for j, o in enumerate(options_tabs):
        if o == 'lines':
            options_widgets.append(
                line_options(viewer_options_default[0]['lines'],
                             toggle_show_visible=False,
                             toggle_show_default=True,
                             plot_function=plot_function,
                             show_checkbox_title='Render lines',
                             labels=labels[j]))
            tab_titles.append('Lines')
        elif o == 'markers':
            options_widgets.append(
                marker_options(viewer_options_default[0]['markers'],
                               toggle_show_visible=False,
                               toggle_show_default=True,
                               plot_function=plot_function,
                               show_checkbox_title='Render markers'))
            tab_titles.append('Markers')
        elif o == 'image':
            options_widgets.append(
                image_options(viewer_options_default[0]['image'],
                              toggle_show_visible=False,
                              toggle_show_default=True,
                              plot_function=plot_function))
            tab_titles.append('Image')
        elif o == 'numbering':
            options_widgets.append(
                numbering_options(viewer_options_default[0]['numbering'],
                                  toggle_show_visible=False,
                                  toggle_show_default=True,
                                  plot_function=plot_function,
                                  show_checkbox_title='Render numbering'))
            tab_titles.append('Numbering')
        elif o == 'figure_one':
            options_widgets.append(
                figure_options(viewer_options_default[0]['figure'],
                               plot_function=plot_function,
                               figure_scale_bounds=(0.1, 4),
                               figure_scale_step=0.1, figure_scale_visible=True,
                               axes_visible=True, toggle_show_default=True,
                               toggle_show_visible=False))
            tab_titles.append('Figure/Axes')
        elif o == 'figure_two':
            options_widgets.append(
                figure_options_two_scales(
                    viewer_options_default[0]['figure'],
                    plot_function=plot_function, coupled_default=False,
                    figure_scales_bounds=(0.1, 4), figure_scales_step=0.1,
                    figure_scales_visible=True, axes_visible=True,
                    toggle_show_default=True, toggle_show_visible=False))
            tab_titles.append('Figure/Axes')
        elif o == 'legend':
            options_widgets.append(
                legend_options(viewer_options_default[0]['legend'],
                               toggle_show_visible=False,
                               toggle_show_default=True,
                               plot_function=plot_function,
                               show_checkbox_title='Render legend'))
            tab_titles.append('Legend')
        elif o == 'grid':
            options_widgets.append(
                grid_options(viewer_options_default[0]['grid'],
                             toggle_show_visible=False,
                             toggle_show_default=True,
                             plot_function=plot_function,
                             show_checkbox_title='Render grid'))
            tab_titles.append('Grid')
    options = ipywidgets.TabWidget(children=options_widgets)

    # Final widget
    all_options = ipywidgets.ContainerWidget(children=[selection, options])
    viewer_options_wid = ipywidgets.ContainerWidget(children=[but, all_options])

    # save tab titles and options str to widget in order to be passed to the
    # format function
    viewer_options_wid.tab_titles = tab_titles
    viewer_options_wid.options_tabs = options_tabs

    # Assign output list of dicts
    viewer_options_wid.selected_values = viewer_options_default

    # Update widgets' state
    def update_widgets(name, value):
        for i, tab in enumerate(options_tabs):
            if tab == 'lines':
                update_line_options(
                    options_widgets[i],
                    viewer_options_default[value]['lines'],
                    labels=labels[value])
            elif tab == 'markers':
                update_marker_options(
                    options_widgets[i],
                    viewer_options_default[value]['markers'])
            elif tab == 'image':
                update_image_options(
                    options_widgets[i],
                    viewer_options_default[value]['image'])
            elif tab == 'numbering':
                update_numbering_options(
                    options_widgets[i],
                    viewer_options_default[value]['numbering'])
            elif tab == 'figure_one':
                update_figure_options(
                    options_widgets[i],
                    viewer_options_default[value]['figure'])
            elif tab == 'figure_two':
                update_figure_options_two_scales(
                    options_widgets[i],
                    viewer_options_default[value]['figure'])
            elif tab == 'legend':
                update_legend_options(
                    options_widgets[i],
                    viewer_options_default[value]['legend'])
            elif tab == 'grid':
                update_grid_options(
                    options_widgets[i],
                    viewer_options_default[value]['grid'])
    selection.on_trait_change(update_widgets, 'value')

    # Toggle button function
    def toggle_fun(name, value):
        selection.visible = value and selection_visible
        options.visible = value
    toggle_fun('', toggle_show_default)
    but.on_trait_change(toggle_fun, 'value')

    return viewer_options_wid


def format_viewer_options(viewer_options_wid, container_padding='6px',
                          container_margin='6px',
                          container_border='1px solid black',
                          toggle_button_font_weight='bold',
                          border_visible=False, suboptions_border_visible=True):
    r"""
    Function that corrects the align (style format) of a given
    viewer_options widget. Usage example:
        viewer_options_wid = viewer_options(default_options)
        display(viewer_options_wid)
        format_viewer_options(viewer_options_wid)

    Parameters
    ----------
    viewer_options_wid :
        The widget object generated by the `viewer_options()` function.

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

    suboptions_border_visible : `boolean`, optional
        Defines whether to draw the border line around each of the sub options.
    """
    # format widgets
    for k, o in enumerate(viewer_options_wid.options_tabs):
        if o == 'lines':
            format_line_options(
                viewer_options_wid.children[1].children[1].children[k],
                suboptions_border_visible=suboptions_border_visible,
                border_visible=False)
        elif o == 'markers':
            format_marker_options(
                viewer_options_wid.children[1].children[1].children[k],
                suboptions_border_visible=suboptions_border_visible,
                border_visible=False)
        elif o == 'image':
            format_image_options(
                viewer_options_wid.children[1].children[1].children[k],
                border_visible=suboptions_border_visible)
        elif o == 'numbering':
            format_numbering_options(
                viewer_options_wid.children[1].children[1].children[k],
                suboptions_border_visible=suboptions_border_visible,
                border_visible=False)
        elif o == 'figure_one':
            format_figure_options(
                viewer_options_wid.children[1].children[1].children[k],
                border_visible=suboptions_border_visible)
        elif o == 'figure_two':
            format_figure_options_two_scales(
                viewer_options_wid.children[1].children[1].children[k],
                border_visible=suboptions_border_visible)
        elif o == 'legend':
            format_legend_options(
                viewer_options_wid.children[1].children[1].children[k],
                suboptions_border_visible=suboptions_border_visible,
                border_visible=False)
        elif o == 'grid':
            format_grid_options(
                viewer_options_wid.children[1].children[1].children[k],
                suboptions_border_visible=suboptions_border_visible,
                border_visible=False)

    # set titles
    for (k, tl) in enumerate(viewer_options_wid.tab_titles):
        viewer_options_wid.children[1].children[1].set_title(k, tl)

    # set toggle button font bold
    viewer_options_wid.children[0].set_css('font-weight',
                                           toggle_button_font_weight)

    # margin and border around container widget
    viewer_options_wid.set_css('padding', container_padding)
    viewer_options_wid.set_css('margin', container_margin)
    if border_visible:
        viewer_options_wid.set_css('border', container_border)


def update_viewer_options(viewer_options_wid, viewer_options_default,
                          labels=None):
    for k, o in enumerate(viewer_options_wid.options_tabs):
        if o == 'lines' and 'lines' in viewer_options_default:
            update_line_options(
                viewer_options_wid.children[1].children[1].children[k],
                viewer_options_default['lines'], labels=labels)
        elif o == 'markers' and 'markers' in viewer_options_default:
            update_marker_options(
                viewer_options_wid.children[1].children[1].children[k],
                viewer_options_default['markers'])
        elif o == 'image' and 'image' in viewer_options_default:
            update_image_options(
                viewer_options_wid.children[1].children[1].children[k],
                viewer_options_default['image'])
        elif o == 'numbering' and 'numbering' in viewer_options_default:
            update_numbering_options(
                viewer_options_wid.children[1].children[1].children[k],
                viewer_options_default['numbering'])
        elif o == 'figure_one' and 'figure' in viewer_options_default:
            update_figure_options(
                viewer_options_wid.children[1].children[1].children[k],
                viewer_options_default['figure'])
        elif o == 'figure_two' and 'figure' in viewer_options_default:
            update_figure_options(
                viewer_options_wid.children[1].children[1].children[k],
                viewer_options_default['figure'])
        elif o == 'legend' and 'legend' in viewer_options_default:
            update_legend_options(
                viewer_options_wid.children[1].children[1].children[k],
                viewer_options_default['legend'])
        elif o == 'grid' and 'grid' in viewer_options_default:
            update_grid_options(
                viewer_options_wid.children[1].children[1].children[k],
                viewer_options_default['grid'])


def save_figure_options(renderer, format_default='png', dpi_default=None,
                        orientation_default='portrait',
                        papertype_default='letter', transparent_default=False,
                        facecolour_default='w', edgecolour_default='w',
                        pad_inches_default=0.5, overwrite_default=False,
                        toggle_show_default=True, toggle_show_visible=True):
    r"""
    Creates a widget with Save Figure Options.

    The structure of the widgets is the following:
        save_figure_wid.children = [toggle_button, options, save_button]
        options.children = [path, page_setup, image_colour]
        path.children = [filename, format, papertype]
        page_setup.children = [orientation, dpi, pad_inches]
        image_colour.children = [facecolour, edgecolour, transparent]

    To fix the alignment within this widget please refer to
    `format_save_figure_options()` function.

    Parameters
    ----------
    figure_id : matplotlib.pyplot.Figure instance
        The handle of the figure to be saved.

    format_default : `str`, optional
        The default value of the format.

    dpi_default : `float`, optional
        The default value of the dpi.

    orientation_default : `str`, optional
        The default value of the orientation. 'portrait' or 'landscape'.

    papertype_default : `str`, optional
        The default value of the papertype.

    transparent_default : `boolean`, optional
        The default value of the transparency flag.

    facecolour_default : `str` or `list` of `float`, optional
        The default value of the facecolour.

    edgecolour_default : `str` or `list` of `float`, optional
        The default value of the edgecolour.

    pad_inches_default : `float`, optional
        The default value of the figure padding in inches.

    toggle_show_default : `boolean`, optional
        Defines whether the options will be visible upon construction.

    toggle_show_visible : `boolean`, optional
        The visibility of the toggle button.
    """
    import IPython.html.widgets as ipywidgets
    from os import getcwd
    from os.path import join, splitext

    # create widgets
    but = ipywidgets.ToggleButtonWidget(description='Save Figure',
                                        value=toggle_show_default,
                                        visible=toggle_show_visible)
    format_dict = OrderedDict()
    format_dict['png'] = 'png'
    format_dict['jpg'] = 'jpg'
    format_dict['pdf'] = 'pdf'
    format_dict['eps'] = 'eps'
    format_dict['postscript'] = 'ps'
    format_dict['svg'] = 'svg'
    format_wid = ipywidgets.SelectWidget(values=format_dict,
                                         value=format_default,
                                         description='Format')

    def papertype_visibility(name, value):
        papertype_wid.disabled = not value == 'ps'

    format_wid.on_trait_change(papertype_visibility, 'value')

    def set_extension(name, value):
        fileName, fileExtension = splitext(filename.value)
        filename.value = fileName + '.' + value

    format_wid.on_trait_change(set_extension, 'value')
    if dpi_default is None:
        dpi_default = 0
    dpi_wid = ipywidgets.FloatTextWidget(description='DPI', value=dpi_default)
    orientation_dict = OrderedDict()
    orientation_dict['portrait'] = 'portrait'
    orientation_dict['landscape'] = 'landscape'
    orientation_wid = ipywidgets.DropdownWidget(values=orientation_dict,
                                                value=orientation_default,
                                                description='Orientation')
    papertype_dict = OrderedDict()
    papertype_dict['letter'] = 'letter'
    papertype_dict['legal'] = 'legal'
    papertype_dict['executive'] = 'executive'
    papertype_dict['ledger'] = 'ledger'
    papertype_dict['a0'] = 'a0'
    papertype_dict['a1'] = 'a1'
    papertype_dict['a2'] = 'a2'
    papertype_dict['a3'] = 'a3'
    papertype_dict['a4'] = 'a4'
    papertype_dict['a5'] = 'a5'
    papertype_dict['a6'] = 'a6'
    papertype_dict['a7'] = 'a7'
    papertype_dict['a8'] = 'a8'
    papertype_dict['a9'] = 'a9'
    papertype_dict['a10'] = 'a10'
    papertype_dict['b0'] = 'b0'
    papertype_dict['b1'] = 'b1'
    papertype_dict['b2'] = 'b2'
    papertype_dict['b3'] = 'b3'
    papertype_dict['b4'] = 'b4'
    papertype_dict['b5'] = 'b5'
    papertype_dict['b6'] = 'b6'
    papertype_dict['b7'] = 'b7'
    papertype_dict['b8'] = 'b8'
    papertype_dict['b9'] = 'b9'
    papertype_dict['b10'] = 'b10'
    is_ps_type = not format_default == 'ps'
    papertype_wid = ipywidgets.DropdownWidget(values=papertype_dict,
                                              value=papertype_default,
                                              description='Paper type',
                                              disabled=is_ps_type)
    transparent_wid = ipywidgets.CheckboxWidget(description='Transparent',
                                                value=transparent_default)
    facecolour_wid = colour_selection([facecolour_default], title='Face colour')
    edgecolour_wid = colour_selection([edgecolour_default], title='Edge colour')
    pad_inches_wid = ipywidgets.FloatTextWidget(description='Pad (inch)',
                                                value=pad_inches_default)
    filename = ipywidgets.TextWidget(description='Path',
                                     value=join(getcwd(),
                                                'Untitled.' + format_default))
    overwrite = ipywidgets.CheckboxWidget(
        description='Overwrite if file exists',
        value=overwrite_default)
    error_str = ipywidgets.LatexWidget(value="")
    save_but = ipywidgets.ButtonWidget(description='Save')

    # create final widget
    path_wid = ipywidgets.ContainerWidget(
        children=[filename, format_wid, overwrite,
                  papertype_wid])
    page_wid = ipywidgets.ContainerWidget(children=[orientation_wid, dpi_wid,
                                                    pad_inches_wid])
    colour_wid = ipywidgets.ContainerWidget(
        children=[facecolour_wid, edgecolour_wid,
                  transparent_wid])
    options_wid = ipywidgets.TabWidget(
        children=[path_wid, page_wid, colour_wid])
    save_wid = ipywidgets.ContainerWidget(children=[save_but, error_str])
    save_figure_wid = ipywidgets.ContainerWidget(
        children=[but, options_wid, save_wid])

    # Assign renderer
    save_figure_wid.renderer = [renderer]

    # save function
    def save_function(name):
        # set save button state
        error_str.value = ''
        save_but.description = 'Saving...'
        save_but.disabled = True

        # save figure
        selected_dpi = dpi_wid.value
        if dpi_wid.value == 0:
            selected_dpi = None
        try:
            save_figure_wid.renderer[0].save_figure(
                filename=filename.value, dpi=selected_dpi,
                face_colour=facecolour_wid.selected_values['colour'][0],
                edge_colour=edgecolour_wid.selected_values['colour'][0],
                orientation=orientation_wid.value,
                paper_type=papertype_wid.value, format=format_wid.value,
                transparent=transparent_wid.value,
                pad_inches=pad_inches_wid.value, overwrite=overwrite.value)
            error_str.value = ''
        except ValueError as e:
            if (e.message == 'File already exists. Please set the overwrite '
                             'kwarg if you wish to overwrite the file.'):
                error_str.value = 'File exists! Select overwrite to replace.'
            else:
                error_str.value = e.message

        # set save button state
        save_but.description = 'Save'
        save_but.disabled = False
    save_but.on_click(save_function)

    # Toggle button function
    def show_options(name, value):
        options_wid.visible = value
        save_but.visible = value
    show_options('', toggle_show_default)
    but.on_trait_change(show_options, 'value')

    return save_figure_wid


def format_save_figure_options(save_figure_wid, container_padding='6px',
                               container_margin='6px',
                               container_border='1px solid black',
                               toggle_button_font_weight='bold',
                               tab_top_margin='0.3cm',
                               border_visible=True):
    r"""
    Function that corrects the align (style format) of a given
    save_figure_options widget. Usage example:
        save_figure_wid = save_figure_options()
        display(save_figure_wid)
        format_save_figure_options(save_figure_wid)

    Parameters
    ----------
    save_figure_wid :
        The widget object generated by the `save_figure_options()` function.

    container_padding : `str`, optional
        The padding around the widget, e.g. '6px'

    container_margin : `str`, optional
        The margin around the widget, e.g. '6px'

    tab_top_margin : `str`, optional
        The margin around the tab options' widget, e.g. '0.3cm'

    container_border : `str`, optional
        The border around the widget, e.g. '1px solid black'

    toggle_button_font_weight : `str`
        The font weight of the toggle button, e.g. 'bold'

    border_visible : `boolean`, optional
        Defines whether to draw the border line around the widget.
    """
    # add margin on top of tabs widget
    save_figure_wid.children[1].set_css('margin-top', tab_top_margin)

    # align path options to the right
    save_figure_wid.children[1].children[0].add_class('align-end')

    # align save button and error message horizontally
    save_figure_wid.children[2].remove_class('vbox')
    save_figure_wid.children[2].add_class('hbox')
    save_figure_wid.children[2].children[1].set_css({'margin-left': '0.5cm',
                                                     'background-color': 'red'})

    # set final tab titles
    tab_titles = ['Path', 'Page setup', 'Image colour']
    for (k, tl) in enumerate(tab_titles):
        save_figure_wid.children[1].set_title(k, tl)

    format_colour_selection(save_figure_wid.children[1].children[2].children[0])
    format_colour_selection(save_figure_wid.children[1].children[2].children[1])
    save_figure_wid.children[1].children[0].children[0].set_css('width', '6cm')
    save_figure_wid.children[1].children[0].children[1].set_css('width', '6cm')

    # set toggle button font bold
    save_figure_wid.children[0].set_css('font-weight',
                                        toggle_button_font_weight)

    # margin and border around container widget
    save_figure_wid.set_css('padding', container_padding)
    save_figure_wid.set_css('margin', container_margin)
    if border_visible:
        save_figure_wid.set_css('border', container_border)


def features_options(toggle_show_default=True, toggle_show_visible=True):
    r"""
    Creates a widget with Features Options.

    The structure of the widgets is the following:
        features_options_wid.children = [toggle_button, tab_options]
        tab_options.children = [features_radiobuttons, per_feature_options,
                                preview]
        per_feature_options.children = [hog_options, igo_options, lbp_options,
                                        daisy_options, no_options]
        preview.children = [input_size_text, lenna_image, output_size_text,
                            elapsed_time]

    To fix the alignment within this widget please refer to
    `format_features_options()` function.

    Parameters
    ----------
    toggle_show_default : `boolean`, optional
        Defines whether the options will be visible upon construction.

    toggle_show_visible : `boolean`, optional
        The visibility of the toggle button.
    """
    # import features methods and time
    import time
    from menpo.feature.features import hog, lbp, igo, es, daisy, gradient, no_op
    from menpo.image import Image
    import menpo.io as mio
    from menpo.visualize.image import glyph
    import IPython.html.widgets as ipywidgets

    # Toggle button that controls options' visibility
    but = ipywidgets.ToggleButtonWidget(description='Features Options',
                                        value=toggle_show_default,
                                        visible=toggle_show_visible)

    # feature type
    tmp = OrderedDict()
    tmp['HOG'] = hog
    tmp['IGO'] = igo
    tmp['ES'] = es
    tmp['Daisy'] = daisy
    tmp['LBP'] = lbp
    tmp['Gradient'] = gradient
    tmp['None'] = no_op
    feature = ipywidgets.RadioButtonsWidget(value=no_op, values=tmp,
                                            description='Feature type:')

    # feature-related options
    hog_options_wid = hog_options(toggle_show_default=True,
                                  toggle_show_visible=False)
    igo_options_wid = igo_options(toggle_show_default=True,
                                  toggle_show_visible=False)
    lbp_options_wid = lbp_options(toggle_show_default=True,
                                  toggle_show_visible=False)
    daisy_options_wid = daisy_options(toggle_show_default=True,
                                      toggle_show_visible=False)
    no_options_wid = ipywidgets.LatexWidget(value='No options available.')

    # load and rescale preview image (lenna)
    image = mio.import_builtin_asset.lenna_png()
    image.crop_to_landmarks_proportion_inplace(0.18)
    image = image.as_greyscale()

    # per feature options widget
    per_feature_options = ipywidgets.ContainerWidget(
        children=[hog_options_wid, igo_options_wid, lbp_options_wid,
                  daisy_options_wid, no_options_wid])

    # preview tab widget
    preview_img = ipywidgets.ImageWidget(value=_convert_image_to_bytes(image),
                                         visible=False)
    preview_input = ipywidgets.LatexWidget(
        value="Input: {}W x {}H x {}C".format(
            image.width, image.height, image.n_channels), visible=False)
    preview_output = ipywidgets.LatexWidget(value="")
    preview_time = ipywidgets.LatexWidget(value="")
    preview = ipywidgets.ContainerWidget(children=[preview_img, preview_input,
                                                   preview_output,
                                                   preview_time])

    # options tab widget
    all_options = ipywidgets.TabWidget(
        children=[feature, per_feature_options, preview])

    # Widget container
    features_options_wid = ipywidgets.ContainerWidget(
        children=[but, all_options])

    # Initialize output dictionary
    options = {}
    features_options_wid.function = partial(no_op, **options)
    features_options_wid.features_function = no_op
    features_options_wid.features_options = options

    # options visibility
    def per_feature_options_visibility(name, value):
        if value == hog:
            igo_options_wid.visible = False
            lbp_options_wid.visible = False
            daisy_options_wid.visible = False
            no_options_wid.visible = False
            hog_options_wid.visible = True
        elif value == igo:
            hog_options_wid.visible = False
            lbp_options_wid.visible = False
            daisy_options_wid.visible = False
            no_options_wid.visible = False
            igo_options_wid.visible = True
        elif value == lbp:
            hog_options_wid.visible = False
            igo_options_wid.visible = False
            daisy_options_wid.visible = False
            no_options_wid.visible = False
            lbp_options_wid.visible = True
        elif value == daisy:
            hog_options_wid.visible = False
            igo_options_wid.visible = False
            lbp_options_wid.visible = False
            no_options_wid.visible = False
            daisy_options_wid.visible = True
        else:
            hog_options_wid.visible = False
            igo_options_wid.visible = False
            lbp_options_wid.visible = False
            daisy_options_wid.visible = False
            no_options_wid.visible = True
            for name, f in tmp.iteritems():
                if f == value:
                    no_options_wid.value = "{}: No available " \
                                           "options.".format(name)
    feature.on_trait_change(per_feature_options_visibility, 'value')
    per_feature_options_visibility('', no_op)

    # get function
    def get_function(name, value):
        # get options
        if feature.value == hog:
            opts = hog_options_wid.options
        elif feature.value == igo:
            opts = igo_options_wid.options
        elif feature.value == lbp:
            opts = lbp_options_wid.options
        elif feature.value == daisy:
            opts = daisy_options_wid.options
        else:
            opts = {}
        # get features function closure
        func = partial(feature.value, **opts)
        # store function
        features_options_wid.function = func
        features_options_wid.features_function = value
        features_options_wid.features_options = opts
    feature.on_trait_change(get_function, 'value')
    all_options.on_trait_change(get_function, 'selected_index')

    # preview function
    def preview_function(name, old_value, value):
        if value == 2:
            # extracting features message
            for name, f in tmp.iteritems():
                if f == features_options_wid.function.func:
                    val1 = name
            preview_output.value = "Previewing {} features...".format(val1)
            preview_time.value = ""
            # extract feature and time it
            t = time.time()
            feat_image = features_options_wid.function(image)
            t = time.time() - t
            # store feature image shape and n_channels
            val2 = feat_image.width
            val3 = feat_image.height
            val4 = feat_image.n_channels
            # compute sum of feature image and normalize its pixels in range
            # (0, 1) because it is required by as_PILImage
            feat_image = glyph(feat_image, vectors_block_size=1,
                               use_negative=False)
            # feat_image = np.sum(feat_image.pixels, axis=2)
            feat_image = feat_image.pixels
            feat_image -= np.min(feat_image)
            feat_image /= np.max(feat_image)
            feat_image = Image(feat_image)
            # update preview
            preview_img.value = _convert_image_to_bytes(feat_image)
            preview_input.visible = True
            preview_img.visible = True
            # set info
            preview_output.value = "{}: {}W x {}H x {}C".format(val1, val2,
                                                                val3, val4)
            preview_time.value = "{0:.2f} secs elapsed".format(t)
        if old_value == 2:
            preview_input.visible = False
            preview_img.visible = False
    all_options.on_trait_change(preview_function, 'selected_index')

    # Toggle button function
    def toggle_options(name, value):
        all_options.visible = value
    but.on_trait_change(toggle_options, 'value')

    return features_options_wid


def format_features_options(features_options_wid, container_padding='6px',
                            container_margin='6px',
                            container_border='1px solid black',
                            toggle_button_font_weight='bold',
                            border_visible=True):
    r"""
    Function that corrects the align (style format) of a given features_options
    widget. Usage example:
        features_options_wid = features_options()
        display(features_options_wid)
        format_features_options(features_options_wid)

    Parameters
    ----------
    features_options_wid :
        The widget object generated by the `features_options()` function.

    container_padding : `str`, optional
        The padding around the widget, e.g. '6px'

    container_margin : `str`, optional
        The margin around the widget, e.g. '6px'

    tab_top_margin : `str`, optional
        The margin around the tab options' widget, e.g. '0.3cm'

    container_border : `str`, optional
        The border around the widget, e.g. '1px solid black'

    toggle_button_font_weight : `str`
        The font weight of the toggle button, e.g. 'bold'

    border_visible : `boolean`, optional
        Defines whether to draw the border line around the widget.
    """
    # format per feature options
    format_hog_options(features_options_wid.children[1].children[1].children[0],
                       border_visible=False)
    format_igo_options(features_options_wid.children[1].children[1].children[1],
                       border_visible=False)
    format_lbp_options(features_options_wid.children[1].children[1].children[2],
                       border_visible=False)
    format_daisy_options(
        features_options_wid.children[1].children[1].children[3],
        border_visible=False)

    # set final tab titles
    tab_titles = ['Feature', 'Options', 'Preview']
    for (k, tl) in enumerate(tab_titles):
        features_options_wid.children[1].set_title(k, tl)

    # set margin above tab widget
    features_options_wid.children[1].set_css('margin', '10px')

    # set toggle button font bold
    features_options_wid.children[0].set_css('font-weight',
                                             toggle_button_font_weight)

    # margin and border around container widget
    features_options_wid.set_css('padding', container_padding)
    features_options_wid.set_css('margin', container_margin)
    if border_visible:
        features_options_wid.set_css('border', container_border)


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
        len_match = len(l1) == len(l2)
        return len_match and np.all([g1 == g2 for g1, g2 in zip(l1, l2)])

    # comparison of the given groups
    groups_same = comp_lists(groups1, groups2)

    # if groups are the same, compare the labels
    if groups_same:
        len_match = len(labels1) == len(labels2)
        return len_match and np.all([comp_lists(g1, g2)
                                     for g1, g2 in zip(labels1, labels2)])
    else:
        return False
