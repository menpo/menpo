from collections import OrderedDict
from functools import partial
import numpy as np

import IPython.html.widgets as ipywidgets
from IPython.utils.traitlets import link

from .tools import _format_box, _format_font


class ChannelOptionsWidget(ipywidgets.Box):
    r"""
    Creates a widget for selecting channel options when rendering an image.
    Specifically, it consists of:

        1) ToggleButton [`self.toggle_visible`]: toggle buttons that controls
           the options' visibility
        2) RadioButtons [`self.mode_radiobuttons`]: 'Single' or 'Multiple'
        3) Checkbox [`self.masked_checkbox`]: enable masked mode
        4) IntSlider [`self.single_slider`]: channel selection
        5) IntRangeSlider [`self.multiple_slider`]: channels range selection
        6) Checkbox [`self.rgb_checkbox`]: view as RGB
        7) Checkbox [`self.sum_checkbox`]: view sum of channels
        8) Checkbox [`self.glyph_checkbox`]: view glyph
        9) BoundedIntText [`self.glyph_block_size_text`]: glyph block size
        10) Checkbox [`self.glyph_use_negative_checkbox`]: use negative values
        11) VBox [`self.glyph_options_box`]: box that contains (9) and (10)
        12) VBox [`self.glyph_box`]: box that contains (8) and (11)
        13) HBox [`self.multiple_options_box`]: box that contains (7), (12), (6)
        14) Box [`self.sliders_box`]: box that contains (4) and (5)
        15) Box [`self.sliders_and_multiple_options_box`]: box that contains
            (14) and (13)
        16) VBox [`self.mode_and_masked_box`]: box that contains (2) and (3)
        17) HBox [`self.options_box`]: box that contains (16) and (15)

    The selected values are stored in `self.selected_values` `dict`. To set the
    styling of this widget please refer to the `style()` method. To update the
    state and function of the widget, please refer to the `set_widget_state()`
    and `set_render_function()` methods.

    Parameters
    ----------
    channel_options : `dict`
        The initial options. For example ::

            channel_options = {'n_channels': 10,
                               'image_is_masked': True,
                               'channels': 0,
                               'glyph_enabled': False,
                               'glyph_block_size': 3,
                               'glyph_use_negative': False,
                               'sum_enabled': False,
                               'masked_enabled': True}

    render_function : `function` or ``None``, optional
        The render function that is executed when a widgets' value changes.
        If ``None``, then nothing is assigned.
    toggle_show_default : `bool`, optional
        Defines whether the options will be visible upon construction.
    toggle_show_visible : `bool`, optional
        The visibility of the toggle button.
    toggle_title : `str`, optional
        The title of the toggle button.
    """
    def __init__(self, channel_options, render_function=None,
                 toggle_show_visible=True, toggle_show_default=True,
                 toggle_title='Channels Options'):
        # If image_is_masked is False, then masked_enabled should be False too
        if not channel_options['image_is_masked']:
            channel_options['masked_enabled'] = False

        # Parse channels
        mode_default, single_slider_default, multiple_slider_default = \
            self._parse_options(channel_options['channels'])

        # Check sum and glyph options
        channel_options['sum_enabled'], channel_options['glyph_enabled'] = \
            self._parse_sum_glyph(mode_default, channel_options['sum_enabled'],
                                  channel_options['glyph_enabled'])

        # Create widgets
        self.toggle_visible = ipywidgets.ToggleButton(
            description=toggle_title, value=toggle_show_default,
            visible=toggle_show_visible)
        self.mode_radiobuttons = ipywidgets.RadioButtons(
            options=['Single', 'Multiple'], value=mode_default,
            description='Mode:', visible=toggle_show_default,
            disabled=channel_options['n_channels'] == 1)
        self.masked_checkbox = ipywidgets.Checkbox(
            value=channel_options['masked_enabled'], description='Masked',
            visible=toggle_show_default and channel_options['image_is_masked'])
        self.single_slider = ipywidgets.IntSlider(
            min=0, max=channel_options['n_channels']-1, step=1,
            value=single_slider_default, description='Channel',
            visible=self._single_slider_visible(toggle_show_default,
                                                mode_default),
            disabled=channel_options['n_channels'] == 1)
        self.multiple_slider = ipywidgets.IntRangeSlider(
            min=0, max=channel_options['n_channels']-1, step=1,
            value=multiple_slider_default, description='Channels',
            visible=self._multiple_slider_visible(toggle_show_default,
                                                  mode_default))
        self.rgb_checkbox = ipywidgets.Checkbox(
            value=(channel_options['n_channels'] == 3 and
                   channel_options['channels'] is None),
            description='RGB',
            visible=self._rgb_checkbox_visible(toggle_show_default,
                                               mode_default,
                                               channel_options['n_channels']))
        self.sum_checkbox = ipywidgets.Checkbox(
            value=channel_options['sum_enabled'], description='Sum',
            visible=self._sum_checkbox_visible(toggle_show_default,
                                               mode_default,
                                               channel_options['n_channels']))
        self.glyph_checkbox = ipywidgets.Checkbox(
            value=channel_options['glyph_enabled'], description='Glyph',
            visible=self._glyph_checkbox_visible(toggle_show_default,
                                                 mode_default,
                                                 channel_options['n_channels']))
        self.glyph_block_size_text = ipywidgets.BoundedIntText(
            description='Block size', min=1, max=25,
            value=channel_options['glyph_block_size'],
            visible=self._glyph_options_visible(
                toggle_show_default, mode_default,
                channel_options['n_channels'],
                channel_options['glyph_enabled']), width='1.5cm')
        self.glyph_use_negative_checkbox = ipywidgets.Checkbox(
            description='Negative',
            value=channel_options['glyph_use_negative'],
            visible=self._glyph_options_visible(
                toggle_show_default, mode_default,
                channel_options['n_channels'],
                channel_options['glyph_enabled']))

        # Group widgets
        self.glyph_options_box = ipywidgets.VBox(
            children=[self.glyph_block_size_text,
                      self.glyph_use_negative_checkbox])
        self.glyph_box = ipywidgets.VBox(children=[self.glyph_checkbox,
                                                   self.glyph_options_box],
                                         align='start')
        self.multiple_options_box = ipywidgets.HBox(
            children=[self.sum_checkbox, self.glyph_box, self.rgb_checkbox])
        self.sliders_box = ipywidgets.Box(
            children=[self.single_slider, self.multiple_slider])
        self.sliders_and_multiple_options_box = ipywidgets.Box(
            children=[self.sliders_box, self.multiple_options_box])
        self.mode_and_masked_box = ipywidgets.VBox(
            children=[self.mode_radiobuttons, self.masked_checkbox])
        self.options_box = ipywidgets.HBox(
            children=[self.mode_and_masked_box,
                      self.sliders_and_multiple_options_box])
        super(ChannelOptionsWidget, self).__init__(
            children=[self.toggle_visible, self.options_box])

        # Assign output
        self.selected_values = channel_options

        # Set functionality
        def mode_selection(name, value):
            # Temporarily remove render function
            self.sum_checkbox.on_trait_change(self._render_function, 'value',
                                              remove=True)
            self.glyph_checkbox.on_trait_change(self._render_function, 'value',
                                                remove=True)
            # Control visibility of widgets
            if value == 'Single':
                self.multiple_slider.visible = False
                self.single_slider.visible = True
                self.sum_checkbox.visible = False
                self.sum_checkbox.value = False
                self.glyph_checkbox.visible = False
                self.glyph_checkbox.value = False
                self.glyph_block_size_text.visible = False
                self.glyph_use_negative_checkbox.visible = False
                self.rgb_checkbox.visible = \
                    self.selected_values['n_channels'] == 3
            else:
                self.single_slider.visible = False
                self.multiple_slider.visible = True
                self.sum_checkbox.visible = True
                self.sum_checkbox.value = False
                self.glyph_checkbox.visible = \
                    self.selected_values['n_channels'] > 1
                self.glyph_checkbox.value = False
                self.glyph_block_size_text.visible = False
                self.glyph_use_negative_checkbox.visible = False
                self.rgb_checkbox.visible = False
            # Add render function
            if self._render_function is not None:
                self.sum_checkbox.on_trait_change(self._render_function,
                                                  'value')
                self.glyph_checkbox.on_trait_change(self._render_function,
                                                    'value')
        self.mode_radiobuttons.on_trait_change(mode_selection, 'value')

        def glyph_options_visibility(name, value):
            # Temporarily remove render function
            self.sum_checkbox.on_trait_change(self._render_function, 'value',
                                              remove=True)
            # Check value of sum checkbox
            if value:
                self.sum_checkbox.value = False
            # Add render function
            if self._render_function is not None:
                self.sum_checkbox.on_trait_change(self._render_function,
                                                  'value')
            # Control glyph options visibility
            self.glyph_block_size_text.visible = value
            self.glyph_use_negative_checkbox.visible = value
        self.glyph_checkbox.on_trait_change(glyph_options_visibility, 'value')

        self.link_rgb_checkbox_and_single_slider = link(
            (self.rgb_checkbox, 'value'), (self.single_slider, 'disabled'))

        def sum_fun(name, value):
            # Temporarily remove render function
            self.glyph_checkbox.on_trait_change(self._render_function, 'value',
                                                remove=True)
            # Check value of glyph checkbox
            if value:
                self.glyph_checkbox.value = False
            # Add render function
            if self._render_function is not None:
                self.glyph_checkbox.on_trait_change(self._render_function,
                                                    'value')
        self.sum_checkbox.on_trait_change(sum_fun, 'value')

        def get_glyph_options(name, value):
            self.selected_values['glyph_enabled'] = self.glyph_checkbox.value
            self.selected_values['sum_enabled'] = self.sum_checkbox.value
            self.selected_values['glyph_use_negative'] = \
                self.glyph_use_negative_checkbox.value
            self.selected_values['glyph_block_size'] = \
                self.glyph_block_size_text.value
            if self.selected_values['sum_enabled']:
                self.selected_values['glyph_block_size'] = 1
        self.glyph_checkbox.on_trait_change(get_glyph_options, 'value')
        self.sum_checkbox.on_trait_change(get_glyph_options, 'value')
        self.glyph_use_negative_checkbox.on_trait_change(get_glyph_options,
                                                         'value')
        self.glyph_block_size_text.on_trait_change(get_glyph_options, 'value')

        def get_channels(name, value):
            if self.mode_radiobuttons.value == "Single":
                if self.rgb_checkbox.value:
                    self.selected_values['channels'] = None
                else:
                    self.selected_values['channels'] = self.single_slider.value
            else:
                self.selected_values['channels'] = range(
                    self.multiple_slider.lower, self.multiple_slider.upper + 1)
        self.single_slider.on_trait_change(get_channels, 'value')
        self.multiple_slider.on_trait_change(get_channels, 'value')
        self.rgb_checkbox.on_trait_change(get_channels, 'value')
        self.mode_radiobuttons.on_trait_change(get_channels, 'value')

        def get_masked(name, value):
            self.selected_values['masked_enabled'] = value
        self.masked_checkbox.on_trait_change(get_masked, 'value')

        def toggle_function(name, value):
            # get values
            mode_val = self.mode_radiobuttons.value
            n_channels = self.selected_values['n_channels']
            glyph_enabled = self.selected_values['glyph_enabled']
            # set visibility
            self.options_box.visible = value
            self.mode_radiobuttons.visible = value
            self.masked_checkbox.visible = \
                value and self.selected_values['image_is_masked']
            self.single_slider.visible = self._single_slider_visible(value,
                                                                     mode_val)
            self.multiple_slider.visible = self._multiple_slider_visible(
                value, mode_val)
            self.rgb_checkbox.visible = self._rgb_checkbox_visible(
                value, mode_val, n_channels)
            self.sum_checkbox.visible = self._sum_checkbox_visible(
                value, mode_val, n_channels)
            self.glyph_checkbox.visible = self._glyph_checkbox_visible(
                value, mode_val, n_channels)
            self.glyph_block_size_text.visible = self._glyph_options_visible(
                value, mode_val, n_channels, glyph_enabled)
            self.glyph_use_negative_checkbox.visible = \
                self._glyph_options_visible(value, mode_val, n_channels,
                                            glyph_enabled)
        self.toggle_visible.on_trait_change(toggle_function, 'value')

        # Set render function
        self._render_function = None
        self.add_render_function(render_function)

    def _parse_options(self, channels):
        if isinstance(channels, list):
            if len(channels) == 1:
                mode_value = 'Single'
                single_slider_value = channels[0]
                multiple_slider_value = (0, 1)
            else:
                mode_value = 'Multiple'
                single_slider_value = 0
                multiple_slider_value = (min(channels), max(channels))
        elif channels is None:
            mode_value = 'Single'
            single_slider_value = 0
            multiple_slider_value = (0, 1)
        else:
            mode_value = 'Single'
            single_slider_value = channels
            multiple_slider_value = (0, 1)
        return mode_value, single_slider_value, multiple_slider_value

    def _parse_sum_glyph(self, mode_value, sum_enabled, glyph_enabled):
        if mode_value == 'Single' or (sum_enabled and glyph_enabled):
            sum_enabled = False
            glyph_enabled = False
        return sum_enabled, glyph_enabled

    def _single_slider_visible(self, toggle, mode):
        return toggle and mode == 'Single'

    def _multiple_slider_visible(self, toggle, mode):
        return toggle and mode == 'Multiple'

    def _rgb_checkbox_visible(self, toggle, mode, n_channels):
        return toggle and mode == 'Single' and n_channels == 3

    def _sum_checkbox_visible(self, toggle, mode, n_channels):
        return toggle and mode == 'Multiple' and n_channels > 1

    def _glyph_checkbox_visible(self, toggle, mode, n_channels):
        return toggle and mode == 'Multiple' and n_channels > 1

    def _glyph_options_visible(self, toggle, mode, n_channels, glyph_value):
        return toggle and mode == 'Multiple' and n_channels > 1 and glyph_value

    def style(self, outer_box_style=None, outer_border_visible=False,
              outer_border_color='black', outer_border_style='solid',
              outer_border_width=1, outer_padding=0, outer_margin=0,
              inner_box_style=None, inner_border_visible=True,
              inner_border_color='black', inner_border_style='solid',
              inner_border_width=1, inner_padding=0, inner_margin=0,
              font_family='', font_size=None, font_style='',
              font_weight='', slider_width=''):
        r"""
        Function that defines the styling of the widget.

        Parameters
        ----------
        outer_box_style : `str` or ``None`` (see below), optional
            Outer box style options ::

                {``'success'``, ``'info'``, ``'warning'``, ``'danger'``, ``''``}
                or
                ``None``

        outer_border_visible : `bool`, optional
            Defines whether to draw the border line around the outer box.
        outer_border_color : `str`, optional
            The color of the border around the outer box.
        outer_border_style : `str`, optional
            The line style of the border around the outer box.
        outer_border_width : `float`, optional
            The line width of the border around the outer box.
        outer_padding : `float`, optional
            The padding around the outer box.
        outer_margin : `float`, optional
            The margin around the outer box.
        inner_box_style : `str` or ``None`` (see below), optional
            Inner box style options ::

                {``'success'``, ``'info'``, ``'warning'``, ``'danger'``, ``''``}
                or
                ``None``

        inner_border_visible : `bool`, optional
            Defines whether to draw the border line around the inner box.
        inner_border_color : `str`, optional
            The color of the border around the inner box.
        inner_border_style : `str`, optional
            The line style of the border around the inner box.
        inner_border_width : `float`, optional
            The line width of the border around the inner box.
        inner_padding : `float`, optional
            The padding around the inner box.
        inner_margin : `float`, optional
            The margin around the inner box.
        font_family : See Below, optional
            The font family to be used.
            Example options ::

                {``'serif'``, ``'sans-serif'``, ``'cursive'``, ``'fantasy'``,
                 ``'monospace'``, ``'helvetica'``}

        font_size : `int`, optional
            The font size.
        font_style : {``'normal'``, ``'italic'``, ``'oblique'``}, optional
            The font style.
        font_weight : See Below, optional
            The font weight.
            Example options ::

                {``'ultralight'``, ``'light'``, ``'normal'``, ``'regular'``,
                 ``'book'``, ``'medium'``, ``'roman'``, ``'semibold'``,
                 ``'demibold'``, ``'demi'``, ``'bold'``, ``'heavy'``,
                 ``'extra bold'``, ``'black'``}

        slider_width : `str`, optional
            The width of the slider.
        """
        _format_box(self, outer_box_style, outer_border_visible,
                    outer_border_color, outer_border_style, outer_border_width,
                    outer_padding, outer_margin)
        _format_box(self.options_box, inner_box_style, inner_border_visible,
                    inner_border_color, inner_border_style, inner_border_width,
                    inner_padding, inner_margin)
        self.single_slider.width = slider_width
        self.multiple_slider.width = slider_width
        _format_font(self, font_family, font_size, font_style, font_weight)
        _format_font(self.mode_radiobuttons, font_family, font_size, font_style,
                     font_weight)
        _format_font(self.single_slider, font_family, font_size, font_style,
                     font_weight)
        _format_font(self.multiple_slider, font_family, font_size, font_style,
                     font_weight)
        _format_font(self.masked_checkbox, font_family, font_size, font_style,
                     font_weight)
        _format_font(self.rgb_checkbox, font_family, font_size, font_style,
                     font_weight)
        _format_font(self.sum_checkbox, font_family, font_size, font_style,
                     font_weight)
        _format_font(self.glyph_checkbox, font_family, font_size, font_style,
                     font_weight)
        _format_font(self.glyph_use_negative_checkbox, font_family, font_size,
                     font_style, font_weight)
        _format_font(self.glyph_block_size_text, font_family, font_size,
                     font_style, font_weight)
        _format_font(self.toggle_visible, font_family, font_size, font_style,
                     font_weight)

    def add_render_function(self, render_function):
        r"""
        Method that adds a `render_function()` to the widget. The signature of
        the given function is also stored in `self._render_function`.

        Parameters
        ----------
        render_function : `function` or ``None``, optional
            The render function that behaves as a callback. If ``None``, then
            nothing is added.
        """
        self._render_function = render_function
        if self._render_function is not None:
            self.mode_radiobuttons.on_trait_change(self._render_function,
                                                   'value')
            self.masked_checkbox.on_trait_change(self._render_function, 'value')
            self.single_slider.on_trait_change(self._render_function, 'value')
            self.multiple_slider.on_trait_change(self._render_function, 'value')
            self.rgb_checkbox.on_trait_change(self._render_function, 'value')
            self.sum_checkbox.on_trait_change(self._render_function, 'value')
            self.glyph_checkbox.on_trait_change(self._render_function, 'value')
            self.glyph_block_size_text.on_trait_change(self._render_function,
                                                       'value')
            self.glyph_use_negative_checkbox.on_trait_change(
                self._render_function, 'value')

    def remove_render_function(self):
        r"""
        Method that removes the current `self._render_function()` from the
        widget and sets ``self._render_function = None``.
        """
        self.mode_radiobuttons.on_trait_change(self._render_function, 'value',
                                               remove=True)
        self.masked_checkbox.on_trait_change(self._render_function, 'value',
                                             remove=True)
        self.single_slider.on_trait_change(self._render_function, 'value',
                                           remove=True)
        self.multiple_slider.on_trait_change(self._render_function, 'value',
                                             remove=True)
        self.rgb_checkbox.on_trait_change(self._render_function, 'value',
                                          remove=True)
        self.sum_checkbox.on_trait_change(self._render_function, 'value',
                                          remove=True)
        self.glyph_checkbox.on_trait_change(self._render_function, 'value',
                                            remove=True)
        self.glyph_block_size_text.on_trait_change(self._render_function,
                                                   'value', remove=True)
        self.glyph_use_negative_checkbox.on_trait_change(self._render_function,
                                                         'value', remove=True)
        self._render_function = None

    def replace_render_function(self, render_function):
        r"""
        Method that replaces the current `self._render_function()` of the widget
        with the given `render_function()`.

        Parameters
        ----------
        render_function : `function` or ``None``, optional
            The render function that behaves as a callback. If ``None``, then
            nothing is happening.
        """
        # remove old function
        self.remove_render_function()

        # add new function
        self.add_render_function(render_function)

    def set_widget_state(self, channel_options, allow_callback=True):
        r"""
        Method that updates the state of the widget with a new set of values.

        Parameter
        ---------
        channel_options : `dict`
            The initial options. For example ::

                channel_options = {'n_channels': 10,
                                   'image_is_masked': True,
                                   'channels': 0,
                                   'glyph_enabled': False,
                                   'glyph_block_size': 3,
                                   'glyph_use_negative': False,
                                   'sum_enabled': False,
                                   'masked_enabled': True}

        allow_callback : `bool`, optional
            If ``True``, it allows triggering of any callback functions.
        """
        # temporarily remove render callback
        render_function = self._render_function
        self.remove_render_function()

        # If image_is_masked is False, then masked_enabled should be False too
        if not channel_options['image_is_masked']:
            channel_options['masked_enabled'] = False

        # Parse channels
        mode_default, single_slider_default, multiple_slider_default = \
            self._parse_options(channel_options['channels'])

        # Check sum and glyph options
        channel_options['sum_enabled'], channel_options['glyph_enabled'] = \
            self._parse_sum_glyph(mode_default, channel_options['sum_enabled'],
                                  channel_options['glyph_enabled'])

        # Update widgets' state
        self.mode_radiobuttons.value = mode_default
        self.mode_radiobuttons.disabled = channel_options['n_channels'] == 1
        self.mode_radiobuttons.visible = self.toggle_visible.value

        self.masked_checkbox.value = channel_options['masked_enabled']
        self.masked_checkbox.visible = (self.toggle_visible.value and
                                        channel_options['image_is_masked'])

        self.single_slider.max = channel_options['n_channels'] - 1
        self.single_slider.value = single_slider_default
        self.single_slider.visible = self._single_slider_visible(
            self.toggle_visible.value, mode_default)
        self.single_slider.disabled = channel_options['n_channels'] == 1

        self.multiple_slider.max = channel_options['n_channels'] - 1
        self.multiple_slider.value = multiple_slider_default
        self.multiple_slider.visible = self._multiple_slider_visible(
            self.toggle_visible.value, mode_default)

        self.rgb_checkbox.value = (channel_options['n_channels'] == 3 and
                                   channel_options['channels'] is None)
        self.rgb_checkbox.visible = self._rgb_checkbox_visible(
            self.toggle_visible.value, mode_default,
            channel_options['n_channels'])

        self.sum_checkbox.value = channel_options['sum_enabled']
        self.sum_checkbox.visible = self._sum_checkbox_visible(
            self.toggle_visible.value, mode_default,
            channel_options['n_channels'])

        self.glyph_checkbox.value = channel_options['glyph_enabled']
        self.glyph_checkbox.visible = self._glyph_checkbox_visible(
            self.toggle_visible.value, mode_default,
            channel_options['n_channels'])

        self.glyph_block_size_text.value = channel_options['glyph_block_size']
        self.glyph_block_size_text.visible = self._glyph_options_visible(
            self.toggle_visible.value, mode_default,
            channel_options['n_channels'], channel_options['glyph_enabled'])

        self.glyph_use_negative_checkbox.value = \
            channel_options['glyph_use_negative']
        self.glyph_use_negative_checkbox.visible = self._glyph_options_visible(
            self.toggle_visible.value, mode_default,
            channel_options['n_channels'], channel_options['glyph_enabled'])

        # Re-assign render callback
        self.add_render_function(render_function)

        # Assign new options dict to selected_values
        self.selected_values = channel_options

        # trigger render function if allowed
        if allow_callback:
            self._render_function('', True)


class LandmarkOptionsWidget(ipywidgets.Box):
    r"""
    Creates a widget for selecting landmark options when rendering an image.
    Specifically, it consists of:

        1) ToggleButton [`self.toggle_visible`]: toggle buttons that controls
           the options' visibility
        2) Latex [`self.no_landmarks_msg`]: Message in case there are no
           landmarks available.
        3) Checkbox [`self.render_landmarks_checkbox`]: render landmarks
        4) Box [`self.landmarks_checkbox_and_msg_box`]: box that contains (3)
           and (2)
        5) Dropdown [`self.group_dropdown`]: group selector
        6) ToggleButtons [`self.labels_toggles`]: `list` of `list`s with the
           labels per group
        7) Latex [`self.labels_text`]: labels title text
        8) HBox [`self.labels_box`]: box that contains all (6)
        9) HBox [`self.labels_and_text_box`]: box contains (7) and (8)
        10) VBox [`self.group_and_labels_and_text_box`]: box that contains (5)
            and (9)
        11) VBox [`self.options_box`]: box that contains (4) and (10)

    The selected values are stored in `self.selected_values` `dict`. To set the
    styling of this widget please refer to the `style()` method. To update the
    state and function of the widget, please refer to the `set_widget_state()`
    and `set_render_function()` methods.

    Parameters
    ----------
    landmark_options : `dict`
        The initial options. For example ::

            landmark_options = {'has_landmarks': True,
                                'render_landmarks': True,
                                'group_keys': ['PTS', 'ibug_face_68'],
                                'labels_keys': [['all'], ['jaw', 'eye']],
                                'group': 'PTS',
                                'with_labels': ['all']}

    render_function : `function` or ``None``, optional
        The render function that is executed when a widgets' value changes.
        If ``None``, then nothing is assigned.
    toggle_show_default : `bool`, optional
        Defines whether the options will be visible upon construction.
    toggle_show_visible : `bool`, optional
        The visibility of the toggle button.
    toggle_title : `str`, optional
        The title of the toggle button.
    """
    def __init__(self, landmark_options, render_function=None,
                 toggle_show_visible=True, toggle_show_default=True,
                 toggle_title='Landmarks Options'):
        # Check given options
        landmark_options = self._parse_landmark_options_dict(landmark_options)

        # Create widgets
        self.toggle_visible = ipywidgets.ToggleButton(
            description=toggle_title, value=toggle_show_default,
            visible=toggle_show_visible)
        self.no_landmarks_msg = ipywidgets.Latex(
            value='No landmarks available.',
            visible=self._no_landmarks_msg_visible(
                toggle_show_default, landmark_options['has_landmarks']))
        # temporarily store visible and disabled values
        tmp_visible = self._options_visible(toggle_show_default,
                                            landmark_options['has_landmarks'])
        tmp_disabled = not landmark_options['render_landmarks']
        self.render_landmarks_checkbox = ipywidgets.Checkbox(
            description='Render landmarks',
            value=landmark_options['render_landmarks'], visible=tmp_visible)
        self.landmarks_checkbox_and_msg_box = ipywidgets.Box(
            children=[self.render_landmarks_checkbox, self.no_landmarks_msg])
        self.group_dropdown = ipywidgets.Dropdown(
            options=landmark_options['group_keys'], description='Group',
            visible=tmp_visible, value=landmark_options['group'],
            disabled=tmp_disabled)
        self.labels_toggles = [
            [ipywidgets.ToggleButton(description=k, value=True,
                                     visible=tmp_visible, disabled=tmp_disabled)
             for k in s_keys]
            for s_keys in landmark_options['labels_keys']]
        self.labels_text = ipywidgets.Latex(value='Labels', visible=tmp_visible)
        group_idx = landmark_options['group_keys'].index(
            landmark_options['group'])
        self.labels_box = ipywidgets.HBox(
            children=self.labels_toggles[group_idx])
        self.labels_and_text_box = ipywidgets.HBox(children=[self.labels_text,
                                                             self.labels_box],
                                                   align='center')
        self._set_labels_toggles_values(landmark_options['with_labels'])
        self.group_and_labels_and_text_box = ipywidgets.VBox(
            children=[self.group_dropdown, self.labels_and_text_box])
        self.options_box = ipywidgets.VBox(
            children=[self.landmarks_checkbox_and_msg_box,
                      self.group_and_labels_and_text_box],
            visible=toggle_show_default)
        super(LandmarkOptionsWidget, self).__init__(
            children=[self.toggle_visible, self.options_box])

        # Assign output
        self.selected_values = landmark_options

        # Set functionality
        def render_landmarks_fun(name, value):
            # save render_landmarks value
            self.selected_values['render_landmarks'] = value
            # disable group drop down menu
            self.group_dropdown.disabled = not value
            # set disability of all labels toggles
            for s_keys in self.labels_toggles:
                for k in s_keys:
                    k.disabled = not value
            # if all currently selected labels toggles are False, set them all
            # to True
            self._all_labels_false_1()
        self.render_landmarks_checkbox.on_trait_change(render_landmarks_fun,
                                                       'value')

        def group_fun(name, value):
            # save group value
            self.selected_values['group'] = value
            # assign the correct children to the labels toggles
            group_idx = self.selected_values['group_keys'].index(
                self.selected_values['group'])
            self.labels_box.children = self.labels_toggles[group_idx]
            # save with_labels value
            self._save_with_labels()
        self.group_dropdown.on_trait_change(group_fun, 'value')

        def labels_fun(name, value):
            # if all labels toggles are False, set render landmarks checkbox to
            # False
            self._all_labels_false_2()
            # save with_labels value
            self._save_with_labels()
        # assign labels_fun to all labels toggles (even hidden ones)
        self._add_function_to_labels_toggles(labels_fun)

        def toggle_function(name, value):
            self.options_box.visible = value
            self.no_landmarks_msg.visible = self._no_landmarks_msg_visible(
                value, self.selected_values['has_landmarks'])
            tmp_visible = self._options_visible(
                value, self.selected_values['has_landmarks'])
            self.render_landmarks_checkbox.visible = tmp_visible
            self.group_dropdown.visible = tmp_visible
            for i1 in self.labels_toggles:
                for lt in i1:
                    lt.visible = tmp_visible
            self.labels_text.visible = tmp_visible
        self.toggle_visible.on_trait_change(toggle_function, 'value')

        # Store functions
        self._render_landmarks_fun = render_landmarks_fun
        self._group_fun = group_fun
        self._labels_fun = labels_fun

        # Set render function
        self._render_function = None
        self.add_render_function(render_function)

    def _parse_landmark_options_dict(self, landmark_options):
        if (len(landmark_options['group_keys']) == 1 and
                landmark_options['group_keys'][0] == ' '):
            landmark_options['has_landmarks'] = False
        if not landmark_options['has_landmarks']:
            landmark_options['render_landmarks'] = False
            landmark_options['group_keys'] = [' ']
            landmark_options['group'] = ' '
            landmark_options['labels_keys'] = [[' ']]
            landmark_options['with_labels'] = [' ']
        else:
            if len(landmark_options['with_labels']) == 0:
                group_idx = landmark_options['group_keys'].index(
                    landmark_options['group'])
                landmark_options['with_labels'] = \
                    landmark_options['labels_keys'][group_idx]
        return landmark_options

    def _no_landmarks_msg_visible(self, toggle, has_landmarks):
        return toggle and not has_landmarks

    def _options_visible(self, toggle, has_landmarks):
        return toggle and has_landmarks

    def _all_labels_false_1(self):
        r"""
        If all currently selected labels toggles are ``False``, set them all to
        ``True``.
        """
        # get all values of current labels toggles
        all_values = [ww.value for ww in self.labels_box.children]
        # if all of them are False
        if all(item is False for item in all_values):
            for ww in self.labels_box.children:
                # temporarily remove render function
                ww.on_trait_change(self._render_function, 'value', remove=True)
                # set value
                ww.value = True
                # re-add render function
                ww.on_trait_change(self._render_function, 'value')

    def _all_labels_false_2(self):
        r"""
        If all currently selected labels toggles are ``False``, set
        `render_landmarks_checkbox` to ``False``.
        """
        # get all values of current labels toggles
        all_values = [ww.value for ww in self.labels_box.children]
        # if all of them are False
        if all(item is False for item in all_values):
            # temporarily remove render function
            self.render_landmarks_checkbox.on_trait_change(
                self._render_function, 'value', remove=True)
            # set value
            self.render_landmarks_checkbox.value = False
            # re-add render function
            self.render_landmarks_checkbox.on_trait_change(
                self._render_function, 'value')

    def _save_with_labels(self):
        r"""
        Saves `with_labels` value to the `selected_values` dictionary.
        """
        self.selected_values['with_labels'] = []
        for ww in self.labels_box.children:
            if ww.value:
                self.selected_values['with_labels'].append(
                    str(ww.description))

    def _set_labels_toggles_values(self, with_labels):
        for w in self.labels_box.children:
            if w.description not in with_labels:
                w.value = False

    def _add_function_to_labels_toggles(self, fun):
        r"""
        Adds a function callback to all labels toggles.
        """
        for s_group in self.labels_toggles:
            for w in s_group:
                w.on_trait_change(fun, 'value')

    def _remove_function_from_labels_toggles(self, fun):
        r"""
        Removes a function callback from all labels toggles.
        """
        for s_group in self.labels_toggles:
            for w in s_group:
                w.on_trait_change(fun, 'value', remove=True)

    def style(self, outer_box_style=None, outer_border_visible=False,
              outer_border_color='black', outer_border_style='solid',
              outer_border_width=1, outer_padding=0, outer_margin=0,
              inner_box_style=None, inner_border_visible=True,
              inner_border_color='black', inner_border_style='solid',
              inner_border_width=1, inner_padding=0, inner_margin=0,
              font_family='', font_size=None, font_style='',
              font_weight=''):
        r"""
        Function that defines the styling of the widget.

        Parameters
        ----------
        outer_box_style : `str` or ``None`` (see below), optional
            Outer box style options ::

                {``'success'``, ``'info'``, ``'warning'``, ``'danger'``, ``''``}
                or
                ``None``

        outer_border_visible : `bool`, optional
            Defines whether to draw the border line around the outer box.
        outer_border_color : `str`, optional
            The color of the border around the outer box.
        outer_border_style : `str`, optional
            The line style of the border around the outer box.
        outer_border_width : `float`, optional
            The line width of the border around the outer box.
        outer_padding : `float`, optional
            The padding around the outer box.
        outer_margin : `float`, optional
            The margin around the outer box.
        inner_box_style : `str` or ``None`` (see below), optional
            Inner box style options ::

                {``'success'``, ``'info'``, ``'warning'``, ``'danger'``, ``''``}
                or
                ``None``

        inner_border_visible : `bool`, optional
            Defines whether to draw the border line around the inner box.
        inner_border_color : `str`, optional
            The color of the border around the inner box.
        inner_border_style : `str`, optional
            The line style of the border around the inner box.
        inner_border_width : `float`, optional
            The line width of the border around the inner box.
        inner_padding : `float`, optional
            The padding around the inner box.
        inner_margin : `float`, optional
            The margin around the inner box.
        font_family : See Below, optional
            The font family to be used.
            Example options ::

                {``'serif'``, ``'sans-serif'``, ``'cursive'``, ``'fantasy'``,
                 ``'monospace'``, ``'helvetica'``}

        font_size : `int`, optional
            The font size.
        font_style : {``'normal'``, ``'italic'``, ``'oblique'``}, optional
            The font style.
        font_weight : See Below, optional
            The font weight.
            Example options ::

                {``'ultralight'``, ``'light'``, ``'normal'``, ``'regular'``,
                 ``'book'``, ``'medium'``, ``'roman'``, ``'semibold'``,
                 ``'demibold'``, ``'demi'``, ``'bold'``, ``'heavy'``,
                 ``'extra bold'``, ``'black'``}

        """
        _format_box(self, outer_box_style, outer_border_visible,
                    outer_border_color, outer_border_style, outer_border_width,
                    outer_padding, outer_margin)
        _format_box(self.options_box, inner_box_style, inner_border_visible,
                    inner_border_color, inner_border_style, inner_border_width,
                    inner_padding, inner_margin)
        _format_font(self, font_family, font_size, font_style, font_weight)
        _format_font(self.render_landmarks_checkbox, font_family, font_size,
                     font_style, font_weight)
        _format_font(self.group_dropdown, font_family, font_size, font_style,
                     font_weight)
        for s_group in self.labels_toggles:
            for w in s_group:
                _format_font(w, font_family, font_size, font_style, font_weight)
        _format_font(self.labels_text, font_family, font_size, font_style,
                     font_weight)
        _format_font(self.toggle_visible, font_family, font_size, font_style,
                     font_weight)

    def add_render_function(self, render_function):
        r"""
        Method that adds a `render_function()` to the widget. The signature of
        the given function is also stored in `self._render_function`.

        Parameters
        ----------
        render_function : `function` or ``None``, optional
            The render function that behaves as a callback. If ``None``, then
            nothing is added.
        """
        self._render_function = render_function
        if self._render_function is not None:
            self.render_landmarks_checkbox.on_trait_change(
                self._render_function, 'value')
            self.group_dropdown.on_trait_change(self._render_function, 'value')
            self._add_function_to_labels_toggles(self._render_function)

    def remove_render_function(self):
        r"""
        Method that removes the current `self._render_function()` from the
        widget and sets ``self._render_function = None``.
        """
        self.render_landmarks_checkbox.on_trait_change(self._render_function,
                                                       'value', remove=True)
        self.group_dropdown.on_trait_change(self._render_function, 'value',
                                            remove=True)
        self._remove_function_from_labels_toggles(self._render_function)
        self._render_function = None

    def replace_render_function(self, render_function):
        r"""
        Method that replaces the current `self._render_function()` of the widget
        with the given `render_function()`.

        Parameters
        ----------
        render_function : `function` or ``None``, optional
            The render function that behaves as a callback. If ``None``, then
            nothing is happening.
        """
        # remove old function
        self.remove_render_function()

        # add new function
        self.add_render_function(render_function)

    def _compare_groups_and_labels(self, groups, labels):
        r"""
        Function that compares the provided landmarks groups and labels with
        `self.selected_values['group_keys']` and
        `self.selected_values['labels_keys']`.

        Parameters
        ----------
        groups : `list` of `str`
            The new `list` of landmark groups.
        labels : `list` of `list` of `str`
            The new `list` of `list`s of each landmark group's labels.

        Returns
        -------
        _compare_groups_and_labels : `bool`
            ``True`` if the groups and labels are identical with the ones stored
            in `self.selected_values['group_keys']` and
            `self.selected_values['labels_keys']`.
        """
        # function that compares two lists without taking into account the order
        def comp_lists(l1, l2):
            len_match = len(l1) == len(l2)
            return len_match and np.all([g1 == g2 for g1, g2 in zip(l1, l2)])

        # comparison of the given groups
        groups_same = comp_lists(groups, self.selected_values['group_keys'])

        # if groups are the same, then compare the labels
        if groups_same:
            len_match = len(labels) == len(self.selected_values['labels_keys'])
            tmp = [comp_lists(g1, g2)
                   for g1, g2 in zip(labels,
                                     self.selected_values['labels_keys'])]
            return len_match and np.all(tmp)
        else:
            return False

    def set_widget_state(self, landmark_options, allow_callback=True):
        r"""
        Method that updates the state of the widget with a new set of values.

        Parameter
        ---------
        landmark_options : `dict`
            The initial options. For example ::

                landmark_options = {'has_landmarks': True,
                                    'render_landmarks': True,
                                    'group_keys': ['PTS', 'ibug_face_68'],
                                    'labels_keys': [['all'], ['jaw', 'eye']],
                                    'group': 'PTS',
                                    'with_labels': ['all']}

        allow_callback : `bool`, optional
            If ``True``, it allows triggering of any callback functions.
        """
        # temporarily remove render callback
        render_function = self._render_function
        self.remove_render_function()

        # temporarily remove the rest of the callbacks
        self.render_landmarks_checkbox.on_trait_change(
            self._render_landmarks_fun, 'value', remove=True)
        self.group_dropdown.on_trait_change(self._group_fun, 'value',
                                            remove=True)
        self._remove_function_from_labels_toggles(self._labels_fun)

        # Check given options
        landmark_options = self._parse_landmark_options_dict(landmark_options)

        # Update widgets
        self.no_landmarks_msg.visible = self._no_landmarks_msg_visible(
            self.toggle_visible.value, landmark_options['has_landmarks'])
        # temporarily store visible and disabled values
        tmp_visible = self._options_visible(self.toggle_visible.value,
                                            landmark_options['has_landmarks'])
        tmp_disabled = not landmark_options['render_landmarks']
        self.render_landmarks_checkbox.value = \
            landmark_options['render_landmarks']
        self.render_landmarks_checkbox.visible = tmp_visible
        self.labels_text.visible = tmp_visible

        # Check if group_keys and labels_keys are the same with the existing
        # ones
        if not self._compare_groups_and_labels(landmark_options['group_keys'],
                                               landmark_options['labels_keys']):
            self.group_dropdown.options = landmark_options['group_keys']
            self.group_dropdown.visible = tmp_visible
            self.group_dropdown.disabled = tmp_disabled
            self.group_dropdown.value = landmark_options['group']

            self.labels_toggles = [
                [ipywidgets.ToggleButton(description=k, disabled=tmp_disabled,
                                         visible=tmp_visible, value=True)
                 for k in s_keys]
                for s_keys in landmark_options['labels_keys']]
            group_idx = landmark_options['group_keys'].index(
                landmark_options['group'])
            self.labels_box.children = self.labels_toggles[group_idx]
            self._set_labels_toggles_values(landmark_options['with_labels'])
        else:
            self.group_dropdown.visible = tmp_visible
            self.group_dropdown.disabled = tmp_disabled
            self.group_dropdown.value = landmark_options['group']

            self._set_labels_toggles_values(landmark_options['with_labels'])
            for w in self.labels_toggles:
                w.disabled = tmp_disabled
                w.visible = tmp_visible

        # Re-assign the rest of the callbacks
        self.render_landmarks_checkbox.on_trait_change(
            self._render_landmarks_fun, 'value')
        self.group_dropdown.on_trait_change(self._group_fun, 'value')
        self._add_function_to_labels_toggles(self._labels_fun)

        # Re-assign render callback
        self.add_render_function(render_function)

        # Assign new options dict to selected_values
        self.selected_values = landmark_options

        # trigger render function if allowed
        if allow_callback:
            self._render_function('', True)


# def info_print(n_bullets, toggle_show_default=True, toggle_show_visible=True):
#     r"""
#     Creates a widget that can print information. Specifically, it has:
#         1) A latex widget where user can write the info text in latex format.
#         2) A toggle button that controls the visibility of all the above, i.e.
#            the info printing.
#
#     The structure of the widgets is the following:
#         info_wid.children = [toggle_button, text_widget]
#
#     Parameters
#     ----------
#     toggle_show_default : `boolean`, optional
#         Defines whether the info will be visible upon construction.
#
#     toggle_show_visible : `boolean`, optional
#         The visibility of the toggle button.
#     """
#     import IPython.html.widgets as ipywidgets
#     # Create toggle button
#     but = ipywidgets.ToggleButton(description='Info',
#                                         value=toggle_show_default,
#                                         visible=toggle_show_visible)
#
#     # Create text widget
#     children = [ipywidgets.Latex(value="> menpo")
#                 for _ in range(n_bullets)]
#     text_wid = ipywidgets.Box(children=children)
#
#     # Toggle button function
#     def show_options(name, value):
#         text_wid.visible = value
#
#     show_options('', toggle_show_default)
#     but.on_trait_change(show_options, 'value')
#
#     # Group widgets
#     info_wid = ipywidgets.Box(children=[but, text_wid])
#
#     return info_wid
#
#
# def format_info_print(info_wid, font_size_in_pt='9pt', container_padding='6px',
#                       container_margin='6px',
#                       container_border='1px solid black',
#                       toggle_button_font_weight='bold',
#                       border_visible=True):
#     r"""
#     Function that corrects the align (style format) of a given info widget.
#     Usage example:
#         info_wid = info_print()
#         display(info_wid)
#         format_info_print(info_wid)
#
#     Parameters
#     ----------
#     info_wid :
#         The widget object generated by the `info_print()` function.
#
#     font_size_in_pt : `str`, optional
#         The font size of the latex text, e.g. '9pt'
#
#     container_padding : `str`, optional
#         The padding around the widget, e.g. '6px'
#
#     container_margin : `str`, optional
#         The margin around the widget, e.g. '6px'
#
#     container_border : `str`, optional
#         The border around the widget, e.g. '1px solid black'
#
#     toggle_button_font_weight : `str`
#         The font weight of the toggle button, e.g. 'bold'
#
#     border_visible : `boolean`, optional
#         Defines whether to draw the border line around the widget.
#     """
#     # text widget formatting
#     info_wid.children[1].border = container_border
#     info_wid.children[1].padding = '4px'
#     info_wid.children[1].margin_top = '1px'
#
#     # set font size
#     for w in info_wid.children[1].children:
#         w.font_size = font_size_in_pt
#
#     # set toggle button font bold
#     info_wid.children[0].font_weight = toggle_button_font_weight
#
#     # margin and border around container widget
#     info_wid.padding = container_padding
#     info_wid.margin = container_margin
#     if border_visible:
#         info_wid.border = container_border
#
#
# def animation_options(index_selection_default, plot_function=None,
#                       update_function=None, index_description='Image Number',
#                       index_minus_description='-', index_plus_description='+',
#                       index_style='buttons', index_text_editable=True,
#                       loop_default=False, interval_default=0.5,
#                       toggle_show_title='Image Options',
#                       toggle_show_default=True, toggle_show_visible=True):
#     r"""
#     Creates a widget for selecting an index and creating animations.
#     Specifically, it has:
#         1) An index selection widget. It can either be a slider or +/- buttons.
#         2) A play toggle button.
#         3) A stop toggle button.
#         4) An options toggle button.
#         If the options toggle is pressed, the following appear:
#         5) An interval text area.
#         6) A loop check box.
#
#     The structure of the widget is the following:
#         animation_options_wid.children = [toggle_button, options]
#         options.children = [index_selection, animation]
#         if index_style == 'buttons':
#             index_selection.children = [title, minus_button, index_text,
#                                         plus_button] (index_selection_buttons())
#         elif index_style == 'slider':
#             index_selection = index_slider (index_selection_slider())
#         animation.children = [buttons, animation_options]
#         buttons.children = [play_button, stop_button, play_options_button]
#         animation_options.children = [interval_text, loop_checkbox]
#
#     The returned widget saves the selected values in the following dictionary:
#         animation_options_wid.selected_values
#         animation_options_wid.index_style
#
#     To fix the alignment within this widget please refer to
#     `format_animation_options()` function.
#
#     To update the state of this widget, please refer to
#     `update_animation_options()` function.
#
#     Parameters
#     ----------
#     index_selection_default : `dict`
#         The dictionary with the default options. For example:
#             index_selection_default = {'min':0,
#                                        'max':100,
#                                        'step':1,
#                                        'index':10}
#
#     plot_function : `function` or None, optional
#         The plot function that is executed when the index value changes.
#         If None, then nothing is assigned.
#
#     update_function : `function` or None, optional
#         The update function that is executed when the index value changes.
#         If None, then nothing is assigned.
#
#     index_description : `str`, optional
#         The title of the index widget.
#
#     index_minus_description : `str`, optional
#         The title of the button that decreases the index.
#
#     index_plus_description : `str`, optional
#         The title of the button that increases the index.
#
#     index_style : {``buttons`` or ``slider``}, optional
#         If 'buttons', then 'index_selection_buttons()' is called.
#         If 'slider', then 'index_selection_slider()' is called.
#
#     index_text_editable : `boolean`, optional
#         Flag that determines whether the index text will be editable.
#
#     loop_default : `boolean`, optional
#         If True, the animation makes loop.
#         If False, the animation stops when reaching the index_max_value.
#
#     interval_default : `float`, optional
#         The interval between the animation frames.
#
#     toggle_show_title : `str`, optional
#         The title of the toggle button.
#
#     toggle_show_default : `boolean`, optional
#         Defines whether the options will be visible upon construction.
#
#     toggle_show_visible : `boolean`, optional
#         The visibility of the toggle button.
#     """
#     from time import sleep
#     from IPython import get_ipython
#     import IPython.html.widgets as ipywidgets
#
#     # get the kernel to use it later in order to make sure that the widgets'
#     # traits changes are passed during a while-loop
#     kernel = get_ipython().kernel
#
#     # Create index widget
#     if index_style == 'slider':
#         index_wid = index_selection_slider(index_selection_default,
#                                            plot_function=plot_function,
#                                            update_function=update_function,
#                                            description=index_description)
#     elif index_style == 'buttons':
#         index_wid = index_selection_buttons(
#             index_selection_default, plot_function=plot_function,
#             update_function=update_function, description=index_description,
#             minus_description=index_minus_description,
#             plus_description=index_plus_description, loop=loop_default,
#             text_editable=index_text_editable)
#
#     # Create other widgets
#     but = ipywidgets.ToggleButton(description=toggle_show_title,
#                                         value=toggle_show_default,
#                                         visible=toggle_show_visible)
#     play_but = ipywidgets.ToggleButton(description='Play >', value=False)
#     stop_but = ipywidgets.ToggleButton(description='Stop', value=True,
#                                              disabled=True)
#     play_options = ipywidgets.ToggleButton(description='Options',
#                                                  value=False)
#     loop = ipywidgets.Checkbox(description='Loop', value=loop_default,
#                                      visible=False)
#     interval = ipywidgets.FloatText(description='Interval (sec)',
#                                           value=interval_default, visible=False)
#
#     # Widget container
#     tmp_options = ipywidgets.Box(children=[interval, loop])
#     buttons = ipywidgets.Box(
#         children=[play_but, stop_but, play_options])
#     animation = ipywidgets.Box(children=[buttons, tmp_options])
#     cont = ipywidgets.Box(children=[index_wid, animation])
#     animation_options_wid = ipywidgets.Box(children=[but, cont])
#
#     # Initialize variables
#     animation_options_wid.selected_values = index_selection_default
#     animation_options_wid.index_style = index_style
#
#     # Play button pressed
#     def play_press(name, value):
#         stop_but.value = not value
#         play_but.disabled = value
#         play_options.disabled = value
#         if value:
#             play_options.value = False
#     play_but.on_trait_change(play_press, 'value')
#
#     # Stop button pressed
#     def stop_press(name, value):
#         play_but.value = not value
#         stop_but.disabled = value
#         play_options.disabled = not value
#     stop_but.on_trait_change(stop_press, 'value')
#
#     # show animation options checkbox function
#     def play_options_fun(name, value):
#         interval.visible = value
#         loop.visible = value
#     play_options.on_trait_change(play_options_fun, 'value')
#
#     # animation function
#     def play_fun(name, value):
#         if loop.value:
#             # loop is enabled
#             i = animation_options_wid.selected_values['index']
#             if i < animation_options_wid.selected_values['max']:
#                 i += animation_options_wid.selected_values['step']
#             else:
#                 i = animation_options_wid.selected_values['min']
#
#             ani_max_selected = animation_options_wid.selected_values['max']
#             while i <= ani_max_selected and not stop_but.value:
#                 # update index value
#                 if index_style == 'slider':
#                     index_wid.value = i
#                 else:
#                     index_wid.children[2].value = i
#
#                 # Run IPython iteration.
#                 # This is the code that makes this operation non-blocking. This
#                 # will allow widget messages and callbacks to be processed.
#                 kernel.do_one_iteration()
#
#                 # update counter
#                 if i < animation_options_wid.selected_values['max']:
#                     i += animation_options_wid.selected_values['step']
#                 else:
#                     i = animation_options_wid.selected_values['min']
#
#                 # wait
#                 sleep(interval.value)
#         else:
#             # loop is disabled
#             i = animation_options_wid.selected_values['index']
#             i += animation_options_wid.selected_values['step']
#             while (i <= animation_options_wid.selected_values['max'] and
#                        not stop_but.value):
#                 # update value
#                 if index_style == 'slider':
#                     index_wid.value = i
#                 else:
#                     index_wid.children[2].value = i
#
#                 # Run IPython iteration.
#                 # This is the code that makes this operation non-blocking. This
#                 # will allow widget messages and callbacks to be processed.
#                 kernel.do_one_iteration()
#
#                 # update counter
#                 i += animation_options_wid.selected_values['step']
#
#                 # wait
#                 sleep(interval.value)
#             if i > index_selection_default['max']:
#                 stop_but.value = True
#     play_but.on_trait_change(play_fun, 'value')
#
#     # Toggle button function
#     def show_options(name, value):
#         index_wid.visible = value
#         buttons.visible = value
#         interval.visible = False
#         loop.visible = False
#         if value:
#             play_options.value = False
#     show_options('', toggle_show_default)
#     but.on_trait_change(show_options, 'value')
#
#     return animation_options_wid
#
#
# def format_animation_options(animation_options_wid, index_text_width='0.5cm',
#                              container_padding='6px', container_margin='6px',
#                              container_border='1px solid black',
#                              toggle_button_font_weight='bold',
#                              border_visible=True):
#     r"""
#     Function that corrects the align (style format) of a given animation_options
#     widget. Usage example:
#         animation_options_wid = animation_options()
#         display(animation_options_wid)
#         format_animation_options(animation_options_wid)
#
#     Parameters
#     ----------
#     animation_options_wid :
#         The widget object generated by the `animation_options()`
#         function.
#
#     index_text_width : `str`, optional
#         The width of the index value text area.
#
#     container_padding : `str`, optional
#         The padding around the widget, e.g. '6px'
#
#     container_margin : `str`, optional
#         The margin around the widget, e.g. '6px'
#
#     container_border : `str`, optional
#         The border around the widget, e.g. '1px solid black'
#
#     toggle_button_font_weight : `str`
#         The font weight of the toggle button, e.g. 'bold'
#
#     border_visible : `boolean`, optional
#         Defines whether to draw the border line around the widget.
#     """
#     # format index widget
#     format_index_selection(animation_options_wid.children[1].children[0],
#                            text_width=index_text_width)
#
#     # align play/stop button with animation options button
#     remove_class(animation_options_wid.children[1].children[1].children[0], 'vbox')
#     add_class(animation_options_wid.children[1].children[1].children[0], 'hbox')
#     add_class(animation_options_wid.children[1].children[1], 'align-end')
#
#     # add margin on the right of the play button
#     animation_options_wid.children[1].children[1]. \
#         children[0].children[1].margin_right = container_margin
#
#     if animation_options_wid.index_style == 'slider':
#         # align animation on the right of slider
#         add_class(animation_options_wid.children[1], 'align-end')
#     else:
#         # align animation and index buttons
#         remove_class(animation_options_wid.children[1], 'vbox')
#         add_class(animation_options_wid.children[1], 'hbox')
#         add_class(animation_options_wid.children[1], 'align-center')
#         animation_options_wid.children[1].children[0].margin_right = '1cm'
#
#     # set interval width
#     animation_options_wid.children[1].children[1].children[1].children[0]. \
#         width = '20px'
#
#     # set toggle button font bold
#     animation_options_wid.children[0].font_weight = toggle_button_font_weight
#
#     # margin and border around container widget
#     animation_options_wid.padding = container_padding
#     animation_options_wid.margin = container_margin
#     if border_visible:
#         animation_options_wid.border = container_border
#
#
# def update_animation_options(animation_options_wid, index_selection_default,
#                              plot_function=None, update_function=None):
#     r"""
#     Function that updates the state of a given animation_options widget if the
#     index bounds have changed. Usage example:
#         index_selection_default = {'min':0,
#                                    'max':100,
#                                    'step':1,
#                                    'index':10}
#         animation_options_wid = animation_options(index_selection_default)
#         display(animation_options_wid)
#         format_animation_options(animation_options_wid)
#         index_selection_default = {'min':0,
#                                    'max':10,
#                                    'step':5,
#                                    'index':5}
#         update_animation_options(animation_options_wid, index_selection_default)
#
#     Parameters
#     ----------
#     animation_options_wid :
#         The widget object generated by either the `animation_options()`
#         function.
#
#     index_selection_default : `dict`
#         The dictionary with the default options. For example:
#             index_selection_default = {'min':0,
#                                        'max':100,
#                                        'step':1,
#                                        'index':10}
#
#     plot_function : `function` or None, optional
#         The plot function that is executed when the index value changes.
#         If None, then nothing is assigned.
#
#     update_function : `function` or None, optional
#         The update function that is executed when the index value changes.
#         If None, then nothing is assigned.
#     """
#     update_index_selection(animation_options_wid.children[1].children[0],
#                            index_selection_default,
#                            plot_function=plot_function,
#                            update_function=update_function)
#
#
# def viewer_options(viewer_options_default, options_tabs, objects_names=None,
#                    labels=None, plot_function=None, toggle_show_visible=True,
#                    toggle_show_default=True):
#     r"""
#     Creates a widget with Viewer Options. Specifically, it has:
#         1) A drop down menu for object selection.
#         2) A tab widget with any of line, marker, numbers and feature options
#         3) A toggle button that controls the visibility of all the above, i.e.
#            the viewer options.
#
#     The structure of the widgets is the following:
#         viewer_options_wid.children = [toggle_button, options]
#         options.children = [selection_menu, tab_options]
#         tab_options.children = [line_options, marker_options,
#                                 numbers_options, figure_options, legend_options]
#
#     The returned widget saves the selected values in the following dictionary:
#         viewer_options_wid.selected_values
#
#     To fix the alignment within this widget please refer to
#     `format_viewer_options()` function.
#
#     Parameters
#     ----------
#     viewer_options_default : list of `dict`
#         A list of dictionaries with the initial selected viewer options per
#         object. Example:
#
#             lines_options = {'render_lines': True,
#                              'line_width': 1,
#                              'line_colour': ['b'],
#                              'line_style': '-'}
#
#             markers_options = {'render_markers':True,
#                                'marker_size':20,
#                                'marker_face_colour':['r'],
#                                'marker_edge_colour':['k'],
#                                'marker_style':'o',
#                                'marker_edge_width':1}
#
#             numbers_options = {'render_numbering': True,
#                                'numbers_font_name': 'serif',
#                                'numbers_font_size': 10,
#                                'numbers_font_style': 'normal',
#                                'numbers_font_weight': 'normal',
#                                'numbers_font_colour': ['k'],
#                                'numbers_horizontal_align': 'center',
#                                'numbers_vertical_align': 'bottom'}
#
#             legend_options = {'render_legend':True,
#                               'legend_title':'',
#                               'legend_font_name':'serif',
#                               'legend_font_style':'normal',
#                               'legend_font_size':10,
#                               'legend_font_weight':'normal',
#                               'legend_marker_scale':1.,
#                               'legend_location':2,
#                               'legend_bbox_to_anchor':(1.05, 1.),
#                               'legend_border_axes_pad':1.,
#                               'legend_n_columns':1,
#                               'legend_horizontal_spacing':1.,
#                               'legend_vertical_spacing':1.,
#                               'legend_border':True,
#                               'legend_border_padding':0.5,
#                               'legend_shadow':False,
#                               'legend_rounded_corners':True}
#
#             figure_options = {'x_scale': 1.,
#                               'y_scale': 1.,
#                               'render_axes': True,
#                               'axes_font_name': 'serif',
#                               'axes_font_size': 10,
#                               'axes_font_style': 'normal',
#                               'axes_font_weight': 'normal',
#                               'axes_x_limits': None,
#                               'axes_y_limits': None}
#
#             grid_options = {'render_grid': True,
#                             'grid_line_style': '--',
#                             'grid_line_width': 0.5}
#
#             viewer_options_default = {'lines': lines_options,
#                                       'markers': markers_options,
#                                       'numbering': numbering_options,
#                                       'legend': legend_options,
#                                       'figure': figure_options,
#                                       'grid': grid_options}
#
#     options_tabs : `list` of `str`
#         List that defines the ordering of the options tabs. It can take one of
#         {``lines``, ``markers``, ``numbering``, ``figure_one``, ``figure_two``,
#         ``legend``, ``grid``}
#
#     objects_names : `list` of `str`, optional
#         A list with the names of the objects that will be used in the selection
#         dropdown menu. If None, then the names will have the form ``%d``.
#
#     plot_function : `function` or None, optional
#         The plot function that is executed when a widgets' value changes.
#         If None, then nothing is assigned.
#
#     toggle_show_default : `boolean`, optional
#         Defines whether the options will be visible upon construction.
#
#     toggle_show_visible : `boolean`, optional
#         The visibility of the toggle button.
#     """
#     import IPython.html.widgets as ipywidgets
#     # make sure that viewer_options_default is list even with one member
#     if not isinstance(viewer_options_default, list):
#         viewer_options_default = [viewer_options_default]
#
#     # find number of objects
#     n_objects = len(viewer_options_default)
#     selection_visible = n_objects > 1
#
#     # Create widgets
#     # toggle button
#     but = ipywidgets.ToggleButton(description='Viewer Options',
#                                   value=toggle_show_default,
#                                   visible=toggle_show_visible)
#
#     # select object drop down menu
#     objects_dict = OrderedDict()
#     if objects_names is None:
#         for k in range(n_objects):
#             objects_dict[str(k)] = k
#     else:
#         for k, g in enumerate(objects_names):
#             objects_dict[g] = k
#     selection = ipywidgets.Dropdown(options=objects_dict, value=0,
#                                     description='Select',
#                                     visible=(selection_visible and
#                                              toggle_show_default))
#
#     # options widgets
#     options_widgets = []
#     tab_titles = []
#     if labels is None:
#         labels = [str(j) for j in range(len(options_tabs))]
#     for j, o in enumerate(options_tabs):
#         if o == 'lines':
#             options_widgets.append(
#                 line_options(viewer_options_default[0]['lines'],
#                              toggle_show_visible=False,
#                              toggle_show_default=True,
#                              plot_function=plot_function,
#                              show_checkbox_title='Render lines',
#                              labels=labels[j]))
#             tab_titles.append('Lines')
#         elif o == 'markers':
#             options_widgets.append(
#                 marker_options(viewer_options_default[0]['markers'],
#                                toggle_show_visible=False,
#                                toggle_show_default=True,
#                                plot_function=plot_function,
#                                show_checkbox_title='Render markers'))
#             tab_titles.append('Markers')
#         elif o == 'image':
#             options_widgets.append(
#                 image_options(viewer_options_default[0]['image'],
#                               toggle_show_visible=False,
#                               toggle_show_default=True,
#                               plot_function=plot_function))
#             tab_titles.append('Image')
#         elif o == 'numbering':
#             options_widgets.append(
#                 numbering_options(viewer_options_default[0]['numbering'],
#                                   toggle_show_visible=False,
#                                   toggle_show_default=True,
#                                   plot_function=plot_function,
#                                   show_checkbox_title='Render numbering'))
#             tab_titles.append('Numbering')
#         elif o == 'figure_one':
#             options_widgets.append(
#                 figure_options(viewer_options_default[0]['figure'],
#                                plot_function=plot_function,
#                                figure_scale_bounds=(0.1, 4),
#                                figure_scale_step=0.1, figure_scale_visible=True,
#                                axes_visible=True, toggle_show_default=True,
#                                toggle_show_visible=False))
#             tab_titles.append('Figure/Axes')
#         elif o == 'figure_two':
#             options_widgets.append(
#                 figure_options_two_scales(
#                     viewer_options_default[0]['figure'],
#                     plot_function=plot_function, coupled_default=False,
#                     figure_scales_bounds=(0.1, 4), figure_scales_step=0.1,
#                     figure_scales_visible=True, axes_visible=True,
#                     toggle_show_default=True, toggle_show_visible=False))
#             tab_titles.append('Figure/Axes')
#         elif o == 'legend':
#             options_widgets.append(
#                 legend_options(viewer_options_default[0]['legend'],
#                                toggle_show_visible=False,
#                                toggle_show_default=True,
#                                plot_function=plot_function,
#                                show_checkbox_title='Render legend'))
#             tab_titles.append('Legend')
#         elif o == 'grid':
#             options_widgets.append(
#                 grid_options(viewer_options_default[0]['grid'],
#                              toggle_show_visible=False,
#                              toggle_show_default=True,
#                              plot_function=plot_function,
#                              show_checkbox_title='Render grid'))
#             tab_titles.append('Grid')
#     options = ipywidgets.Tab(children=options_widgets)
#
#     # Final widget
#     all_options = ipywidgets.Box(children=[selection, options])
#     viewer_options_wid = ipywidgets.Box(children=[but, all_options])
#
#     # save tab titles and options str to widget in order to be passed to the
#     # format function
#     viewer_options_wid.tab_titles = tab_titles
#     viewer_options_wid.options_tabs = options_tabs
#
#     # Assign output list of dicts
#     viewer_options_wid.selected_values = viewer_options_default
#
#     # Update widgets' state
#     def update_widgets(name, value):
#         for i, tab in enumerate(options_tabs):
#             if tab == 'lines':
#                 update_line_options(
#                     options_widgets[i],
#                     viewer_options_default[value]['lines'],
#                     labels=labels[value])
#             elif tab == 'markers':
#                 update_marker_options(
#                     options_widgets[i],
#                     viewer_options_default[value]['markers'])
#             elif tab == 'image':
#                 update_image_options(
#                     options_widgets[i],
#                     viewer_options_default[value]['image'])
#             elif tab == 'numbering':
#                 update_numbering_options(
#                     options_widgets[i],
#                     viewer_options_default[value]['numbering'])
#             elif tab == 'figure_one':
#                 update_figure_options(
#                     options_widgets[i],
#                     viewer_options_default[value]['figure'])
#             elif tab == 'figure_two':
#                 update_figure_options_two_scales(
#                     options_widgets[i],
#                     viewer_options_default[value]['figure'])
#             elif tab == 'legend':
#                 update_legend_options(
#                     options_widgets[i],
#                     viewer_options_default[value]['legend'])
#             elif tab == 'grid':
#                 update_grid_options(
#                     options_widgets[i],
#                     viewer_options_default[value]['grid'])
#     selection.on_trait_change(update_widgets, 'value')
#
#     # Toggle button function
#     def toggle_fun(name, value):
#         selection.visible = value and selection_visible
#         options.visible = value
#     toggle_fun('', toggle_show_default)
#     but.on_trait_change(toggle_fun, 'value')
#
#     return viewer_options_wid
#
#
# def format_viewer_options(viewer_options_wid, container_padding='6px',
#                           container_margin='6px',
#                           container_border='1px solid black',
#                           toggle_button_font_weight='bold',
#                           border_visible=False, suboptions_border_visible=True):
#     r"""
#     Function that corrects the align (style format) of a given
#     viewer_options widget. Usage example:
#         viewer_options_wid = viewer_options(default_options)
#         display(viewer_options_wid)
#         format_viewer_options(viewer_options_wid)
#
#     Parameters
#     ----------
#     viewer_options_wid :
#         The widget object generated by the `viewer_options()` function.
#
#     container_padding : `str`, optional
#         The padding around the widget, e.g. '6px'
#
#     container_margin : `str`, optional
#         The margin around the widget, e.g. '6px'
#
#     container_border : `str`, optional
#         The border around the widget, e.g. '1px solid black'
#
#     toggle_button_font_weight : `str`
#         The font weight of the toggle button, e.g. 'bold'
#
#     border_visible : `boolean`, optional
#         Defines whether to draw the border line around the widget.
#
#     suboptions_border_visible : `boolean`, optional
#         Defines whether to draw the border line around each of the sub options.
#     """
#     # format widgets
#     for k, o in enumerate(viewer_options_wid.options_tabs):
#         if o == 'lines':
#             format_line_options(
#                 viewer_options_wid.children[1].children[1].children[k],
#                 suboptions_border_visible=suboptions_border_visible,
#                 border_visible=False)
#         elif o == 'markers':
#             format_marker_options(
#                 viewer_options_wid.children[1].children[1].children[k],
#                 suboptions_border_visible=suboptions_border_visible,
#                 border_visible=False)
#         elif o == 'image':
#             format_image_options(
#                 viewer_options_wid.children[1].children[1].children[k],
#                 border_visible=suboptions_border_visible)
#         elif o == 'numbering':
#             format_numbering_options(
#                 viewer_options_wid.children[1].children[1].children[k],
#                 suboptions_border_visible=suboptions_border_visible,
#                 border_visible=False)
#         elif o == 'figure_one':
#             format_figure_options(
#                 viewer_options_wid.children[1].children[1].children[k],
#                 border_visible=suboptions_border_visible)
#         elif o == 'figure_two':
#             format_figure_options_two_scales(
#                 viewer_options_wid.children[1].children[1].children[k],
#                 border_visible=suboptions_border_visible)
#         elif o == 'legend':
#             format_legend_options(
#                 viewer_options_wid.children[1].children[1].children[k],
#                 suboptions_border_visible=suboptions_border_visible,
#                 border_visible=False)
#         elif o == 'grid':
#             format_grid_options(
#                 viewer_options_wid.children[1].children[1].children[k],
#                 suboptions_border_visible=suboptions_border_visible,
#                 border_visible=False)
#
#     # set titles
#     for (k, tl) in enumerate(viewer_options_wid.tab_titles):
#         viewer_options_wid.children[1].children[1].set_title(k, tl)
#
#     # set toggle button font bold
#     viewer_options_wid.children[0].font_weight = toggle_button_font_weight
#
#     # margin and border around container widget
#     viewer_options_wid.padding = container_padding
#     viewer_options_wid.margin = container_margin
#     if border_visible:
#         viewer_options_wid.border = container_border
#
#
# def update_viewer_options(viewer_options_wid, viewer_options_default,
#                           labels=None):
#     for k, o in enumerate(viewer_options_wid.options_tabs):
#         if o == 'lines' and 'lines' in viewer_options_default:
#             update_line_options(
#                 viewer_options_wid.children[1].children[1].children[k],
#                 viewer_options_default['lines'], labels=labels)
#         elif o == 'markers' and 'markers' in viewer_options_default:
#             update_marker_options(
#                 viewer_options_wid.children[1].children[1].children[k],
#                 viewer_options_default['markers'])
#         elif o == 'image' and 'image' in viewer_options_default:
#             update_image_options(
#                 viewer_options_wid.children[1].children[1].children[k],
#                 viewer_options_default['image'])
#         elif o == 'numbering' and 'numbering' in viewer_options_default:
#             update_numbering_options(
#                 viewer_options_wid.children[1].children[1].children[k],
#                 viewer_options_default['numbering'])
#         elif o == 'figure_one' and 'figure' in viewer_options_default:
#             update_figure_options(
#                 viewer_options_wid.children[1].children[1].children[k],
#                 viewer_options_default['figure'])
#         elif o == 'figure_two' and 'figure' in viewer_options_default:
#             update_figure_options(
#                 viewer_options_wid.children[1].children[1].children[k],
#                 viewer_options_default['figure'])
#         elif o == 'legend' and 'legend' in viewer_options_default:
#             update_legend_options(
#                 viewer_options_wid.children[1].children[1].children[k],
#                 viewer_options_default['legend'])
#         elif o == 'grid' and 'grid' in viewer_options_default:
#             update_grid_options(
#                 viewer_options_wid.children[1].children[1].children[k],
#                 viewer_options_default['grid'])
#
#
# def save_figure_options(renderer, format_default='png', dpi_default=None,
#                         orientation_default='portrait',
#                         papertype_default='letter', transparent_default=False,
#                         facecolour_default='w', edgecolour_default='w',
#                         pad_inches_default=0.5, overwrite_default=False,
#                         toggle_show_default=True, toggle_show_visible=True):
#     r"""
#     Creates a widget with Save Figure Options.
#
#     The structure of the widgets is the following:
#         save_figure_wid.children = [toggle_button, options, save_button]
#         options.children = [path, page_setup, image_colour]
#         path.children = [filename, format, papertype]
#         page_setup.children = [orientation, dpi, pad_inches]
#         image_colour.children = [facecolour, edgecolour, transparent]
#
#     To fix the alignment within this widget please refer to
#     `format_save_figure_options()` function.
#
#     Parameters
#     ----------
#     figure_id : matplotlib.pyplot.Figure instance
#         The handle of the figure to be saved.
#
#     format_default : `str`, optional
#         The default value of the format.
#
#     dpi_default : `float`, optional
#         The default value of the dpi.
#
#     orientation_default : `str`, optional
#         The default value of the orientation. 'portrait' or 'landscape'.
#
#     papertype_default : `str`, optional
#         The default value of the papertype.
#
#     transparent_default : `boolean`, optional
#         The default value of the transparency flag.
#
#     facecolour_default : `str` or `list` of `float`, optional
#         The default value of the facecolour.
#
#     edgecolour_default : `str` or `list` of `float`, optional
#         The default value of the edgecolour.
#
#     pad_inches_default : `float`, optional
#         The default value of the figure padding in inches.
#
#     toggle_show_default : `boolean`, optional
#         Defines whether the options will be visible upon construction.
#
#     toggle_show_visible : `boolean`, optional
#         The visibility of the toggle button.
#     """
#     import IPython.html.widgets as ipywidgets
#     from os import getcwd
#     from os.path import join, splitext
#
#     # create widgets
#     but = ipywidgets.ToggleButton(description='Save Figure',
#                                   value=toggle_show_default,
#                                   visible=toggle_show_visible)
#     format_dict = OrderedDict()
#     format_dict['png'] = 'png'
#     format_dict['jpg'] = 'jpg'
#     format_dict['pdf'] = 'pdf'
#     format_dict['eps'] = 'eps'
#     format_dict['postscript'] = 'ps'
#     format_dict['svg'] = 'svg'
#     format_wid = ipywidgets.Select(options=format_dict,
#                                    value=format_default,
#                                    description='Format')
#
#     def papertype_visibility(name, value):
#         papertype_wid.disabled = not value == 'ps'
#
#     format_wid.on_trait_change(papertype_visibility, 'value')
#
#     def set_extension(name, value):
#         fileName, fileExtension = splitext(filename.value)
#         filename.value = fileName + '.' + value
#
#     format_wid.on_trait_change(set_extension, 'value')
#     if dpi_default is None:
#         dpi_default = 0
#     dpi_wid = ipywidgets.FloatText(description='DPI', value=dpi_default)
#     orientation_dict = OrderedDict()
#     orientation_dict['portrait'] = 'portrait'
#     orientation_dict['landscape'] = 'landscape'
#     orientation_wid = ipywidgets.Dropdown(options=orientation_dict,
#                                           value=orientation_default,
#                                           description='Orientation')
#     papertype_dict = OrderedDict()
#     papertype_dict['letter'] = 'letter'
#     papertype_dict['legal'] = 'legal'
#     papertype_dict['executive'] = 'executive'
#     papertype_dict['ledger'] = 'ledger'
#     papertype_dict['a0'] = 'a0'
#     papertype_dict['a1'] = 'a1'
#     papertype_dict['a2'] = 'a2'
#     papertype_dict['a3'] = 'a3'
#     papertype_dict['a4'] = 'a4'
#     papertype_dict['a5'] = 'a5'
#     papertype_dict['a6'] = 'a6'
#     papertype_dict['a7'] = 'a7'
#     papertype_dict['a8'] = 'a8'
#     papertype_dict['a9'] = 'a9'
#     papertype_dict['a10'] = 'a10'
#     papertype_dict['b0'] = 'b0'
#     papertype_dict['b1'] = 'b1'
#     papertype_dict['b2'] = 'b2'
#     papertype_dict['b3'] = 'b3'
#     papertype_dict['b4'] = 'b4'
#     papertype_dict['b5'] = 'b5'
#     papertype_dict['b6'] = 'b6'
#     papertype_dict['b7'] = 'b7'
#     papertype_dict['b8'] = 'b8'
#     papertype_dict['b9'] = 'b9'
#     papertype_dict['b10'] = 'b10'
#     is_ps_type = not format_default == 'ps'
#     papertype_wid = ipywidgets.Dropdown(options=papertype_dict,
#                                         value=papertype_default,
#                                         description='Paper type',
#                                         disabled=is_ps_type)
#     transparent_wid = ipywidgets.Checkbox(description='Transparent',
#                                           value=transparent_default)
#     facecolour_wid = colour_selection([facecolour_default], title='Face colour')
#     edgecolour_wid = colour_selection([edgecolour_default], title='Edge colour')
#     pad_inches_wid = ipywidgets.FloatText(description='Pad (inch)',
#                                                 value=pad_inches_default)
#     filename = ipywidgets.Text(description='Path',
#                                      value=join(getcwd(),
#                                                 'Untitled.' + format_default))
#     overwrite = ipywidgets.Checkbox(
#         description='Overwrite if file exists',
#         value=overwrite_default)
#     error_str = ipywidgets.Latex(value="")
#     save_but = ipywidgets.Button(description='Save')
#
#     # create final widget
#     path_wid = ipywidgets.Box(
#         children=[filename, format_wid, overwrite,
#                   papertype_wid])
#     page_wid = ipywidgets.Box(children=[orientation_wid, dpi_wid,
#                                                     pad_inches_wid])
#     colour_wid = ipywidgets.Box(
#         children=[facecolour_wid, edgecolour_wid,
#                   transparent_wid])
#     options_wid = ipywidgets.Tab(
#         children=[path_wid, page_wid, colour_wid])
#     save_wid = ipywidgets.Box(children=[save_but, error_str])
#     save_figure_wid = ipywidgets.Box(
#         children=[but, options_wid, save_wid])
#
#     # Assign renderer
#     save_figure_wid.renderer = [renderer]
#
#     # save function
#     def save_function(name):
#         # set save button state
#         error_str.value = ''
#         save_but.description = 'Saving...'
#         save_but.disabled = True
#
#         # save figure
#         selected_dpi = dpi_wid.value
#         if dpi_wid.value == 0:
#             selected_dpi = None
#         try:
#             save_figure_wid.renderer[0].save_figure(
#                 filename=filename.value, dpi=selected_dpi,
#                 face_colour=facecolour_wid.selected_values['colour'][0],
#                 edge_colour=edgecolour_wid.selected_values['colour'][0],
#                 orientation=orientation_wid.value,
#                 paper_type=papertype_wid.value, format=format_wid.value,
#                 transparent=transparent_wid.value,
#                 pad_inches=pad_inches_wid.value, overwrite=overwrite.value)
#             error_str.value = ''
#         except ValueError as e:
#             if (e.message == 'File already exists. Please set the overwrite '
#                              'kwarg if you wish to overwrite the file.'):
#                 error_str.value = 'File exists! Select overwrite to replace.'
#             else:
#                 error_str.value = e.message
#
#         # set save button state
#         save_but.description = 'Save'
#         save_but.disabled = False
#     save_but.on_click(save_function)
#
#     # Toggle button function
#     def show_options(name, value):
#         options_wid.visible = value
#         save_but.visible = value
#     show_options('', toggle_show_default)
#     but.on_trait_change(show_options, 'value')
#
#     return save_figure_wid
#
#
# def format_save_figure_options(save_figure_wid, container_padding='6px',
#                                container_margin='6px',
#                                container_border='1px solid black',
#                                toggle_button_font_weight='bold',
#                                tab_top_margin='0.3cm',
#                                border_visible=True):
#     r"""
#     Function that corrects the align (style format) of a given
#     save_figure_options widget. Usage example:
#         save_figure_wid = save_figure_options()
#         display(save_figure_wid)
#         format_save_figure_options(save_figure_wid)
#
#     Parameters
#     ----------
#     save_figure_wid :
#         The widget object generated by the `save_figure_options()` function.
#
#     container_padding : `str`, optional
#         The padding around the widget, e.g. '6px'
#
#     container_margin : `str`, optional
#         The margin around the widget, e.g. '6px'
#
#     tab_top_margin : `str`, optional
#         The margin around the tab options' widget, e.g. '0.3cm'
#
#     container_border : `str`, optional
#         The border around the widget, e.g. '1px solid black'
#
#     toggle_button_font_weight : `str`
#         The font weight of the toggle button, e.g. 'bold'
#
#     border_visible : `boolean`, optional
#         Defines whether to draw the border line around the widget.
#     """
#     # add margin on top of tabs widget
#     save_figure_wid.children[1].margin_top = tab_top_margin
#
#     # align path options to the right
#     add_class(save_figure_wid.children[1].children[0], 'align-end')
#
#     # align save button and error message horizontally
#     remove_class(save_figure_wid.children[2], 'vbox')
#     add_class(save_figure_wid.children[2], 'hbox')
#     save_figure_wid.children[2].children[1].margin_left = '0.5cm'
#     save_figure_wid.children[2].children[1].background_color = 'red'
#
#     # set final tab titles
#     tab_titles = ['Path', 'Page setup', 'Image colour']
#     for (k, tl) in enumerate(tab_titles):
#         save_figure_wid.children[1].set_title(k, tl)
#
#     format_colour_selection(save_figure_wid.children[1].children[2].children[0])
#     format_colour_selection(save_figure_wid.children[1].children[2].children[1])
#     save_figure_wid.children[1].children[0].children[0].width = '6cm'
#     save_figure_wid.children[1].children[0].children[1].width = '6cm'
#
#     # set toggle button font bold
#     save_figure_wid.children[0].font_weight = toggle_button_font_weight
#
#     # margin and border around container widget
#     save_figure_wid.padding = container_padding
#     save_figure_wid.margin = container_margin
#     if border_visible:
#         save_figure_wid.border = container_border
#
#
# def features_options(toggle_show_default=True, toggle_show_visible=True):
#     r"""
#     Creates a widget with Features Options.
#
#     The structure of the widgets is the following:
#         features_options_wid.children = [toggle_button, tab_options]
#         tab_options.children = [features_radiobuttons, per_feature_options,
#                                 preview]
#         per_feature_options.children = [hog_options, igo_options, lbp_options,
#                                         daisy_options, no_options]
#         preview.children = [input_size_text, lenna_image, output_size_text,
#                             elapsed_time]
#
#     To fix the alignment within this widget please refer to
#     `format_features_options()` function.
#
#     Parameters
#     ----------
#     toggle_show_default : `boolean`, optional
#         Defines whether the options will be visible upon construction.
#
#     toggle_show_visible : `boolean`, optional
#         The visibility of the toggle button.
#     """
#     # import features methods and time
#     import time
#     from menpo.feature.features import hog, lbp, igo, es, daisy, gradient, no_op
#     from menpo.image import Image
#     import menpo.io as mio
#     from menpo.visualize.image import glyph
#     import IPython.html.widgets as ipywidgets
#
#     # Toggle button that controls options' visibility
#     but = ipywidgets.ToggleButton(description='Features Options',
#                                         value=toggle_show_default,
#                                         visible=toggle_show_visible)
#
#     # feature type
#     tmp = OrderedDict()
#     tmp['HOG'] = hog
#     tmp['IGO'] = igo
#     tmp['ES'] = es
#     tmp['Daisy'] = daisy
#     tmp['LBP'] = lbp
#     tmp['Gradient'] = gradient
#     tmp['None'] = no_op
#     feature = ipywidgets.RadioButtons(value=no_op, options=tmp,
#                                       description='Feature type:')
#
#     # feature-related options
#     hog_options_wid = hog_options(toggle_show_default=True,
#                                   toggle_show_visible=False)
#     igo_options_wid = igo_options(toggle_show_default=True,
#                                   toggle_show_visible=False)
#     lbp_options_wid = lbp_options(toggle_show_default=True,
#                                   toggle_show_visible=False)
#     daisy_options_wid = daisy_options(toggle_show_default=True,
#                                       toggle_show_visible=False)
#     no_options_wid = ipywidgets.Latex(value='No options available.')
#
#     # load and rescale preview image (lenna)
#     image = mio.import_builtin_asset.lenna_png()
#     image.crop_to_landmarks_proportion_inplace(0.18)
#     image = image.as_greyscale()
#
#     # per feature options widget
#     per_feature_options = ipywidgets.Box(
#         children=[hog_options_wid, igo_options_wid, lbp_options_wid,
#                   daisy_options_wid, no_options_wid])
#
#     # preview tab widget
#     preview_img = ipywidgets.Image(value=_convert_image_to_bytes(image),
#                                          visible=False)
#     preview_input = ipywidgets.Latex(
#         value="Input: {}W x {}H x {}C".format(
#             image.width, image.height, image.n_channels), visible=False)
#     preview_output = ipywidgets.Latex(value="")
#     preview_time = ipywidgets.Latex(value="")
#     preview = ipywidgets.Box(children=[preview_img, preview_input,
#                                                    preview_output,
#                                                    preview_time])
#
#     # options tab widget
#     all_options = ipywidgets.Tab(
#         children=[feature, per_feature_options, preview])
#
#     # Widget container
#     features_options_wid = ipywidgets.Box(
#         children=[but, all_options])
#
#     # Initialize output dictionary
#     options = {}
#     features_options_wid.function = partial(no_op, **options)
#     features_options_wid.features_function = no_op
#     features_options_wid.features_options = options
#
#     # options visibility
#     def per_feature_options_visibility(name, value):
#         if value == hog:
#             igo_options_wid.visible = False
#             lbp_options_wid.visible = False
#             daisy_options_wid.visible = False
#             no_options_wid.visible = False
#             hog_options_wid.visible = True
#         elif value == igo:
#             hog_options_wid.visible = False
#             lbp_options_wid.visible = False
#             daisy_options_wid.visible = False
#             no_options_wid.visible = False
#             igo_options_wid.visible = True
#         elif value == lbp:
#             hog_options_wid.visible = False
#             igo_options_wid.visible = False
#             daisy_options_wid.visible = False
#             no_options_wid.visible = False
#             lbp_options_wid.visible = True
#         elif value == daisy:
#             hog_options_wid.visible = False
#             igo_options_wid.visible = False
#             lbp_options_wid.visible = False
#             no_options_wid.visible = False
#             daisy_options_wid.visible = True
#         else:
#             hog_options_wid.visible = False
#             igo_options_wid.visible = False
#             lbp_options_wid.visible = False
#             daisy_options_wid.visible = False
#             no_options_wid.visible = True
#             for name, f in tmp.items():
#                 if f == value:
#                     no_options_wid.value = "{}: No available " \
#                                            "options.".format(name)
#     feature.on_trait_change(per_feature_options_visibility, 'value')
#     per_feature_options_visibility('', no_op)
#
#     # get function
#     def get_function(name, value):
#         # get options
#         if feature.value == hog:
#             opts = hog_options_wid.options
#         elif feature.value == igo:
#             opts = igo_options_wid.options
#         elif feature.value == lbp:
#             opts = lbp_options_wid.options
#         elif feature.value == daisy:
#             opts = daisy_options_wid.options
#         else:
#             opts = {}
#         # get features function closure
#         func = partial(feature.value, **opts)
#         # store function
#         features_options_wid.function = func
#         features_options_wid.features_function = value
#         features_options_wid.features_options = opts
#     feature.on_trait_change(get_function, 'value')
#     all_options.on_trait_change(get_function, 'selected_index')
#
#     # preview function
#     def preview_function(name, old_value, value):
#         if value == 2:
#             # extracting features message
#             for name, f in tmp.items():
#                 if f == features_options_wid.function.func:
#                     val1 = name
#             preview_output.value = "Previewing {} features...".format(val1)
#             preview_time.value = ""
#             # extract feature and time it
#             t = time.time()
#             feat_image = features_options_wid.function(image)
#             t = time.time() - t
#             # store feature image shape and n_channels
#             val2 = feat_image.width
#             val3 = feat_image.height
#             val4 = feat_image.n_channels
#             # compute sum of feature image and normalize its pixels in range
#             # (0, 1) because it is required by as_PILImage
#             feat_image = glyph(feat_image, vectors_block_size=1,
#                                use_negative=False)
#             # feat_image = np.sum(feat_image.pixels, axis=2)
#             feat_image = feat_image.pixels
#             feat_image -= np.min(feat_image)
#             feat_image /= np.max(feat_image)
#             feat_image = Image(feat_image)
#             # update preview
#             preview_img.value = _convert_image_to_bytes(feat_image)
#             preview_input.visible = True
#             preview_img.visible = True
#             # set info
#             preview_output.value = "{}: {}W x {}H x {}C".format(val1, val2,
#                                                                 val3, val4)
#             preview_time.value = "{0:.2f} secs elapsed".format(t)
#         if old_value == 2:
#             preview_input.visible = False
#             preview_img.visible = False
#     all_options.on_trait_change(preview_function, 'selected_index')
#
#     # Toggle button function
#     def toggle_options(name, value):
#         all_options.visible = value
#     but.on_trait_change(toggle_options, 'value')
#
#     return features_options_wid
#
#
# def format_features_options(features_options_wid, container_padding='6px',
#                             container_margin='6px',
#                             container_border='1px solid black',
#                             toggle_button_font_weight='bold',
#                             border_visible=True):
#     r"""
#     Function that corrects the align (style format) of a given features_options
#     widget. Usage example:
#         features_options_wid = features_options()
#         display(features_options_wid)
#         format_features_options(features_options_wid)
#
#     Parameters
#     ----------
#     features_options_wid :
#         The widget object generated by the `features_options()` function.
#
#     container_padding : `str`, optional
#         The padding around the widget, e.g. '6px'
#
#     container_margin : `str`, optional
#         The margin around the widget, e.g. '6px'
#
#     tab_top_margin : `str`, optional
#         The margin around the tab options' widget, e.g. '0.3cm'
#
#     container_border : `str`, optional
#         The border around the widget, e.g. '1px solid black'
#
#     toggle_button_font_weight : `str`
#         The font weight of the toggle button, e.g. 'bold'
#
#     border_visible : `boolean`, optional
#         Defines whether to draw the border line around the widget.
#     """
#     # format per feature options
#     format_hog_options(features_options_wid.children[1].children[1].children[0],
#                        border_visible=False)
#     format_igo_options(features_options_wid.children[1].children[1].children[1],
#                        border_visible=False)
#     format_lbp_options(features_options_wid.children[1].children[1].children[2],
#                        border_visible=False)
#     format_daisy_options(
#         features_options_wid.children[1].children[1].children[3],
#         border_visible=False)
#
#     # set final tab titles
#     tab_titles = ['Feature', 'Options', 'Preview']
#     for (k, tl) in enumerate(tab_titles):
#         features_options_wid.children[1].set_title(k, tl)
#
#     # set margin above tab widget
#     features_options_wid.children[1].margin = '10px'
#
#     # set toggle button font bold
#     features_options_wid.children[0].font_weight = toggle_button_font_weight
#
#     # margin and border around container widget
#     features_options_wid.padding = container_padding
#     features_options_wid.margin = container_margin
#     if border_visible:
#         features_options_wid.border = container_border

