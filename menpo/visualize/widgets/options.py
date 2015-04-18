from collections import OrderedDict
from functools import partial
import numpy as np

import IPython.html.widgets as ipywidgets
from IPython.utils.traitlets import link

from .tools import (_format_box, _format_font, _convert_image_to_bytes,
                    IndexButtonsWidget, IndexSliderWidget, LineOptionsWidget,
                    MarkerOptionsWidget, ImageOptionsWidget,
                    NumberingOptionsWidget, FigureOptionsOneScaleWidget,
                    FigureOptionsTwoScalesWidget, LegendOptionsWidget,
                    GridOptionsWidget, ColourSelectionWidget, HOGOptionsWidget,
                    DaisyOptionsWidget, LBPOptionsWidget, IGOOptionsWidget,
                    _map_styles_to_hex_colours)


class ChannelOptionsWidget(ipywidgets.FlexBox):
    r"""
    Creates a widget for selecting channel options when rendering an image.
    Specifically, it consists of:

        1) RadioButtons [`self.mode_radiobuttons`]: 'Single' or 'Multiple'
        2) Checkbox [`self.masked_checkbox`]: enable masked mode
        3) IntSlider [`self.single_slider`]: channel selection
        4) IntRangeSlider [`self.multiple_slider`]: channels range selection
        5) Checkbox [`self.rgb_checkbox`]: view as RGB
        6) Checkbox [`self.sum_checkbox`]: view sum of channels
        7) Checkbox [`self.glyph_checkbox`]: view glyph
        8) BoundedIntText [`self.glyph_block_size_text`]: glyph block size
        9) Checkbox [`self.glyph_use_negative_checkbox`]: use negative values
        10) VBox [`self.glyph_options_box`]: box that contains (8) and (9)
        11) VBox [`self.glyph_box`]: box that contains (7) and (10)
        12) HBox [`self.multiple_options_box`]: box that contains (6), (11), (5)
        13) Box [`self.sliders_box`]: box that contains (3) and (4)
        14) Box [`self.sliders_and_multiple_options_box`]: box that contains
            (13) and (12)
        15) VBox [`self.mode_and_masked_box`]: box that contains (1) and (2)

    The selected values are stored in `self.selected_values` `dict`. To set the
    styling of this widget please refer to the `style()` method. To update the
    state and function of the widget, please refer to the `set_widget_state()`
    and `replace_render_function()` methods.

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
    style : `str` (see below)
        Sets a predefined style at the widget. Possible options are ::

            {``'minimal'``, ``'success'``, ``'info'``, ``'warning'``,
             ``'danger'``, ``''``}

    """
    def __init__(self, channel_options, render_function=None, style='minimal'):
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
        self.mode_radiobuttons = ipywidgets.RadioButtons(
            options=['Single', 'Multiple'], value=mode_default,
            description='Mode:', disabled=channel_options['n_channels'] == 1)
        self.masked_checkbox = ipywidgets.Checkbox(
            value=channel_options['masked_enabled'], description='Masked',
            visible=channel_options['image_is_masked'])
        self.single_slider = ipywidgets.IntSlider(
            min=0, max=channel_options['n_channels']-1, step=1,
            value=single_slider_default, description='Channel',
            visible=self._single_slider_visible(mode_default),
            disabled=channel_options['n_channels'] == 1)
        self.multiple_slider = ipywidgets.IntRangeSlider(
            min=0, max=channel_options['n_channels']-1, step=1,
            value=multiple_slider_default, description='Channels',
            visible=self._multiple_slider_visible(mode_default))
        self.rgb_checkbox = ipywidgets.Checkbox(
            value=(channel_options['n_channels'] == 3 and
                   channel_options['channels'] is None),
            description='RGB',
            visible=self._rgb_checkbox_visible(
                mode_default, channel_options['n_channels']))
        self.sum_checkbox = ipywidgets.Checkbox(
            value=channel_options['sum_enabled'], description='Sum',
            visible=self._sum_checkbox_visible(
                mode_default, channel_options['n_channels']))
        self.glyph_checkbox = ipywidgets.Checkbox(
            value=channel_options['glyph_enabled'], description='Glyph',
            visible=self._glyph_checkbox_visible(
                mode_default, channel_options['n_channels']))
        self.glyph_block_size_text = ipywidgets.BoundedIntText(
            description='Block size', min=1, max=25,
            value=channel_options['glyph_block_size'], width='1.5cm')
        self.glyph_use_negative_checkbox = ipywidgets.Checkbox(
            description='Negative', value=channel_options['glyph_use_negative'])

        # Group widgets
        self.glyph_options_box = ipywidgets.VBox(
            children=[self.glyph_block_size_text,
                      self.glyph_use_negative_checkbox],
            visible=self._glyph_options_visible(
                mode_default, channel_options['n_channels'],
                channel_options['glyph_enabled']))
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
        super(ChannelOptionsWidget, self).__init__(
            children=[self.mode_and_masked_box,
                      self.sliders_and_multiple_options_box])
        self.align = 'start'
        self.orientation = 'horizontal'

        # Assign output
        self.selected_values = channel_options

        # Set style
        self.predefined_style(style)

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
                self.glyph_options_box.visible = False
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
                self.glyph_options_box.visible = False
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
            self.glyph_options_box.visible = value
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

    def _single_slider_visible(self, mode):
        return mode == 'Single'

    def _multiple_slider_visible(self, mode):
        return mode == 'Multiple'

    def _rgb_checkbox_visible(self, mode, n_channels):
        return mode == 'Single' and n_channels == 3

    def _sum_checkbox_visible(self, mode, n_channels):
        return mode == 'Multiple' and n_channels > 1

    def _glyph_checkbox_visible(self, mode, n_channels):
        return mode == 'Multiple' and n_channels > 1

    def _glyph_options_visible(self, mode, n_channels, glyph_value):
        return mode == 'Multiple' and n_channels > 1 and glyph_value

    def style(self, box_style=None, border_visible=False, border_color='black',
              border_style='solid', border_width=1, border_radius=0, padding=0,
              margin=0, font_family='', font_size=None, font_style='',
              font_weight='', slider_width='', slider_colour=''):
        r"""
        Function that defines the styling of the widget.

        Parameters
        ----------
        box_style : `str` or ``None`` (see below), optional
            Style options ::

                {``'success'``, ``'info'``, ``'warning'``, ``'danger'``, ``''``}
                or
                ``None``

        border_visible : `bool`, optional
            Defines whether to draw the border line around the widget.
        border_color : `str`, optional
            The color of the border around the widget.
        border_style : `str`, optional
            The line style of the border around the widget.
        border_width : `float`, optional
            The line width of the border around the widget.
        border_radius : `float`, optional
            The radius of the corners of the box.
        padding : `float`, optional
            The padding around the widget.
        margin : `float`, optional
            The margin around the widget.
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
        slider_colour : `str`, optional
            The colour of the sliders.
        """
        _format_box(self, box_style, border_visible, border_color, border_style,
                    border_width, border_radius, padding, margin)
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
        self.single_slider.slider_color = slider_colour
        self.single_slider.background_color = slider_colour
        self.multiple_slider.slider_color = slider_colour
        self.multiple_slider.background_color = slider_colour

    def predefined_style(self, style):
        r"""
        Function that set a predefined styling on the widget.

        Parameter
        ---------
        style : `str` (see below)
            Style options ::

                {``'minimal'``, ``'success'``, ``'info'``, ``'warning'``,
                 ``'danger'``, ``''``}

        """
        if style == 'minimal':
            self.style(box_style=None, border_visible=True,
                       border_color='black', border_style='solid',
                       border_width=1, border_radius=0, padding='0.2cm',
                       margin='0.3cm', font_family='', font_size=None,
                       font_style='', font_weight='', slider_width='5cm',
                       slider_colour='')
            _format_box(self.glyph_options_box, box_style='',
                        border_visible=False, border_color='',
                        border_style='solid', border_width=1, border_radius=0,
                        padding='0.1cm', margin=0)
        elif (style == 'info' or style == 'success' or style == 'danger' or
              style == 'warning'):
            self.style(box_style=style, border_visible=True,
                       border_color=_map_styles_to_hex_colours(style),
                       border_style='solid', border_width=1, border_radius=10,
                       padding='0.2cm', margin='0.3cm', font_family='',
                       font_size=None, font_style='', font_weight='',
                       slider_width='5cm',
                       slider_colour=_map_styles_to_hex_colours(style))
            _format_box(self.glyph_options_box, box_style=style,
                        border_visible=True,
                        border_color=_map_styles_to_hex_colours(style),
                        border_style='solid', border_width=1, border_radius=10,
                        padding='0.1cm', margin=0)
        else:
            raise ValueError('style must be minimal or info or success or '
                             'danger or warning')

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

        self.masked_checkbox.value = channel_options['masked_enabled']
        self.masked_checkbox.visible = channel_options['image_is_masked']

        self.single_slider.max = channel_options['n_channels'] - 1
        self.single_slider.value = single_slider_default
        self.single_slider.visible = self._single_slider_visible(mode_default)
        self.single_slider.disabled = channel_options['n_channels'] == 1

        self.multiple_slider.max = channel_options['n_channels'] - 1
        self.multiple_slider.value = multiple_slider_default
        self.multiple_slider.visible = self._multiple_slider_visible(
            mode_default)

        self.rgb_checkbox.value = (channel_options['n_channels'] == 3 and
                                   channel_options['channels'] is None)
        self.rgb_checkbox.visible = self._rgb_checkbox_visible(
            mode_default, channel_options['n_channels'])

        self.sum_checkbox.value = channel_options['sum_enabled']
        self.sum_checkbox.visible = self._sum_checkbox_visible(
            mode_default, channel_options['n_channels'])

        self.glyph_checkbox.value = channel_options['glyph_enabled']
        self.glyph_checkbox.visible = self._glyph_checkbox_visible(
            mode_default, channel_options['n_channels'])

        self.glyph_block_size_text.value = channel_options['glyph_block_size']

        self.glyph_use_negative_checkbox.value = \
            channel_options['glyph_use_negative']

        self.glyph_options_box.visible = self._glyph_options_visible(
            mode_default, channel_options['n_channels'],
            channel_options['glyph_enabled'])

        # Re-assign render callback
        self.add_render_function(render_function)

        # Assign new options dict to selected_values
        self.selected_values = channel_options

        # trigger render function if allowed
        if allow_callback:
            self._render_function('', True)


class LandmarkOptionsWidget(ipywidgets.FlexBox):
    r"""
    Creates a widget for selecting landmark options when rendering an image.
    Specifically, it consists of:

        1) Latex [`self.no_landmarks_msg`]: Message in case there are no
           landmarks available.
        2) Checkbox [`self.render_landmarks_checkbox`]: render landmarks
        3) Box [`self.landmarks_checkbox_and_msg_box`]: box that contains (2)
           and (1)
        4) Dropdown [`self.group_dropdown`]: group selector
        5) ToggleButtons [`self.labels_toggles`]: `list` of `list`s with the
           labels per group
        6) Latex [`self.labels_text`]: labels title text
        7) HBox [`self.labels_box`]: box that contains all (5)
        8) HBox [`self.labels_and_text_box`]: box contains (6) and (7)
        9) VBox [`self.group_and_labels_and_text_box`]: box that contains (4)
            and (8)

    The selected values are stored in `self.selected_values` `dict`. To set the
    styling of this widget please refer to the `style()` method. To update the
    state and function of the widget, please refer to the `set_widget_state()`
    and `replace_render_function()` methods.

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
    update_function : `function` or ``None``, optional
        The update function that is executed when the index value changes.
        If ``None``, then nothing is assigned.
    style : `str` (see below)
        Sets a predefined style at the widget. Possible options are ::

            {``'minimal'``, ``'success'``, ``'info'``, ``'warning'``,
             ``'danger'``, ``''``}

    """
    def __init__(self, landmark_options, render_function=None,
                 update_function=None, style='minimal'):
        # Check given options
        landmark_options = self._parse_landmark_options_dict(landmark_options)

        # Create widgets
        self.no_landmarks_msg = ipywidgets.Latex(
            value='No landmarks available.',
            visible=self._no_landmarks_msg_visible(
                landmark_options['has_landmarks']))
        # temporarily store visible and disabled values
        tmp_visible = self._options_visible(landmark_options['has_landmarks'])
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
        super(LandmarkOptionsWidget, self).__init__(
            children=[self.landmarks_checkbox_and_msg_box,
                      self.group_and_labels_and_text_box])
        self.align = 'start'
        self.labels_box.padding = '0.3cm'

        # Assign output
        self.selected_values = landmark_options

        # Set style
        self.predefined_style(style)

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

        # Store functions
        self._render_landmarks_fun = render_landmarks_fun
        self._group_fun = group_fun
        self._labels_fun = labels_fun

        # Set render function
        self._update_function = None
        self.add_update_function(update_function)
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

    def _no_landmarks_msg_visible(self, has_landmarks):
        return not has_landmarks

    def _options_visible(self, has_landmarks):
        return has_landmarks

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

    def style(self, box_style=None, border_visible=False, border_color='black',
              border_style='solid', border_width=1, border_radius=0, padding=0,
              margin=0, font_family='', font_size=None, font_style='',
              font_weight='', labels_buttons_style=''):
        r"""
        Function that defines the styling of the widget.

        Parameters
        ----------
        box_style : `str` or ``None`` (see below), optional
            Style options ::

                {``'success'``, ``'info'``, ``'warning'``, ``'danger'``, ``''``}
                or
                ``None``

        border_visible : `bool`, optional
            Defines whether to draw the border line around the widget.
        border_color : `str`, optional
            The color of the border around the widget.
        border_style : `str`, optional
            The line style of the border around the widget.
        border_width : `float`, optional
            The line width of the border around the widget.
        border_radius : `float`, optional
            The radius of the corners of the box.
        padding : `float`, optional
            The padding around the widget.
        margin : `float`, optional
            The margin around the widget.
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

        labels_buttons_style : `str` or ``None`` (see below), optional
            Style options ::

                {``'primary'``, ``'success'``, ``'info'``, ``'warning'``,
                 ``'danger'``, ``''``}
                or
                ``None``

        """
        _format_box(self, box_style, border_visible, border_color, border_style,
                    border_width, border_radius, padding, margin)
        _format_font(self, font_family, font_size, font_style, font_weight)
        _format_font(self.render_landmarks_checkbox, font_family, font_size,
                     font_style, font_weight)
        _format_font(self.group_dropdown, font_family, font_size, font_style,
                     font_weight)
        for s_group in self.labels_toggles:
            for w in s_group:
                _format_font(w, font_family, font_size, font_style, font_weight)
                w.button_style = labels_buttons_style
        _format_font(self.labels_text, font_family, font_size, font_style,
                     font_weight)

    def predefined_style(self, style):
        r"""
        Function that set a predefined styling on the widget.

        Parameter
        ---------
        style : `str` (see below)
            Style options ::

                {``'minimal'``, ``'success'``, ``'info'``, ``'warning'``,
                 ``'danger'``, ``''``}

        """
        if style == 'minimal':
            self.style(box_style=None, border_visible=True,
                       border_color='black', border_style='solid',
                       border_width=1, border_radius=0, padding='0.2cm',
                       margin='0.3cm', font_family='', font_size=None,
                       font_style='', font_weight='', labels_buttons_style='')
        elif (style == 'info' or style == 'success' or style == 'danger' or
              style == 'warning'):
            self.style(box_style=style, border_visible=True,
                       border_color=_map_styles_to_hex_colours(style),
                       border_style='solid', border_width=1, border_radius=10,
                       padding='0.2cm', margin='0.3cm', font_family='',
                       font_size=None, font_style='', font_weight='',
                       labels_buttons_style='primary')
        else:
            raise ValueError('style must be minimal or info or success or '
                             'danger or warning')

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

    def add_update_function(self, update_function):
        r"""
        Method that adds an `update_function()` to the widget. The signature of
        the given function is also stored in `self._update_function`.

        Parameters
        ----------
        update_function : `function` or ``None``, optional
            The update function that behaves as a callback. If ``None``, then
            nothing is added.
        """
        self._update_function = update_function
        if self._update_function is not None:
            self.render_landmarks_checkbox.on_trait_change(
                self._update_function, 'value')
            self.group_dropdown.on_trait_change(self._update_function, 'value')
            self._add_function_to_labels_toggles(self._update_function)

    def remove_update_function(self):
        r"""
        Method that removes the current `self._update_function()` from the
        widget and sets ``self._update_function = None``.
        """
        self.render_landmarks_checkbox.on_trait_change(self._update_function,
                                                       'value', remove=True)
        self.group_dropdown.on_trait_change(self._update_function, 'value',
                                            remove=True)
        self._remove_function_from_labels_toggles(self._update_function)
        self._update_function = None

    def replace_update_function(self, update_function):
        r"""
        Method that replaces the current `self._update_function()` of the widget
        with the given `update_function()`.

        Parameters
        ----------
        update_function : `function` or ``None``, optional
            The update function that behaves as a callback. If ``None``, then
            nothing is happening.
        """
        # remove old function
        self.remove_update_function()

        # add new function
        self.add_update_function(update_function)

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
        update_function = self._update_function
        self.remove_update_function()

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
            landmark_options['has_landmarks'])
        # temporarily store visible and disabled values
        tmp_visible = self._options_visible(landmark_options['has_landmarks'])
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
            for w in self.labels_box.children:
                w.disabled = tmp_disabled
                w.visible = tmp_visible

        # Re-assign the rest of the callbacks
        self.render_landmarks_checkbox.on_trait_change(
            self._render_landmarks_fun, 'value')
        self.group_dropdown.on_trait_change(self._group_fun, 'value')
        self._add_function_to_labels_toggles(self._labels_fun)

        # Re-assign render callback
        self.add_update_function(update_function)
        self.add_render_function(render_function)

        # Assign new options dict to selected_values
        self.selected_values = landmark_options

        # trigger render function if allowed
        if allow_callback:
            self._update_function('', True)
            self._render_function('', True)


class TextPrintWidget(ipywidgets.FlexBox):
    r"""
    Creates a widget for printing text. Specifically, it consists of:

        1) Latex [`self.latex_texts`]: the text lines

    To set the styling of this widget please refer to the `style()` method. To
    update the state of the widget, please refer to the `set_widget_state()`
    method.

    Parameters
    ----------
    n_lines : `int`
        The number of lines of the text to be printed.
    text_per_line : `list` of length `n_lines`
        The text to be printed per line.
    style : `str` (see below)
        Sets a predefined style at the widget. Possible options are ::

            {``'minimal'``, ``'success'``, ``'info'``, ``'warning'``,
             ``'danger'``, ``''``}

    """
    def __init__(self, n_lines, text_per_line, style='minimal'):
        self.latex_texts = [ipywidgets.Latex(value=text_per_line[i])
                            for i in range(n_lines)]
        super(TextPrintWidget, self).__init__(children=self.latex_texts)
        self.align = 'start'

        # Assign options
        self.n_lines = n_lines
        self.text_per_line = text_per_line

        # Set style
        self.predefined_style(style)

    def style(self, box_style=None, border_visible=False, border_color='black',
              border_style='solid', border_width=1, border_radius=0, padding=0,
              margin=0, font_family='', font_size=None, font_style='',
              font_weight=''):
        r"""
        Function that defines the styling of the widget.

        Parameters
        ----------
        box_style : `str` or ``None`` (see below), optional
            Style options ::

                {``'success'``, ``'info'``, ``'warning'``, ``'danger'``, ``''``}
                or
                ``None``

        border_visible : `bool`, optional
            Defines whether to draw the border line around the widget.
        border_color : `str`, optional
            The color of the border around the widget.
        border_style : `str`, optional
            The line style of the border around the widget.
        border_width : `float`, optional
            The line width of the border around the widget.
        border_radius : `float`, optional
            The radius of the corners of the box.
        padding : `float`, optional
            The padding around the widget.
        margin : `float`, optional
            The margin around the widget.
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
        _format_box(self, box_style, border_visible, border_color, border_style,
                    border_width, border_radius, padding, margin)
        _format_font(self, font_family, font_size, font_style, font_weight)
        for i in range(self.n_lines):
            _format_font(self.latex_texts[i], font_family, font_size,
                         font_style, font_weight)

    def predefined_style(self, style):
        r"""
        Function that set a predefined styling on the widget.

        Parameter
        ---------
        style : `str` (see below)
            Style options ::

                {``'minimal'``, ``'success'``, ``'info'``, ``'warning'``,
                 ``'danger'``, ``''``}

        """
        if style == 'minimal':
            self.style(box_style=None, border_visible=True,
                       border_color='black', border_style='solid',
                       border_width=1, border_radius=0, padding='0.1cm',
                       margin='0.3cm', font_family='', font_size=None,
                       font_style='', font_weight='')
        elif (style == 'info' or style == 'success' or style == 'danger' or
              style == 'warning'):
            self.style(box_style=style, border_visible=True,
                       border_color=_map_styles_to_hex_colours(style),
                       border_style='solid', border_width=1, border_radius=10,
                       padding='0.1cm', margin='0.3cm', font_family='',
                       font_size=None, font_style='', font_weight='')
        else:
            raise ValueError('style must be minimal or info or success or '
                             'danger or warning')

    def set_widget_state(self, n_lines, text_per_line):
        r"""
        Method that updates the state of the widget with a new set of values.

        Parameter
        ---------
        n_lines : `int`
            The number of lines of the text to be printed.
        text_per_line : `list` of length `n_lines`
            The text to be printed per line.
        """
        # Check if n_lines has changed
        if n_lines != self.n_lines:
            self.latex_texts = [ipywidgets.Latex(value=text_per_line[i])
                                for i in range(n_lines)]
            self.children = self.latex_texts
        else:
            for i in range(n_lines):
                self.latex_texts[i].value = text_per_line[i]
        self.n_lines = n_lines
        self.text_per_line = text_per_line


class AnimationOptionsWidget(ipywidgets.FlexBox):
    r"""
    Creates a widget for selecting channel options when rendering an image.
    Specifically, it consists of:

        1) ToggleButton [`self.play_toggle`]: the play button
        2) ToggleButton [`self.stop_toggle`]: the stop button
        3) ToggleButton [`self.play_options_toggle`]: the play options
        4) Checkbox [`self.loop_checkbox`]: where animation loops
        5) FloatText [`self.interval_text`]: animation interval in seconds
        6) VBox [`self.play_options_box`]: box that contains (3), (4) and (5)
        7) HBox [`self.animation_box`]: box that contains (1), (2) and (6)
        8) `IndexButtonsWidget` or `IndexSliderWidget`: the index object

    The selected values are stored in `self.selected_values` `dict`. To set the
    styling of this widget please refer to the `style()` method. To update the
    state and function of the widget, please refer to the `set_widget_state()`,
    `replace_render_function()` and `replace_update_function()` methods.

    Parameters
    ----------
    index : `dict`
        The dictionary with the default options. For example ::

            index = {'min': 0, 'max': 100, 'step': 1, 'index': 10}

    render_function : `function` or ``None``, optional
        The render function that is executed when a widgets' value changes.
        If ``None``, then nothing is assigned.
    update_function : `function` or ``None``, optional
        The update function that is executed when the index value changes.
        If ``None``, then nothing is assigned.
    index_style : {``'buttons'``, ``'slider'``}, optional
        If ``'buttons'``, then `IndexButtonsWidget()` class is called. If
        ``'slider'``, then 'IndexSliderWidget()' class is called.
    interval : `float`, optional
        The interval between the animation progress.
    description : `str`, optional
        The title of the widget.
    minus_description : `str`, optional
        The title of the button that decreases the index.
    plus_description : `str`, optional
        The title of the button that increases the index.
    loop_enabled : `bool`, optional
        If ``True``, then if by pressing the buttons we reach the minimum
        (maximum) index values, then the counting will continue from the end
        (beginning). If ``False``, the counting will stop at the minimum
        (maximum) value.
    text_editable : `bool`, optional
        Flag that determines whether the index text will be editable.
    style : `str` (see below)
        Sets a predefined style at the widget. Possible options are ::

            {``'minimal'``, ``'success'``, ``'info'``, ``'warning'``,
             ``'danger'``, ``''``}

    """
    def __init__(self, index, render_function=None, update_function=None,
                 index_style='buttons', interval=0.5, description='Index: ',
                 minus_description='-', plus_description='+', loop_enabled=True,
                 text_editable=True, style='minimal'):
        from time import sleep
        from IPython import get_ipython

        # Get the kernel to use it later in order to make sure that the widgets'
        # traits changes are passed during a while-loop
        kernel = get_ipython().kernel

        # Create index widget
        if index_style == 'slider':
            self.index_wid = IndexSliderWidget(index, description=description)
        elif index_style == 'buttons':
            self.index_wid = IndexButtonsWidget(
                index, description=description,
                minus_description=minus_description,
                plus_description=plus_description, loop_enabled=loop_enabled,
                text_editable=text_editable)
        else:
            raise ValueError('index_style should be either slider or buttons')
        self.index_wid.style(box_style=None, border_visible=False,
                             padding=0, margin='0.15cm')

        # Create other widgets
        self.play_toggle = ipywidgets.ToggleButton(description='Play >',
                                                   value=False)
        self.stop_toggle = ipywidgets.ToggleButton(description='Stop',
                                                   value=True, disabled=True)
        self.play_options_toggle = ipywidgets.ToggleButton(
            description='Options', value=False)
        self.loop_checkbox = ipywidgets.Checkbox(description='Loop',
                                                 value=loop_enabled)
        self.interval_text = ipywidgets.FloatText(description='Interval (sec)',
                                                  value=interval)
        self.loop_interval_box = ipywidgets.VBox(
            children=[self.interval_text, self.loop_checkbox], visible=False,
            margin='0.1cm', padding='0.1cm', border_color='black',
            border_style='solid', border_width=1)
        self.play_options_box = ipywidgets.VBox(
            children=[self.play_options_toggle, self.loop_interval_box])
        self.animation_box = ipywidgets.HBox(
            children=[self.play_toggle, self.stop_toggle,
                      self.play_options_box], margin='0.15cm', padding=0)
        super(AnimationOptionsWidget, self).__init__(
            children=[self.index_wid, self.animation_box])
        self.align = 'start'
        self.orientation = 'horizontal'

        # Assign output
        self.selected_values = index
        self.index_style = index_style

        # Set style
        self.predefined_style(style)

        # Set functionality
        def play_pressed(name, value):
            self.stop_toggle.value = not value
            self.play_toggle.disabled = value
            self.play_options_toggle.disabled = value
            if value:
                self.play_options_toggle.value = False
        self.play_toggle.on_trait_change(play_pressed, 'value')

        def stop_pressed(name, value):
            self.play_toggle.value = not value
            self.stop_toggle.disabled = value
            self.play_options_toggle.disabled = not value
        self.stop_toggle.on_trait_change(stop_pressed, 'value')

        def play_options_visibility(name, value):
            self.loop_interval_box.visible = value
        self.play_options_toggle.on_trait_change(play_options_visibility,
                                                 'value')

        def animate(name, value):
            if self.loop_checkbox.value:
                # loop is enabled
                i = self.selected_values['index']
                if i < self.selected_values['max']:
                    i += self.selected_values['step']
                else:
                    i = self.selected_values['min']

                while (i <= self.selected_values['max'] and
                       not self.stop_toggle.value):
                    # update index value
                    if index_style == 'slider':
                        self.index_wid.slider.value = i
                    else:
                        self.index_wid.index_text.value = i

                    # Run IPython iteration.
                    # This is the code that makes this operation non-blocking.
                    # This allows widget messages and callbacks to be processed.
                    kernel.do_one_iteration()

                    # update counter
                    if i < self.selected_values['max']:
                        i += self.selected_values['step']
                    else:
                        i = self.selected_values['min']

                    # wait
                    sleep(self.interval_text.value)
            else:
                # loop is disabled
                i = self.selected_values['index']
                i += self.selected_values['step']
                while (i <= self.selected_values['max'] and
                       not self.stop_toggle.value):
                    # update index value
                    if index_style == 'slider':
                        self.index_wid.slider.value = i
                    else:
                        self.index_wid.index_text.value = i

                    # Run IPython iteration.
                    # This is the code that makes this operation non-blocking.
                    # This allows widget messages and callbacks to be processed.
                    kernel.do_one_iteration()

                    # update counter
                    i += self.selected_values['step']

                    # wait
                    sleep(self.interval_text.value)
                if i > self.selected_values['max']:
                    self.stop_toggle.value = True
        self.play_toggle.on_trait_change(animate, 'value')

        # Set render and update functions
        self._update_function = None
        self.add_update_function(update_function)
        self._render_function = None
        self.add_render_function(render_function)

    def style(self, box_style=None, border_visible=False, border_color='black',
              border_style='solid', border_width=1, border_radius=0, padding=0,
              margin=0, font_family='', font_size=None, font_style='',
              font_weight=''):
        r"""
        Function that defines the styling of the widget.

        Parameters
        ----------
        box_style : `str` or ``None`` (see below), optional
            Style options ::

                {``'success'``, ``'info'``, ``'warning'``, ``'danger'``, ``''``}
                or
                ``None``

        border_visible : `bool`, optional
            Defines whether to draw the border line around the widget.
        border_color : `str`, optional
            The color of the border around the widget.
        border_style : `str`, optional
            The line style of the border around the widget.
        border_width : `float`, optional
            The line width of the border around the widget.
        border_radius : `float`, optional
            The radius of the corners of the box.
        padding : `float`, optional
            The padding around the widget.
        margin : `float`, optional
            The margin around the widget.
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
        _format_box(self, box_style, border_visible, border_color, border_style,
                    border_width, border_radius, padding, margin)
        _format_font(self, font_family, font_size, font_style, font_weight)
        _format_font(self.play_toggle, font_family, font_size, font_style,
                     font_weight)
        _format_font(self.stop_toggle, font_family, font_size, font_style,
                     font_weight)
        _format_font(self.play_options_toggle, font_family, font_size,
                     font_style, font_weight)
        _format_font(self.loop_checkbox, font_family, font_size, font_style,
                     font_weight)
        _format_font(self.interval_text, font_family, font_size, font_style,
                     font_weight)
        if self.index_style == 'buttons':
            self.index_wid.style(
                box_style=None, border_visible=False, padding=0,
                margin='0.15cm', font_family=font_family, font_size=font_size,
                font_style=font_style, font_weight=font_weight)
        else:
            self.index_wid.style(
                box_style=None, border_visible=False, padding=0,
                margin='0.15cm', font_family=font_family, font_size=font_size,
                font_style=font_style, font_weight=font_weight)

    def predefined_style(self, style):
        r"""
        Function that set a predefined styling on the widget.

        Parameter
        ---------
        style : `str` (see below)
            Style options ::

                {``'minimal'``, ``'success'``, ``'info'``, ``'warning'``,
                 ``'danger'``, ``''``}

        """
        if style == 'minimal':
            self.style(box_style='', border_visible=False)
            self.play_toggle.button_style = ''
            self.play_toggle.font_weight = 'normal'
            self.stop_toggle.button_style = ''
            self.stop_toggle.font_weight = 'normal'
            self.play_options_toggle.button_style = ''
            _format_box(self.loop_interval_box, '', False, 'black', 'solid', 1,
                        10, '0.1cm', '0.1cm')
            if self.index_style == 'buttons':
                self.index_wid.button_plus.button_style = ''
                self.index_wid.button_plus.font_weight = 'normal'
                self.index_wid.button_minus.button_style = ''
                self.index_wid.button_minus.font_weight = 'normal'
                self.index_wid.index_text.background_color = ''
            elif self.index_style == 'slider':
                self.index_wid.slider.slider_color = ''
                self.index_wid.slider.background_color = ''
        elif (style == 'info' or style == 'success' or style == 'danger' or
              style == 'warning'):
            self.style(box_style=style, border_visible=False)
            self.play_toggle.button_style = 'success'
            self.play_toggle.font_weight = 'bold'
            self.stop_toggle.button_style = 'danger'
            self.stop_toggle.font_weight = 'bold'
            self.play_options_toggle.button_style = 'info'
            _format_box(self.loop_interval_box, 'info', True,
                        _map_styles_to_hex_colours('info'), 'solid', 1, 10,
                        '0.1cm', '0.1cm')
            if self.index_style == 'buttons':
                self.index_wid.button_plus.button_style = 'primary'
                self.index_wid.button_plus.font_weight = 'bold'
                self.index_wid.button_minus.button_style = 'primary'
                self.index_wid.button_minus.font_weight = 'bold'
                self.index_wid.index_text.background_color = \
                    _map_styles_to_hex_colours(style, True)
            elif self.index_style == 'slider':
                self.index_wid.slider.slider_color = \
                    _map_styles_to_hex_colours(style)
                self.index_wid.slider.background_color = \
                    _map_styles_to_hex_colours(style)
        else:
            raise ValueError('style must be minimal or info or success or '
                             'danger or warning')

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
            self.index_wid.add_render_function(self._render_function)

    def remove_render_function(self):
        r"""
        Method that removes the current `self._render_function()` from the
        widget and sets ``self._render_function = None``.
        """
        self.index_wid.remove_render_function()
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

    def add_update_function(self, update_function):
        r"""
        Method that adds an `update_function()` to the widget. The signature of
        the given function is also stored in `self._update_function`.

        Parameters
        ----------
        update_function : `function` or ``None``, optional
            The update function that behaves as a callback. If ``None``, then
            nothing is added.
        """
        self._update_function = update_function
        if self._update_function is not None:
            self.index_wid.add_update_function(self._update_function)

    def remove_update_function(self):
        r"""
        Method that removes the current `self._update_function()` from the
        widget and sets ``self._update_function = None``.
        """
        self.index_wid.remove_update_function()
        self._update_function = None

    def replace_update_function(self, update_function):
        r"""
        Method that replaces the current `self._update_function()` of the widget
        with the given `update_function()`.

        Parameters
        ----------
        update_function : `function` or ``None``, optional
            The update function that behaves as a callback. If ``None``, then
            nothing is happening.
        """
        # remove old function
        self.remove_update_function()

        # add new function
        self.add_update_function(update_function)

    def set_widget_state(self, index, allow_callback=True):
        r"""
        Method that updates the state of the widget with a new set of values.

        Parameter
        ---------
        index : `dict`
            The dictionary with the default options. For example ::

                index = {'min': 0, 'max': 100, 'step': 1, 'index': 10}

        allow_callback : `bool`, optional
            If ``True``, it allows triggering of any callback functions.
        """
        if self.play_toggle.value:
            self.play_toggle.value = False
        if self.index_style == 'slider':
            self.index_wid.set_widget_state(index,
                                            allow_callback=allow_callback)
        else:
            self.index_wid.set_widget_state(
                index, loop_enabled=self.index_wid.loop_enabled,
                text_editable=self.index_wid.text_editable,
                allow_callback=allow_callback)
        self.selected_values = index


class RendererOptionsWidget(ipywidgets.FlexBox):
    r"""
    Creates a widget for selecting rendering options when rendering an object.
    Specifically, it consists of:

        1) Dropdown [`self.object_selection_dropdown`]: the object selection
           dropdown
        2) {Line,Marker,Image,Numbering,Figure,Legend,Grid}OptionsWidget
           [`self.options_widgets`]: the various rendering sub-options widgets
        3) Tab [`self.suboptions_tab`]: box that contains (2)

    The selected values are stored in `self.selected_values` `dict`. To set the
    styling of this widget please refer to the `style()` method. To update the
    state and function of the widget, please refer to the `set_widget_state()`
    and `replace_render_function()` methods.

    Parameters
    ----------
    renderer_options : `list` of `dict`
        The selected rendering options per object. The `list` must have length
        `n_objects` and contain a `dict` of rendering options per object.
        For example, in case we had two objects to render ::

            lines_options = {'render_lines': True,
                             'line_width': 1,
                             'line_colour': ['b', 'r'],
                             'line_style': '-'}
            markers_options = {'render_markers': True,
                               'marker_size': 20,
                               'marker_face_colour': ['w', 'w'],
                               'marker_edge_colour': ['b', 'r'],
                               'marker_style': 'o',
                               'marker_edge_width': 1}
            numbering_options = {'render_numbering': True,
                                 'numbers_font_name': 'serif',
                                 'numbers_font_size': 10,
                                 'numbers_font_style': 'normal',
                                 'numbers_font_weight': 'normal',
                                 'numbers_font_colour': ['k'],
                                 'numbers_horizontal_align': 'center',
                                 'numbers_vertical_align': 'bottom'}
            legend_options = {'render_legend': True,
                              'legend_title': '',
                              'legend_font_name': 'serif',
                              'legend_font_style': 'normal',
                              'legend_font_size': 10,
                              'legend_font_weight': 'normal',
                              'legend_marker_scale': 1.,
                              'legend_location': 2,
                              'legend_bbox_to_anchor': (1.05, 1.),
                              'legend_border_axes_pad': 1.,
                              'legend_n_columns': 1,
                              'legend_horizontal_spacing': 1.,
                              'legend_vertical_spacing': 1.,
                              'legend_border': True,
                              'legend_border_padding': 0.5,
                              'legend_shadow': False,
                              'legend_rounded_corners': True}
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
            image_options = {'alpha': 1.,
                             'interpolation': 'bilinear',
                             'cmap_name': 'gray'}
            rendering_dict = {'lines': lines_options,
                              'markers': markers_options,
                              'numbering': numbering_options,
                              'legend': legend_options,
                              'figure': figure_options,
                              'grid': grid_options,
                              'image': image_options}
            renderer_options = [rendering_dict, rendering_dict]

    options_tabs : `list` of `str`
        `List` that defines the ordering of the options tabs. Possible values
        are

            ============= ===============================
            Value         Returned class
            ============= ===============================
            'lines'       `LineOptionsWidget`
            'markers'     `MarkerOptionsWidget`
            'numbering'   `NumberingOptionsWidget`
            'figure_one'  `FigureOptionsOneScaleWidget`
            'figure_two'  `FigureOptionsTwoScalesWidget`
            'legend'      `LegendOptionsWidget`
            'grid'        `GridOptionsWidget`
            'image'       `ImageOptionsWidget`
            ============= ===============================

    objects_names : `list` of `str` or ``None``, optional
        A `list` with the names of the objects that will be used in the
        selection dropdown menu. If ``None``, then the names will have the
        format ``%d``.
    labels_per_object : `list` of `list` or ``None``, optional
        A `list` that contains a `list` of labels for each object. Those
        `labels` are employed by the `ColourSelectionWidget`. An example for
        which this option is useful is in the case we wish to create rendering
        options for multiple :map:`LandmarkGroup` objects and each one of them
        has a different set of `labels`. If ``None``, then `labels_per_object`
        is a `list` of length `n_objects` with ``None``.
    selected_object : `int`, optional
        The object for which to show the rendering options in the beginning,
        when the widget is created.
    object_selection_dropdown_visible : `bool`, optional
        Controls the visibility of the object selection dropdown
        (`self.object_selection_dropdown`).
    render_function : `function` or ``None``, optional
        The render function that is executed when a widgets' value changes.
        If ``None``, then nothing is assigned.
    style : `str` (see below)
        Sets a predefined style at the widget. Possible options are ::

            {``'minimal'``, ``'success'``, ``'info'``, ``'warning'``,
             ``'danger'``, ``''``}

    tabs_style : `str` (see below)
        Sets a predefined style at the tabs of the widget. Possible options
        are ::

            {``'minimal'``, ``'success'``, ``'info'``, ``'warning'``,
             ``'danger'``, ``''``}

    """
    def __init__(self, renderer_options, options_tabs, objects_names=None,
                 labels_per_object=None, selected_object=0,
                 object_selection_dropdown_visible=True, render_function=None,
                 style='minimal', tabs_style='minimal'):
        # Make sure that renderer_options is a list even with one member
        if not isinstance(renderer_options, list):
            renderer_options = [renderer_options]

        # Get number of objects to be rendered
        self.n_objects = len(renderer_options)

        # Check labels_per_object
        if labels_per_object is None:
            labels_per_object = [None] * self.n_objects

        # Check objects_names
        if objects_names is None:
            objects_names = [str(k) for k in range(self.n_objects)]

        # Create widgets
        # object selection dropdown
        objects_dict = OrderedDict()
        for k, g in enumerate(objects_names):
            objects_dict[g] = k
        tmp_visible = self._selection_dropdown_visible(
            object_selection_dropdown_visible)
        self.object_selection_dropdown = ipywidgets.Dropdown(
            options=objects_dict, value=selected_object, description='Select',
            visible=tmp_visible, margin='0.1cm')
        # options widgets
        options_widgets = []
        tab_titles = []
        for o in options_tabs:
            # get the options to pass to the sub-options constructors
            if o == 'figure_one' or o == 'figure_two':
                tmp_options = renderer_options[selected_object]['figure']
            else:
                tmp_options = renderer_options[selected_object][o]
            # get the labels to pass in where required
            tmp_labels = labels_per_object[selected_object]
            # call sub-options classes
            if o == 'lines':
                options_widgets.append(LineOptionsWidget(
                    tmp_options, render_function=render_function,
                    render_checkbox_title='Render lines', labels=tmp_labels))
                tab_titles.append('Lines')
            elif o == 'markers':
                options_widgets.append(MarkerOptionsWidget(
                    tmp_options, render_function=render_function,
                    render_checkbox_title='Render markers', labels=tmp_labels))
                tab_titles.append('Markers')
            elif o == 'image':
                options_widgets.append(ImageOptionsWidget(
                    tmp_options, render_function=render_function))
                tab_titles.append('Image')
            elif o == 'numbering':
                options_widgets.append(NumberingOptionsWidget(
                    tmp_options, render_function=render_function,
                    render_checkbox_title='Render numbering'))
                tab_titles.append('Numbering')
            elif o == 'figure_one':
                options_widgets.append(FigureOptionsOneScaleWidget(
                    tmp_options, render_function=render_function,
                    figure_scale_visible=True, axes_visible=True))
                tab_titles.append('Figure/Axes')
            elif o == 'figure_two':
                options_widgets.append(FigureOptionsTwoScalesWidget(
                    tmp_options, render_function=render_function,
                    figure_scale_visible=True, axes_visible=True,
                    coupled_default=False))
                tab_titles.append('Figure/Axes')
            elif o == 'legend':
                options_widgets.append(LegendOptionsWidget(
                    tmp_options, render_function=render_function,
                    render_checkbox_title='Render legend'))
                tab_titles.append('Legend')
            elif o == 'grid':
                options_widgets.append(GridOptionsWidget(
                    tmp_options, render_function=render_function,
                    render_checkbox_title='Render grid'))
                tab_titles.append('Grid')
        self.options_widgets = options_widgets
        self.tab_titles = tab_titles
        self.suboptions_tab = ipywidgets.Tab(children=options_widgets)
        # set titles
        for (k, tl) in enumerate(self.tab_titles):
            self.suboptions_tab.set_title(k, tl)
        self.options_box = ipywidgets.VBox(
            children=[self.object_selection_dropdown, self.suboptions_tab],
            align='center', padding='0.2cm')
        super(RendererOptionsWidget, self).__init__(children=[self.options_box])
        self.align = 'start'

        # Assign output
        self.selected_values = renderer_options
        self.options_tabs = options_tabs
        self.objects_names = objects_names
        self.labels_per_object = labels_per_object
        self.object_selection_dropdown_visible = \
            object_selection_dropdown_visible

        # Set style
        self.predefined_style(style, tabs_style)

        # Set functionality
        def update_widgets(name, value):
            for i, tab in enumerate(self.options_tabs):
                # get the options to pass to the sub-options update functions
                if tab == 'figure_one' or tab == 'figure_two':
                    tmp_options = self.selected_values[value]['figure']
                else:
                    tmp_options = self.selected_values[value][tab]
                # call sub-options classes
                if tab == 'lines' or tab == 'markers':
                    self.options_widgets[i].set_widget_state(
                        tmp_options, labels=self.labels_per_object[value],
                        allow_callback=False)
                else:
                    self.options_widgets[i].set_widget_state(
                        tmp_options, allow_callback=False)
        self.object_selection_dropdown.on_trait_change(update_widgets, 'value')

        # Set render function
        self._render_function = render_function

    def _selection_dropdown_visible(self, object_selection_dropdown_visible):
        return object_selection_dropdown_visible and self.n_objects > 1

    def style(self, box_style=None, border_visible=False, border_color='black',
              border_style='solid', border_width=1, border_radius=0,
              padding='0.2cm', margin=0, tabs_box_style=None,
              tabs_border_visible=True, tabs_border_color='black',
              tabs_border_style='solid', tabs_border_width=1,
              tabs_border_radius=1, tabs_padding=0, tabs_margin=0,
              font_family='', font_size=None, font_style='', font_weight=''):
        r"""
        Function that defines the styling of the widget.

        Parameters
        ----------
        box_style : `str` or ``None`` (see below), optional
            Style options ::

                {``'success'``, ``'info'``, ``'warning'``, ``'danger'``, ``''``}
                or
                ``None``

        border_visible : `bool`, optional
            Defines whether to draw the border line around the widget.
        border_color : `str`, optional
            The color of the border around the widget.
        border_style : `str`, optional
            The line style of the border around the widget.
        border_width : `float`, optional
            The line width of the border around the widget.
        border_radius : `float`, optional
            The radius of the corners of the box.
        padding : `float`, optional
            The padding around the widget.
        margin : `float`, optional
            The margin around the widget.
        tabs_box_style : `str` or ``None`` (see below), optional
            Style options ::

                {``'success'``, ``'info'``, ``'warning'``, ``'danger'``, ``''``}
                or
                ``None``

        tabs_border_visible : `bool`, optional
            Defines whether to draw the border line around the tab widgets.
        tabs_border_color : `str`, optional
            The color of the border around the tab widgets.
        tabs_border_style : `str`, optional
            The line style of the border around the tab widgets.
        tabs_border_width : `float`, optional
            The line width of the border around the tab widgets.
        tabs_border_radius : `float`, optional
            The radius of the corners of the box of the tab widgets.
        tabs_padding : `float`, optional
            The padding around the tab widgets.
        tabs_margin : `float`, optional
            The margin around the tab widgets.
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
        _format_box(self, box_style, border_visible, border_color, border_style,
                    border_width, border_radius, padding, margin)
        for wid in self.options_widgets:
            wid.style(box_style=tabs_box_style,
                      border_visible=tabs_border_visible,
                      border_color=tabs_border_color,
                      border_style=tabs_border_style,
                      border_width=tabs_border_width,
                      border_radius=tabs_border_radius, padding=tabs_padding,
                      margin=tabs_margin, font_family=font_family,
                      font_size=font_size, font_style=font_style,
                      font_weight=font_weight)
        _format_font(self, font_family, font_size, font_style, font_weight)
        _format_font(self.object_selection_dropdown, font_family, font_size,
                     font_style, font_weight)

    def predefined_style(self, style, tabs_style='minimal'):
        r"""
        Function that set a predefined styling on the widget.

        Parameter
        ---------
        style : `str` (see below)
            Style options ::

                {``'minimal'``, ``'success'``, ``'info'``, ``'warning'``,
                 ``'danger'``, ``''``}

        tabs_style : `str` (see below), optional
            Style options ::

                {``'minimal'``, ``'success'``, ``'info'``, ``'warning'``,
                 ``'danger'``, ``''``}

        """
        if tabs_style == 'minimal' or tabs_style=='':
            tabs_style = ''
            tabs_border_visible = False
            tabs_border_color = 'black'
            tabs_border_radius = 0
            tabs_padding = 0
        else:
            tabs_style = tabs_style
            tabs_border_visible = True
            tabs_border_color = _map_styles_to_hex_colours(tabs_style)
            tabs_border_radius = 10
            tabs_padding = '0.3cm'

        if style == 'minimal':
            self.style(box_style='', border_visible=True, border_color='black',
                       border_style='solid', border_width=1, border_radius=0,
                       padding='0.2cm', margin='0.5cm', font_family='',
                       font_size=None, font_style='', font_weight='',
                       tabs_box_style=tabs_style,
                       tabs_border_visible=tabs_border_visible,
                       tabs_border_color=tabs_border_color,
                       tabs_border_style='solid', tabs_border_width=1,
                       tabs_border_radius=tabs_border_radius,
                       tabs_padding=tabs_padding, tabs_margin='0.1cm')
        elif (style == 'info' or style == 'success' or style == 'danger' or
              style == 'warning'):
            self.style(box_style=style, border_visible=True,
                       border_color=_map_styles_to_hex_colours(style),
                       border_style='solid', border_width=1, border_radius=10,
                       padding='0.2cm', margin='0.5cm', font_family='',
                       font_size=None, font_style='', font_weight='',
                       tabs_box_style=tabs_style,
                       tabs_border_visible=tabs_border_visible,
                       tabs_border_color=tabs_border_color,
                       tabs_border_style='solid', tabs_border_width=1,
                       tabs_border_radius=tabs_border_radius,
                       tabs_padding=tabs_padding, tabs_margin='0.1cm')
        else:
            raise ValueError('style must be minimal or info or success or '
                             'danger or warning')

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
            for wid in self.options_widgets:
                wid.add_render_function(self._render_function)

    def remove_render_function(self):
        r"""
        Method that removes the current `self._render_function()` from the
        widget and sets ``self._render_function = None``.
        """
        for wid in self.options_widgets:
            wid.remove_render_function()
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

    def set_widget_state(self, renderer_options, labels_per_object,
                         selected_object=None,
                         object_selection_dropdown_visible=None,
                         allow_callback=True):
        r"""
        Method that updates the state of the widget with a new set of values.
        Note that the number of objects should not change.

        Parameter
        ---------
        renderer_options : `list` of `dict`
            The selected rendering options per object. The `list` must have
            length `n_objects` and contain a `dict` of rendering options per
            object. For example, in case we had two objects to render ::

                lines_options = {'render_lines': True,
                                 'line_width': 1,
                                 'line_colour': ['b', 'r'],
                                 'line_style': '-'}
                markers_options = {'render_markers': True,
                                   'marker_size': 20,
                                   'marker_face_colour': ['w', 'w'],
                                   'marker_edge_colour': ['b', 'r'],
                                   'marker_style': 'o',
                                   'marker_edge_width': 1}
                numbering_options = {'render_numbering': True,
                                     'numbers_font_name': 'serif',
                                     'numbers_font_size': 10,
                                     'numbers_font_style': 'normal',
                                     'numbers_font_weight': 'normal',
                                     'numbers_font_colour': ['k'],
                                     'numbers_horizontal_align': 'center',
                                     'numbers_vertical_align': 'bottom'}
                legend_options = {'render_legend': True,
                                  'legend_title': '',
                                  'legend_font_name': 'serif',
                                  'legend_font_style': 'normal',
                                  'legend_font_size': 10,
                                  'legend_font_weight': 'normal',
                                  'legend_marker_scale': 1.,
                                  'legend_location': 2,
                                  'legend_bbox_to_anchor': (1.05, 1.),
                                  'legend_border_axes_pad': 1.,
                                  'legend_n_columns': 1,
                                  'legend_horizontal_spacing': 1.,
                                  'legend_vertical_spacing': 1.,
                                  'legend_border': True,
                                  'legend_border_padding': 0.5,
                                  'legend_shadow': False,
                                  'legend_rounded_corners': True}
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
                image_options = {'alpha': 1.,
                                 'interpolation': 'bilinear',
                                 'cmap_name': 'gray'}
                rendering_dict = {'lines': lines_options,
                                  'markers': markers_options,
                                  'numbering': numbering_options,
                                  'legend': legend_options,
                                  'figure': figure_options,
                                  'grid': grid_options
                                  'image': image_options}
                renderer_options = [rendering_dict, rendering_dict]

        labels_per_object : `list` of `list` or ``None``, optional
            A `list` that contains a `list` of labels for each object. Those
            `labels` are employed by the `ColourSelectionWidget`. An example for
             which this option is useful is in the case we wish to create
             rendering options for multiple :map:`LandmarkGroup` objects and
             each one of them has a different set of `labels`. If ``None``, then
             `labels_per_object` is a `list` of lenth `n_objects` with ``None``.
        selected_object : `int`, optional
            The object for which to show the rendering options in the beginning,
            when the widget is created.
        object_selection_dropdown_visible : `bool`, optional
            Controls the visibility of the object selection dropdown
            (`self.object_selection_dropdown`).
        allow_callback : `bool`, optional
            If ``True``, it allows triggering of any callback functions.
        """
        # Check options
        if selected_object is None:
            selected_object = self.object_selection_dropdown.value
        if object_selection_dropdown_visible is not None:
            self.object_selection_dropdown.visible = \
                self._selection_dropdown_visible(
                    object_selection_dropdown_visible)
            self.object_selection_dropdown_visible = \
                object_selection_dropdown_visible

        # Update sub-options widgets
        for i, tab in enumerate(self.options_tabs):
            # get the options to pass to the sub-options update functions
            if tab == 'figure_one' or tab == 'figure_two':
                tmp_options = renderer_options[selected_object]['figure']
            else:
                tmp_options = renderer_options[selected_object][tab]
            # call sub-options classes
            if tab == 'lines' or tab == 'markers':
                self.options_widgets[i].set_widget_state(
                    tmp_options, labels=labels_per_object[selected_object],
                    allow_callback=False)
            else:
                self.options_widgets[i].set_widget_state(
                    tmp_options, allow_callback=False)

        # Assign new options dict to selected_values
        self.selected_values = renderer_options
        self.labels_per_object = labels_per_object

        # trigger render function if allowed
        if allow_callback:
            self._render_function('', True)


class SaveFigureOptionsWidget(ipywidgets.FlexBox):
    r"""
    Creates a widget for saving a figure to file. Specifically, it consists of:

        1) Select [`self.file_format_select`]: image format selection
        2) FloatText [`self.dpi_text`]: set dpi
        3) Dropdown [`self.orientation_dropdown`]: paper orientation selection
        4) Select [`self.papertype_select`]: paper type selection
        5) Checkbox [`self.transparent_checkbox`]: set transparency
        6) ColourSelectionWidget [`self.facecolour_widget`]: set face colour
        7) ColourSelectionWidget [`self.edgecolour_widget`]: set edge colour
        8) FloatText [`self.pad_inches_text`]: set padding in inches
        9) Text [`self.filename_text`]: set path and filename
        10) Checkbox [`self.overwrite_checkbox`]: overwrite flag
        11) Latex [`self.error_latex`]: error message area
        12) Button [`self.save_button`]: save button
        13) VBox [`self.path_box`]: box that contains (9), (1), (10) and (4)
        14) VBox [`self.page_box`]: box that contains (3), (2) and (8)
        15) VBox [`self.colour_box`]: box that contains (6), (7) and (5)
        16) Tab [`self.options_tabs`]: box that contains (13), (14) and (15)
        17) HBox [`self.save_box`]: box that contains (12) and (11)
        18) VBox [`self.options_box`]: box that contains (16) and (17)

    To set the styling of this widget please refer to the `style()` method.

    Parameters
    ----------
    renderer : :map:`Renderer` class or subclass
        The renderer object that was used to render the figure.
    file_format : `str`, optional
        The initial value of the file format.
    dpi : `float` or ``None``, optional
        The initial value of the dpi. If ``None``, then dpi is set to ``0``.
    orientation : {``portrait``, ``landscape``}, optional
        The initial value of the orientation.
    papertype : `str`, optional
        The initial value of the paper type. Possible options are ::

            {``'letter'``, ``'legal'``, ``'executive'``, ``'ledger'``, ``'a0'``,
             ``'a1'``, ``'a2'``, ``'a3'``, ``'a4'``, ``'a5'``, ``'a6'``,
             ``'a7'``, ``'a8'``, ``'a9'``, ``'a10'``, ``'b0'``, ``'b1'``,
             ``'b2'``, ``'b3'``, ``'b4'``, ``'b5'``, ``'b6'``, ``'b7'``,
             ``'b8'``, ``'b9'``, ``'b10'``}
    transparent : `bool`, optional
        The initial value of the transparency flag.
    facecolour : `str` or `list` of `float`, optional
        The initial value of the face colour.
    edgecolour : `str` or `list` of `float`, optional
        The initial value of the edge colour.
    pad_inches : `float`, optional
        The initial value of the figure padding in inches.
    overwrite : `bool`, optional
        The initial value of the overwrite flag.
    style : `str` (see below)
        Sets a predefined style at the widget. Possible options are ::

            {``'minimal'``, ``'success'``, ``'info'``, ``'warning'``,
             ``'danger'``, ``''``}

    """
    def __init__(self, renderer, file_format='png', dpi=None,
                 orientation='portrait', papertype='letter', transparent=False,
                 facecolour='w', edgecolour='w', pad_inches=0.,
                 overwrite=False, style='minimal'):
        from os import getcwd
        from os.path import join, splitext

        # Create widgets
        file_format_dict = OrderedDict()
        file_format_dict['png'] = 'png'
        file_format_dict['jpg'] = 'jpg'
        file_format_dict['pdf'] = 'pdf'
        file_format_dict['eps'] = 'eps'
        file_format_dict['postscript'] = 'ps'
        file_format_dict['svg'] = 'svg'
        self.file_format_select = ipywidgets.Select(
            options=file_format_dict, value=file_format, description='Format',
            width='3cm')
        if dpi is None:
            dpi = 0
        self.dpi_text = ipywidgets.FloatText(description='DPI', value=dpi)
        orientation_dict = OrderedDict()
        orientation_dict['portrait'] = 'portrait'
        orientation_dict['landscape'] = 'landscape'
        self.orientation_dropdown = ipywidgets.Dropdown(
            options=orientation_dict, value=orientation,
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
        self.papertype_select = ipywidgets.Select(
            options=papertype_dict, value=papertype, description='Paper type',
            visible=file_format == 'ps', width='3cm')
        self.transparent_checkbox = ipywidgets.Checkbox(
            description='Transparent', value=transparent)
        self.facecolour_widget = ColourSelectionWidget(
            [facecolour], description='Face colour')
        self.edgecolour_widget = ColourSelectionWidget(
            [edgecolour], description='Edge colour')
        self.pad_inches_text = ipywidgets.FloatText(description='Pad (inch)',
                                                    value=pad_inches)
        self.filename_text = ipywidgets.Text(
            description='Path', value=join(getcwd(), 'Untitled.' + file_format),
            width='10cm')
        self.overwrite_checkbox = ipywidgets.Checkbox(
            description='Overwrite if file exists', value=overwrite)
        self.error_latex = ipywidgets.Latex(value="", font_weight='bold',
                                            font_style='italic')
        self.save_button = ipywidgets.Button(description='Save',
                                             margin='0.2cm')

        # Group widgets
        self.path_box = ipywidgets.VBox(
            children=[self.filename_text, self.file_format_select,
                      self.papertype_select, self.overwrite_checkbox],
            align='end', margin='0.2cm')
        self.page_box = ipywidgets.VBox(
            children=[self.orientation_dropdown, self.dpi_text,
                      self.pad_inches_text], margin='0.2cm')
        self.colour_box = ipywidgets.VBox(
            children=[self.facecolour_widget, self.edgecolour_widget,
                      self.transparent_checkbox], margin='0.2cm')
        self.options_tabs = ipywidgets.Tab(
            children=[self.path_box, self.page_box, self.colour_box],
            margin=0, padding='0.1cm')
        self.options_tabs_box = ipywidgets.Box(
            children=[self.options_tabs], border_width=1, border_color='black',
            margin='0.3cm', padding='0.2cm')
        tab_titles = ['Path', 'Page setup', 'Image colour']
        for (k, tl) in enumerate(tab_titles):
            self.options_tabs.set_title(k, tl)
        self.save_box = ipywidgets.HBox(
            children=[self.save_button, self.error_latex], align='center')
        self.options_box = ipywidgets.VBox(
            children=[self.options_tabs, self.save_box], align='center')
        super(SaveFigureOptionsWidget, self).__init__(
            children=[self.options_box])
        self.align = 'start'

        # Assign renderer
        self.renderer = renderer

        # Set style
        self.predefined_style(style)

        # Set functionality
        def papertype_visibility(name, value):
            self.papertype_select.visible = value == 'ps'
        self.file_format_select.on_trait_change(papertype_visibility, 'value')

        def set_extension(name, value):
            file_name, file_extension = splitext(self.filename_text.value)
            self.filename_text.value = file_name + '.' + value
        self.file_format_select.on_trait_change(set_extension, 'value')

        def save_function(name):
            # set save button state
            self.error_latex.value = ''
            self.save_button.description = 'Saving...'
            self.save_button.disabled = True

            # save figure
            selected_dpi = self.dpi_text.value
            if self.dpi_text.value == 0:
                selected_dpi = None
            try:
                self.renderer.save_figure(
                    filename=self.filename_text.value, dpi=selected_dpi,
                    face_colour=
                    self.facecolour_widget.selected_values['colour'][0],
                    edge_colour=
                    self.edgecolour_widget.selected_values['colour'][0],
                    orientation=self.orientation_dropdown.value,
                    paper_type=self.papertype_select.value,
                    format=self.file_format_select.value,
                    transparent=self.transparent_checkbox.value,
                    pad_inches=self.pad_inches_text.value,
                    overwrite=self.overwrite_checkbox.value)
                self.error_latex.value = ''
            except ValueError as e:
                if (e.message == 'File already exists. Please set the '
                                 'overwrite kwarg if you wish to overwrite '
                                 'the file.'):
                    self.error_latex.value = 'File exists! ' \
                                             'Tick overwrite to replace it.'
                else:
                    self.error_latex.value = e.message

            # set save button state
            self.save_button.description = 'Save'
            self.save_button.disabled = False
        self.save_button.on_click(save_function)

    def style(self, box_style=None, border_visible=False, border_color='black',
              border_style='solid', border_width=1, border_radius=0, padding=0,
              margin=0, font_family='', font_size=None, font_style='',
              font_weight=''):
        r"""
        Function that defines the styling of the widget.

        Parameters
        ----------
        box_style : `str` or ``None`` (see below), optional
            Style options ::

                {``'success'``, ``'info'``, ``'warning'``, ``'danger'``, ``''``}
                or
                ``None``

        border_visible : `bool`, optional
            Defines whether to draw the border line around the widget.
        border_color : `str`, optional
            The color of the border around the widget.
        border_style : `str`, optional
            The line style of the border around the widget.
        border_width : `float`, optional
            The line width of the border around the widget.
        border_radius : `float`, optional
            The radius of the corners of the box.
        padding : `float`, optional
            The padding around the widget.
        margin : `float`, optional
            The margin around the widget.
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
        _format_box(self, box_style, border_visible, border_color, border_style,
                    border_width, border_radius, padding, margin)
        _format_font(self, font_family, font_size, font_style, font_weight)
        _format_font(self.file_format_select, font_family, font_size,
                     font_style, font_weight)
        _format_font(self.dpi_text, font_family, font_size, font_style,
                     font_weight)
        _format_font(self.orientation_dropdown, font_family, font_size,
                     font_style, font_weight)
        _format_font(self.papertype_select, font_family, font_size,
                     font_style, font_weight)
        _format_font(self.transparent_checkbox, font_family, font_size,
                     font_style, font_weight)
        _format_font(self.pad_inches_text, font_family, font_size, font_style,
                     font_weight)
        _format_font(self.filename_text, font_family, font_size, font_style,
                     font_weight)
        _format_font(self.overwrite_checkbox, font_family, font_size,
                     font_style, font_weight)
        _format_font(self.save_button, font_family, font_size, font_style,
                     font_weight)
        self.facecolour_widget.style(
            box_style=None, border_visible=False, font_family=font_family,
            font_size=font_size, font_weight=font_weight, font_style=font_style)
        self.edgecolour_widget.style(
            box_style=None, border_visible=False, font_family=font_family,
            font_size=font_size, font_weight=font_weight, font_style=font_style)

    def predefined_style(self, style):
        r"""
        Function that set a predefined styling on the widget.

        Parameter
        ---------
        style : `str` (see below)
            Style options ::

                {``'minimal'``, ``'success'``, ``'info'``, ``'warning'``,
                 ``'danger'``, ``''``}

        """
        if style == 'minimal':
            self.style(box_style='', border_visible=True, border_color='black',
                       border_style='solid', border_width=1, border_radius=0,
                       padding='0.2cm', margin='0.3cm', font_family='',
                       font_size=None, font_style='', font_weight='')
            self.save_button.button_style = ''
            self.save_button.font_weight = 'normal'
        elif (style == 'info' or style == 'success' or style == 'danger' or
              style == 'warning'):
            self.style(box_style=style, border_visible=True,
                       border_color= _map_styles_to_hex_colours(style),
                       border_style='solid', border_width=1, border_radius=10,
                       padding='0.2cm', margin='0.3cm', font_family='',
                       font_size=None, font_style='', font_weight='')
            self.save_button.button_style = 'primary'
            self.save_button.font_weight = 'bold'
        else:
            raise ValueError('style must be minimal or info or success or '
                             'danger or warning')


class FeatureOptionsWidget(ipywidgets.FlexBox):
    r"""
    Creates a widget for selecting feature options. Specifically, it consists
    of:

        1) RadioButtons [`self.feature_radiobuttons`]: select feature type
        2) HOGOptionsWidget [`self.hog_options_widget`]: hog options widget
        3) IGOOptionsWidget [`self.igo_options_widget`]: igo options widget
        4) LBPOptionsWidget [`self.lbp_options_widget`]: lbp options widget
        5) DaisyOptionsWidget [`self.daisy_options_widget`]: daisy options
           widget
        6) Latex [`self.no_options_widget`]: message for no options available
        7) Box [`self.per_feature_options_box`]: box that contains (2), (3),
           (4), (5) and (6)
        8) Image [`self.preview_image`]: lenna image
        9) Latex [`self.preview_input_latex`]: the initial image information
        10) Latex [`self.preview_output_latex`]: the output image information
        11) Latex [`self.preview_time_latex`]: the timing information
        12) VBox [`self.preview_box`]: box that contains (8), (9), (10), (11)
        13) Tab [`self.options_box`]: box that contains (1), (7) and (12)

    To set the styling of this widget please refer to the `style()` method. The
    widget stores the features `function` to `self.features_function`, the
    features options `dict` in `self.features_options` and the `partial`
    function with the options as `self.function`.

    Parameter
    ---------
    style : `str` (see below)
        Sets a predefined style at the widget. Possible options are ::

            {``'minimal'``, ``'success'``, ``'info'``, ``'warning'``,
             ``'danger'``, ``''``}

    """
    def __init__(self, style='minimal'):
        # import features methods and time
        import time
        from menpo.feature.features import (hog, lbp, igo, es, daisy, gradient,
                                            no_op)
        from menpo.image import Image
        import menpo.io as mio
        from menpo.visualize.image import glyph

        # Create widgets
        tmp = OrderedDict()
        tmp['HOG'] = hog
        tmp['IGO'] = igo
        tmp['ES'] = es
        tmp['Daisy'] = daisy
        tmp['LBP'] = lbp
        tmp['Gradient'] = gradient
        tmp['None'] = no_op
        self.feature_radiobuttons = ipywidgets.RadioButtons(
            value=no_op, options=tmp, description='Feature type:')
        hog_options_dict = {'mode': 'dense', 'algorithm': 'dalaltriggs',
                            'num_bins': 9, 'cell_size': 8, 'block_size': 2,
                            'signed_gradient': True, 'l2_norm_clip': 0.2,
                            'window_height': 1, 'window_width': 1,
                            'window_unit': 'blocks', 'window_step_vertical': 1,
                            'window_step_horizontal': 1,
                            'window_step_unit': 'pixels', 'padding': True}
        self.hog_options_widget = HOGOptionsWidget(hog_options_dict)
        self.hog_options_widget.style(box_style=None, border_visible=False,
                                      margin='0.2cm')
        igo_options_dict = {'double_angles': True}
        self.igo_options_widget = IGOOptionsWidget(igo_options_dict)
        self.igo_options_widget.style(box_style=None, border_visible=False,
                                      margin='0.2cm')
        lbp_options_dict = {'radius': range(1, 5), 'samples': [8] * 4,
                            'mapping_type': 'u2', 'window_step_vertical': 1,
                            'window_step_horizontal': 1,
                            'window_step_unit': 'pixels', 'padding': True}
        self.lbp_options_widget = LBPOptionsWidget(lbp_options_dict)
        self.lbp_options_widget.style(box_style=None, border_visible=False,
                                      margin='0.2cm')
        daisy_options_dict = {'step': 1, 'radius': 15, 'rings': 2,
                              'histograms': 2, 'orientations': 8,
                              'normalization': 'l1', 'sigmas': None,
                              'ring_radii': None}
        self.daisy_options_widget = DaisyOptionsWidget(daisy_options_dict)
        self.daisy_options_widget.style(box_style=None, border_visible=False,
                                        margin='0.2cm')
        self.no_options_widget = ipywidgets.Latex(value='No options available.')

        # Load and rescale preview image (lenna)
        self.image = mio.import_builtin_asset.lenna_png()
        self.image.crop_to_landmarks_proportion_inplace(0.18)
        self.image = self.image.as_greyscale()

        # Group widgets
        self.per_feature_options_box = ipywidgets.Box(
            children=[self.hog_options_widget, self.igo_options_widget,
                      self.lbp_options_widget, self.daisy_options_widget,
                      self.no_options_widget])
        self.preview_image = ipywidgets.Image(
            value=_convert_image_to_bytes(self.image), visible=False)
        self.preview_input_latex = ipywidgets.Latex(
            value="Input: {}W x {}H x {}C".format(
                self.image.width, self.image.height, self.image.n_channels),
            visible=False)
        self.preview_output_latex = ipywidgets.Latex(value="")
        self.preview_time_latex = ipywidgets.Latex(value="")
        self.preview_box = ipywidgets.VBox(
            children=[self.preview_image, self.preview_input_latex,
                      self.preview_output_latex, self.preview_time_latex])
        self.options_box = ipywidgets.Tab(
            children=[self.feature_radiobuttons, self.per_feature_options_box,
                      self.preview_box])
        tab_titles = ['Feature', 'Options', 'Preview']
        for (k, tl) in enumerate(tab_titles):
            self.options_box.set_title(k, tl)
        super(FeatureOptionsWidget, self).__init__(children=[self.options_box])
        self.align = 'start'

        # Initialize output
        options = {}
        self.function = partial(no_op, **options)
        self.features_function = no_op
        self.features_options = options

        # Set style
        self.predefined_style(style)

        # Set functionality
        def per_feature_options_visibility(name, value):
            if value == hog:
                self.igo_options_widget.visible = False
                self.lbp_options_widget.visible = False
                self.daisy_options_widget.visible = False
                self.no_options_widget.visible = False
                self.hog_options_widget.visible = True
            elif value == igo:
                self.hog_options_widget.visible = False
                self.lbp_options_widget.visible = False
                self.daisy_options_widget.visible = False
                self.no_options_widget.visible = False
                self.igo_options_widget.visible = True
            elif value == lbp:
                self.hog_options_widget.visible = False
                self.igo_options_widget.visible = False
                self.daisy_options_widget.visible = False
                self.no_options_widget.visible = False
                self.lbp_options_widget.visible = True
            elif value == daisy:
                self.hog_options_widget.visible = False
                self.igo_options_widget.visible = False
                self.lbp_options_widget.visible = False
                self.no_options_widget.visible = False
                self.daisy_options_widget.visible = True
            else:
                self.hog_options_widget.visible = False
                self.igo_options_widget.visible = False
                self.lbp_options_widget.visible = False
                self.daisy_options_widget.visible = False
                self.no_options_widget.visible = True
                for name, f in tmp.items():
                    if f == value:
                        self.no_options_widget.value = \
                            "{}: No available options.".format(name)
        self.feature_radiobuttons.on_trait_change(
            per_feature_options_visibility, 'value')
        per_feature_options_visibility('', no_op)

        def get_function(name, value):
            # get options
            if self.feature_radiobuttons.value == hog:
                opts = self.hog_options_widget.selected_values
            elif self.feature_radiobuttons.value == igo:
                opts = self.igo_options_widget.selected_values
            elif self.feature_radiobuttons.value == lbp:
                opts = self.lbp_options_widget.selected_values
            elif self.feature_radiobuttons.value == daisy:
                opts = self.daisy_options_widget.selected_values
            else:
                opts = {}
            # get features function closure
            func = partial(self.feature_radiobuttons.value, **opts)
            # store function
            self.function = func
            self.features_function = value
            self.features_options = opts
        self.feature_radiobuttons.on_trait_change(get_function, 'value')
        self.options_box.on_trait_change(get_function, 'selected_index')

        def preview_function(name, old_value, value):
            if value == 2:
                # extracting features message
                for name, f in tmp.items():
                    if f == self.function.func:
                        val1 = name
                self.preview_output_latex.value = \
                    "Previewing {} features...".format(val1)
                self.preview_time_latex.value = ""
                # extract feature and time it
                t = time.time()
                feat_image = self.function(self.image)
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
                self.preview_image.value = _convert_image_to_bytes(feat_image)
                self.preview_input_latex.visible = True
                self.preview_image.visible = True
                # set info
                self.preview_output_latex.value = \
                    "{}: {}W x {}H x {}C".format(val1, val2, val3, val4)
                self.preview_time_latex.value = "{0:.2f} secs elapsed".format(t)
            if old_value == 2:
                self.preview_input_latex.visible = False
                self.preview_image.visible = False
        self.options_box.on_trait_change(preview_function, 'selected_index')

    def style(self, box_style=None, border_visible=False, border_color='black',
              border_style='solid', border_width=1, border_radius=0, padding=0,
              margin=0, font_family='', font_size=None, font_style='',
              font_weight=''):
        r"""
        Function that defines the styling of the widget.

        Parameters
        ----------
        box_style : `str` or ``None`` (see below), optional
            Style options ::

                {``'success'``, ``'info'``, ``'warning'``, ``'danger'``, ``''``}
                or
                ``None``

        border_visible : `bool`, optional
            Defines whether to draw the border line around the widget.
        border_color : `str`, optional
            The color of the border around the widget.
        border_style : `str`, optional
            The line style of the border around the widget.
        border_width : `float`, optional
            The line width of the border around the widget.
        border_radius : `float`, optional
            The radius of the corners of the box.
        padding : `float`, optional
            The padding around the widget.
        margin : `float`, optional
            The margin around the widget.
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
        _format_box(self, box_style, border_visible, border_color, border_style,
                    border_width, border_radius, padding, margin)
        _format_font(self, font_family, font_size, font_style, font_weight)
        _format_font(self.feature_radiobuttons, font_family, font_size,
                     font_style, font_weight)
        _format_font(self.no_options_widget, font_family, font_size, font_style,
                     font_weight)
        _format_font(self.preview_input_latex, font_family, font_size,
                     font_style, font_weight)
        _format_font(self.preview_output_latex, font_family, font_size,
                     font_style, font_weight)
        _format_font(self.preview_time_latex, font_family, font_size,
                     font_style, font_weight)
        self.hog_options_widget.style(
            box_style=None, border_visible=False, margin='0.2cm',
            font_family=font_family, font_size=font_size, font_style=font_style,
            font_weight=font_weight)
        self.igo_options_widget.style(
            box_style=None, border_visible=False, margin='0.2cm',
            font_family=font_family, font_size=font_size, font_style=font_style,
            font_weight=font_weight)
        self.lbp_options_widget.style(
            box_style=None, border_visible=False, margin='0.2cm',
            font_family=font_family, font_size=font_size, font_style=font_style,
            font_weight=font_weight)
        self.daisy_options_widget.style(
            box_style=None, border_visible=False, margin='0.2cm',
            font_family=font_family, font_size=font_size, font_style=font_style,
            font_weight=font_weight)
        self.no_options_widget.margin = '0.2cm'

    def predefined_style(self, style):
        r"""
        Function that set a predefined styling on the widget.

        Parameter
        ---------
        style : `str` (see below)
            Style options ::

                {``'minimal'``, ``'success'``, ``'info'``, ``'warning'``,
                 ``'danger'``, ``''``}

        """
        if style == 'minimal':
            self.style(box_style='', border_visible=True, border_color='black',
                       border_style='solid', border_width=1, border_radius=0,
                       padding='0.2cm', margin='0.3cm', font_family='',
                       font_size=None, font_style='', font_weight='')
        elif (style == 'info' or style == 'success' or style == 'danger' or
              style == 'warning'):
            self.style(box_style=style, border_visible=True,
                       border_color= _map_styles_to_hex_colours(style),
                       border_style='solid', border_width=1, border_radius=10,
                       padding='0.2cm', margin='0.3cm', font_family='',
                       font_size=None, font_style='', font_weight='')
        else:
            raise ValueError('style must be minimal or info or success or '
                             'danger or warning')
