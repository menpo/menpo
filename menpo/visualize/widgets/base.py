import numpy as np
from collections import Sized
from matplotlib.pyplot import show as pltshow

import IPython.html.widgets as ipywidgets
import IPython.display as ipydisplay

from menpo.visualize.viewmatplotlib import (MatplotlibImageViewer2d,
                                            sample_colours_from_colourmap)

from .options import (RendererOptionsWidget, TextPrintWidget,
                      SaveFigureOptionsWidget, AnimationOptionsWidget,
                      LandmarkOptionsWidget, ChannelOptionsWidget,
                      FeatureOptionsWidget, GraphOptionsWidget)
from .tools import _format_box, LogoWidget, _map_styles_to_hex_colours


# This glyph import is called frequently during visualisation, so we ensure
# that we only import it once. The same for the sum_channels method.
glyph = None
sum_channels = None


def visualize_pointclouds(pointclouds, figure_size=(10, 8), style='coloured',
                          browser_style='buttons'):
    r"""
    Widget that allows browsing through a `list` of :map:`PointCloud`,
    :map:`PointUndirectedGraph`, :map:`PointDirectedGraph`, :map:`PointTree`,
    :map:`TriMesh` or subclasses. All the above can be combined in the `list`.

    The widget has options tabs regarding the renderer (lines, markers, figure,
    axes) and saving the figure to file.

    Parameters
    ----------
    pointclouds : `list`
        The `list` of objects to be visualized. It can contain a combination of
        :map:`PointCloud`, :map:`PointUndirectedGraph`,
        :map:`PointDirectedGraph`, :map:`PointTree`, :map:`TriMesh` or
        subclasses of those.
    figure_size : (`int`, `int`), optional
        The initial size of the rendered figure.
    style : {``'coloured'``, ``'minimal'``}, optional
        If ``'coloured'``, then the style of the widget will be coloured. If
        ``minimal``, then the style is simple using black and white colours.
    browser_style : {``'buttons'``, ``'slider'``}, optional
        It defines whether the selector of the objects will have the form of
        plus/minus buttons or a slider.
    """
    print('Initializing...')

    # Make sure that pointclouds is a list even with one pointcloud member
    if not isinstance(pointclouds, Sized):
        pointclouds = [pointclouds]

    # Get the number of pointclouds
    n_pointclouds = len(pointclouds)

    # Define the styling options
    if style == 'coloured':
        logo_style = 'warning'
        widget_box_style = 'warning'
        widget_border_radius = 10
        widget_border_width = 1
        animation_style = 'warning'
        info_style = 'info'
        renderer_box_style = 'info'
        renderer_box_border_colour = _map_styles_to_hex_colours('info')
        renderer_box_border_radius = 10
        renderer_style = 'danger'
        renderer_tabs_style = 'minimal'
        save_figure_style = 'danger'
    else:
        logo_style = 'minimal'
        widget_box_style = ''
        widget_border_radius = 0
        widget_border_width = 0
        animation_style = 'minimal'
        info_style = 'minimal'
        renderer_box_style = ''
        renderer_box_border_colour = 'black'
        renderer_box_border_radius = 0
        renderer_style = 'minimal'
        renderer_tabs_style = 'minimal'
        save_figure_style = 'minimal'

    # Initial options dictionaries
    line_options = {'render_lines': True, 'line_width': 1, 'line_colour': ['r'],
                    'line_style': '-'}
    marker_options = {'render_markers': True, 'marker_size': 20,
                      'marker_face_colour': ['r'], 'marker_edge_colour': ['k'],
                      'marker_style': 'o', 'marker_edge_width': 1}
    figure_options = {'x_scale': 1., 'y_scale': 1., 'render_axes': False,
                      'axes_font_name': 'sans-serif', 'axes_font_size': 10,
                      'axes_font_style': 'normal', 'axes_font_weight': 'normal',
                      'axes_x_limits': None, 'axes_y_limits': None}
    renderer_options = {'lines': line_options, 'markers': marker_options,
                        'figure': figure_options}
    index = {'min': 0, 'max': n_pointclouds-1, 'step': 1, 'index': 0}

    # Define render function
    def render_function(name, value):
        # Clear current figure, but wait until the generation of the new data
        # that will be rendered
        ipydisplay.clear_output(wait=True)

        # Get selected pointcloud index
        im = 0
        if n_pointclouds > 1:
            im = pointcloud_number_wid.selected_values['index']

        # Update info text widget
        update_info(pointclouds[im])

        # Render pointcloud with selected options
        tmp1 = renderer_options_wid.selected_values[0]['lines']
        tmp2 = renderer_options_wid.selected_values[0]['markers']
        tmp3 = renderer_options_wid.selected_values[0]['figure']
        new_figure_size = (tmp3['x_scale'] * figure_size[0],
                           tmp3['y_scale'] * figure_size[1])
        renderer = pointclouds[im].view(
            figure_id=save_figure_wid.renderer.figure_id,
            new_figure=False, image_view=axes_mode_wid.value == 1,
            render_lines=tmp1['render_lines'],
            line_colour=tmp1['line_colour'][0],
            line_style=tmp1['line_style'], line_width=tmp1['line_width'],
            render_markers=tmp2['render_markers'],
            marker_style=tmp2['marker_style'], marker_size=tmp2['marker_size'],
            marker_face_colour=tmp2['marker_face_colour'][0],
            marker_edge_colour=tmp2['marker_edge_colour'][0],
            marker_edge_width=tmp2['marker_edge_width'],
            render_axes=tmp3['render_axes'],
            axes_font_name=tmp3['axes_font_name'],
            axes_font_size=tmp3['axes_font_size'],
            axes_font_style=tmp3['axes_font_style'],
            axes_font_weight=tmp3['axes_font_weight'],
            axes_x_limits=tmp3['axes_x_limits'],
            axes_y_limits=tmp3['axes_y_limits'], figure_size=new_figure_size,
            label=None)
        pltshow()

        # Save the current figure id
        save_figure_wid.renderer = renderer

    # Define function that updates the info text
    def update_info(pointcloud):
        min_b, max_b = pointcloud.bounds()
        rang = pointcloud.range()
        cm = pointcloud.centre()
        text_per_line = [
            "> {} points".format(pointcloud.n_points),
            "> Bounds: [{0:.1f}-{1:.1f}]W, [{2:.1f}-{3:.1f}]H".format(
                min_b[0], max_b[0], min_b[1], max_b[1]),
            "> Range: {0:.1f}W, {1:.1f}H".format(rang[0], rang[1]),
            "> Centre of mass: ({0:.1f}, {1:.1f})".format(cm[0], cm[1]),
            "> Norm: {0:.2f}".format(pointcloud.norm())]
        info_wid.set_widget_state(n_lines=5, text_per_line=text_per_line)

    # Create widgets
    axes_mode_wid = ipywidgets.RadioButtons(
        options={'Image': 1, 'Point cloud': 2}, description='Axes mode:',
        value=2)
    axes_mode_wid.on_trait_change(render_function, 'value')
    renderer_options_wid = RendererOptionsWidget(
        renderer_options, ['lines', 'markers', 'figure_one'],
        object_selection_dropdown_visible=False,
        render_function=render_function, style=renderer_style,
        tabs_style=renderer_tabs_style)
    renderer_options_box = ipywidgets.VBox(
        children=[axes_mode_wid, renderer_options_wid], align='center',
        margin='0.1cm')
    info_wid = TextPrintWidget(n_lines=5, text_per_line=[''] * 5,
                               style=info_style)
    initial_renderer = MatplotlibImageViewer2d(figure_id=None, new_figure=True,
                                               image=np.zeros((10, 10)))
    save_figure_wid = SaveFigureOptionsWidget(initial_renderer,
                                              style=save_figure_style)

    # Group widgets
    if n_pointclouds > 1:
        # Pointcloud selection slider
        pointcloud_number_wid = AnimationOptionsWidget(
            index, render_function=render_function, index_style=browser_style,
            interval=0.3, description='Pointcloud ', minus_description='<',
            plus_description='>', loop_enabled=True, text_editable=True,
            style=animation_style)

        # Header widget
        header_wid = ipywidgets.HBox(
            children=[LogoWidget(style=logo_style), pointcloud_number_wid],
            align='start')
    else:
        # Header widget
        header_wid = LogoWidget(style=logo_style)
    header_wid.margin = '0.1cm'
    options_box = ipywidgets.Tab(children=[info_wid, renderer_options_box,
                                           save_figure_wid], margin='0.1cm')
    tab_titles = ['Info', 'Renderer', 'Export']
    for (k, tl) in enumerate(tab_titles):
        options_box.set_title(k, tl)
    if n_pointclouds > 1:
        wid = ipywidgets.VBox(children=[header_wid, options_box], align='start')
    else:
        wid = ipywidgets.HBox(children=[header_wid, options_box], align='start')

    # Set widget's style
    wid.box_style = widget_box_style
    wid.border_radius = widget_border_radius
    wid.border_width = widget_border_width
    wid.border_color = _map_styles_to_hex_colours(widget_box_style)
    _format_box(renderer_options_box, renderer_box_style, True,
                renderer_box_border_colour, 'solid', 1,
                renderer_box_border_radius, '0.1cm', '0.2cm')

    # Display final widget
    ipydisplay.display(wid)

    # Reset value to trigger initial visualization
    axes_mode_wid.value = 1


def visualize_landmarkgroups(landmarkgroups, figure_size=(10, 8),
                             style='coloured', browser_style='buttons'):
    r"""
    Widget that allows browsing through a `list` of :map:`LandmarkGroup`
    (or subclass) objects.

    The landmark groups can have a combination of different attributes, e.g.
    different labels, number of points etc. The widget has options tabs
    regarding the landmarks, the renderer (lines, markers, numbering, legend,
    figure, axes) and saving the figure to file.

    Parameters
    ----------
    landmarkgroups : `list` of :map:`LandmarkGroup` or subclass
        The `list` of landmark groups to be visualized.
    figure_size : (`int`, `int`), optional
        The initial size of the rendered figure.
    style : {``'coloured'``, ``'minimal'``}, optional
        If ``'coloured'``, then the style of the widget will be coloured. If
        ``minimal``, then the style is simple using black and white colours.
    browser_style : {``'buttons'``, ``'slider'``}, optional
        It defines whether the selector of the objects will have the form of
        plus/minus buttons or a slider.
    """
    print('Initializing...')

    # Make sure that landmarkgroups is a list even with one landmark group
    # member
    if not isinstance(landmarkgroups, list):
        landmarkgroups = [landmarkgroups]

    # Get the number of landmarkgroups
    n_landmarkgroups = len(landmarkgroups)

    # Define the styling options
    if style == 'coloured':
        logo_style = 'success'
        widget_box_style = 'success'
        widget_border_radius = 10
        widget_border_width = 1
        animation_style = 'success'
        landmarks_style = 'info'
        info_style = 'info'
        renderer_box_style = 'info'
        renderer_box_border_colour = _map_styles_to_hex_colours('info')
        renderer_box_border_radius = 10
        renderer_style = 'danger'
        renderer_tabs_style = 'minimal'
        save_figure_style = 'danger'
    else:
        logo_style = 'minimal'
        widget_box_style = ''
        widget_border_radius = 0
        widget_border_width = 0
        landmarks_style = 'minimal'
        animation_style = 'minimal'
        info_style = 'minimal'
        renderer_box_style = ''
        renderer_box_border_colour = 'black'
        renderer_box_border_radius = 0
        renderer_style = 'minimal'
        renderer_tabs_style = 'minimal'
        save_figure_style = 'minimal'

    # Find all available groups and the respective labels from all the landmarks
    # that are passed in
    all_groups = []
    all_labels = []
    groups_l = 0
    for i, l in enumerate(landmarkgroups):
        labels_l = l.labels
        if labels_l not in all_labels:
            groups_l += 1
            all_groups.append(str(groups_l))
            all_labels.append(labels_l)

    # Get initial line and marker colours for each available group
    colours = []
    for l in all_labels:
        if len(l) == 1:
            colours.append(['r'])
        else:
            colours.append(sample_colours_from_colourmap(len(l), 'jet'))

    # initial options dictionaries
    landmark_options = {'has_landmarks': True, 'render_landmarks': True,
                        'group_keys': ['0'],
                        'labels_keys': [landmarkgroups[0].labels],
                        'group': '0', 'with_labels': landmarkgroups[0].labels}
    index = {'min': 0, 'max': n_landmarkgroups-1, 'step': 1, 'index': 0}
    numbering_options = {'render_numbering': False,
                         'numbers_font_name': 'sans-serif',
                         'numbers_font_size': 10,
                         'numbers_font_style': 'normal',
                         'numbers_font_weight': 'normal',
                         'numbers_font_colour': ['k'],
                         'numbers_horizontal_align': 'center',
                         'numbers_vertical_align': 'bottom'}
    legend_options = {'render_legend': True, 'legend_title': '',
                      'legend_font_name': 'sans-serif',
                      'legend_font_style': 'normal', 'legend_font_size': 10,
                      'legend_font_weight': 'normal', 'legend_marker_scale': 1.,
                      'legend_location': 2, 'legend_bbox_to_anchor': (1.05, 1.),
                      'legend_border_axes_pad': 1., 'legend_n_columns': 1,
                      'legend_horizontal_spacing': 1.,
                      'legend_vertical_spacing': 1., 'legend_border': True,
                      'legend_border_padding': 0.5, 'legend_shadow': False,
                      'legend_rounded_corners': False}
    figure_options = {'x_scale': 1., 'y_scale': 1., 'render_axes': False,
                      'axes_font_name': 'sans-serif', 'axes_font_size': 10,
                      'axes_font_style': 'normal', 'axes_font_weight': 'normal',
                      'axes_x_limits': None, 'axes_y_limits': None}
    renderer_options = []
    for i in range(len(all_labels)):
        lines_options = {'render_lines': True, 'line_width': 1,
                         'line_colour': colours[i], 'line_style': '-'}
        marker_options = {'render_markers': True, 'marker_size': 20,
                          'marker_face_colour': list(colours[i]),
                          'marker_edge_colour': list(colours[i]),
                          'marker_style': 'o', 'marker_edge_width': 1}
        tmp = {'lines': lines_options, 'markers': marker_options,
               'numbering': numbering_options, 'legend': legend_options,
               'figure': figure_options}
        renderer_options.append(tmp)

    # Define render function
    def render_function(name, value):
        # Clear current figure, but wait until the generation of the new data
        # that will be rendered
        ipydisplay.clear_output(wait=True)

        # get selected index
        im = 0
        if n_landmarkgroups > 1:
            im = landmark_number_wid.selected_values['index']

        # update info text widget
        update_info(landmarkgroups[im])

        # show landmarks with selected options
        group_idx = all_labels.index(landmarkgroups[im].labels)
        tmp1 = renderer_options_wid.selected_values[group_idx]['lines']
        tmp2 = renderer_options_wid.selected_values[group_idx]['markers']
        tmp3 = renderer_options_wid.selected_values[group_idx]['numbering']
        tmp4 = renderer_options_wid.selected_values[group_idx]['legend']
        tmp5 = renderer_options_wid.selected_values[group_idx]['figure']
        new_figure_size = (tmp5['x_scale'] * figure_size[0],
                           tmp5['y_scale'] * figure_size[1])

        # find the with_labels' indices
        with_labels_idx = [
            landmark_options_wid.selected_values['labels_keys'][0].index(lbl)
            for lbl in landmark_options_wid.selected_values['with_labels']]

        # get line and marker colours
        line_colour = [tmp1['line_colour'][lbl_idx]
                       for lbl_idx in with_labels_idx]
        marker_face_colour = [tmp2['marker_face_colour'][lbl_idx]
                              for lbl_idx in with_labels_idx]
        marker_edge_colour = [tmp2['marker_edge_colour'][lbl_idx]
                              for lbl_idx in with_labels_idx]

        if landmark_options_wid.selected_values['render_landmarks']:
            renderer = landmarkgroups[im].view(
                with_labels=landmark_options_wid.selected_values['with_labels'],
                figure_id=save_figure_wid.renderer.figure_id, new_figure=False,
                image_view=axes_mode_wid.value == 1,
                render_lines=tmp1['render_lines'], line_colour=line_colour,
                line_style=tmp1['line_style'], line_width=tmp1['line_width'],
                render_markers=tmp2['render_markers'],
                marker_style=tmp2['marker_style'],
                marker_size=tmp2['marker_size'],
                marker_face_colour=marker_face_colour,
                marker_edge_colour=marker_edge_colour,
                marker_edge_width=tmp2['marker_edge_width'],
                render_numbering=tmp3['render_numbering'],
                numbers_font_name=tmp3['numbers_font_name'],
                numbers_font_size=tmp3['numbers_font_size'],
                numbers_font_style=tmp3['numbers_font_style'],
                numbers_font_weight=tmp3['numbers_font_weight'],
                numbers_font_colour=tmp3['numbers_font_colour'][0],
                numbers_horizontal_align=tmp3['numbers_horizontal_align'],
                numbers_vertical_align=tmp3['numbers_vertical_align'],
                legend_n_columns=tmp4['legend_n_columns'],
                legend_border_axes_pad=tmp4['legend_border_axes_pad'],
                legend_rounded_corners=tmp4['legend_rounded_corners'],
                legend_title=tmp4['legend_title'],
                legend_horizontal_spacing=tmp4['legend_horizontal_spacing'],
                legend_shadow=tmp4['legend_shadow'],
                legend_location=tmp4['legend_location'],
                legend_font_name=tmp4['legend_font_name'],
                legend_bbox_to_anchor=tmp4['legend_bbox_to_anchor'],
                legend_border=tmp4['legend_border'],
                legend_marker_scale=tmp4['legend_marker_scale'],
                legend_vertical_spacing=tmp4['legend_vertical_spacing'],
                legend_font_weight=tmp4['legend_font_weight'],
                legend_font_size=tmp4['legend_font_size'],
                render_legend=tmp4['render_legend'],
                legend_font_style=tmp4['legend_font_style'],
                legend_border_padding=tmp4['legend_border_padding'],
                render_axes=tmp5['render_axes'],
                axes_font_name=tmp5['axes_font_name'],
                axes_font_size=tmp5['axes_font_size'],
                axes_font_style=tmp5['axes_font_style'],
                axes_font_weight=tmp5['axes_font_weight'],
                axes_x_limits=tmp5['axes_x_limits'],
                axes_y_limits=tmp5['axes_y_limits'],
                figure_size=new_figure_size)
            pltshow()

            # Save the current figure id
            save_figure_wid.renderer = renderer
        else:
            ipydisplay.clear_output()

    # Define function that updates the info text
    def update_info(landmarkgroup):
        min_b, max_b = landmarkgroup.lms.bounds()
        rang = landmarkgroup.lms.range()
        cm = landmarkgroup.lms.centre()
        text_per_line = [
            "> {} landmark points".format(landmarkgroup.n_landmarks),
            "> Bounds: [{0:.1f}-{1:.1f}]W, [{2:.1f}-{3:.1f}]H".format(
                min_b[0], max_b[0], min_b[1], max_b[1]),
            "> Range: {0:.1f}W, {1:.1f}H".format(rang[0], rang[1]),
            "> Centre of mass: ({0:.1f}, {1:.1f})".format(cm[0], cm[1]),
            "> Norm: {0:.2f}".format(landmarkgroup.lms.norm())]
        info_wid.set_widget_state(n_lines=5, text_per_line=text_per_line)

    # Create widgets
    landmark_options_wid = LandmarkOptionsWidget(
        landmark_options, render_function=render_function,
        style=landmarks_style)
    axes_mode_wid = ipywidgets.RadioButtons(
        options={'Image': 1, 'Point cloud': 2}, description='Axes mode:',
        value=2)
    axes_mode_wid.on_trait_change(render_function, 'value')
    renderer_options_wid = RendererOptionsWidget(
        renderer_options,
        ['lines', 'markers', 'numbering', 'legend', 'figure_one'],
        object_selection_dropdown_visible=False, labels_per_object=all_labels,
        render_function=render_function, selected_object=0,
        style=renderer_style, tabs_style=renderer_tabs_style)
    renderer_options_box = ipywidgets.VBox(
        children=[axes_mode_wid, renderer_options_wid], align='center',
        margin='0.1cm')
    info_wid = TextPrintWidget(n_lines=5, text_per_line=[''] * 5,
                               style=info_style)
    initial_renderer = MatplotlibImageViewer2d(figure_id=None, new_figure=True,
                                               image=np.zeros((10, 10)))
    save_figure_wid = SaveFigureOptionsWidget(initial_renderer,
                                              style=save_figure_style)

    # Define function that updates options' widgets state
    def update_widgets(name, value):
        # Get new groups and labels, then update landmark options
        im = 0
        if n_landmarkgroups > 1:
            im = landmark_number_wid.selected_values['index']
        landmark_options = {
            'has_landmarks': True,
            'render_landmarks':
                landmark_options_wid.selected_values['render_landmarks'],
            'group_keys': ['0'], 'labels_keys': [landmarkgroups[im].labels],
            'group': '0', 'with_labels': None}
        landmark_options_wid.set_widget_state(landmark_options, False)
        landmark_options_wid.predefined_style(landmarks_style)

        # Set correct group to renderer options' selection
        renderer_options_wid.object_selection_dropdown.value = \
            all_labels.index(landmarkgroups[im].labels)

    # Group widgets
    if n_landmarkgroups > 1:
        # Landmark selection slider
        landmark_number_wid = AnimationOptionsWidget(
            index, render_function=render_function,
            update_function=update_widgets, index_style=browser_style,
            interval=0.3, description='Shape', minus_description='<',
            plus_description='>', loop_enabled=True, text_editable=True,
            style=animation_style)

        # Header widget
        header_wid = ipywidgets.HBox(
            children=[LogoWidget(style=logo_style), landmark_number_wid],
            align='start')
    else:
        # Header widget
        header_wid = LogoWidget(style=logo_style)
    header_wid.margin = '0.2cm'
    options_box = ipywidgets.Tab(
        children=[info_wid, landmark_options_wid, renderer_options_box,
                  save_figure_wid], margin='0.2cm')
    tab_titles = ['Info', 'Landmarks', 'Renderer', 'Export']
    for (k, tl) in enumerate(tab_titles):
        options_box.set_title(k, tl)
    if n_landmarkgroups > 1:
        wid = ipywidgets.VBox(children=[header_wid, options_box], align='start')
    else:
        wid = ipywidgets.HBox(children=[header_wid, options_box], align='start')

    # Set widget's style
    wid.box_style = widget_box_style
    wid.border_radius = widget_border_radius
    wid.border_width = widget_border_width
    wid.border_color = _map_styles_to_hex_colours(widget_box_style)
    _format_box(renderer_options_box, renderer_box_style, True,
                renderer_box_border_colour, 'solid', 1,
                renderer_box_border_radius, '0.1cm', '0.2cm')

    # Display final widget
    ipydisplay.display(wid)

    # Reset value to trigger initial visualization
    axes_mode_wid.value = 1


def visualize_landmarks(landmarks, figure_size=(10, 8), style='coloured',
                        browser_style='buttons'):
    r"""
    Widget that allows browsing through a `list` of :map:`LandmarkManager`
    (or subclass) objects.

    The landmark managers can have a combination of different attributes, e.g.
    landmark groups and labels etc. The widget has options tabs regarding the
    landmarks, the renderer (lines, markers, numbering, legend, figure, axes)
    and saving the figure to file.

    Parameters
    ----------
    landmarks : `list` of :map:`LandmarkManager` or subclass
        The `list` of landmark managers to be visualized.
    figure_size : (`int`, `int`), optional
        The initial size of the rendered figure.
    style : {``'coloured'``, ``'minimal'``}, optional
        If ``'coloured'``, then the style of the widget will be coloured. If
        ``minimal``, then the style is simple using black and white colours.
    browser_style : {``'buttons'``, ``'slider'``}, optional
        It defines whether the selector of the objects will have the form of
        plus/minus buttons or a slider.
    """
    print('Initializing...')

    # Make sure that landmarks is a list even with one landmark manager member
    if not isinstance(landmarks, list):
        landmarks = [landmarks]

    # Get the number of landmark managers
    n_landmarks = len(landmarks)

    # Define the styling options
    if style == 'coloured':
        logo_style = 'info'
        widget_box_style = 'info'
        widget_border_radius = 10
        widget_border_width = 1
        animation_style = 'info'
        landmarks_style = 'danger'
        info_style = 'danger'
        renderer_box_style = 'danger'
        renderer_box_border_colour = _map_styles_to_hex_colours('danger')
        renderer_box_border_radius = 10
        renderer_style = 'warning'
        renderer_tabs_style = 'minimal'
        save_figure_style = 'danger'
    else:
        logo_style = 'minimal'
        widget_box_style = ''
        widget_border_radius = 0
        widget_border_width = 0
        landmarks_style = 'minimal'
        animation_style = 'minimal'
        info_style = 'minimal'
        renderer_box_style = ''
        renderer_box_border_colour = 'black'
        renderer_box_border_radius = 0
        renderer_style = 'minimal'
        renderer_tabs_style = 'minimal'
        save_figure_style = 'minimal'

    # Find all available groups and the respective labels from all the landmarks
    # that are passed in
    all_groups = []
    all_labels = []
    for l in landmarks:
        groups_l, labels_l = _extract_group_labels_landmarks(l)
        for i, g in enumerate(groups_l):
            if g not in all_groups:
                all_groups.append(g)
                all_labels.append(labels_l[i])

    # Get initial line and marker colours for each available group
    colours = []
    for l in all_labels:
        if len(l) == 1:
            colours.append(['r'])
        else:
            colours.append(sample_colours_from_colourmap(len(l), 'jet'))

    # Initial options dictionaries
    initial_groups_keys, initial_labels_keys = \
        _extract_group_labels_landmarks(landmarks[0])
    landmark_options = {'has_landmarks': landmarks[0].has_landmarks,
                        'render_landmarks': True,
                        'group_keys': initial_groups_keys,
                        'labels_keys': initial_labels_keys,
                        'group': initial_groups_keys[0],
                        'with_labels': initial_labels_keys[0]}
    index = {'min': 0, 'max': n_landmarks-1, 'step': 1, 'index': 0}
    numbering_options = {'render_numbering': False,
                         'numbers_font_name': 'sans-serif',
                         'numbers_font_size': 10,
                         'numbers_font_style': 'normal',
                         'numbers_font_weight': 'normal',
                         'numbers_font_colour': ['k'],
                         'numbers_horizontal_align': 'center',
                         'numbers_vertical_align': 'bottom'}
    legend_options = {'render_legend': True, 'legend_title': '',
                      'legend_font_name': 'sans-serif',
                      'legend_font_style': 'normal', 'legend_font_size': 10,
                      'legend_font_weight': 'normal', 'legend_marker_scale': 1.,
                      'legend_location': 2, 'legend_bbox_to_anchor': (1.05, 1.),
                      'legend_border_axes_pad': 1., 'legend_n_columns': 1,
                      'legend_horizontal_spacing': 1.,
                      'legend_vertical_spacing': 1., 'legend_border': True,
                      'legend_border_padding': 0.5, 'legend_shadow': False,
                      'legend_rounded_corners': False}
    figure_options = {'x_scale': 1., 'y_scale': 1., 'render_axes': False,
                      'axes_font_name': 'sans-serif', 'axes_font_size': 10,
                      'axes_font_style': 'normal', 'axes_font_weight': 'normal',
                      'axes_x_limits': None, 'axes_y_limits': None}
    renderer_options = []
    for i in range(len(all_labels)):
        lines_options = {'render_lines': True, 'line_width': 1,
                         'line_colour': colours[i], 'line_style': '-'}
        marker_options = {'render_markers': True, 'marker_size': 20,
                          'marker_face_colour': list(colours[i]),
                          'marker_edge_colour': list(colours[i]),
                          'marker_style': 'o', 'marker_edge_width': 1}
        tmp = {'lines': lines_options, 'markers': marker_options,
               'numbering': numbering_options, 'legend': legend_options,
               'figure': figure_options}
        renderer_options.append(tmp)

    # Define render function
    def render_function(name, value):
        # Clear current figure, but wait until the generation of the new data
        # that will be rendered
        ipydisplay.clear_output(wait=True)

        # get selected index
        im = 0
        if n_landmarks > 1:
            im = landmark_number_wid.selected_values['index']

        # update info text widget
        update_info(landmarks[im],
                    landmark_options_wid.selected_values['group'])

        # show landmarks with selected options
        group_idx = all_groups.index(
            landmark_options_wid.selected_values['group'])
        tmp1 = renderer_options_wid.selected_values[group_idx]['lines']
        tmp2 = renderer_options_wid.selected_values[group_idx]['markers']
        tmp3 = renderer_options_wid.selected_values[group_idx]['numbering']
        tmp4 = renderer_options_wid.selected_values[group_idx]['legend']
        tmp5 = renderer_options_wid.selected_values[group_idx]['figure']
        new_figure_size = (tmp5['x_scale'] * figure_size[0],
                           tmp5['y_scale'] * figure_size[1])

        # get selected group
        sel_group = landmark_options_wid.selected_values['group']
        sel_group_idx = landmark_options_wid.selected_values[
            'group_keys'].index(sel_group)

        # find the with_labels' indices
        with_labels_idx = [
            landmark_options_wid.selected_values['labels_keys'][
                sel_group_idx].index(lbl)
            for lbl in landmark_options_wid.selected_values['with_labels']]

        # get line and marker colours
        line_colour = [tmp1['line_colour'][lbl_idx]
                       for lbl_idx in with_labels_idx]
        marker_face_colour = [tmp2['marker_face_colour'][lbl_idx]
                              for lbl_idx in with_labels_idx]
        marker_edge_colour = [tmp2['marker_edge_colour'][lbl_idx]
                              for lbl_idx in with_labels_idx]

        if landmark_options_wid.selected_values['render_landmarks']:
            renderer = landmarks[im][sel_group].view(
                with_labels=landmark_options_wid.selected_values['with_labels'],
                figure_id=save_figure_wid.renderer.figure_id, new_figure=False,
                image_view=axes_mode_wid.value == 1,
                render_lines=tmp1['render_lines'], line_colour=line_colour,
                line_style=tmp1['line_style'], line_width=tmp1['line_width'],
                render_markers=tmp2['render_markers'],
                marker_style=tmp2['marker_style'],
                marker_size=tmp2['marker_size'],
                marker_face_colour=marker_face_colour,
                marker_edge_colour=marker_edge_colour,
                marker_edge_width=tmp2['marker_edge_width'],
                render_numbering=tmp3['render_numbering'],
                numbers_font_name=tmp3['numbers_font_name'],
                numbers_font_size=tmp3['numbers_font_size'],
                numbers_font_style=tmp3['numbers_font_style'],
                numbers_font_weight=tmp3['numbers_font_weight'],
                numbers_font_colour=tmp3['numbers_font_colour'][0],
                numbers_horizontal_align=tmp3['numbers_horizontal_align'],
                numbers_vertical_align=tmp3['numbers_vertical_align'],
                legend_n_columns=tmp4['legend_n_columns'],
                legend_border_axes_pad=tmp4['legend_border_axes_pad'],
                legend_rounded_corners=tmp4['legend_rounded_corners'],
                legend_title=tmp4['legend_title'],
                legend_horizontal_spacing=tmp4['legend_horizontal_spacing'],
                legend_shadow=tmp4['legend_shadow'],
                legend_location=tmp4['legend_location'],
                legend_font_name=tmp4['legend_font_name'],
                legend_bbox_to_anchor=tmp4['legend_bbox_to_anchor'],
                legend_border=tmp4['legend_border'],
                legend_marker_scale=tmp4['legend_marker_scale'],
                legend_vertical_spacing=tmp4['legend_vertical_spacing'],
                legend_font_weight=tmp4['legend_font_weight'],
                legend_font_size=tmp4['legend_font_size'],
                render_legend=tmp4['render_legend'],
                legend_font_style=tmp4['legend_font_style'],
                legend_border_padding=tmp4['legend_border_padding'],
                render_axes=tmp5['render_axes'],
                axes_font_name=tmp5['axes_font_name'],
                axes_font_size=tmp5['axes_font_size'],
                axes_font_style=tmp5['axes_font_style'],
                axes_font_weight=tmp5['axes_font_weight'],
                axes_x_limits=tmp5['axes_x_limits'],
                axes_y_limits=tmp5['axes_y_limits'],
                figure_size=new_figure_size)
            pltshow()

            # Save the current figure id
            save_figure_wid.renderer = renderer
        else:
            ipydisplay.clear_output()

    # Define function that updates the info text
    def update_info(landmarks, group):
        if group != ' ':
            min_b, max_b = landmarks[group][None].bounds()
            rang = landmarks[group][None].range()
            cm = landmarks[group][None].centre()
            text_per_line = [
                "> {} landmark points".format(landmarks[group][None].n_points),
                "> Bounds: [{0:.1f}-{1:.1f}]W, [{2:.1f}-{3:.1f}]H".
                    format(min_b[0], max_b[0], min_b[1], max_b[1]),
                "> Range: {0:.1f}W, {1:.1f}H".format(rang[0], rang[1]),
                "> Centre of mass: ({0:.1f}, {1:.1f})".format(cm[0], cm[1]),
                "> Norm: {0:.2f}".format(landmarks[group][None].norm())]
            n_lines = 5
        else:
            text_per_line = ["No landmarks available."]
            n_lines = 1
        info_wid.set_widget_state(n_lines=n_lines, text_per_line=text_per_line)

    # Define update function of landmark widget
    def update_renderer_widget(name, value):
        renderer_options_wid.object_selection_dropdown.value = \
            all_groups.index(landmark_options_wid.selected_values['group'])

    # Create widgets
    landmark_options_wid = LandmarkOptionsWidget(
        landmark_options, render_function=render_function,
        update_function=update_renderer_widget, style=landmarks_style)
    axes_mode_wid = ipywidgets.RadioButtons(
        options={'Image': 1, 'Point cloud': 2}, description='Axes mode:',
        value=2)
    axes_mode_wid.on_trait_change(render_function, 'value')
    renderer_options_wid = RendererOptionsWidget(
        renderer_options,
        ['lines', 'markers', 'numbering', 'legend', 'figure_one'],
        objects_names=all_groups,
        object_selection_dropdown_visible=False, labels_per_object=all_labels,
        render_function=render_function, selected_object=0,
        style=renderer_style, tabs_style=renderer_tabs_style)
    renderer_options_box = ipywidgets.VBox(
        children=[axes_mode_wid, renderer_options_wid], align='center',
        margin='0.1cm')
    info_wid = TextPrintWidget(n_lines=5, text_per_line=[''] * 5,
                               style=info_style)
    initial_renderer = MatplotlibImageViewer2d(figure_id=None, new_figure=True,
                                               image=np.zeros((10, 10)))
    save_figure_wid = SaveFigureOptionsWidget(initial_renderer,
                                              style=save_figure_style)

    # Define function that updates options' widgets state
    def update_widgets(name, value):
        # Get new groups and labels, then update landmark options
        im = 0
        if n_landmarks > 1:
            im = landmark_number_wid.selected_values['index']
        group_keys, labels_keys = _extract_group_labels_landmarks(landmarks[im])
        landmark_options = {
            'has_landmarks': landmarks[im].has_landmarks,
            'render_landmarks':
                landmark_options_wid.selected_values['render_landmarks'],
            'group_keys': group_keys, 'labels_keys': labels_keys,
            'group': None, 'with_labels': None}
        landmark_options_wid.set_widget_state(landmark_options, False)
        landmark_options_wid.predefined_style(landmarks_style)

        # Set correct group to renderer options' selection
        renderer_options_wid.object_selection_dropdown.value = \
            all_groups.index(landmark_options_wid.selected_values['group'])

    # Group widgets
    if n_landmarks > 1:
        # Landmark selection slider
        landmark_number_wid = AnimationOptionsWidget(
            index, render_function=render_function,
            update_function=update_widgets, index_style=browser_style,
            interval=0.3, description='Shape', minus_description='<',
            plus_description='>', loop_enabled=True, text_editable=True,
            style=animation_style)

        # Header widget
        header_wid = ipywidgets.HBox(
            children=[LogoWidget(style=logo_style), landmark_number_wid],
            align='start')
    else:
        # Header widget
        header_wid = LogoWidget(style=logo_style)
    header_wid.margin = '0.2cm'
    options_box = ipywidgets.Tab(
        children=[info_wid, landmark_options_wid, renderer_options_box,
                  save_figure_wid], margin='0.2cm')
    tab_titles = ['Info', 'Landmarks', 'Renderer', 'Export']
    for (k, tl) in enumerate(tab_titles):
        options_box.set_title(k, tl)
    if n_landmarks > 1:
        wid = ipywidgets.VBox(children=[header_wid, options_box], align='start')
    else:
        wid = ipywidgets.HBox(children=[header_wid, options_box], align='start')

    # Set widget's style
    wid.box_style = widget_box_style
    wid.border_radius = widget_border_radius
    wid.border_width = widget_border_width
    wid.border_color = _map_styles_to_hex_colours(widget_box_style)
    _format_box(renderer_options_box, renderer_box_style, True,
                renderer_box_border_colour, 'solid', 1,
                renderer_box_border_radius, '0.1cm', '0.2cm')

    # Display final widget
    ipydisplay.display(wid)

    # Reset value to trigger initial visualization
    axes_mode_wid.value = 1


def visualize_images(images, figure_size=(10, 8), style='coloured',
                     browser_style='buttons'):
    r"""
    Widget that allows browsing through a `list` of :map:`Image` (or subclass)
    objects.

    The images can have a combination of different attributes, e.g. masked or
    not, landmarked or not, without multiple landmark groups and labels etc.
    The widget has options tabs regarding the visualized channels, the
    landmarks, the renderer (lines, markers, numbering, legend, figure, axes)
    and saving the figure to file.

    Parameters
    ----------
    images : `list` of :map:`Image` or subclass
        The `list` of images to be visualized.
    figure_size : (`int`, `int`), optional
        The initial size of the rendered figure.
    style : {``'coloured'``, ``'minimal'``}, optional
        If ``'coloured'``, then the style of the widget will be coloured. If
        ``minimal``, then the style is simple using black and white colours.
    browser_style : {``'buttons'``, ``'slider'``}, optional
        It defines whether the selector of the objects will have the form of
        plus/minus buttons or a slider.
    """
    from menpo.image import MaskedImage
    print('Initializing...')

    # Make sure that images is a list even with one image member
    if not isinstance(images, Sized):
        images = [images]

    # Get the number of images
    n_images = len(images)

    # Define the styling options
    if style == 'coloured':
        logo_style = 'info'
        widget_box_style = 'info'
        widget_border_radius = 10
        widget_border_width = 1
        animation_style = 'info'
        channels_style = 'danger'
        landmarks_style = 'danger'
        info_style = 'danger'
        renderer_style = 'danger'
        renderer_tabs_style = 'minimal'
        save_figure_style = 'danger'
    else:
        logo_style = 'minimal'
        widget_box_style = ''
        widget_border_radius = 0
        widget_border_width = 0
        channels_style = 'minimal'
        landmarks_style = 'minimal'
        animation_style = 'minimal'
        info_style = 'minimal'
        renderer_style = 'minimal'
        renderer_tabs_style = 'minimal'
        save_figure_style = 'minimal'

    # Find all available groups and the respective labels from all the landmarks
    # that are passed in
    all_groups = []
    all_labels = []
    for l in images:
        groups_l, labels_l = _extract_groups_labels(l)
        for i, g in enumerate(groups_l):
            if g not in all_groups:
                all_groups.append(g)
                all_labels.append(labels_l[i])

    # Get initial line and marker colours for each available group
    colours = []
    for l in all_labels:
        if len(l) == 1:
            colours.append(['r'])
        else:
            colours.append(sample_colours_from_colourmap(len(l), 'jet'))

    # Initial options dictionaries
    channels_default = 0
    if images[0].n_channels == 3:
        channels_default = None
    channel_options = {'n_channels': images[0].n_channels,
                       'image_is_masked': isinstance(images[0], MaskedImage),
                       'channels': channels_default, 'glyph_enabled': False,
                       'glyph_block_size': 3, 'glyph_use_negative': False,
                       'sum_enabled': False,
                       'masked_enabled': isinstance(images[0], MaskedImage)}
    initial_groups_keys, initial_labels_keys = _extract_groups_labels(images[0])
    landmark_options = {'has_landmarks': images[0].has_landmarks,
                        'render_landmarks': True,
                        'group_keys': initial_groups_keys,
                        'labels_keys': initial_labels_keys,
                        'group': initial_groups_keys[0],
                        'with_labels': initial_labels_keys[0]}
    index = {'min': 0, 'max': n_images-1, 'step': 1, 'index': 0}
    image_options = {'alpha': 1.0, 'interpolation': 'bilinear',
                     'cmap_name': None}
    numbering_options = {'render_numbering': False,
                         'numbers_font_name': 'sans-serif',
                         'numbers_font_size': 10,
                         'numbers_font_style': 'normal',
                         'numbers_font_weight': 'normal',
                         'numbers_font_colour': ['k'],
                         'numbers_horizontal_align': 'center',
                         'numbers_vertical_align': 'bottom'}
    legend_options = {'render_legend': True, 'legend_title': '',
                      'legend_font_name': 'sans-serif',
                      'legend_font_style': 'normal', 'legend_font_size': 10,
                      'legend_font_weight': 'normal', 'legend_marker_scale': 1.,
                      'legend_location': 2, 'legend_bbox_to_anchor': (1.05, 1.),
                      'legend_border_axes_pad': 1., 'legend_n_columns': 1,
                      'legend_horizontal_spacing': 1.,
                      'legend_vertical_spacing': 1., 'legend_border': True,
                      'legend_border_padding': 0.5, 'legend_shadow': False,
                      'legend_rounded_corners': False}
    figure_options = {'x_scale': 1., 'y_scale': 1., 'render_axes': False,
                      'axes_font_name': 'sans-serif', 'axes_font_size': 10,
                      'axes_font_style': 'normal', 'axes_font_weight': 'normal',
                      'axes_x_limits': None, 'axes_y_limits': None}
    renderer_options = []
    for i in range(len(all_labels)):
        lines_options = {'render_lines': True, 'line_width': 1,
                         'line_colour': colours[i], 'line_style': '-'}
        marker_options = {'render_markers': True, 'marker_size': 20,
                          'marker_face_colour': list(colours[i]),
                          'marker_edge_colour': list(colours[i]),
                          'marker_style': 'o', 'marker_edge_width': 1}
        tmp = {'lines': lines_options, 'markers': marker_options,
               'numbering': numbering_options, 'legend': legend_options,
               'figure': figure_options, 'image': image_options}
        renderer_options.append(tmp)

    # Define render function
    def render_function(name, value):
        # Clear current figure, but wait until the generation of the new data
        # that will be rendered
        ipydisplay.clear_output(wait=True)

        # get selected index
        im = 0
        if n_images > 1:
            im = image_number_wid.selected_values['index']

        # update info text widget
        image_is_masked = isinstance(images[im], MaskedImage)
        update_info(images[im], image_is_masked,
                    landmark_options_wid.selected_values['group'])

        # show image with selected options
        group_idx = all_groups.index(
            landmark_options_wid.selected_values['group'])
        tmp1 = renderer_options_wid.selected_values[group_idx]['lines']
        tmp2 = renderer_options_wid.selected_values[group_idx]['markers']
        tmp3 = renderer_options_wid.selected_values[group_idx]['numbering']
        tmp4 = renderer_options_wid.selected_values[group_idx]['legend']
        tmp5 = renderer_options_wid.selected_values[group_idx]['figure']
        tmp6 = renderer_options_wid.selected_values[group_idx]['image']
        new_figure_size = (tmp5['x_scale'] * figure_size[0],
                           tmp5['y_scale'] * figure_size[1])

        # get selected group index
        sel_group_idx = landmark_options_wid.selected_values[
            'group_keys'].index(landmark_options_wid.selected_values['group'])

        # find the with_labels' indices
        with_labels_idx = [
            landmark_options_wid.selected_values['labels_keys'][
                sel_group_idx].index(lbl)
            for lbl in landmark_options_wid.selected_values['with_labels']]

        # get line and marker colours
        line_colour = [tmp1['line_colour'][lbl_idx]
                       for lbl_idx in with_labels_idx]
        marker_face_colour = [tmp2['marker_face_colour'][lbl_idx]
                              for lbl_idx in with_labels_idx]
        marker_edge_colour = [tmp2['marker_edge_colour'][lbl_idx]
                              for lbl_idx in with_labels_idx]

        renderer = _visualize(
            images[im], save_figure_wid.renderer,
            landmark_options_wid.selected_values['render_landmarks'],
            image_is_masked,
            channel_options_wid.selected_values['masked_enabled'],
            channel_options_wid.selected_values['channels'],
            channel_options_wid.selected_values['glyph_enabled'],
            channel_options_wid.selected_values['glyph_block_size'],
            channel_options_wid.selected_values['glyph_use_negative'],
            channel_options_wid.selected_values['sum_enabled'],
            landmark_options_wid.selected_values['group'],
            landmark_options_wid.selected_values['with_labels'],
            tmp1['render_lines'], tmp1['line_style'], tmp1['line_width'],
            line_colour, tmp2['render_markers'], tmp2['marker_style'],
            tmp2['marker_size'], tmp2['marker_edge_width'], marker_edge_colour,
            marker_face_colour, tmp3['render_numbering'],
            tmp3['numbers_font_name'], tmp3['numbers_font_size'],
            tmp3['numbers_font_style'], tmp3['numbers_font_weight'],
            tmp3['numbers_font_colour'][0], tmp3['numbers_horizontal_align'],
            tmp3['numbers_vertical_align'], tmp4['legend_n_columns'],
            tmp4['legend_border_axes_pad'], tmp4['legend_rounded_corners'],
            tmp4['legend_title'], tmp4['legend_horizontal_spacing'],
            tmp4['legend_shadow'], tmp4['legend_location'],
            tmp4['legend_font_name'], tmp4['legend_bbox_to_anchor'],
            tmp4['legend_border'], tmp4['legend_marker_scale'],
            tmp4['legend_vertical_spacing'], tmp4['legend_font_weight'],
            tmp4['legend_font_size'], tmp4['render_legend'],
            tmp4['legend_font_style'], tmp4['legend_border_padding'],
            new_figure_size, tmp5['render_axes'], tmp5['axes_font_name'],
            tmp5['axes_font_size'], tmp5['axes_font_style'],
            tmp5['axes_x_limits'], tmp5['axes_y_limits'],
            tmp5['axes_font_weight'], tmp6['interpolation'], tmp6['alpha'],
            tmp6['cmap_name'])

        # Save the current figure id
        save_figure_wid.renderer = renderer

    # Define function that updates the info text
    def update_info(img, image_is_masked, group):
        # Prepare masked (or non-masked) string
        masked_str = 'Masked Image' if image_is_masked else 'Image'
        # Get image path, if available
        path_str = img.path if hasattr(img, 'path') else 'No path available'
        # Create text lines
        text_per_line = [
            "> {} of size {} with {} channel{}".format(
                masked_str, img._str_shape, img.n_channels,
                's' * (img.n_channels > 1)),
            "> Path: '{}'".format(path_str)]
        n_lines = 2
        if image_is_masked:
            text_per_line.append(
                "> {} masked pixels (attached mask {:.1%} true)".format(
                    img.n_true_pixels(), img.mask.proportion_true()))
            n_lines += 1
        text_per_line.append("> min={:.3f}, max={:.3f}".format(
            img.pixels.min(), img.pixels.max()))
        n_lines += 1
        if img.has_landmarks:
            text_per_line.append("> {} landmark points".format(
                img.landmarks[group].lms.n_points))
            n_lines += 1
        info_wid.set_widget_state(n_lines=n_lines, text_per_line=text_per_line)

    # Define update function of renderer widget
    def update_renderer_widget(name, value):
        renderer_options_wid.object_selection_dropdown.value = \
            all_groups.index(landmark_options_wid.selected_values['group'])

    # Create widgets
    channel_options_wid = ChannelOptionsWidget(
        channel_options, render_function=render_function, style=channels_style)
    landmark_options_wid = LandmarkOptionsWidget(
        landmark_options, render_function=render_function,
        update_function=update_renderer_widget, style=landmarks_style)
    renderer_options_wid = RendererOptionsWidget(
        renderer_options,
        ['lines', 'markers', 'numbering', 'legend', 'figure_one', 'image'],
        objects_names=all_groups,
        object_selection_dropdown_visible=False, labels_per_object=all_labels,
        render_function=render_function, selected_object=0,
        style=renderer_style, tabs_style=renderer_tabs_style)
    info_wid = TextPrintWidget(n_lines=1, text_per_line=[''],
                               style=info_style)
    initial_renderer = MatplotlibImageViewer2d(figure_id=None, new_figure=True,
                                               image=np.zeros((10, 10)))
    save_figure_wid = SaveFigureOptionsWidget(initial_renderer,
                                              style=save_figure_style)

    # Define function that updates options' widgets state
    def update_widgets(name, value):
        # Get new groups and labels, then update landmark options
        im = 0
        if n_images > 1:
            im = image_number_wid.selected_values['index']
        group_keys, labels_keys = _extract_groups_labels(images[im])
        landmark_options = {
            'has_landmarks': images[im].has_landmarks,
            'render_landmarks':
                landmark_options_wid.selected_values['render_landmarks'],
            'group_keys': group_keys, 'labels_keys': labels_keys,
            'group': None, 'with_labels': None}
        landmark_options_wid.set_widget_state(landmark_options, False)
        landmark_options_wid.predefined_style(landmarks_style)

        # Update channel options
        tmp_channels = channel_options_wid.selected_values['channels']
        tmp_glyph_enabled = channel_options_wid.selected_values['glyph_enabled']
        tmp_sum_enabled = channel_options_wid.selected_values['sum_enabled']
        if np.max(tmp_channels) > images[im].n_channels - 1:
            tmp_channels = 0
            tmp_glyph_enabled = False
            tmp_sum_enabled = False
        tmp_glyph_block_size = \
            channel_options_wid.selected_values['glyph_block_size']
        tmp_glyph_use_negative = \
            channel_options_wid.selected_values['glyph_use_negative']
        if not(images[im].n_channels == 3) and tmp_channels is None:
            tmp_channels = 0
        channel_options = {
            'n_channels': images[im].n_channels,
            'image_is_masked': isinstance(images[im], MaskedImage),
            'channels': tmp_channels, 'glyph_enabled': tmp_glyph_enabled,
            'glyph_block_size': tmp_glyph_block_size,
            'glyph_use_negative': tmp_glyph_use_negative,
            'sum_enabled': tmp_sum_enabled,
            'masked_enabled': isinstance(images[im], MaskedImage)}
        channel_options_wid.set_widget_state(channel_options, False)

        # Set correct group to renderer options' selection
        renderer_options_wid.object_selection_dropdown.value = \
            all_groups.index(landmark_options_wid.selected_values['group'])

    # Group widgets
    if n_images > 1:
        # Image selection slider
        image_number_wid = AnimationOptionsWidget(
            index, render_function=render_function,
            update_function=update_widgets, index_style=browser_style,
            interval=0.3, description='Image', minus_description='<',
            plus_description='>', loop_enabled=True, text_editable=True,
            style=animation_style)

        # Header widget
        header_wid = ipywidgets.HBox(
            children=[LogoWidget(style=logo_style), image_number_wid],
            align='start')
    else:
        # Header widget
        header_wid = LogoWidget(style=logo_style)
    header_wid.margin = '0.2cm'
    options_box = ipywidgets.Tab(
        children=[info_wid, channel_options_wid, landmark_options_wid,
                  renderer_options_wid, save_figure_wid], margin='0.2cm')
    tab_titles = ['Info', 'Channels', 'Landmarks', 'Renderer', 'Export']
    for (k, tl) in enumerate(tab_titles):
        options_box.set_title(k, tl)
    if n_images > 1:
        wid = ipywidgets.VBox(children=[header_wid, options_box], align='start')
    else:
        wid = ipywidgets.HBox(children=[header_wid, options_box], align='start')

    # Set widget's style
    wid.box_style = widget_box_style
    wid.border_radius = widget_border_radius
    wid.border_width = widget_border_width
    wid.border_color = _map_styles_to_hex_colours(widget_box_style)

    # Display final widget
    ipydisplay.display(wid)

    # Reset value to trigger initial visualization
    renderer_options_wid.options_widgets[3].render_legend_checkbox.value = False


def plot_graph(x_axis, y_axis, legend_entries=None, title=None, x_label=None,
               y_label=None, x_axis_limits=None, y_axis_limits=None,
               figure_size=(10, 6), style='coloured'):
    r"""
    Widget that allows plotting various curves in a graph using
    :map:`GraphPlotter`.

    The widget has options tabs regarding the graph and the renderer (lines,
    markers, legend, figure, axes, grid) and saving the figure to file.

    Parameters
    ----------
    x_axis : `list` of `float`
        The values of the horizontal axis. Note that these values are common for
        all the curves.
    y_axis : `list` of `lists` of `float`
        A `list` that stores a `list` of values to be plotted for each curve.
    legend_entries : `list` or `str` or ``None``, optional
        The `list` of names that will appear on the legend for each curve. If
        ``None``, then the names format is ``curve {}.format(i)``.
    title : `str` or ``None``, optional
        The title of the graph.
    x_label : `str` or ``None``, optional
        The label on the horizontal axis of the graph.
    y_label : `str` or ``None``, optional
        The label on the vertical axis of the graph.
    x_axis_limits : (`float`, `float`) or ``None``, optional
        The limits of the horizontal axis. If ``None``, the limits are set
        based on the min and max values of `x_axis`.
    y_axis_limits : (`float`, `float`), optional
        The limits of the vertical axis. If ``None``, the limits are set based
        on the min and max values of `y_axis`.
    figure_size : (`int`, `int`), optional
        The initial size of the rendered figure.
    style : {``'coloured'``, ``'minimal'``}, optional
        If ``'coloured'``, then the style of the widget will be coloured. If
        ``minimal``, then the style is simple using black and white colours.
    """
    from menpo.visualize import GraphPlotter
    print('Initializing...')

    # Get number of curves to be plotted
    n_curves = len(y_axis)

    # Define the styling options
    if style == 'coloured':
        logo_style = 'danger'
        widget_box_style = 'danger'
        tabs_style = 'warning'
        renderer_tabs_style = 'info'
        save_figure_style = 'warning'
    else:
        logo_style = 'minimal'
        widget_box_style = 'minimal'
        tabs_style = 'minimal'
        renderer_tabs_style = 'minimal'
        save_figure_style = 'minimal'

    # Parse options
    if legend_entries is None:
        legend_entries = ["curve {}".format(i) for i in range(n_curves)]
    if title is None:
        title = ''
    if x_label is None:
        x_label = ''
    if y_label is None:
        y_label = ''
    x_min = np.floor(np.min(x_axis))
    x_max = np.ceil(np.max(x_axis))
    x_step = 0.05 * (x_max - x_min + 1)
    x_slider_options = (x_min - 2 * x_step, x_max + 2 * x_step, x_step)
    y_min = np.floor(np.min([np.min(i) for i in y_axis]))
    y_max = np.ceil(np.max([np.max(i) for i in y_axis]))
    y_step = 0.05 * (y_max - y_min + 1)
    y_slider_options = (y_min - 2 * y_step, y_max + 2 * y_step, y_step)
    if x_axis_limits is None:
        x_axis_limits = (x_min, x_max)
    if y_axis_limits is None:
        y_axis_limits = (y_min, y_max)

    # Get initial line and marker colours for each curve
    if n_curves == 1:
        line_colours = ['b']
        marker_edge_colours = ['b']
    else:
        colours_tmp = sample_colours_from_colourmap(n_curves, 'jet')
        line_colours = [list(i) for i in colours_tmp]
        marker_edge_colours = [list(i) for i in colours_tmp]

    # Initial options dictionaries
    graph_options = {'legend_entries': legend_entries, 'x_label': x_label,
                     'y_label': y_label, 'title': title,
                     'x_axis_limits': x_axis_limits,
                     'y_axis_limits': y_axis_limits,
                     'render_lines': [True] * n_curves,
                     'line_colour': line_colours,
                     'line_style': ['-'] * n_curves,
                     'line_width': [2] * n_curves,
                     'render_markers': [True] * n_curves,
                     'marker_style': ['s'] * n_curves,
                     'marker_size': [8] * n_curves,
                     'marker_face_colour': ['w'] * n_curves,
                     'marker_edge_colour': marker_edge_colours,
                     'marker_edge_width': [2] * n_curves,
                     'render_legend': n_curves > 1, 'legend_title': '',
                     'legend_font_name': 'sans-serif',
                     'legend_font_style': 'normal', 'legend_font_size': 10,
                     'legend_font_weight': 'normal', 'legend_marker_scale': 1.,
                     'legend_location': 2, 'legend_bbox_to_anchor': (1.05, 1.),
                     'legend_border_axes_pad': 1., 'legend_n_columns': 1,
                     'legend_horizontal_spacing': 1.,
                     'legend_vertical_spacing': 1., 'legend_border': True,
                     'legend_border_padding': 0.5, 'legend_shadow': False,
                     'legend_rounded_corners': False, 'render_axes': False,
                     'axes_font_name': 'sans-serif', 'axes_font_size': 10,
                     'axes_font_style': 'normal', 'axes_font_weight': 'normal',
                     'figure_size': figure_size, 'render_grid': True,
                     'grid_line_style': '--', 'grid_line_width': 1}

    # Define render function
    def render_function(name, value):
        # Clear current figure, but wait until the generation of the new data
        # that will be rendered
        ipydisplay.clear_output(wait=True)

        # plot with selected options
        opts = wid.selected_values
        plotter = GraphPlotter(
            figure_id=save_figure_wid.renderer.figure_id, new_figure=False,
            x_axis=x_axis, y_axis=y_axis, title=opts['title'],
            legend_entries=opts['legend_entries'], x_label=opts['x_label'],
            y_label=opts['y_label'], x_axis_limits=opts['x_axis_limits'],
            y_axis_limits=opts['y_axis_limits'])
        renderer = plotter.render(
            render_lines=opts['render_lines'], line_colour=opts['line_colour'],
            line_style=opts['line_style'],  line_width=opts['line_width'],
            render_markers=opts['render_markers'],
            marker_style=opts['marker_style'], marker_size=opts['marker_size'],
            marker_face_colour=opts['marker_face_colour'],
            marker_edge_colour=opts['marker_edge_colour'],
            marker_edge_width=opts['marker_edge_width'],
            render_legend=opts['render_legend'],
            legend_title=opts['legend_title'],
            legend_font_name=opts['legend_font_name'],
            legend_font_style=opts['legend_font_style'],
            legend_font_size=opts['legend_font_size'],
            legend_font_weight=opts['legend_font_weight'],
            legend_marker_scale=opts['legend_marker_scale'],
            legend_location=opts['legend_location'],
            legend_bbox_to_anchor=opts['legend_bbox_to_anchor'],
            legend_border_axes_pad=opts['legend_border_axes_pad'],
            legend_n_columns=opts['legend_n_columns'],
            legend_horizontal_spacing=opts['legend_horizontal_spacing'],
            legend_vertical_spacing=opts['legend_vertical_spacing'],
            legend_border=opts['legend_border'],
            legend_border_padding=opts['legend_border_padding'],
            legend_shadow=opts['legend_shadow'],
            legend_rounded_corners=opts['legend_rounded_corners'],
            render_axes=opts['render_axes'],
            axes_font_name=opts['axes_font_name'],
            axes_font_size=opts['axes_font_size'],
            axes_font_style=opts['axes_font_style'],
            axes_font_weight=opts['axes_font_weight'],
            figure_size=opts['figure_size'], render_grid=opts['render_grid'],
            grid_line_style=opts['grid_line_style'],
            grid_line_width=opts['grid_line_width'])

        # show plot
        pltshow()

        # Save the current figure id
        save_figure_wid.renderer = renderer

    # Create widgets
    wid = GraphOptionsWidget(graph_options, x_slider_options, y_slider_options,
                             render_function=render_function,
                             style=widget_box_style, tabs_style=tabs_style,
                             renderer_tabs_style=renderer_tabs_style)
    initial_renderer = MatplotlibImageViewer2d(figure_id=None, new_figure=True,
                                               image=np.zeros((10, 10)))
    save_figure_wid = SaveFigureOptionsWidget(initial_renderer,
                                              style=save_figure_style)

    # Group widgets
    logo = LogoWidget(style=logo_style)
    logo.margin = '0.1cm'
    wid.options_tab.children = [wid.graph_related_options, wid.renderer_widget,
                                save_figure_wid]
    wid.options_tab.set_title(0, 'Graph')
    wid.options_tab.set_title(1, 'Renderer')
    wid.options_tab.set_title(2, 'Export')
    wid.children = [logo, wid.options_tab]
    wid.align = 'start'

    # Display final widget
    ipydisplay.display(wid)

    # Reset value to trigger initial visualization
    wid.renderer_widget.options_widgets[3].render_axes_checkbox.value = True


def save_matplotlib_figure(renderer, style='coloured'):
    r"""
    Widget that allows to save a figure, which was generated with Matplotlib,
    to file.

    Parameters
    ----------
    renderer : :map:`MatplotlibRenderer`
        The Matplotlib renderer object.
    style : {``'coloured'``, ``'minimal'``}, optional
        If ``'coloured'``, then the style of the widget will be coloured. If
        ``minimal``, then the style is simple using black and white colours.
    """
    # Create sub-widgets
    if style == 'coloured':
        style = 'warning'
    logo_wid = LogoWidget(style='minimal')
    save_figure_wid = SaveFigureOptionsWidget(renderer, style=style)
    save_figure_wid.margin = '0.1cm'
    logo_wid.margin = '0.1cm'
    wid = ipywidgets.HBox(children=[logo_wid, save_figure_wid])

    # Display widget
    ipydisplay.display(wid)


def features_selection(style='coloured'):
    r"""
    Widget that allows selecting a features function and its options. The
    widget supports all features from :ref:`api-feature-index` and has a
    preview tab. It returns a `list` of length 1 with the selected features
    function closure.

    Parameters
    ----------
    style : {``'coloured'``, ``'minimal'``}, optional
        If ``'coloured'``, then the style of the widget will be coloured. If
        ``minimal``, then the style is simple using black and white colours.

    Returns
    -------
    features_function : `list` of length ``1``
        The function closure of the features function using `functools.partial`.
        So the function can be called as: ::

            features_image = features_function[0](image)

    """
    # Styling options
    if style == 'coloured':
        logo_style = 'info'
        outer_style = 'info'
        inner_style = 'warning'
        but_style = 'primary'
        rad = 10
    elif style == 'minimal':
        logo_style = 'minimal'
        outer_style = ''
        inner_style = 'minimal'
        but_style = ''
        rad = 0
    else:
        raise ValueError('style must be either coloured or minimal')

    # Create sub-widgets
    logo_wid = LogoWidget(style=logo_style)
    features_options_wid = FeatureOptionsWidget(style=inner_style)
    features_wid = ipywidgets.HBox(children=[logo_wid, features_options_wid])
    select_but = ipywidgets.Button(description='Select')

    # Create final widget
    wid = ipywidgets.VBox(children=[features_wid, select_but])
    _format_box(wid, outer_style, True,
                _map_styles_to_hex_colours(outer_style), 'solid', 1, rad, 0, 0)
    logo_wid.margin = '0.3cm'
    features_options_wid.margin = '0.3cm'
    select_but.margin = '0.2cm'
    select_but.button_style = but_style
    wid.align = 'center'

    # function for select button
    def select_function(name):
        wid.close()
        output.pop(0)
        output.append(features_options_wid.function)
    select_but.on_click(select_function)

    # Display widget
    ipydisplay.display(wid)

    # Initialize output with empty list. It needs to be a list so that
    # it's mutable and synchronizes with frontend.
    output = [features_options_wid.function]

    return output


def _visualize(image, renderer, render_landmarks, image_is_masked,
               masked_enabled, channels, glyph_enabled, glyph_block_size,
               glyph_use_negative, sum_enabled, group, with_labels,
               render_lines, line_style, line_width, line_colour,
               render_markers, marker_style, marker_size,
               marker_edge_width, marker_edge_colour, marker_face_colour,
               render_numbering, numbers_font_name, numbers_font_size,
               numbers_font_style, numbers_font_weight, numbers_font_colour,
               numbers_horizontal_align, numbers_vertical_align,
               legend_n_columns, legend_border_axes_pad, legend_rounded_corners,
               legend_title, legend_horizontal_spacing, legend_shadow,
               legend_location, legend_font_name, legend_bbox_to_anchor,
               legend_border, legend_marker_scale, legend_vertical_spacing,
               legend_font_weight, legend_font_size, render_legend,
               legend_font_style, legend_border_padding, figure_size,
               render_axes, axes_font_name, axes_font_size, axes_font_style,
               axes_x_limits, axes_y_limits, axes_font_weight, interpolation,
               alpha, cmap_name):
    global glyph
    if glyph is None:
        from menpo.feature.visualize import glyph
    global sum_channels
    if sum_channels is None:
        from menpo.feature.visualize import sum_channels

    # This makes the code shorter for dealing with masked images vs non-masked
    # images
    mask_arguments = ({'masked': masked_enabled}
                      if image_is_masked else {})

    # plot
    if render_landmarks and not group == ' ':
        # show image with landmarks
        if glyph_enabled:
            # image, landmarks, masked, glyph
            renderer = glyph(image, vectors_block_size=glyph_block_size,
                             use_negative=glyph_use_negative,
                             channels=channels).view_landmarks(
                group=group, with_labels=with_labels, without_labels=None,
                figure_id=renderer.figure_id, new_figure=False,
                render_lines=render_lines, line_colour=line_colour,
                line_style=line_style, line_width=line_width,
                render_markers=render_markers, marker_style=marker_style,
                marker_size=marker_size, marker_face_colour=marker_face_colour,
                marker_edge_colour=marker_edge_colour,
                marker_edge_width=marker_edge_width,
                render_numbering=render_numbering,
                numbers_horizontal_align=numbers_horizontal_align,
                numbers_vertical_align=numbers_vertical_align,
                numbers_font_name=numbers_font_name,
                numbers_font_size=numbers_font_size,
                numbers_font_style=numbers_font_style,
                numbers_font_weight=numbers_font_weight,
                numbers_font_colour=numbers_font_colour,
                render_legend=render_legend, legend_title=legend_title,
                legend_font_name=legend_font_name,
                legend_font_style=legend_font_style,
                legend_font_size=legend_font_size,
                legend_font_weight=legend_font_weight,
                legend_marker_scale=legend_marker_scale,
                legend_location=legend_location,
                legend_bbox_to_anchor=legend_bbox_to_anchor,
                legend_border_axes_pad=legend_border_axes_pad,
                legend_n_columns=legend_n_columns,
                legend_horizontal_spacing=legend_horizontal_spacing,
                legend_vertical_spacing=legend_vertical_spacing,
                legend_border=legend_border,
                legend_border_padding=legend_border_padding,
                legend_shadow=legend_shadow,
                legend_rounded_corners=legend_rounded_corners,
                render_axes=render_axes, axes_font_name=axes_font_name,
                axes_font_size=axes_font_size, axes_font_style=axes_font_style,
                axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
                axes_y_limits=axes_y_limits, figure_size=figure_size,
                interpolation=interpolation, alpha=alpha, cmap_name=cmap_name,
                **mask_arguments)
        elif sum_enabled:
            # image, landmarks, masked, sum
            renderer = sum_channels(image, channels=channels).view_landmarks(
                group=group, with_labels=with_labels, without_labels=None,
                figure_id=renderer.figure_id, new_figure=False,
                render_lines=render_lines, line_colour=line_colour,
                line_style=line_style, line_width=line_width,
                render_markers=render_markers, marker_style=marker_style,
                marker_size=marker_size, marker_face_colour=marker_face_colour,
                marker_edge_colour=marker_edge_colour,
                marker_edge_width=marker_edge_width,
                render_numbering=render_numbering,
                numbers_horizontal_align=numbers_horizontal_align,
                numbers_vertical_align=numbers_vertical_align,
                numbers_font_name=numbers_font_name,
                numbers_font_size=numbers_font_size,
                numbers_font_style=numbers_font_style,
                numbers_font_weight=numbers_font_weight,
                numbers_font_colour=numbers_font_colour,
                render_legend=render_legend, legend_title=legend_title,
                legend_font_name=legend_font_name,
                legend_font_style=legend_font_style,
                legend_font_size=legend_font_size,
                legend_font_weight=legend_font_weight,
                legend_marker_scale=legend_marker_scale,
                legend_location=legend_location,
                legend_bbox_to_anchor=legend_bbox_to_anchor,
                legend_border_axes_pad=legend_border_axes_pad,
                legend_n_columns=legend_n_columns,
                legend_horizontal_spacing=legend_horizontal_spacing,
                legend_vertical_spacing=legend_vertical_spacing,
                legend_border=legend_border,
                legend_border_padding=legend_border_padding,
                legend_shadow=legend_shadow,
                legend_rounded_corners=legend_rounded_corners,
                render_axes=render_axes, axes_font_name=axes_font_name,
                axes_font_size=axes_font_size, axes_font_style=axes_font_style,
                axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
                axes_y_limits=axes_y_limits, figure_size=figure_size,
                interpolation=interpolation, alpha=alpha, cmap_name=cmap_name,
                **mask_arguments)
        else:
            renderer = image.view_landmarks(
                channels=channels, group=group, with_labels=with_labels,
                without_labels=None, figure_id=renderer.figure_id,
                new_figure=False, render_lines=render_lines,
                line_colour=line_colour, line_style=line_style,
                line_width=line_width, render_markers=render_markers,
                marker_style=marker_style, marker_size=marker_size,
                marker_face_colour=marker_face_colour,
                marker_edge_colour=marker_edge_colour,
                marker_edge_width=marker_edge_width,
                render_numbering=render_numbering,
                numbers_horizontal_align=numbers_horizontal_align,
                numbers_vertical_align=numbers_vertical_align,
                numbers_font_name=numbers_font_name,
                numbers_font_size=numbers_font_size,
                numbers_font_style=numbers_font_style,
                numbers_font_weight=numbers_font_weight,
                numbers_font_colour=numbers_font_colour,
                render_legend=render_legend, legend_title=legend_title,
                legend_font_name=legend_font_name,
                legend_font_style=legend_font_style,
                legend_font_size=legend_font_size,
                legend_font_weight=legend_font_weight,
                legend_marker_scale=legend_marker_scale,
                legend_location=legend_location,
                legend_bbox_to_anchor=legend_bbox_to_anchor,
                legend_border_axes_pad=legend_border_axes_pad,
                legend_n_columns=legend_n_columns,
                legend_horizontal_spacing=legend_horizontal_spacing,
                legend_vertical_spacing=legend_vertical_spacing,
                legend_border=legend_border,
                legend_border_padding=legend_border_padding,
                legend_shadow=legend_shadow,
                legend_rounded_corners=legend_rounded_corners,
                render_axes=render_axes, axes_font_name=axes_font_name,
                axes_font_size=axes_font_size, axes_font_style=axes_font_style,
                axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
                axes_y_limits=axes_y_limits, figure_size=figure_size,
                interpolation=interpolation, alpha=alpha, cmap_name=cmap_name,
                **mask_arguments)
    else:
        # either there are not any landmark groups selected or they won't
        # be displayed
        if glyph_enabled:
            # image, not landmarks, masked, glyph
            renderer = glyph(image, vectors_block_size=glyph_block_size,
                             use_negative=glyph_use_negative,
                             channels=channels).view(
                render_axes=render_axes, axes_font_name=axes_font_name,
                axes_font_size=axes_font_size, axes_font_style=axes_font_style,
                axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
                axes_y_limits=axes_y_limits, figure_size=figure_size,
                interpolation=interpolation, alpha=alpha, cmap_name=cmap_name,
                **mask_arguments)
        elif sum_enabled:
            # image, not landmarks, masked, sum
            renderer = sum_channels(image, channels=channels).view(
                render_axes=render_axes, axes_font_name=axes_font_name,
                axes_font_size=axes_font_size, axes_font_style=axes_font_style,
                axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
                axes_y_limits=axes_y_limits, figure_size=figure_size,
                interpolation=interpolation, alpha=alpha, cmap_name=cmap_name,
                **mask_arguments)
        else:
            # image, not landmarks, masked, not glyph/sum
            renderer = image.view(
                channels=channels, render_axes=render_axes,
                axes_font_name=axes_font_name, axes_font_size=axes_font_size,
                axes_font_style=axes_font_style,
                axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
                axes_y_limits=axes_y_limits, figure_size=figure_size,
                interpolation=interpolation, alpha=alpha, cmap_name=cmap_name,
                **mask_arguments)

    # show plot
    pltshow()

    return renderer


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
    groups_keys, labels_keys = _extract_group_labels_landmarks(image.landmarks)
    return groups_keys, labels_keys


def _extract_group_labels_landmarks(landmark_manager):
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
    if landmark_manager.has_landmarks:
        groups_keys = landmark_manager.keys()
        labels_keys = [landmark_manager[g].keys() for g in groups_keys]
    else:
        groups_keys = [' ']
        labels_keys = [[' ']]
    return groups_keys, labels_keys
