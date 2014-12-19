from IPython.html.widgets import (PopupWidget, ContainerWidget, TabWidget,
                                  RadioButtonsWidget, ButtonWidget,
                                  CheckboxWidget)
from IPython.display import display, clear_output
import matplotlib.pylab as plt
import numpy as np
from collections import Sized

from menpo.visualize.viewmatplotlib import (MatplotlibSubplots,
                                            MatplotlibImageViewer2d,
                                            sample_colours_from_colourmap)

from .helpers import (channel_options, format_channel_options,
                      update_channel_options,
                      landmark_options, format_landmark_options,
                      update_landmark_options, info_print, format_info_print,
                      animation_options, format_animation_options,
                      save_figure_options, format_save_figure_options,
                      features_options, format_features_options, viewer_options,
                      format_viewer_options)
from .lowlevelhelpers import logo, format_logo

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


def visualize_pointclouds(pointclouds, figure_size=(6, 4), popup=False,
                          browser_style='buttons'):
    r"""
    Widget that allows browsing through a list of pointclouds and pointgraphs.

    The widget has options tabs regarding the renderer (lines, markers, figure,
    axes) and saving the figure to file.

    Parameters
    -----------
    pointclouds : `list` of :map:`PointCloud` or :map:`PointGraph` or subclass
        The `list` of pointclouds to be visualized.
    figure_size : (`int`, `int`), optional
        The initial size of the rendered figure.
    popup : `bool`, optional
        If ``True``, the widget will appear as a popup window.
    browser_style : {``buttons``, ``slider``}, optional
        It defines whether the selector of the pointclouds will have the form of
        plus/minus buttons or a slider.
    """
    # make sure that pointclouds is a list even with one pointcloud member
    if not isinstance(pointclouds, Sized):
        pointclouds = [pointclouds]

    # find number of pointclouds
    n_pointclouds = len(pointclouds)

    # initial options dictionaries
    lines_options = {'render_lines': False,
                     'line_width': 1,
                     'line_colour': ['r'],
                     'line_style': '-'}
    markers_options = {'render_markers': True,
                       'marker_size': 20,
                       'marker_face_colour': ['r'],
                       'marker_edge_colour': ['k'],
                       'marker_style': 'o',
                       'marker_edge_width': 1}
    figure_options = {'x_scale': 1.,
                      'y_scale': 1.,
                      'render_axes': False,
                      'axes_font_name': 'sans-serif',
                      'axes_font_size': 10,
                      'axes_font_style': 'normal',
                      'axes_font_weight': 'normal',
                      'axes_x_limits': None,
                      'axes_y_limits': None}
    viewer_options_default = {'lines': lines_options,
                              'markers': markers_options,
                              'figure': figure_options}
    index_selection_default = {'min': 0,
                               'max': n_pointclouds-1,
                               'step': 1,
                               'index': 0}

    # define plot function
    def plot_function(name, value):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        clear_output(wait=True)

        # get selected pointcloud number
        im = 0
        if n_pointclouds > 1:
            im = pointcloud_number_wid.selected_values['index']

        # update info text widget
        update_info(pointclouds[im])

        # show image with selected options
        tmp1 = viewer_options_wid.selected_values[0]['lines']
        tmp2 = viewer_options_wid.selected_values[0]['markers']
        tmp3 = viewer_options_wid.selected_values[0]['figure']
        new_figure_size = (tmp3['x_scale'] * figure_size[0],
                           tmp3['y_scale'] * figure_size[1])

        renderer = pointclouds[im].view(
            figure_id=save_figure_wid.renderer[0].figure_id,
            new_figure=False, image_view=image_view_wid.value,
            render_lines=tmp1['render_lines'],
            line_colour=tmp1['line_colour'][0],
            line_style=tmp1['line_style'], line_width=tmp1['line_width'],
            render_markers=tmp2['render_markers'],
            marker_style=tmp2['marker_style'], marker_size=tmp2['marker_size'],
            marker_face_colour=tmp2['marker_face_colour'],
            marker_edge_colour=tmp2['marker_edge_colour'],
            marker_edge_width=tmp2['marker_edge_width'],
            render_axes=tmp3['render_axes'],
            axes_font_name=tmp3['axes_font_name'],
            axes_font_size=tmp3['axes_font_size'],
            axes_font_style=tmp3['axes_font_style'],
            axes_font_weight=tmp3['axes_font_weight'],
            axes_x_limits=tmp3['axes_x_limits'],
            axes_y_limits=tmp3['axes_y_limits'],
            figure_size=new_figure_size,
            label=None)

        plt.show()

        # save the current figure id
        save_figure_wid.renderer[0] = renderer

    # define function that updates info text
    def update_info(pointcloud):
        min_b, max_b = pointcloud.bounds()
        rang = pointcloud.range()
        cm = pointcloud.centre()
        # Create info string
        info_txt = r"""
             {0} points.
             Bounds: [{1:.1f}-{2:.1f}]W, [{3:.1f}-{4:.1f}]H.
             Range: {5:.1f}W, {6:.1f}H
             Centre of mass: ({7:.1f}, {8:.1f})
        """.format(pointcloud.n_points, min_b[0], max_b[0], min_b[1],
                   max_b[1], rang[0], rang[1], cm[0], cm[1])

        # update info widget text
        info_wid.children[1].value = _raw_info_string_to_latex(info_txt)

    # viewer options widget
    image_view_wid = CheckboxWidget(description='Image mode', value=True)
    image_view_wid.on_trait_change(plot_function, 'value')
    viewer_options_wid = viewer_options(viewer_options_default,
                                        ['lines', 'markers', 'figure_one'],
                                        objects_names=None,
                                        plot_function=plot_function,
                                        toggle_show_visible=False,
                                        toggle_show_default=True)
    viewer_options_all = ContainerWidget(children=[image_view_wid,
                                                   viewer_options_wid])
    info_wid = info_print(toggle_show_default=True,
                          toggle_show_visible=False)

    # save figure widget
    initial_renderer = MatplotlibImageViewer2d(figure_id=None, new_figure=True,
                                               image=np.zeros((10, 10)))
    save_figure_wid = save_figure_options(initial_renderer,
                                          toggle_show_default=True,
                                          toggle_show_visible=False)

    # create final widget
    if n_pointclouds > 1:
        # pointcloud selection slider
        pointcloud_number_wid = animation_options(
            index_selection_default, plot_function=plot_function,
            index_description='Pointcloud Number', index_minus_description='<',
            index_plus_description='>', index_style=browser_style,
            index_text_editable=True, loop_default=True, interval_default=0.3,
            toggle_show_title='Pointcloud Options', toggle_show_default=True,
            toggle_show_visible=False)

        # final widget
        logo_wid = ContainerWidget(children=[logo(), pointcloud_number_wid])
        button_title = 'Pointclouds Menu'
    else:
        # final widget
        logo_wid = logo()
        button_title = 'Pointcloud Menu'
    # create popup widget if asked
    cont_wid = TabWidget(children=[info_wid, viewer_options_all,
                                   save_figure_wid])
    if popup:
        wid = PopupWidget(children=[logo_wid, cont_wid],
                          button_text=button_title)
    else:
        wid = ContainerWidget(children=[logo_wid, cont_wid])

    # display final widget
    display(wid)

    # set final tab titles
    tab_titles = ['Pointcloud info', 'Viewer options', 'Save figure']
    for (k, tl) in enumerate(tab_titles):
        wid.children[1].set_title(k, tl)

    # align-start the pointcloud number widget and the rest
    if n_pointclouds > 1:
        wid.add_class('align-start')

    # format options' widgets
    if n_pointclouds > 1:
        wid.children[0].remove_class('vbox')
        wid.children[0].add_class('hbox')
        format_animation_options(pointcloud_number_wid,
                                 index_text_width='1.0cm',
                                 container_padding='6px',
                                 container_margin='6px',
                                 container_border='1px solid black',
                                 toggle_button_font_weight='bold',
                                 border_visible=False)
    format_viewer_options(viewer_options_wid, container_padding='6px',
                          container_margin='6px',
                          container_border='1px solid black',
                          toggle_button_font_weight='bold',
                          border_visible=False,
                          suboptions_border_visible=True)
    format_info_print(info_wid, font_size_in_pt='9pt', container_padding='6px',
                      container_margin='6px',
                      container_border='1px solid black',
                      toggle_button_font_weight='bold', border_visible=False)
    format_save_figure_options(save_figure_wid, container_padding='6px',
                               container_margin='6px',
                               container_border='1px solid black',
                               toggle_button_font_weight='bold',
                               tab_top_margin='0cm', border_visible=False)

    # Reset value to trigger initial visualization
    viewer_options_wid.children[1].children[1].children[0].children[1].children[0].value = True


def visualize_landmarkgroups(landmarkgroups, figure_size=(6, 4), popup=False,
                             browser_style='buttons'):
    r"""
    Widget that allows browsing through a list of landmark groups.

    The landmark groups can have a combination of different attributes, e.g.
    different labels, number of points etc. The widget has options tabs
    regarding the landmarks, the renderer (lines, markers, numbering, legend,
    figure, axes) and saving the figure to file.

    Parameters
    -----------
    landmarkgroups : `list` of :map:`LandmarkGroup` or subclass
        The `list` of landmark groups to be visualized.
    figure_size : (`int`, `int`), optional
        The initial size of the rendered figure.
    popup : `bool`, optional
        If ``True``, the widget will appear as a popup window.
    browser_style : {``buttons``, ``slider``}, optional
        It defines whether the selector of the landmark managers will have the
        form of plus/minus buttons or a slider.
    """
    # make sure that landmarkgroups is a list even with one landmark group
    # member
    if not isinstance(landmarkgroups, list):
        landmarkgroups = [landmarkgroups]

    # find number of landmarkgroups
    n_landmarkgroups = len(landmarkgroups)

    # find all available groups and the respective labels from all the landmarks
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

    # get initial line colours for each available group
    line_colours = []
    for l in all_labels:
        if len(l) == 1:
            line_colours.append(['r'])
        else:
            line_colours.append(sample_colours_from_colourmap(len(l), 'jet'))

    # initial options dictionaries
    landmark_options_default = {'render_landmarks': True,
                                'group_keys': ['0'],
                                'labels_keys': [landmarkgroups[0].labels],
                                'group': None,
                                'with_labels': None}
    index_selection_default = {'min': 0,
                               'max': n_landmarkgroups-1,
                               'step': 1,
                               'index': 0}
    markers_options = {'render_markers': True,
                       'marker_size': 20,
                       'marker_face_colour': ['r'],
                       'marker_edge_colour': ['k'],
                       'marker_style': 'o',
                       'marker_edge_width': 1}
    numbering_options = {'render_numbering': False,
                         'numbers_font_name': 'sans-serif',
                         'numbers_font_size': 10,
                         'numbers_font_style': 'normal',
                         'numbers_font_weight': 'normal',
                         'numbers_font_colour': ['k'],
                         'numbers_horizontal_align': 'center',
                         'numbers_vertical_align': 'bottom'}
    legend_options = {'render_legend': True,
                      'legend_title': '',
                      'legend_font_name': 'sans-serif',
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
                      'legend_rounded_corners': False}
    figure_options = {'x_scale': 1.,
                      'y_scale': 1.,
                          'render_axes': False,
                      'axes_font_name': 'sans-serif',
                      'axes_font_size': 10,
                      'axes_font_style': 'normal',
                      'axes_font_weight': 'normal',
                      'axes_x_limits': None,
                      'axes_y_limits': None}
    viewer_options_default = []
    for i in range(len(all_labels)):
        lines_options_default = {'render_lines': True,
                                 'line_width': 1,
                                 'line_colour': line_colours[i],
                                 'line_style': '-'}
        tmp = {'lines': lines_options_default,
               'markers': markers_options,
               'numbering': numbering_options,
               'legend': legend_options,
               'figure': figure_options}
        viewer_options_default.append(tmp)

    # Define plot function
    def plot_function(name, value):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        clear_output(wait=True)

        # get selected pointcloud number
        im = 0
        if n_landmarkgroups > 1:
            im = landmark_number_wid.selected_values['index']

        # update info text widget
        update_info(landmarkgroups[im])

        # show landmarks with selected options
        group_idx = all_labels.index(landmarkgroups[im].labels)
        tmp1 = viewer_options_wid.selected_values[group_idx]['lines']
        tmp2 = viewer_options_wid.selected_values[group_idx]['markers']
        tmp3 = viewer_options_wid.selected_values[group_idx]['numbering']
        tmp4 = viewer_options_wid.selected_values[group_idx]['legend']
        tmp5 = viewer_options_wid.selected_values[group_idx]['figure']
        new_figure_size = (tmp5['x_scale'] * figure_size[0],
                           tmp5['y_scale'] * figure_size[1])
        n_labels = len(landmark_options_wid.selected_values['with_labels'])

        renderer = landmarkgroups[im].view(
            with_labels=landmark_options_wid.selected_values['with_labels'],
            figure_id=save_figure_wid.renderer[0].figure_id,
            new_figure=False, image_view=axes_mode_wid.value == 1,
            render_lines=tmp1['render_lines'],
            line_colour=tmp1['line_colour'][:n_labels],
            line_style=tmp1['line_style'], line_width=tmp1['line_width'],
            render_markers=tmp2['render_markers'],
            marker_style=tmp2['marker_style'], marker_size=tmp2['marker_size'],
            marker_face_colour=tmp2['marker_face_colour'],
            marker_edge_colour=tmp2['marker_edge_colour'],
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

        plt.show()

        # save the current figure id
        save_figure_wid.renderer[0] = renderer

    # define function that updates info text
    def update_info(landmarkgroup):
        min_b, max_b = landmarkgroup.values()[0].bounds()
        rang = landmarkgroup.values()[0].range()
        cm = landmarkgroup.values()[0].centre()
        info_txt = r"""
            {0} landmark points.
            Bounds: [{1:.1f}-{2:.1f}]W, [{3:.1f}-{4:.1f}]H.
            Range: {5:.1f}W, {6:.1f}H
            Centre of mass: ({7:.1f}, {8:.1f})
            Shape norm is {9:.2f}.
        """.format(landmarkgroup.values()[0].n_points,
                   min_b[0], max_b[0], min_b[1],
                   max_b[1], rang[0], rang[1], cm[0], cm[1],
                   landmarkgroup.values()[0].norm())

        # update info widget text
        info_wid.children[1].value = _raw_info_string_to_latex(info_txt)

    # create options widgets
    # The landmarks checkbox default value if the first image doesn't have
    # landmarks
    landmark_options_wid = landmark_options(landmark_options_default,
                                            plot_function=plot_function,
                                            toggle_show_default=True,
                                            toggle_show_visible=False)

    # viewer options widget
    axes_mode_wid = RadioButtonsWidget(values={'Image': 1, 'Point cloud': 2},
                                       description='Axes mode:', value=2)
    axes_mode_wid.on_trait_change(plot_function, 'value')
    viewer_options_wid = viewer_options(viewer_options_default,
                                        ['lines', 'markers', 'numbering',
                                         'legend', 'figure_one'],
                                        objects_names=all_groups,
                                        plot_function=plot_function,
                                        toggle_show_visible=False,
                                        toggle_show_default=True,
                                        labels=all_labels)
    viewer_options_all = ContainerWidget(children=[axes_mode_wid,
                                                   viewer_options_wid])
    info_wid = info_print(toggle_show_default=True, toggle_show_visible=False)

    # save figure widget
    initial_renderer = MatplotlibImageViewer2d(figure_id=None, new_figure=True,
                                               image=np.zeros((10, 10)))
    save_figure_wid = save_figure_options(initial_renderer,
                                          toggle_show_default=True,
                                          toggle_show_visible=False)

    # define function that updates options' widgets state
    def update_widgets(name, value):
        # get new groups and labels, update landmark options and format them
        im = 0
        if n_landmarkgroups > 1:
            im = landmark_number_wid.selected_values['index']
        group_l = all_labels.index(landmarkgroups[im].labels)
        update_landmark_options(landmark_options_wid, [str(group_l)],
                                [landmarkgroups[im].labels], plot_function)
        format_landmark_options(landmark_options_wid, container_padding='6px',
                                container_margin='6px',
                                container_border='1px solid black',
                                toggle_button_font_weight='bold',
                                border_visible=False)

        # set correct group to viewer options' selection
        viewer_options_wid.children[1].children[0].value = \
            all_labels.index(landmarkgroups[im].labels)
    landmark_options_wid.children[2].children[0].on_trait_change(update_widgets,
                                                                 'value')

    # create final widget
    if n_landmarkgroups > 1:
        # landmark selection slider
        landmark_number_wid = animation_options(
            index_selection_default, plot_function=plot_function,
            update_function=update_widgets, index_description='Shape Number',
            index_minus_description='<', index_plus_description='>',
            index_style=browser_style, index_text_editable=True,
            loop_default=True, interval_default=0.3,
            toggle_show_title='Shape Options', toggle_show_default=True,
            toggle_show_visible=False)

        # final widget
        logo_wid = ContainerWidget(children=[logo(), landmark_number_wid])
        button_title = 'Shapes Menu'
    else:
        # final widget
        logo_wid = logo()
        button_title = 'Shape Menu'
    # create popup widget if asked
    cont_wid = TabWidget(children=[info_wid, landmark_options_wid,
                                   viewer_options_all, save_figure_wid])
    if popup:
        wid = PopupWidget(children=[logo_wid, cont_wid],
                          button_text=button_title)
    else:
        wid = ContainerWidget(children=[logo_wid, cont_wid])

    # display final widget
    display(wid)

    # set final tab titles
    tab_titles = ['Shape info', 'Landmarks options', 'Viewer options',
                  'Save figure']
    for (k, tl) in enumerate(tab_titles):
        wid.children[1].set_title(k, tl)

    # align-start the image number widget and the rest
    if n_landmarkgroups > 1:
        wid.add_class('align-start')

    # format options' widgets
    if n_landmarkgroups > 1:
        wid.children[0].remove_class('vbox')
        wid.children[0].add_class('hbox')
        format_animation_options(landmark_number_wid, index_text_width='1.0cm',
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
    format_viewer_options(viewer_options_wid, container_padding='6px',
                          container_margin='6px',
                          container_border='1px solid black',
                          toggle_button_font_weight='bold',
                          border_visible=False,
                          suboptions_border_visible=True)
    format_info_print(info_wid, font_size_in_pt='9pt', container_padding='6px',
                      container_margin='6px',
                      container_border='1px solid black',
                      toggle_button_font_weight='bold', border_visible=False)
    format_save_figure_options(save_figure_wid, container_padding='6px',
                               container_margin='6px',
                               container_border='1px solid black',
                               toggle_button_font_weight='bold',
                               tab_top_margin='0cm', border_visible=False)

    # update widgets' state for shape number 0
    update_widgets('', 0)

    # Reset value to trigger initial visualization
    axes_mode_wid.value = 1


def visualize_landmarks(landmarks, figure_size=(6, 4), popup=False,
                        browser_style='buttons'):
    r"""
    Widget that allows browsing through a list of landmark managers.

    The managers can have a combination of different attributes, e.g. different
    landmark groups and labels etc. The widget has options tabs regarding the
    landmarks, the renderer (lines, markers, numbering, legend, figure, axes)
    and saving the figure to file.

    Parameters
    -----------
    landmarks : `list` of :map:`LandmarkManager` or subclass
        The `list` of landmarks to be visualized.
    figure_size : (`int`, `int`), optional
        The initial size of the rendered figure.
    popup : `bool`, optional
        If ``True``, the widget will appear as a popup window.
    browser_style : {``buttons``, ``slider``}, optional
        It defines whether the selector of the landmark managers will have the
        form of plus/minus buttons or a slider.
    """
    # make sure that landmarks (groups) is a list even with one landmark group
    # member
    if not isinstance(landmarks, list):
        landmarks = [landmarks]

    # find number of landmarks
    n_landmarks = len(landmarks)

    # find initial groups and labels that will be passed to the landmark options
    # widget creation
    first_has_landmarks = landmarks[0].n_groups != 0
    if first_has_landmarks:
        initial_groups_keys, initial_labels_keys = \
            _extract_group_labels_landmarks(landmarks[0])
    else:
        initial_groups_keys = [' ']
        initial_labels_keys = [[' ']]

    # find all available groups and the respective labels from all the landmarks
    # that are passed in
    all_groups = []
    all_labels = []
    for l in landmarks:
        groups_l, labels_l = _extract_group_labels_landmarks(l)
        for i, g in enumerate(groups_l):
            if g not in all_groups:
                all_groups.append(g)
                all_labels.append(labels_l[i])

    # get initial line colours for each available group
    line_colours = []
    for l in all_labels:
        if len(l) == 1:
            line_colours.append(['r'])
        else:
            line_colours.append(sample_colours_from_colourmap(len(l), 'jet'))

    # initial options dictionaries
    landmark_options_default = {'render_landmarks': first_has_landmarks,
                                'group_keys': initial_groups_keys,
                                'labels_keys': initial_labels_keys,
                                'group': None,
                                'with_labels': None}
    index_selection_default = {'min': 0,
                               'max': n_landmarks-1,
                               'step': 1,
                               'index': 0}
    markers_options = {'render_markers': True,
                       'marker_size': 20,
                       'marker_face_colour': ['r'],
                       'marker_edge_colour': ['k'],
                       'marker_style': 'o',
                       'marker_edge_width': 1}
    numbering_options = {'render_numbering': False,
                         'numbers_font_name': 'sans-serif',
                         'numbers_font_size': 10,
                         'numbers_font_style': 'normal',
                         'numbers_font_weight': 'normal',
                         'numbers_font_colour': ['k'],
                         'numbers_horizontal_align': 'center',
                         'numbers_vertical_align': 'bottom'}
    legend_options = {'render_legend': True,
                      'legend_title': '',
                      'legend_font_name': 'sans-serif',
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
                      'legend_rounded_corners': False}
    figure_options = {'x_scale': 1.,
                      'y_scale': 1.,
                          'render_axes': False,
                      'axes_font_name': 'sans-serif',
                      'axes_font_size': 10,
                      'axes_font_style': 'normal',
                      'axes_font_weight': 'normal',
                      'axes_x_limits': None,
                      'axes_y_limits': None}
    viewer_options_default = []
    for i in range(len(all_groups)):
        lines_options_default = {'render_lines': True,
                                 'line_width': 1,
                                 'line_colour': line_colours[i],
                                 'line_style': '-'}
        tmp = {'lines': lines_options_default,
               'markers': markers_options,
               'numbering': numbering_options,
               'legend': legend_options,
               'figure': figure_options}
        viewer_options_default.append(tmp)

    # Define plot function
    def plot_function(name, value):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        clear_output(wait=True)

        # get selected pointcloud number
        im = 0
        if n_landmarks > 1:
            im = landmark_number_wid.selected_values['index']

        # update info text widget
        update_info(landmarks[im],
                    landmark_options_wid.selected_values['group'])

        # show landmarks with selected options
        group_idx = all_groups.index(landmark_options_wid.selected_values['group'])
        tmp1 = viewer_options_wid.selected_values[group_idx]['lines']
        tmp2 = viewer_options_wid.selected_values[group_idx]['markers']
        tmp3 = viewer_options_wid.selected_values[group_idx]['numbering']
        tmp4 = viewer_options_wid.selected_values[group_idx]['legend']
        tmp5 = viewer_options_wid.selected_values[group_idx]['figure']
        new_figure_size = (tmp5['x_scale'] * figure_size[0],
                           tmp5['y_scale'] * figure_size[1])
        n_labels = len(landmark_options_wid.selected_values['with_labels'])

        renderer = landmarks[im].view(
            group=landmark_options_wid.selected_values['group'],
            with_labels=landmark_options_wid.selected_values['with_labels'],
            figure_id=save_figure_wid.renderer[0].figure_id,
            new_figure=False, image_view=axes_mode_wid.value == 1,
            render_lines=tmp1['render_lines'],
            line_colour=tmp1['line_colour'][:n_labels],
            line_style=tmp1['line_style'], line_width=tmp1['line_width'],
            render_markers=tmp2['render_markers'],
            marker_style=tmp2['marker_style'], marker_size=tmp2['marker_size'],
            marker_face_colour=tmp2['marker_face_colour'],
            marker_edge_colour=tmp2['marker_edge_colour'],
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

        plt.show()

        # save the current figure id
        save_figure_wid.renderer[0] = renderer

    # define function that updates info text
    def update_info(landmarks, group):
        if group != ' ':
            min_b, max_b = landmarks[group][None].bounds()
            rang = landmarks[group][None].range()
            cm = landmarks[group][None].centre()
            info_txt = r"""
                {0} landmark points.
                Bounds: [{1:.1f}-{2:.1f}]W, [{3:.1f}-{4:.1f}]H.
                Range: {5:.1f}W, {6:.1f}H
                Centre of mass: ({7:.1f}, {8:.1f})
                Shape norm is {9:.2f}.
            """.format(landmarks[group][None].n_points,
                       min_b[0], max_b[0], min_b[1],
                       max_b[1], rang[0], rang[1], cm[0], cm[1],
                       landmarks[group][None].norm())
        else:
            info_txt = "There are no landmarks."

        # update info widget text
        info_wid.children[1].value = _raw_info_string_to_latex(info_txt)

    # create options widgets
    # The landmarks checkbox default value if the first image doesn't have
    # landmarks
    landmark_options_wid = landmark_options(landmark_options_default,
                                            plot_function=plot_function,
                                            toggle_show_default=True,
                                            toggle_show_visible=False)
    # if only a single landmark manager is passed in and it doesn't have
    # landmarks, then landmarks checkbox should be disabled
    landmark_options_wid.children[1].disabled = not first_has_landmarks

    # viewer options widget
    axes_mode_wid = RadioButtonsWidget(values={'Image': 1, 'Point cloud': 2},
                                       description='Axes mode:', value=2)
    axes_mode_wid.on_trait_change(plot_function, 'value')
    viewer_options_wid = viewer_options(viewer_options_default,
                                        ['lines', 'markers', 'numbering',
                                         'legend', 'figure_one'],
                                        objects_names=all_groups,
                                        plot_function=plot_function,
                                        toggle_show_visible=False,
                                        toggle_show_default=True,
                                        labels=all_labels)
    viewer_options_all = ContainerWidget(children=[axes_mode_wid,
                                                   viewer_options_wid])
    info_wid = info_print(toggle_show_default=True, toggle_show_visible=False)

    # save figure widget
    initial_renderer = MatplotlibImageViewer2d(figure_id=None, new_figure=True,
                                               image=np.zeros((10, 10)))
    save_figure_wid = save_figure_options(initial_renderer,
                                          toggle_show_default=True,
                                          toggle_show_visible=False)

    # define function that updates options' widgets state
    def update_widgets(name, value):
        # get new groups and labels, update landmark options and format them
        im = 0
        if n_landmarks > 1:
            im = landmark_number_wid.selected_values['index']
        group_keys, labels_keys = _extract_group_labels_landmarks(landmarks[im])
        update_landmark_options(landmark_options_wid, group_keys,
                                labels_keys, plot_function)
        format_landmark_options(landmark_options_wid, container_padding='6px',
                                container_margin='6px',
                                container_border='1px solid black',
                                toggle_button_font_weight='bold',
                                border_visible=False)

        # set correct group to viewer options' selection
        viewer_options_wid.children[1].children[0].value = \
            all_groups.index(landmark_options_wid.selected_values['group'])
    landmark_options_wid.children[2].children[0].on_trait_change(update_widgets,
                                                                 'value')

    # create final widget
    if n_landmarks > 1:
        # landmark selection slider
        landmark_number_wid = animation_options(
            index_selection_default, plot_function=plot_function,
            update_function=update_widgets, index_description='Shape Number',
            index_minus_description='<', index_plus_description='>',
            index_style=browser_style, index_text_editable=True,
            loop_default=True, interval_default=0.3,
            toggle_show_title='Shape Options', toggle_show_default=True,
            toggle_show_visible=False)

        # final widget
        logo_wid = ContainerWidget(children=[logo(), landmark_number_wid])
        button_title = 'Shapes Menu'
    else:
        # final widget
        logo_wid = logo()
        button_title = 'Shape Menu'
    # create popup widget if asked
    cont_wid = TabWidget(children=[info_wid, landmark_options_wid,
                                   viewer_options_all, save_figure_wid])
    if popup:
        wid = PopupWidget(children=[logo_wid, cont_wid],
                          button_text=button_title)
    else:
        wid = ContainerWidget(children=[logo_wid, cont_wid])

    # display final widget
    display(wid)

    # set final tab titles
    tab_titles = ['Shape info', 'Landmarks options', 'Viewer options',
                  'Save figure']
    for (k, tl) in enumerate(tab_titles):
        wid.children[1].set_title(k, tl)

    # align-start the image number widget and the rest
    if n_landmarks > 1:
        wid.add_class('align-start')

    # format options' widgets
    if n_landmarks > 1:
        wid.children[0].remove_class('vbox')
        wid.children[0].add_class('hbox')
        format_animation_options(landmark_number_wid, index_text_width='1.0cm',
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
    format_viewer_options(viewer_options_wid, container_padding='6px',
                          container_margin='6px',
                          container_border='1px solid black',
                          toggle_button_font_weight='bold',
                          border_visible=False,
                          suboptions_border_visible=True)
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
    axes_mode_wid.value = 1


def visualize_images(images, figure_size=(6, 4), popup=False,
                     browser_style='buttons'):
    r"""
    Widget that allows browsing through a list of images.

    The images can have a combination of different attributes, e.g. masked or
    not, landmarked or not, without multiple landmark groups and labels etc.
    The widget has options tabs regarding the visualized channels, the
    landmarks, the renderer (lines, markers, numbering, legend, figure, axes)
    and saving the figure to file.

    Parameters
    -----------
    images : `list` of :map:`Image` or subclass
        The `list` of images to be visualized.
    figure_size : (`int`, `int`), optional
        The initial size of the rendered figure.
    popup : `bool`, optional
        If ``True``, the widget will appear as a popup window.
    browser_style : {``buttons``, ``slider``}, optional
        It defines whether the selector of the images will have the form of
        plus/minus buttons or a slider.
    """
    from menpo.image import MaskedImage

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

    # initial options dictionaries
    channels_default = 0
    if images[0].n_channels == 3:
        channels_default = None
    channels_options_default = {'n_channels': images[0].n_channels,
                                'image_is_masked': isinstance(images[0],
                                                              MaskedImage),
                                'channels': channels_default,
                                'glyph_enabled': False,
                                'glyph_block_size': 3,
                                'glyph_use_negative': False,
                                'sum_enabled': False,
                                'masked_enabled': isinstance(images[0],
                                                             MaskedImage)}
    landmark_options_default = {'render_landmarks': first_has_landmarks,
                                'group_keys': initial_groups_keys,
                                'labels_keys': initial_labels_keys,
                                'group': None,
                                'with_labels': None}
    lines_options = {'render_lines': True,
                     'line_width': 1,
                     'line_colour': ['r'],
                     'line_style': '-'}
    markers_options = {'render_markers': True,
                       'marker_size': 20,
                       'marker_face_colour': ['r'],
                       'marker_edge_colour': ['k'],
                       'marker_style': 'o',
                       'marker_edge_width': 1}
    numbering_options = {'render_numbering': False,
                         'numbers_font_name': 'sans-serif',
                         'numbers_font_size': 10,
                         'numbers_font_style': 'normal',
                         'numbers_font_weight': 'normal',
                         'numbers_font_colour': ['k'],
                         'numbers_horizontal_align': 'center',
                         'numbers_vertical_align': 'bottom'}
    legend_options = {'render_legend': True,
                      'legend_title': '',
                      'legend_font_name': 'sans-serif',
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
                      'legend_rounded_corners': False}
    figure_options = {'x_scale': 1.,
                      'y_scale': 1.,
                      'render_axes': False,
                      'axes_font_name': 'sans-serif',
                      'axes_font_size': 10,
                      'axes_font_style': 'normal',
                      'axes_font_weight': 'normal',
                      'axes_x_limits': None,
                      'axes_y_limits': None}
    viewer_options_default = {'lines': lines_options,
                              'markers': markers_options,
                              'numbering': numbering_options,
                              'legend': legend_options,
                              'figure': figure_options}
    index_selection_default = {'min': 0,
                               'max': n_images-1,
                               'step': 1,
                               'index': 0}

    # define plot function
    def plot_function(name, value):
        # clear current figure, but wait until the new data to be displayed are
        # generated
        clear_output(wait=True)

        # get selected image number
        im = 0
        if n_images > 1:
            im = image_number_wid.selected_values['index']

        # update info text widget
        image_has_landmarks = images[im].landmarks.n_groups != 0
        image_is_masked = isinstance(images[im], MaskedImage)
        update_info(images[im], image_is_masked, image_has_landmarks,
                    landmark_options_wid.selected_values['group'])

        # show image with selected options
        tmp1 = viewer_options_wid.selected_values[0]['lines']
        tmp2 = viewer_options_wid.selected_values[0]['markers']
        tmp3 = viewer_options_wid.selected_values[0]['numbering']
        tmp4 = viewer_options_wid.selected_values[0]['legend']
        tmp5 = viewer_options_wid.selected_values[0]['figure']
        new_figure_size = (tmp5['x_scale'] * figure_size[0],
                           tmp5['y_scale'] * figure_size[1])
        renderer = _visualize(
            images[im], save_figure_wid.renderer[0], True,
            landmark_options_wid.selected_values['render_landmarks'],
            channel_options_wid.selected_values['image_is_masked'],
            channel_options_wid.selected_values['masked_enabled'],
            channel_options_wid.selected_values['channels'],
            channel_options_wid.selected_values['glyph_enabled'],
            channel_options_wid.selected_values['glyph_block_size'],
            channel_options_wid.selected_values['glyph_use_negative'],
            channel_options_wid.selected_values['sum_enabled'],
            [landmark_options_wid.selected_values['group']],
            [landmark_options_wid.selected_values['with_labels']],
            False, dict(), True, False,
            tmp1['render_lines'], tmp1['line_style'], tmp1['line_width'],
            tmp1['line_colour'][0], tmp2['render_markers'],
            tmp2['marker_style'], tmp2['marker_size'], tmp2['marker_edge_width'],
            tmp2['marker_edge_colour'], tmp2['marker_face_colour'],
            tmp3['render_numbering'], tmp3['numbers_font_name'],
            tmp3['numbers_font_size'], tmp3['numbers_font_style'],
            tmp3['numbers_font_weight'], tmp3['numbers_font_colour'][0],
            tmp3['numbers_horizontal_align'], tmp3['numbers_vertical_align'],
            tmp4['legend_n_columns'], tmp4['legend_border_axes_pad'],
            tmp4['legend_rounded_corners'], tmp4['legend_title'],
            tmp4['legend_horizontal_spacing'], tmp4['legend_shadow'],
            tmp4['legend_location'], tmp4['legend_font_name'],
            tmp4['legend_bbox_to_anchor'], tmp4['legend_border'],
            tmp4['legend_marker_scale'], tmp4['legend_vertical_spacing'],
            tmp4['legend_font_weight'], tmp4['legend_font_size'],
            tmp4['render_legend'], tmp4['legend_font_style'],
            tmp4['legend_border_padding'], new_figure_size, tmp5['render_axes'],
            tmp5['axes_font_name'], tmp5['axes_font_size'],
            tmp5['axes_font_style'], tmp5['axes_x_limits'],
            tmp5['axes_y_limits'], tmp5['axes_font_weight'])

        # save the current figure id
        save_figure_wid.renderer[0] = renderer

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

    # channel options widget
    channel_options_wid = channel_options(channels_options_default,
                                          plot_function=plot_function,
                                          toggle_show_default=True,
                                          toggle_show_visible=False)

    # landmarks options widget
    # The landmarks checkbox default value if the first image doesn't have
    # landmarks
    landmark_options_wid = landmark_options(landmark_options_default,
                                            plot_function=plot_function,
                                            toggle_show_default=True,
                                            toggle_show_visible=False)
    # if only a single image is passed in and it doesn't have landmarks, then
    # landmarks checkbox should be disabled
    landmark_options_wid.children[1].disabled = not first_has_landmarks

    # viewer options widget
    viewer_options_wid = viewer_options(viewer_options_default,
                                        ['lines', 'markers', 'numbering',
                                         'legend', 'figure_one'],
                                        objects_names=None,
                                        plot_function=plot_function,
                                        toggle_show_visible=False,
                                        toggle_show_default=True)
    info_wid = info_print(toggle_show_default=True,
                          toggle_show_visible=False)

    # save figure widget
    initial_renderer = MatplotlibImageViewer2d(figure_id=None, new_figure=True,
                                               image=np.zeros((10, 10)))
    save_figure_wid = save_figure_options(initial_renderer,
                                          toggle_show_default=True,
                                          toggle_show_visible=False)

    # define function that updates options' widgets state
    def update_widgets(name, value):
        # set channels = 0 to make sure that when plotting is triggered by the
        # update_landmark_options, we don't get an error for not enough channels
        tmp_channels = channel_options_wid.selected_values['channels']
        channel_options_wid.selected_values['channels'] = 0

        # get new groups and labels, update landmark options and format them
        group_keys, labels_keys = _extract_groups_labels(images[value])
        update_landmark_options(landmark_options_wid, group_keys,
                                labels_keys, plot_function)
        format_landmark_options(landmark_options_wid,
                                container_padding='6px',
                                container_margin='6px',
                                container_border='1px solid black',
                                toggle_button_font_weight='bold',
                                border_visible=False)

        # update channel options
        channel_options_wid.selected_values['channels'] = tmp_channels
        update_channel_options(channel_options_wid,
                               n_channels=images[value].n_channels,
                               image_is_masked=isinstance(images[value],
                                                          MaskedImage))

    # create final widget
    if n_images > 1:
        # image selection slider
        image_number_wid = animation_options(index_selection_default,
                                             plot_function=plot_function,
                                             update_function=update_widgets,
                                             index_description='Image Number',
                                             index_minus_description='<',
                                             index_plus_description='>',
                                             index_style=browser_style,
                                             index_text_editable=True,
                                             loop_default=True,
                                             interval_default=0.3,
                                             toggle_show_title='Image Options',
                                             toggle_show_default=True,
                                             toggle_show_visible=False)

        # final widget
        logo_wid = ContainerWidget(children=[logo(), image_number_wid])
        button_title = 'Images Menu'
    else:
        # final widget
        logo_wid = logo()
        button_title = 'Image Menu'
    # create popup widget if asked
    cont_wid = TabWidget(children=[info_wid, channel_options_wid,
                                   landmark_options_wid,
                                   viewer_options_wid, save_figure_wid])
    if popup:
        wid = PopupWidget(children=[logo_wid, cont_wid],
                          button_text=button_title)
    else:
        wid = ContainerWidget(children=[logo_wid, cont_wid])

    # display final widget
    display(wid)

    # set final tab titles
    tab_titles = ['Image info', 'Channels options', 'Landmarks options',
                  'Viewer options', 'Save figure']
    for (k, tl) in enumerate(tab_titles):
        wid.children[1].set_title(k, tl)

    # align-start the image number widget and the rest
    if n_images > 1:
        wid.add_class('align-start')

    # format options' widgets
    if n_images > 1:
        wid.children[0].remove_class('vbox')
        wid.children[0].add_class('hbox')
        format_animation_options(image_number_wid, index_text_width='1.0cm',
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
    format_viewer_options(viewer_options_wid, container_padding='6px',
                          container_margin='6px',
                          container_border='1px solid black',
                          toggle_button_font_weight='bold',
                          border_visible=False,
                          suboptions_border_visible=True)
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
    viewer_options_wid.children[1].children[1].children[3].children[1].children[0].value = False


def save_matplotlib_figure(renderer, popup=True):
    r"""
    Widget that allows to save a figure generated with Matplotlib to file.

    Parameters
    -----------
    renderer : :map:`MatplotlibRenderer`
        The Matplotlib renderer object.
    popup : `bool`, optional
        If ``True``, the widget will appear as a popup window.
    """
    # Create sub-widgets
    logo_wid = logo()
    save_figure_wid = save_figure_options(renderer, toggle_show_default=True,
                                          toggle_show_visible=False)

    # Create final widget
    if popup:
        wid = PopupWidget(children=[logo_wid, save_figure_wid],
                          button_text='Save Figure')
        # set width of popup widget
        wid.set_css({'width': '11cm'}, selector='modal')
    else:
        wid = ContainerWidget(children=[logo_wid, save_figure_wid])

    # Display widget
    display(wid)

    # Format widgets
    format_save_figure_options(save_figure_wid, container_padding='6px',
                               container_margin='6px',
                               container_border='1px solid black',
                               toggle_button_font_weight='bold',
                               tab_top_margin='0cm', border_visible=True)
    format_logo(logo_wid, border_visible=False)


def features_selection(popup=True):
    r"""
    Widget that allows selecting a features function and its options. The
    widget supports all features from :ref:`api-feature-index` and has a
    preview tab. It returns a list of length one with the selected features
    function closure.

    Parameters
    -----------
    popup : `bool`, optional
        If ``True``, the widget will appear as a popup window.

    Returns
    -------
    features_function : `list` of length 1
        The function closure of the features function using `functools.partial`.
        So the function can be called as:

        ::

            features_image = features_function[0](image)
    """
    # Create sub-widgets
    logo_wid = logo()
    features_options_wid = features_options(toggle_show_default=True,
                                            toggle_show_visible=False)
    features_wid = ContainerWidget(children=[logo_wid, features_options_wid])
    select_but = ButtonWidget(description='Select')

    # Create final widget
    if popup:
        wid = PopupWidget(children=[features_wid, select_but],
                          button_text='Features Selection')
    else:
        wid = ContainerWidget(children=[features_wid, select_but])

    # function for select button
    def select_function(name):
        wid.close()
        output.pop(0)
        output.append(features_options_wid.function)
    select_but.on_click(select_function)

    # Display widget
    display(wid)

    # Format widgets
    format_features_options(features_options_wid, border_visible=True)
    format_logo(logo_wid, border_visible=False)
    # set popup width
    if popup:
        wid.set_css({
            'width': '13cm'}, selector='modal')
    # align logo at the end
    features_wid.add_class('align-end')
    # align select button at the centre
    wid.add_class('align-center')

    # Initialize output with empty list. It needs to be a list so that
    # it's mutable and synchronizes with frontend.
    output = [features_options_wid.function]

    return output


def _visualize(image, renderer, render_image, render_landmarks, image_is_masked,
               masked_enabled, channels, glyph_enabled, glyph_block_size,
               glyph_use_negative, sum_enabled, groups, with_labels,
               subplots_enabled, subplots_titles, image_axes_mode,
               legend_enabled, render_lines, line_style, line_width,
               line_colour, render_markers, marker_style, marker_size,
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
               axes_x_limits, axes_y_limits, axes_font_weight):
    global glyph
    if glyph is None:
        from menpo.visualize.image import glyph

    # plot
    if render_image:
        # image will be displayed
        if render_landmarks and len(groups) > 0:
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
                if glyph_enabled or sum_enabled:
                    # image, landmarks, masked, glyph
                    renderer = glyph(image, vectors_block_size=glyph_block_size,
                                     use_negative=glyph_use_negative,
                                     channels=channels).\
                        view_landmarks(
                            masked=masked_enabled, group=group,
                            with_labels=with_labels[k], without_labels=None,
                            figure_id=renderer.figure_id, new_figure=False,
                            render_lines=render_lines, line_colour=line_colour,
                            line_style=line_style, line_width=line_width,
                            render_markers=render_markers,
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
                            render_legend=render_legend,
                            legend_title=legend_title,
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
                            render_axes=render_axes,
                            axes_font_name=axes_font_name,
                            axes_font_size=axes_font_size,
                            axes_font_style=axes_font_style,
                            axes_font_weight=axes_font_weight,
                            axes_x_limits=axes_x_limits,
                            axes_y_limits=axes_y_limits,
                            figure_size=figure_size)
                else:
                    # image, landmarks, masked, not glyph
                    renderer = image.view_landmarks(
                        channels=channels, masked=masked_enabled, group=group,
                        with_labels=with_labels[k], without_labels=None,
                        figure_id=renderer.figure_id, new_figure=False,
                        render_lines=render_lines, line_colour=line_colour,
                        line_style=line_style, line_width=line_width,
                        render_markers=render_markers,
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
                        axes_font_size=axes_font_size,
                        axes_font_style=axes_font_style,
                        axes_font_weight=axes_font_weight,
                        axes_x_limits=axes_x_limits,
                        axes_y_limits=axes_y_limits, figure_size=figure_size)
        else:
            # either there are not any landmark groups selected or they won't
            # be displayed
            if image_is_masked:
                if glyph_enabled or sum_enabled:
                    # image, not landmarks, masked, glyph
                    renderer = glyph(image, vectors_block_size=glyph_block_size,
                                     use_negative=glyph_use_negative,
                                     channels=channels).view(
                        masked=masked_enabled, render_axes=render_axes,
                        axes_font_name=axes_font_name,
                        axes_font_size=axes_font_size,
                        axes_font_style=axes_font_style,
                        axes_font_weight=axes_font_weight,
                        axes_x_limits=axes_x_limits,
                        axes_y_limits=axes_y_limits,
                        figure_size=figure_size)
                else:
                    # image, not landmarks, masked, not glyph
                    renderer = image.view(
                        masked=masked_enabled, channels=channels,
                        render_axes=render_axes, axes_font_name=axes_font_name,
                        axes_font_size=axes_font_size,
                        axes_font_style=axes_font_style,
                        axes_font_weight=axes_font_weight,
                        axes_x_limits=axes_x_limits,
                        axes_y_limits=axes_y_limits, figure_size=figure_size)
            else:
                if glyph_enabled or sum_enabled:
                    # image, not landmarks, not masked, glyph
                    renderer = glyph(image, vectors_block_size=glyph_block_size,
                                     use_negative=glyph_use_negative,
                                     channels=channels).view(
                        render_axes=render_axes, axes_font_name=axes_font_name,
                        axes_font_size=axes_font_size,
                        axes_font_style=axes_font_style,
                        axes_font_weight=axes_font_weight,
                        axes_x_limits=axes_x_limits,
                        axes_y_limits=axes_y_limits, figure_size=figure_size)
                else:
                    # image, not landmarks, not masked, not glyph
                    renderer = image.view(
                        channels=channels, render_axes=render_axes,
                        axes_font_name=axes_font_name,
                        axes_font_size=axes_font_size,
                        axes_font_style=axes_font_style,
                        axes_font_weight=axes_font_weight,
                        axes_x_limits=axes_x_limits,
                        axes_y_limits=axes_y_limits, figure_size=figure_size)
    else:
        # image won't be displayed
        if render_landmarks and len(groups) > 0:
            # there are selected landmark groups and they will be displayed
            if subplots_enabled:
                # calculate subplots structure
                subplots = MatplotlibSubplots()._subplot_layout(len(groups))
            # not image, landmarks
            for k, group in enumerate(groups):
                if subplots_enabled:
                    # create subplot
                    plt.subplot(subplots[0], subplots[1], k + 1)
                    if legend_enabled:
                        # set subplot's title
                        plt.title(subplots_titles[group])
                image.landmarks[group].lms.view(image_view=image_axes_mode)
            if not subplots_enabled:
                if legend_enabled:
                    # display legend on side
                    plt.legend(groups, loc='best')

    # show plot
    plt.show()

    return renderer


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
