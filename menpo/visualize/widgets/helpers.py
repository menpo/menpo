from IPython.html.widgets import (FloatSliderWidget, ContainerWidget,
                                  LatexWidget, CheckboxWidget,
                                  ToggleButtonWidget)
from IPython.display import display


def figure_options(x_scale_default=1.5, y_scale_default=0.5,
                   coupled_default=False, show_axes_default=True,
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
    but = ToggleButtonWidget(description='Figure Options', value=True)

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
    elif figure_scales_visible and not show_axes_visible:
        def show_options(name, value):
            if value:
                figure_scale.visible = True
            else:
                figure_scale.visible = False
    elif not figure_scales_visible and show_axes_visible:
        def show_options(name, value):
            if value:
                show_axes.visible = True
            else:
                show_axes.visible = False
    else:
        def show_options(name, value):
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

    # display widget
    display(figure_options_wid)

    # format widget alignment
    figure_scale.remove_class('vbox')
    figure_scale.add_class('hbox')
    X_scale.set_css('width', '3cm')
    Y_scale.set_css('width', '3cm')

    return figure_options_wid
