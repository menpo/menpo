from collections import Iterable

import numpy as np

from menpo.shape import PointCloud
from .base import Image


def create_patches_image(patches, patch_centers, patches_indices=None,
                         offset_index=None):
    # Parse inputs
    if offset_index is None:
        offset_index = 0
    if patches_indices is None:
        patches_indices = np.arange(patches.shape[0])
    elif not isinstance(patches_indices, Iterable):
        patches_indices = [patches_indices]

    # Compute patches image's shape
    n_channels = patches.shape[2]
    patch_shape0 = patches.shape[3]
    patch_shape1 = patches.shape[4]
    top, left = np.min(patch_centers.points, 0)
    bottom, right = np.max(patch_centers.points, 0)
    min_0 = np.floor(top - patch_shape0)
    min_1 = np.floor(left - patch_shape1)
    max_0 = np.ceil(bottom + patch_shape0)
    max_1 = np.ceil(right + patch_shape1)
    height = max_0 - min_0 + 1
    width = max_1 - min_1 + 1

    # Translate the patch centers to fit in the new image
    new_patch_centers = patch_centers.copy()
    new_patch_centers.points = patch_centers.points - np.array([[min_0, min_1]])

    # Create temporary pointcloud with the selected patch centers
    tmp_centers = PointCloud(new_patch_centers.points[patches_indices])

    # Create black new image and attach the corrected patch centers
    patches_image = Image.init_blank((height, width), n_channels)
    patches_image.landmarks['all_patch_centers'] = new_patch_centers
    patches_image.landmarks['selected_patch_centers'] = tmp_centers

    # Set the patches
    patches_image.set_patches_around_landmarks(patches[patches_indices],
                                               group='selected_patch_centers',
                                               offset_index=offset_index)

    return patches_image


def render_rectangles_around_patches(centers, patch_shape, axes=None,
                                     image_view=True, line_colour='r',
                                     line_style='-', line_width=1,
                                     interpolation='none'):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # Dictionary with the line styles
    line_style_dict = {'-': 'solid', '--': 'dashed', '-.': 'dashdot',
                       '.': 'dotted'}

    # Get axes object
    if axes is None:
        axes = plt.gca()

    # Need those in order to compute the lower left corner of the rectangle
    half_patch_shape = [patch_shape[0] / 2,
                        patch_shape[1] / 2]

    # Set the view mode
    if image_view:
        xi = 1
        yi = 0
    else:
        xi = 0
        yi = 1

    # Set correct offsets so that the rectangle is tight to the patch
    if interpolation == 'none':
        off_start = 0.5
        off_end = 0.
    else:
        off_start = 1.
        off_end = 0.5

    # Render rectangles
    for p in range(centers.shape[0]):
        xc = np.intp(centers[p, xi] - half_patch_shape[xi]) - off_start
        yc = np.intp(centers[p, yi] - half_patch_shape[yi]) - off_start
        axes.add_patch(Rectangle((xc, yc),
                                 patch_shape[xi] + off_end,
                                 patch_shape[yi] + off_end,
                                 fill=False, edgecolor=line_colour,
                                 linewidth=line_width,
                                 linestyle=line_style_dict[line_style]))


def view_patches(patches, patch_centers, patches_indices=None,
                 offset_index=None, figure_id=None, new_figure=False,
                 channels=None, interpolation='none', cmap_name=None, alpha=1.,
                 render_patches_bboxes=True, bboxes_line_colour='b',
                 bboxes_line_style='-', bboxes_line_width=1, render_axes=False,
                 axes_font_name='sans-serif',  axes_font_size=10,
                 axes_font_style='normal', axes_font_weight='normal',
                 axes_x_limits=None, axes_y_limits=None, figure_size=(10, 8)):
    # Create patches image
    patches_image = create_patches_image(patches, patch_centers,
                                         patches_indices=patches_indices,
                                         offset_index=offset_index)

    # Render patches image
    patches_image.view(figure_id=figure_id, new_figure=new_figure,
                       channels=channels, interpolation=interpolation,
                       cmap_name=cmap_name, alpha=alpha,
                       render_axes=render_axes, axes_font_name=axes_font_name,
                       axes_font_size=axes_font_size,
                       axes_font_style=axes_font_style,
                       axes_font_weight=axes_font_weight,
                       axes_x_limits=axes_x_limits,
                       axes_y_limits=axes_y_limits, figure_size=figure_size)

    # Render rectangles around patches
    if render_patches_bboxes:
        patch_shape = [patches.shape[3], patches.shape[4]]
        render_rectangles_around_patches(
            patches_image.landmarks['selected_patch_centers'].lms.points,
            patch_shape, image_view=True, line_colour=bboxes_line_colour,
            line_style=bboxes_line_style, line_width=bboxes_line_width,
            interpolation=interpolation)


def view_patches_with_landmarks(patches, patch_centers, patches_indices=None,
                                offset_index=None, figure_id=None,
                                new_figure=False, channels=None,
                                interpolation='none', cmap_name=None, alpha=1.,
                                render_patches_bboxes=True,
                                bboxes_line_colour='r', bboxes_line_style='-',
                                bboxes_line_width=1, render_lines=True,
                                line_colour=None, line_style='-', line_width=1,
                                render_markers=True, marker_style='o',
                                marker_size=20, marker_face_colour=None,
                                marker_edge_colour=None, marker_edge_width=1.,
                                render_numbering=False,
                                numbers_horizontal_align='center',
                                numbers_vertical_align='bottom',
                                numbers_font_name='sans-serif',
                                numbers_font_size=10,
                                numbers_font_style='normal',
                                numbers_font_weight='normal',
                                numbers_font_colour='k', render_legend=False,
                                legend_title='', legend_font_name='sans-serif',
                                legend_font_style='normal', legend_font_size=10,
                                legend_font_weight='normal',
                                legend_marker_scale=None, legend_location=2,
                                legend_bbox_to_anchor=(1.05, 1.),
                                legend_border_axes_pad=None, legend_n_columns=1,
                                legend_horizontal_spacing=None,
                                legend_vertical_spacing=None,
                                legend_border=True,
                                legend_border_padding=None, legend_shadow=False,
                                legend_rounded_corners=False, render_axes=False,
                                axes_font_name='sans-serif', axes_font_size=10,
                                axes_font_style='normal',
                                axes_font_weight='normal', axes_x_limits=None,
                                axes_y_limits=None, figure_size=(10, 8)):
    # Create patches image
    patches_image = create_patches_image(patches, patch_centers,
                                         patches_indices=patches_indices,
                                         offset_index=offset_index)

    # Render patches image
    patches_image.view_landmarks(
        channels=channels, group='all_patch_centers', figure_id=figure_id,
        new_figure=new_figure, interpolation=interpolation, cmap_name=cmap_name,
        alpha=alpha, render_lines=render_lines, line_colour=line_colour,
        line_style=line_style, line_width=line_width,
        render_markers=render_markers, marker_style=marker_style,
        marker_size=marker_size, marker_face_colour=marker_face_colour,
        marker_edge_colour=marker_edge_colour,
        marker_edge_width=marker_edge_width, render_numbering=render_numbering,
        numbers_horizontal_align=numbers_horizontal_align,
        numbers_vertical_align=numbers_vertical_align,
        numbers_font_name=numbers_font_name,
        numbers_font_size=numbers_font_size,
        numbers_font_style=numbers_font_style,
        numbers_font_weight=numbers_font_weight,
        numbers_font_colour=numbers_font_colour, render_legend=render_legend,
        legend_title=legend_title, legend_font_name=legend_font_name,
        legend_font_style=legend_font_style, legend_font_size=legend_font_size,
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
        legend_rounded_corners=legend_rounded_corners, render_axes=render_axes,
        axes_font_name=axes_font_name, axes_font_size=axes_font_size,
        axes_font_style=axes_font_style, axes_font_weight=axes_font_weight,
        axes_x_limits=axes_x_limits, axes_y_limits=axes_y_limits,
        figure_size=figure_size)

    # Render rectangles around patches
    if render_patches_bboxes:
        patch_shape = [patches.shape[3], patches.shape[4]]
        render_rectangles_around_patches(
            patches_image.landmarks['selected_patch_centers'].lms.points,
            patch_shape, image_view=True, line_colour=bboxes_line_colour,
            line_style=bboxes_line_style, line_width=bboxes_line_width,
            interpolation=interpolation)
