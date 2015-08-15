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


def render_rectangles_around_patches(centers, patch_shape, axes=None, image_view=True,
                                     line_colour='r', line_style='-', line_width=1):
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

    # Render rectangles
    for p in range(centers.shape[0]):
        xc = np.intp(centers[p, xi] - half_patch_shape[xi]) - 0.5
        yc = np.intp(centers[p, yi] - half_patch_shape[yi]) - 0.5
        axes.add_patch(Rectangle((xc, yc), patch_shape[xi], patch_shape[yi],
                                 fill=False, edgecolor=line_colour,
                                 linewidth=line_width,
                                 linestyle=line_style_dict[line_style]))
