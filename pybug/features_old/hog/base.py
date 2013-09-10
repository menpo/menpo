from pybug.features_old.hog.hog_wrapper import _hog
import numpy as np

# The HOG functions were converted from Matlab and thus assume fortran
# ordering. In order to do maintain this ordering without user intervention,
# we handle marshalling the data type in these methods.


def dense_hog(image, method='dalaltriggs', num_orientations=9, cell_size=4,
              block_size=2, gradient_signed=True, l2_norm_clip=0.2,
              window_height=16, window_width=16, window_unit='pixels',
              window_step_vertical=1, window_step_horizontal=1,
              window_step_unit='pixels', padding_disabled=False, verbose=False):

    options = _parse_options('dense', method, num_orientations, cell_size,
                             block_size, gradient_signed, l2_norm_clip,
                             window_height, window_width, window_unit,
                             window_step_vertical, window_step_horizontal,
                             window_step_unit, padding_disabled, verbose)
    image = np.asfortranarray(image)
    descriptors, centers, opt_info = _hog(image, options)
    if verbose:
        return (np.ascontiguousarray(descriptors),
                np.ascontiguousarray(centers),
                opt_info)
    else:
        return np.ascontiguousarray(descriptors), np.ascontiguousarray(centers)


def sparse_hog(image, method='dalaltriggs', num_orientations=9, cell_size=4,
               block_size=2, gradient_signed=True, l2_norm_clip=0.2,
               verbose=False):
    # We should never use the manually filled out values - but they are
    # given sensible values as a defense.
    options = _parse_options('sparse', method, num_orientations, cell_size,
                             block_size, gradient_signed, l2_norm_clip,
                             0, 0, 0, 0, 0, 0, 0, verbose)
    image = np.asfortranarray(image)
    descriptors, centers, opt_info = _hog(image, options)
    descriptors = np.squeeze(descriptors)
    if verbose:
        return (np.ascontiguousarray(descriptors),
                np.ascontiguousarray(centers),
                opt_info)
    else:
        return np.ascontiguousarray(descriptors), np.ascontiguousarray(centers)


def _parse_options(type, method, num_orientations, cell_size,
                   block_size, gradient_signed, l2_norm_clip,
                   window_height, window_width, window_unit,
                   window_step_vertical, window_step_horizontal,
                   window_step_unit, padding_disabled, verbose):
    # Options only valid for dense HOGs
    if type is 'dense':
        if window_height <= 0:
            raise ValueError("Window height must be > 0.")
        if window_width <= 0:
            raise ValueError("Window width must be > 0.")
        if window_unit not in ['pixels', 'blocks']:
            raise ValueError("Window unit must be either pixels or blocks")
        if window_step_horizontal <= 0:
            raise ValueError("Horizontal window step must be > 0.")
        if window_step_vertical <= 0:
            raise ValueError("Vertical window step must be > 0.")
        if window_step_unit not in ['pixels', 'cells']:
            raise ValueError("Window step unit must be "
                             "either pixels or cells.")

    if method not in ['dalaltriggs', 'zhuramanan']:
        raise ValueError("Method must be either dalaltriggs or zhuramanan.")
    if num_orientations <= 0:
        raise ValueError("Number of orientation bins must be > 0.")
    if cell_size <= 0:
        raise ValueError("Cell size (in pixels) must be > 0.")
    if block_size <= 0:
        raise ValueError("Block size (in cells) must be > 0.")
    if l2_norm_clip <= 0.0:
        raise ValueError("Value for L2-norm clipping must be > 0.0.")

    options = np.zeros(15)
    options[0] = 1 if type is 'sparse' else 2
    options[1] = window_height
    options[2] = window_width
    options[3] = 1 if window_unit is 'blocks' else 2
    options[4] = window_step_horizontal
    options[5] = window_step_vertical
    options[6] = 1 if window_step_unit is 'cells' else 2
    options[7] = padding_disabled
    options[8] = 1 if method is 'dalaltriggs' else 2
    options[9] = num_orientations
    options[10] = cell_size
    options[11] = block_size
    options[12] = gradient_signed
    options[13] = l2_norm_clip
    options[14] = verbose

    return options
