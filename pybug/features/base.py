from pybug.features.cppimagewindowiterator import CppImageWindowIterator
import numpy as np


def hog(image_data, mode='dense', algorithm='dalaltriggs', num_bins=9, cell_size=8,
        block_size=2, signed_gradient=True, l2_norm_clip=0.2,
        window_height=1, window_width=1, window_unit='blocks',
        window_step_vertical=1, window_step_horizontal=1,
        window_step_unit='pixels', padding=True, verbose=False):
    r"""
    Computes a 2-dimensional HOG features image with k number of channels, of
    size ``(M, N, C)`` and data type ``np.float``.

    Parameters
    ----------
    image_data :  ndarray
        The pixel data for the image, where the last axis represents the
        number of channels.
    mask : (M, N) ``np.bool`` ndarray or :class:`BooleanNDImage`, optional
        A binary array representing the mask. Must be the same
        shape as the image. Only one mask is supported for an image (so the
        mask is applied to every channel equally).

        Default: :class:`BooleanNDImage` covering the whole image
    mode : 'dense' or 'sparse'
        The 'sparse' case refers to the traditional usage of HOGs, so default
        parameters values are passed to the ImageWindowIterator. The sparse
        case of 'dalaltriggs' algorithm sets the window height and width equal
        to block size and the window step horizontal and vertical equal to cell
        size. Thse sparse case of 'zhuramanan' algorithm sets the window height
        and width equal to 3 times the cell size and the window step horizontal
        and vertical equal to cell size. In the 'dense' case, the user can
        change the ImageWindowIterator related parameters (window_height,
        window_width, window_unit, window_step_vertical,
        window_step_horizontal, window_step_unit, padding).

        Default: 'dense'
    window_height : float
        Defines the height of the window for the ImageWindowIterator object.
        The metric unit is defined by window_unit.

        Default: 1
    window_width : float
        Defines the width of the window for the ImageWindowIterator object.
        The metric unit is defined by window_unit.

        Default: 1
    window_unit : 'blocks' or 'pixels'
        Defines the metric unit of the window_height and window_width
        parameters for the ImageWindowIterator object.

        Default: 'blocks'
    window_step_vertical : float
        Defines the vertical step by which the window in the
        ImageWindowIterator is moved, thus it controls the features density.
        The metric unit is defined by window_step_unit.

        Default: 1
    window_step_horizontal : float
        Defines the horizontal step by which the window in the
        ImageWindowIterator is moved, thus it controls the features density.
        The metric unit is defined by window_step_unit.

        Default: 1
    window_step_unit : 'pixels' or 'cells'
        Defines the metric unit of the window_step_vertical and
        window_step_horizontal parameters for the ImageWindowIterator object.

        Default: 'pixels'
    padding : bool
        Enables/disables padding for the close-to-boundary windows in the
        ImageWindowIterator object. When padding is enabled, the
        out-of-boundary pixels are set to zero.

        Default: True
    algorithm : 'dalaltriggs' or 'zhuramanan'
        Specifies the algorithm used to compute HOGs.

        Default: 'dalaltriggs'
    cell_size : float
        Defines the cell size in pixels. This value is set to both the width
        and height of the cell. This option is valid for both algorithms.

        Default: 8
    block_size : float
        Defines the block size in cells. This value is set to both the width
        and height of the block. This option is valid only for the
        'dalaltriggs' algorithm.

        Default: 2
    num_bins : float
        Defines the number of orientation histogram bins. This option is valid
        only for the 'dalaltriggs' algorithm.

        Default: 9
    signed_gradient : bool
        Flag that defines whether we use signed or unsigned gradient angles.
        This option is valid only for the 'dalaltriggs' algorithm.

        Default: True
    l2_norm_clip : float
        Defines the clipping value of the gradients' L2-norm. This option is
        valid only for the 'dalaltriggs' algorithm.

        Default: 0.2
    verbose : bool
        Flag to print HOG related information.

        Default: False

    Raises
    -------
    ValueError
        HOG features mode must be either dense or sparse
    ValueError
        Algorithm must be either dalaltriggs or zhuramanan
    ValueError
        Number of orientation bins must be > 0
    ValueError
        Cell size (in pixels) must be > 0
    ValueError
        Block size (in cells) must be > 0
    ValueError
        Value for L2-norm clipping must be > 0.0
    ValueError
        Window height must be >= block size and <= image height
    ValueError
        Window width must be >= block size and <= image width
    ValueError
        Window unit must be either pixels or blocks
    ValueError
        Horizontal window step must be > 0
    ValueError
        Vertical window step must be > 0
    ValueError
        Window step unit must be either pixels or cells
    """
    # Parse options
    if mode not in ['dense', 'sparse']:
        raise ValueError("HOG features mode must be either dense or sparse")
    if algorithm not in ['dalaltriggs', 'zhuramanan']:
        raise ValueError("Algorithm must be either dalaltriggs or zhuramanan")
    if num_bins <= 0:
        raise ValueError("Number of orientation bins must be > 0")
    if cell_size <= 0:
        raise ValueError("Cell size (in pixels) must be > 0")
    if block_size <= 0:
        raise ValueError("Block size (in cells) must be > 0")
    if l2_norm_clip <= 0.0:
        raise ValueError("Value for L2-norm clipping must be > 0.0")
    if mode is 'dense':
        if window_unit not in ['pixels', 'blocks']:
            raise ValueError("Window unit must be either pixels or blocks")
        window_height_temp = window_height
        window_width_temp = window_width
        if window_unit == 'blocks':
            window_height_temp = window_height*block_size*cell_size
            window_width_temp = window_width*block_size*cell_size
        if window_height_temp < block_size*cell_size or \
           window_height_temp > image_data.shape[0]:
            raise ValueError("Window height must be >= block size and <= "
                             "image height")
        if window_width_temp < block_size*cell_size or \
           window_width_temp > image_data.shape[1]:
            raise ValueError("Window width must be >= block size and <= "
                             "image width")
        if window_step_horizontal <= 0:
            raise ValueError("Horizontal window step must be > 0")
        if window_step_vertical <= 0:
            raise ValueError("Vertical window step must be > 0")
        if window_step_unit not in ['pixels', 'cells']:
            raise ValueError("Window step unit must be either pixels or cells")

    # Correct input image_data
    image_data = np.asfortranarray(image_data)
    if image_data.shape[2] == 3:
        image_data *= 255.
    elif image_data.shape[2] == 1:
        if algorithm == 'dalaltriggs':
            image_data = image_data
        elif algorithm == 'zhuramanan':
            image_data *= 255.
            image_data = np.tile(image_data, [1, 1, 3])

    # Dense case
    if mode == 'dense':
        # Iterator parameters
        if algorithm == 'dalaltriggs':
            algorithm = 1
            if window_unit == 'blocks':
                block_in_pixels = cell_size * block_size
                window_height = np.uint32(window_height * block_in_pixels)
                window_width = np.uint32(window_width * block_in_pixels)
            if window_step_unit == 'cells':
                window_step_vertical = np.uint32(window_step_vertical *
                                                 cell_size)
                window_step_horizontal = np.uint32(window_step_horizontal *
                                                   cell_size)
        elif algorithm == 'zhuramanan':
            algorithm = 2
            if window_unit == 'blocks':
                block_in_pixels = 3 * cell_size
                window_height = np.uint32(window_height * block_in_pixels)
                window_width = np.uint32(window_width * block_in_pixels)
            if window_step_unit == 'cells':
                window_step_vertical = np.uint32(window_step_vertical *
                                                 cell_size)
                window_step_horizontal = np.uint32(window_step_horizontal *
                                                   cell_size)
        iterator = CppImageWindowIterator(image_data, window_height,
                                          window_width, window_step_horizontal,
                                          window_step_vertical, padding)
    # Sparse case
    else:
        # Create iterator
        if algorithm == 'dalaltriggs':
            algorithm = 1
            window_size = cell_size * block_size
            step = cell_size
        else:
            algorithm = 2
            window_size = 3*cell_size
            step = cell_size
        iterator = CppImageWindowIterator(image_data, window_size, window_size,
                                          step, step, False)
    # Print iterator's info
    if verbose:
        print iterator
    # Compute HOG
    output_image, windows_centers = iterator.HOG(algorithm, num_bins,
                                                 cell_size, block_size,
                                                 signed_gradient, l2_norm_clip,
                                                 verbose)
    # Destroy iterator and return
    del iterator
    return np.ascontiguousarray(output_image), np.ascontiguousarray(
        windows_centers)