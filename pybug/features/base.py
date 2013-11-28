from pybug.features.cppimagewindowiterator import CppImageWindowIterator
import numpy as np
from scipy.misc import imrotate

# HOG Features
# hog function first creates an iterator object and then applies the hog computation
#
# Window-Related Options:
# -> image : input image
# -> window_height, window_width : size of the window
# -> window_unit : 'pixels' or 'blocks', the metric unit of window_height, window_width
# -> window_step_vertical, window_step_horizontal : the sampling step of the window (image down-sampling factor)
# -> window_step_unit : 'pixels' or 'cells' : the metric unit of window_step_vertical, window_step_horizontal
# -> padding : boolean to enable or disable padding
#
# HOG-Related Options:
# -> method : 'dense' or 'sparse', in the sparse case, the window is the whole image
# -> algorithm : 'dalaltriggs' or 'zhuramanan', the computation method
# -> num_bins : the number of orientation bins
# -> cell_size : the height and width of the rectangular cell in pixels
# -> block_size : the height and width of the rectangular block
# -> signed_gradient : boolean for use of signed or unsigned gradients
# -> l2_norm_clip : the clipping value of L2-norm
#
# General Options:
# -> verbose : boolean to print information
#
# In the DENSE type all options have an effect.
# In the SPARSE type, all the Window-Related options have no effect.
#
# TO-DO:
# -> Maybe we should remove the type option, since the classic sparse hog can be easily obtained from the dense hog
#


def hog(image, method='dense', algorithm='dalaltriggs', num_bins=9, cell_size=8,
        block_size=2, signed_gradient=True, l2_norm_clip=0.2,
        window_height=1, window_width=1, window_unit='blocks',
        window_step_vertical=1, window_step_horizontal=1,
        window_step_unit='pixels', padding=True, verbose=False):

    # Parse options
    if method not in ['dense', 'sparse']:
        raise ValueError("Method must be either dense or sparse.")
    if method is 'dense':
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
            raise ValueError("Window step unit must be either pixels or cells.")
    if algorithm not in ['dalaltriggs', 'zhuramanan']:
        raise ValueError("Algorithm must be either dalaltriggs or zhuramanan.")
    if num_bins <= 0:
        raise ValueError("Number of orientation bins must be > 0.")
    if cell_size <= 0:
        raise ValueError("Cell size (in pixels) must be > 0.")
    if block_size <= 0:
        raise ValueError("Block size (in cells) must be > 0.")
    if l2_norm_clip <= 0.0:
        raise ValueError("Value for L2-norm clipping must be > 0.0.")

    # Correct input image
    image = np.asfortranarray(image)
    if image.shape[2] == 3:
        image *= 255.
    elif image.shape[2] == 1:
        if algorithm == 'dalaltriggs':
            image = image
        elif algorithm == 'zhuramanan':
            image *= 255.
            image = np.tile(image, [1, 1, 3])

    # Dense case
    if method == 'dense':
        # Create iterator
        if algorithm == 'dalaltriggs':
            algorithm = 1
            if window_unit == 'blocks':
                block_in_pixels = cell_size * block_size
                window_height = np.uint32(window_height * block_in_pixels)
                window_width = np.uint32(window_width * block_in_pixels)
            if window_step_unit == 'cells':
                window_step_vertical = np.uint32(window_step_vertical * cell_size)
                window_step_horizontal = np.uint32(window_step_horizontal * cell_size)
        elif algorithm == 'zhuramanan':
            algorithm = 2
            if window_unit == 'blocks':
                block_in_pixels = 3*cell_size
                window_height = np.uint32(window_height * block_in_pixels)
                window_width = np.uint32(window_width * block_in_pixels)
            if window_step_unit == 'cells':
                window_step_vertical = np.uint32(window_step_vertical * cell_size)
                window_step_horizontal = np.uint32(window_step_horizontal * cell_size)
        iterator = CppImageWindowIterator(image, window_height, window_width, window_step_horizontal,
                                          window_step_vertical, padding)
        # Print iterator's info
        if verbose:
            print iterator
        # Compute HOG
        output_image, windows_centers = iterator.HOG(algorithm, num_bins, cell_size, block_size, signed_gradient,
                                                     l2_norm_clip, verbose)
        # Destroy iterator and return
        del iterator
        return np.ascontiguousarray(output_image), np.ascontiguousarray(windows_centers)
    # Sparse case
    elif method == 'sparse':
        # Create iterator
        if algorithm == 'dalaltriggs':
            algorithm = 1
            window_size = cell_size * block_size
            step = cell_size
        else:
            algorithm = 2
            window_size = 3*cell_size
            step = cell_size
        iterator = CppImageWindowIterator(image, window_size, window_size, step, step, False)
        # Compute HOG
        output_image, windows_centers = iterator.HOG(algorithm, num_bins, cell_size, block_size, signed_gradient,
                                                     l2_norm_clip, verbose)
        # Destroy iterator and return
        del iterator
        return np.ascontiguousarray(output_image), np.ascontiguousarray(windows_centers)


def view_hog(weights, block_size=10, num_bins=9):
    weights = weights[:, :, 0:num_bins]
    negative_weights = -weights
    scale = np.maximum(weights.max(), negative_weights.max())
    pos = _hog_picture(weights, block_size, num_bins) * 255/scale
    neg = _hog_picture(-weights, block_size, num_bins) * 255/scale
    if weights.min() < 0:
        hog_image = np.concatenate((pos, neg))
    else:
        hog_image = pos
    return hog_image


def _hog_picture(weights, block_size, num_bins):
    # construct a "glyph" for each orientation
    block_image_temp = np.zeros((block_size, block_size))
    block_image_temp[:, round(block_size/2)-1:round(block_size/2)+1] = 1
    block_image = np.zeros((block_image_temp.shape[0], block_image_temp.shape[1], num_bins))
    block_image[:, :, 0] = block_image_temp
    for i in range(2, num_bins+1):
        block_image[:, :, i-1] = imrotate(block_image_temp, -(i-1)*block_size)
    # make pictures of positive weights by adding up weighted glyphs
    s = weights.shape
    weights[weights < 0] = 0
    hog_picture = np.zeros((block_size*s[0], block_size*s[1]))
    for i in range(1, s[0]+1):
        for j in range(1, s[1]+1):
            for k in range(1, 10):
                hog_picture[(i-1)*block_size:i*block_size][:, (j-1)*block_size:j*block_size] = \
                    hog_picture[(i-1)*block_size:i*block_size][:, (j-1)*block_size:j*block_size] + \
                    block_image[:, :, k-1] * weights[i-1, j-1, k-1]
    return hog_picture