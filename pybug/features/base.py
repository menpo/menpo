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
# -> type : 'dense' or 'sparse', in the sparse case, the window is the whole image
# -> method : 'dalaltriggs' or 'zhuramanan', the computation method
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


def hog(image, type='dense', method='dalaltriggs', num_bins=9, cell_size=8,
        block_size=2, signed_gradient=True, l2_norm_clip=0.2,
        window_height=1, window_width=1, window_unit='blocks',
        window_step_vertical=1, window_step_horizontal=1,
        window_step_unit='pixels', padding=True, verbose=False):

    # Parse options
    if type not in ['dense', 'sparse']:
        raise ValueError("Type must be either dense or sparse.")
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
            raise ValueError("Window step unit must be either pixels or cells.")
    if method not in ['dalaltriggs', 'zhuramanan']:
        raise ValueError("Method must be either dalaltriggs or zhuramanan.")
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
        if method == 'dalaltriggs':
            image = image
        elif method == 'zhuramanan':
            image *= 255.
            image = np.tile(image, [1, 1, 3])

    # Dense case
    if type == 'dense':
        # Create iterator
        if method == 'dalaltriggs':
            method = 1
            if window_unit == 'blocks':
                block_in_pixels = cell_size * block_size
                window_height = np.uint32(window_height * block_in_pixels)
                window_width = np.uint32(window_width * block_in_pixels)
            if window_step_unit == 'cells':
                window_step_vertical = np.uint32(window_step_vertical * cell_size)
                window_step_horizontal = np.uint32(window_step_horizontal * cell_size)
        elif method == 'zhuramanan':
            method = 2
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
        output_image, windows_centers = iterator.HOG(method, num_bins, cell_size, block_size, signed_gradient,
                                                     l2_norm_clip, verbose)
        # Destroy iterator and return
        del iterator
        return np.ascontiguousarray(output_image), np.ascontiguousarray(windows_centers)
    # Sparse case
    elif type == 'sparse':
        # Create iterator
        if method == 'dalaltriggs':
            method = 1
            window_size = cell_size * block_size
            step = cell_size
        else:
            method = 2
            window_size = 3*cell_size
            step = cell_size
        iterator = CppImageWindowIterator(image, window_size, window_size, step, step, False)
        # Compute HOG
        output_image, windows_centers = iterator.HOG(method, num_bins, cell_size, block_size, signed_gradient,
                                                     l2_norm_clip, verbose)
        # Destroy iterator and return
        del iterator
        return np.ascontiguousarray(output_image), np.ascontiguousarray(windows_centers)


def view_hog(w):
    bs = 20
    w = w[:, :, 0:9]
    neg_w = -w
    scale = np.maximum(w.max(), neg_w.max())
    pos = _hog_picture(w, bs) * 255/scale
    neg = _hog_picture(-w, bs) * 255/scale

    # put pictures together and draw
    if w.min() < 0:
        im = np.uint8(np.concatenate((pos, neg)))
    else:
        im = np.uint8(pos)
    return im


# Make picture of positive HOG weights.
def _hog_picture(w, bs):
    # construct a "glyph" for each orientation
    bim1 = np.zeros((bs, bs))
    bim1[:, round(bs/2)-1:round(bs/2)+1] = 1
    bim = np.zeros((bim1.shape[0], bim1.shape[1], 9))
    bim[:, :, 0] = bim1
    for i in range(2, 10):
        bim[:, :, i-1] = imrotate(bim1, -(i-1)*20)

    # make pictures of positive weights bs adding up weighted glyphs
    s = w.shape
    w[w < 0] = 0
    im = np.zeros((bs*s[0], bs*s[1]))
    for i in range(1, s[0]+1):
        for j in range(1, s[1]+1):
            for k in range(1, 10):
                #print '%d:%d, %d:%d\n' % ((i-1)*bs-1, i*bs-1, (j-1)*bs-1, j*bs-1)
                im[(i-1)*bs:i*bs][:, (j-1)*bs:j*bs] = im[(i-1)*bs:i*bs][:, (j-1)*bs:j*bs] + bim[:, :, k-1] * w[i-1, j-1, k-1]
    return im