import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from nose.tools import raises
from menpo.testing import is_same_array
from menpo.image import BooleanImage, MaskedImage, Image


@raises(ValueError)
def test_create_1d_error():
    Image(np.ones(1))


def test_image_n_elements():
    image = Image(np.ones((10, 10, 3)))
    assert(image.n_elements == 10 * 10 * 3)


def test_image_width():
    image = Image(np.ones((6, 4, 3)))
    assert(image.width == 4)


def test_image_height():
    image = Image(np.ones((6, 4, 3)))
    assert(image.height == 6)


def test_image_blank():
    image = Image(np.zeros((6, 4, 1)))
    image_blank = Image.blank((6, 4))
    assert(np.all(image_blank.pixels == image.pixels))


def test_image_blank_fill():
    image = Image(np.ones((6, 4, 1)) * 7)
    image_blank = Image.blank((6, 4), fill=7)
    assert(np.all(image_blank.pixels == image.pixels))


def test_image_blank_n_channels():
    image = Image(np.zeros((6, 4, 7)))
    image_blank = Image.blank((6, 4), n_channels=7)
    assert(np.all(image_blank.pixels == image.pixels))


def test_image_centre():
    pixels = np.ones((10, 20, 1))
    image = Image(pixels)
    assert(np.all(image.centre == np.array([5, 10])))


def test_image_str_shape_4d():
    pixels = np.ones((10, 20, 11, 12, 1))
    image = Image(pixels)
    assert(image._str_shape == '10 x 20 x 11 x 12')


def test_image_str_shape_2d():
    pixels = np.ones((10, 20, 1))
    image = Image(pixels)
    assert(image._str_shape == '20W x 10H')


def test_image_as_vector():
    pixels = np.random.rand(10, 20, 1)
    image = Image(pixels)
    assert(np.all(image.as_vector() == pixels.ravel()))


def test_image_as_vector_keep_channels():
    pixels = np.random.rand(10, 20, 2)
    image = Image(pixels)
    assert(np.all(image.as_vector(keep_channels=True) ==
                  pixels.reshape([-1, 2])))


def test_image_from_vector():
    pixels = np.random.rand(10, 20, 2)
    pixels2 = np.random.rand(10, 20, 2)
    image = Image(pixels)
    image2 = image.from_vector(pixels2.ravel())
    assert(np.all(image2.pixels == pixels2))


def test_image_from_vector_custom_channels():
    pixels = np.random.rand(10, 20, 2)
    pixels2 = np.random.rand(10, 20, 3)
    image = Image(pixels)
    image2 = image.from_vector(pixels2.ravel(), n_channels=3)
    assert(np.all(image2.pixels == pixels2))


def test_image_from_vector_no_copy():
    pixels = np.random.rand(10, 20, 2)
    pixels2 = np.random.rand(10, 20, 2)
    image = Image(pixels)
    image2 = image.from_vector(pixels2.ravel(), copy=False)
    assert(is_same_array(image2.pixels, pixels2))


def test_image_from_vector_inplace_no_copy():
    pixels = np.random.rand(10, 20, 2)
    pixels2 = np.random.rand(10, 20, 2)
    image = Image(pixels)
    image.from_vector_inplace(pixels2.ravel(), copy=False)
    assert(is_same_array(image.pixels, pixels2))


def test_image_from_vector_inplace_no_copy_warning():
        pixels = np.random.rand(10, 20, 2)
        pixels2 = np.random.rand(10, 20, 2)
        image = Image(pixels)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            image.from_vector_inplace(pixels2.ravel()[::-1], copy=False)
            assert len(w) == 1


def test_image_from_vector_inplace_copy_default():
    pixels = np.random.rand(10, 20, 2)
    pixels2 = np.random.rand(10, 20, 2)
    image = Image(pixels)
    image.from_vector_inplace(pixels2.ravel())
    assert(not is_same_array(image.pixels, pixels2))


def test_image_from_vector_inplace_copy_explicit():
    pixels = np.random.rand(10, 20, 2)
    pixels2 = np.random.rand(10, 20, 2)
    image = Image(pixels)
    image.from_vector_inplace(pixels2.ravel(), copy=True)
    assert(not is_same_array(image.pixels, pixels2))


def test_image_from_vector_custom_channels_no_copy():
    pixels = np.random.rand(10, 20, 2)
    pixels2 = np.random.rand(10, 20, 3)
    image = Image(pixels)
    image2 = image.from_vector(pixels2.ravel(), n_channels=3, copy=False)
    assert(is_same_array(image2.pixels, pixels2))


@raises(ValueError)
def test_boolean_image_wrong_round():
    BooleanImage.blank((12, 12), round='ads')


def test_boolean_image_proportion_true():
    image = BooleanImage.blank((10, 10))
    image.pixels[:7] = False
    assert(image.proportion_true() == 0.3)


def test_boolean_image_proportion_false():
    image = BooleanImage.blank((10, 10))
    image.pixels[:7] = False
    assert(image.proportion_false() == 0.7)


def test_boolean_image_proportion_sums():
    image = BooleanImage.blank((10, 10))
    image.pixels[:7] = False
    assert(image.proportion_true() + image.proportion_false() == 1)


def test_boolean_image_false_indices():
    image = BooleanImage.blank((2, 3))
    image.pixels[0, 1] = False
    image.pixels[1, 2] = False
    assert(np.all(image.false_indices() == np.array([[0, 1],
                                                     [1, 2]])))


def test_boolean_image_str():
    image = BooleanImage.blank((2, 3))
    assert(image.__str__() == '3W x 2H 2D mask, 100.0% of which is True')


def test_boolean_image_from_vector():
    vector = np.zeros(16, dtype=np.bool)
    image = BooleanImage.blank((4, 4))
    image2 = image.from_vector(vector)
    assert(np.all(image2.as_vector() == vector))


def test_boolean_image_from_vector_no_copy():
    vector = np.zeros(16, dtype=np.bool)
    image = BooleanImage.blank((4, 4))
    image2 = image.from_vector(vector, copy=False)
    assert(is_same_array(image2.pixels.ravel(), vector))


def test_boolean_image_from_vector_no_copy_raises():
    vector = np.zeros(16, dtype=np.bool)
    image = BooleanImage.blank((4, 4))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        image.from_vector(vector[::-1], copy=False)
        assert len(w) == 1


def test_boolean_image_invert_inplace():
    image = BooleanImage.blank((4, 4))
    image.invert_inplace()
    assert(np.all(~image.pixels))


def test_boolean_image_invert_inplace_double_noop():
    image = BooleanImage.blank((4, 4))
    image.invert_inplace()
    image.invert_inplace()
    assert(np.all(image.pixels))


def test_boolean_image_invert():
    image = BooleanImage.blank((4, 4))
    image2 = image.invert()
    assert(np.all(image.pixels))
    assert(np.all(~image2.pixels))


def test_boolean_bounds_false():
    mask = BooleanImage.blank((8, 8), fill=True)
    mask.pixels[1, 2] = False
    mask.pixels[5, 4] = False
    mask.pixels[3:2, 3] = False
    min_b, max_b = mask.bounds_false()
    assert(np.all(min_b == np.array([1, 2])))
    assert(np.all(max_b == np.array([5, 4])))


@raises(TypeError)
def test_boolean_prevent_order_kwarg():
    mask = BooleanImage.blank((8, 8), fill=True)
    mask.warp_to_mask(mask, None, order=4)


def test_create_image_copy_false():
    pixels = np.ones((100, 100, 1))
    image = Image(pixels, copy=False)
    assert (is_same_array(image.pixels, pixels))


def test_create_image_copy_true():
    pixels = np.ones((100, 100, 1))
    image = Image(pixels)
    assert (not is_same_array(image.pixels, pixels))


def test_create_image_copy_false_not_c_contiguous():
    pixels = np.ones((100, 100, 1), order='F')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Image(pixels, copy=False)
        assert(len(w) == 1)


def mask_image_3d_test():
    mask_shape = (120, 121, 13)
    mask_region = np.ones(mask_shape)
    return BooleanImage(mask_region)


def test_mask_creation_basics():
    mask_shape = (120, 121, 3)
    mask_region = np.ones(mask_shape)
    mask = BooleanImage(mask_region)
    assert_equal(mask.n_channels, 1)
    assert_equal(mask.n_dims, 3)
    assert_equal(mask.shape, mask_shape)


def test_mask_blank():
    mask = BooleanImage.blank((56, 12, 3))
    assert (np.all(mask.pixels))


def test_boolean_copy_false_boolean():
    mask = np.zeros((10, 10), dtype=np.bool)
    boolean_image = BooleanImage(mask, copy=False)
    assert (is_same_array(boolean_image.pixels, mask))


def test_boolean_copy_true():
    mask = np.zeros((10, 10), dtype=np.bool)
    boolean_image = BooleanImage(mask)
    assert (not is_same_array(boolean_image.pixels, mask))


def test_boolean_copy_false_non_boolean():
    mask = np.zeros((10, 10))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BooleanImage(mask, copy=False)
        assert(len(w) == 1)


def test_mask_blank_rounding_floor():
    mask = BooleanImage.blank((56.1, 12.1), round='floor')
    assert_allclose(mask.shape, (56, 12))


def test_mask_blank_rounding_ceil():
    mask = BooleanImage.blank((56.1, 12.1), round='ceil')
    assert_allclose(mask.shape, (57, 13))


def test_mask_blank_rounding_round():
    mask = BooleanImage.blank((56.1, 12.6), round='round')
    assert_allclose(mask.shape, (56, 13))


def test_mask_blank_false_fill():
    mask = BooleanImage.blank((56, 12, 3), fill=False)
    assert (np.all(~mask.pixels))


def test_mask_n_true_n_false():
    mask = BooleanImage.blank((64, 14), fill=False)
    assert_equal(mask.n_true(), 0)
    assert_equal(mask.n_false(), 64 * 14)
    mask.mask[0, 0] = True
    mask.mask[9, 13] = True
    assert_equal(mask.n_true(), 2)
    assert_equal(mask.n_false(), 64 * 14 - 2)


def test_mask_true_indices():
    mask = BooleanImage.blank((64, 14, 51), fill=False)
    mask.mask[0, 2, 5] = True
    mask.mask[5, 13, 4] = True
    true_indices = mask.true_indices()
    true_indices_test = np.array([[0, 2, 5], [5, 13, 4]])
    assert_equal(true_indices, true_indices_test)


def test_mask_false_indices():
    mask = BooleanImage.blank((64, 14, 51), fill=True)
    mask.mask[0, 2, 5] = False
    mask.mask[5, 13, 4] = False
    false_indices = mask.false_indices()
    false_indices_test = np.array([[0, 2, 5], [5, 13, 4]])
    assert_equal(false_indices, false_indices_test)


def test_mask_true_bounding_extent():
    mask = BooleanImage.blank((64, 14, 51), fill=False)
    mask.mask[0, 13, 5] = True
    mask.mask[5, 2, 4] = True
    tbe = mask.bounds_true()
    true_extends_mins = np.array([0, 2, 4])
    true_extends_maxs = np.array([5, 13, 5])
    assert_equal(tbe[0], true_extends_mins)
    assert_equal(tbe[1], true_extends_maxs)


def test_3channel_image_creation():
    pixels = np.ones((120, 120, 3))
    MaskedImage(pixels)


def test_no_channels_image_creation():
    pixels = np.ones((120, 120))
    MaskedImage(pixels)


def test_create_MaskedImage_copy_false_mask_array():
    pixels = np.ones((100, 100, 1))
    mask = np.ones((100, 100), dtype=np.bool)
    image = MaskedImage(pixels, mask=mask, copy=False)
    assert (is_same_array(image.pixels, pixels))
    assert (is_same_array(image.mask.pixels, mask))


def test_create_MaskedImage_copy_false_mask_BooleanImage():
    pixels = np.ones((100, 100, 1))
    mask = np.ones((100, 100), dtype=np.bool)
    mask_image = BooleanImage(mask, copy=False)
    image = MaskedImage(pixels, mask=mask_image, copy=False)
    assert (is_same_array(image.pixels, pixels))
    assert (is_same_array(image.mask.pixels, mask))


def test_create_MaskedImage_copy_true_mask_array():
    pixels = np.ones((100, 100))
    mask = np.ones((100, 100), dtype=np.bool)
    image = MaskedImage(pixels, mask=mask)
    assert (not is_same_array(image.pixels, pixels))
    assert (not is_same_array(image.mask.pixels, mask))


def test_create_MaskedImage_copy_true_mask_BooleanImage():
    pixels = np.ones((100, 100, 1))
    mask = np.ones((100, 100), dtype=np.bool)
    mask_image = BooleanImage(mask, copy=False)
    image = MaskedImage(pixels, mask=mask_image, copy=True)
    assert (not is_same_array(image.pixels, pixels))
    assert (not is_same_array(image.mask.pixels, mask))


def test_2d_crop_without_mask():
    pixels = np.ones((120, 120, 3))
    im = MaskedImage(pixels)

    cropped_im = im.crop([10, 50], [20, 60])

    assert (cropped_im.shape == (10, 10))
    assert (cropped_im.n_channels == 3)
    assert (np.alltrue(cropped_im.shape))


def test_2d_crop_with_mask():
    pixels = np.ones((120, 120, 3))
    mask = np.zeros_like(pixels[..., 0])
    mask[10:100, 20:30] = 1
    im = MaskedImage(pixels, mask=mask)
    cropped_im = im.crop([0, 0], [20, 60])
    assert (cropped_im.shape == (20, 60))
    assert (np.alltrue(cropped_im.shape))


def test_normalize_std_image():
    pixels = np.ones((120, 120, 3))
    pixels[..., 0] = 0.5
    pixels[..., 1] = 0.2345
    image = Image(pixels)
    image.normalize_std_inplace()
    assert_allclose(np.mean(image.pixels), 0, atol=1e-10)
    assert_allclose(np.std(image.pixels), 1)


def test_normalize_norm_image():
    pixels = np.ones((120, 120, 3))
    pixels[..., 0] = 0.5
    pixels[..., 1] = 0.2345
    image = Image(pixels)
    image.normalize_norm_inplace()
    assert_allclose(np.mean(image.pixels), 0, atol=1e-10)
    assert_allclose(np.linalg.norm(image.pixels), 1)


@raises(ValueError)
def test_normalize_std_no_variance_exception():
    pixels = np.ones((120, 120, 3))
    pixels[..., 0] = 0.5
    pixels[..., 1] = 0.2345
    image = MaskedImage(pixels)
    image.normalize_std_inplace(mode='per_channel')


@raises(ValueError)
def test_normalize_norm_zero_norm_exception():
    pixels = np.zeros((120, 120, 3))
    image = MaskedImage(pixels)
    image.normalize_norm_inplace(mode='per_channel')


def test_normalize_std_masked_per_channel():
    pixels = np.random.randn(120, 120, 3)
    pixels[..., 1] *= 7
    pixels[..., 0] += -14
    pixels[..., 2] /= 130
    image = MaskedImage(pixels)
    image.normalize_std_inplace(mode='per_channel')
    assert_allclose(
        np.mean(image.as_vector(keep_channels=True), axis=0), 0, atol=1e-10)
    assert_allclose(
        np.std(image.as_vector(keep_channels=True), axis=0), 1)


def test_normalize_std_image_per_channel():
    pixels = np.random.randn(120, 120, 3)
    pixels[..., 1] *= 9
    pixels[..., 0] += -3
    pixels[..., 2] /= 140
    image = Image(pixels)
    image.normalize_std_inplace(mode='per_channel')
    assert_allclose(
        np.mean(image.as_vector(keep_channels=True), axis=0), 0, atol=1e-10)
    assert_allclose(
        np.std(image.as_vector(keep_channels=True), axis=0), 1)


def test_normalize_norm_image_per_channel():
    pixels = np.random.randn(120, 120, 3)
    pixels[..., 1] *= 17
    pixels[..., 0] += -114
    pixels[..., 2] /= 30
    image = Image(pixels)
    image.normalize_norm_inplace(mode='per_channel')
    assert_allclose(
        np.mean(image.as_vector(keep_channels=True), axis=0), 0, atol=1e-10)
    assert_allclose(
        np.linalg.norm(image.as_vector(keep_channels=True), axis=0), 1)


def test_normalize_norm_masked_per_channel():
    pixels = np.random.randn(120, 120, 3)
    pixels[..., 1] *= 7
    pixels[..., 0] += -14
    pixels[..., 2] /= 130
    image = MaskedImage(pixels)
    image.normalize_norm_inplace(mode='per_channel')
    assert_allclose(
        np.mean(image.as_vector(keep_channels=True), axis=0), 0, atol=1e-10)
    assert_allclose(
        np.linalg.norm(image.as_vector(keep_channels=True), axis=0), 1)


def test_normalize_std_masked():
    pixels = np.random.randn(120, 120, 3)
    pixels[..., 1] *= 7
    pixels[..., 0] += -14
    pixels[..., 2] /= 130
    mask = np.zeros((120, 120))
    mask[30:50, 20:30] = 1
    image = MaskedImage(pixels, mask=mask)
    image.normalize_std_inplace(mode='per_channel', limit_to_mask=True)
    assert_allclose(
        np.mean(image.as_vector(keep_channels=True), axis=0), 0, atol=1e-10)
    assert_allclose(
        np.std(image.as_vector(keep_channels=True), axis=0), 1)


def test_normalize_norm_masked():
    pixels = np.random.randn(120, 120, 3)
    pixels[..., 1] *= 7
    pixels[..., 0] += -14
    pixels[..., 2] /= 130
    mask = np.zeros((120, 120))
    mask[30:50, 20:30] = 1
    image = MaskedImage(pixels, mask=mask)
    image.normalize_norm_inplace(mode='per_channel', limit_to_mask=True)
    assert_allclose(
        np.mean(image.as_vector(keep_channels=True), axis=0), 0, atol=1e-10)
    assert_allclose(
        np.linalg.norm(image.as_vector(keep_channels=True), axis=0), 1)


def test_rescale_single_num():
    image = MaskedImage(np.random.randn(120, 120, 3))
    new_image = image.rescale(0.5)
    assert_allclose(new_image.shape, (60, 60))


def test_rescale_tuple():
    image = MaskedImage(np.random.randn(120, 120, 3))
    new_image = image.rescale([0.5, 2.0])
    assert_allclose(new_image.shape, (60, 240))


@raises(ValueError)
def test_rescale_negative():
    image = MaskedImage(np.random.randn(120, 120, 3))
    image.rescale([0.5, -0.5])


@raises(ValueError)
def test_rescale_negative_single_num():
    image = MaskedImage(np.random.randn(120, 120, 3))
    image.rescale(-0.5)


def test_rescale_boundaries_interpolation():
    image = MaskedImage(np.random.randn(60, 60, 3))
    for i in [x * 0.1 for x in range(1, 31)]:
        image_rescaled = image.rescale(i)
        assert_allclose(image_rescaled.mask.proportion_true(), 1.0)


def test_resize():
    image = MaskedImage(np.random.randn(120, 120, 3))
    new_size = (250, 250)
    new_image = image.resize(new_size)
    assert_allclose(new_image.shape, new_size)


def test_as_greyscale_luminosity():
    image = MaskedImage(np.ones([120, 120, 3]))
    new_image = image.as_greyscale(mode='luminosity')
    assert (new_image.shape == image.shape)
    assert (new_image.n_channels == 1)


def test_as_greyscale_average():
    image = MaskedImage(np.ones([120, 120, 3]))
    new_image = image.as_greyscale(mode='average')
    assert (new_image.shape == image.shape)
    assert (new_image.n_channels == 1)


@raises(ValueError)
def test_as_greyscale_channels_no_index():
    image = MaskedImage(np.ones([120, 120, 3]))
    new_image = image.as_greyscale(mode='channel')
    assert (new_image.shape == image.shape)
    assert (new_image.n_channels == 1)


def test_as_greyscale_channels():
    image = MaskedImage(np.random.randn(120, 120, 3))
    new_image = image.as_greyscale(mode='channel', channel=0)
    assert (new_image.shape == image.shape)
    assert (new_image.n_channels == 1)
    assert_allclose(new_image.pixels[..., 0], image.pixels[..., 0])


def test_as_pil_image_1channel():
    im = MaskedImage(np.ones((120, 120, 1)))
    new_im = im.as_PILImage()
    assert_allclose(np.asarray(new_im.getdata()).reshape(im.pixels.shape),
                    (im.pixels * 255).astype(np.uint8))


@raises(ValueError)
def test_as_pil_image_bad_range():
    im = MaskedImage(np.random.randn(120, 120, 1))
    im.as_PILImage()


def test_as_pil_image_float32():
    im = MaskedImage(np.ones((120, 120, 1)).astype(np.float32))
    new_im = im.as_PILImage()
    assert_allclose(np.asarray(new_im.getdata()).reshape(im.pixels.shape),
                    (im.pixels * 255).astype(np.uint8))


def test_as_pil_image_bool():
    im = BooleanImage(np.ones((120, 120), dtype=np.bool))
    new_im = im.as_PILImage()
    assert_allclose(np.asarray(new_im.getdata()).reshape(im.pixels.shape),
                    im.pixels.astype(np.uint8) * 255)


def test_as_pil_image_uint8():
    im = Image(np.ones((120, 120), dtype=np.uint8))
    new_im = im.as_PILImage()
    assert_allclose(np.asarray(new_im.getdata()).reshape(im.pixels.shape),
                    im.pixels)


def test_as_pil_image_3channels():
    im = MaskedImage(np.ones((120, 120, 3)))
    new_im = im.as_PILImage()
    assert_allclose(np.asarray(new_im.getdata()).reshape(im.pixels.shape),
                    (im.pixels * 255).astype(np.uint8))


def test_image_gradient_sanity():
    # Only a sanity check - does it run and generate sensible output?
    image = Image(np.zeros([120, 120, 3]))
    new_image = image.gradient()
    assert(type(new_image) == Image)
    assert(new_image.shape == image.shape)
    assert(new_image.n_channels == image.n_channels * 2)


def test_image_extract_channels():
    image = Image(np.random.rand(120, 120, 3))
    extracted = image.extract_channels(0)
    assert_equal(extracted.pixels, image.pixels[..., [0]])


def test_image_extract_channels_multiple():
    image = Image(np.random.rand(120, 120, 3))
    extracted = image.extract_channels([0, 2])
    assert_equal(extracted.pixels[..., 0], image.pixels[..., 0])
    assert_equal(extracted.pixels[..., 1], image.pixels[..., 2])


def test_image_extract_channels_multiple_reversed():
    image = Image(np.random.rand(120, 120, 3))
    extracted = image.extract_channels([2, 0])
    assert_equal(extracted.pixels[..., 0], image.pixels[..., 2])
    assert_equal(extracted.pixels[..., 1], image.pixels[..., 0])


def test_diagonal_greyscale():
    image = Image.blank((100, 250), n_channels=1)
    assert image.diagonal == (100 ** 2 + 250 ** 2) ** 0.5


def test_diagonal_color():
    image = Image.blank((100, 250), n_channels=3)
    assert image.diagonal == (100 ** 2 + 250 ** 2) ** 0.5


def test_diagonal_greyscale_ndim():
    image = Image.blank((100, 250, 50), n_channels=1)
    assert image.diagonal == (100 ** 2 + 250 ** 2 + 50 ** 2) ** 0.5


def test_diagonal_kchannel_ndim():
    image = Image.blank((100, 250, 50), n_channels=5)
    assert image.diagonal == (100 ** 2 + 250 ** 2 + 50 ** 2) ** 0.5


def test_rescale_to_diagonal():
    image = Image.blank((8, 6), n_channels=2)
    assert image.diagonal == 10
    rescaled = image.rescale_to_diagonal(5)
    assert rescaled.shape == (4, 3)
    assert rescaled.n_channels == 2

