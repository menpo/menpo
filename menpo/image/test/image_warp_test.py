import numpy as np
import menpo
from numpy.testing import assert_allclose
from menpo.image import BooleanImage, Image, MaskedImage
from menpo.transform import Affine
import menpo.io as mio

# do the import to generate the expected outputs
rgb_image = mio.import_builtin_asset('takeo.ppm')
gray_image = rgb_image.as_greyscale()
gray_template = gray_image.crop(np.array([70, 30]),
                                np.array([169, 129]))
rgb_template = rgb_image.crop(np.array([70, 30]),
                              np.array([169, 129]))
template_mask = BooleanImage.blank(gray_template.shape)

initial_params = np.array([0, 0, 0, 0, 70, 30])
row_indices, col_indices = np.meshgrid(np.arange(50, 100), np.arange(50, 100),
                                       indexing='ij')
row_indices, col_indices = row_indices.flatten(), col_indices.flatten()
multi_expected = rgb_image.crop([50, 50],
                                [100, 100]).pixels.flatten()


def test_warp_gray():
    rgb_image = mio.import_builtin_asset('takeo.ppm')
    gray_image = rgb_image.as_greyscale()
    target_transform = Affine.identity(2).from_vector(initial_params)
    warped_im = gray_image.warp_to_mask(template_mask, target_transform)

    assert(warped_im.shape == gray_template.shape)
    assert_allclose(warped_im.pixels, gray_template.pixels)


def test_warp_multi():
    rgb_image = mio.import_builtin_asset('takeo.ppm')
    target_transform = Affine.identity(2).from_vector(initial_params)
    warped_im = rgb_image.warp_to_mask(template_mask, target_transform)

    assert(warped_im.shape == rgb_template.shape)
    assert_allclose(warped_im.pixels, rgb_template.pixels)


def test_warp_to_mask_boolean():
    b = BooleanImage.blank((10, 10))
    b.pixels[:, :5] = False
    template_mask = BooleanImage.blank((10, 10))
    template_mask.pixels[:5, :] = False
    t = Affine.identity(2)
    warped_mask = b.warp_to_mask(template_mask, t)
    assert(type(warped_mask) == BooleanImage)
    result = template_mask.pixels.copy()
    result[:, :5] = False
    assert(np.all(result == warped_mask.pixels))


def test_warp_to_mask_image():
    img = Image.blank((10, 10), n_channels=2)
    img.pixels[:, :5, :] = 0.5
    template_mask = BooleanImage.blank((10, 10))
    template_mask.pixels[5:, :] = False
    t = Affine.identity(2)
    warped_img = img.warp_to_mask(template_mask, t)
    assert(type(warped_img) == MaskedImage)
    result = Image.blank((10, 10), n_channels=2).pixels
    result[:5, :5, :] = 0.5
    assert(np.all(result == warped_img.pixels))


def test_warp_to_mask_masked_image():
    mask = BooleanImage.blank((10, 10))
    # make a funny mask on the original image
    mask.pixels[2:, :] = False
    img = MaskedImage.blank((10, 10), n_channels=2, mask=mask)
    img.pixels[...] = 2.5
    template_mask = BooleanImage.blank((10, 10), fill=False)
    template_mask.pixels[:5, :5] = True
    t = Affine.identity(2)
    warped_img = img.warp_to_mask(template_mask, t)
    assert(type(warped_img) == MaskedImage)
    result = Image.blank((10, 10), n_channels=2).pixels
    result[:5, :5, :] = 2.5
    result_mask = BooleanImage.blank((10, 10), fill=False).pixels
    result_mask[:2, :5] = True
    assert(warped_img.n_true_pixels() == 10)
    assert(np.all(result == warped_img.pixels))
    assert(np.all(result_mask == warped_img.mask.pixels))


def test_warp_to_shape_equal_warp_to_mask():
    r = menpo.transform.UniformScale(2.0, n_dims=2)
    b = mio.import_builtin_asset('breakingbad.jpg')
    m_shape = b.warp_to_shape((540, 960), r)
    m_mask = b.warp_to_mask(menpo.image.BooleanImage.blank((540, 960)), r)
    assert_allclose(m_shape.pixels, m_mask.pixels)


def test_rescale_boolean():
    mask = BooleanImage.blank((100, 100))
    mask.resize((10, 10))
