import menpo


def test_image_gaussian_pyramid_n_levels():
    lenna = menpo.io.import_builtin_asset.lenna_png()
    assert len(list(lenna.gaussian_pyramid(n_levels=4))) == 4


def test_image_gaussian_pyramid_one_level():
    lenna = menpo.io.import_builtin_asset.lenna_png()
    assert len(list(lenna.gaussian_pyramid(n_levels=1))) == 1


def test_image_gaussian_pyramid_shapes():
    lenna = menpo.io.import_builtin_asset.lenna_png()
    shapes = [(512, 512), (256, 256), (128, 128), (64, 64)]
    for l, expected_shape in zip(lenna.gaussian_pyramid(n_levels=4), shapes):
        assert l.shape == expected_shape


def test_image_pyramid_n_levels():
    lenna = menpo.io.import_builtin_asset.lenna_png()
    assert len(list(lenna.pyramid(n_levels=4))) == 4


def test_image_pyramid_one_level():
    lenna = menpo.io.import_builtin_asset.lenna_png()
    assert len(list(lenna.pyramid(n_levels=1))) == 1


def test_image_pyramid_shapes():
    lenna = menpo.io.import_builtin_asset.lenna_png()
    shapes = [(512, 512), (128, 128), (32, 32)]
    for l, expected_shape in zip(lenna.pyramid(n_levels=3, downscale=4), shapes):
        assert l.shape == expected_shape
