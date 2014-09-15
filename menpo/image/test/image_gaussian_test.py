import menpo


def test_image_gaussian_pyramid_n_levels():
    lenna = menpo.io.import_builtin_asset.lenna_png()
    assert len(list(lenna.gaussian_pyramid(n_levels=4))) == 4


def test_image_gaussian_pyramid_one_level():
    lenna = menpo.io.import_builtin_asset.lenna_png()
    assert len(list(lenna.gaussian_pyramid(n_levels=1))) == 1
