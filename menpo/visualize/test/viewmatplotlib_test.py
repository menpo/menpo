import matplotlib

matplotlib.use("Agg")

import pytest
from menpo.visualize.viewmatplotlib import _parse_cmap
import menpo.io as mio


def test_parse_cmap_by_name():
    # cm.get_cmap, which this used, was removed in Matplotlib 3.9
    assert _parse_cmap("viridis").name == "viridis"


def test_parse_cmap_default_is_gray_for_one_channel():
    assert _parse_cmap(None, image_shape_len=2).name == "gray"
    assert _parse_cmap(None, image_shape_len=3) is None


def test_unknown_cmap_raises():
    with pytest.raises(ValueError):
        _parse_cmap("not-a-colour-map")


@pytest.mark.parametrize("cmap", ["viridis", "hot", "gray"])
def test_view_image_with_a_colour_map(cmap):
    image = mio.import_builtin_asset.takeo_ppm().as_greyscale()
    image.view(cmap_name=cmap)
