from pybug.exceptions import DimensionalityError
from pybug.visualize.base import Viewer


class ViewerImage(Viewer):
    r"""
    A viewer restricted to 2 dimensional image data. The image can have an
    optional number of channels (RGB images). 1 channel images are rendered
    in grayscale.

    Parameters
    ----------
    image : (M, N) ndarray
        A 2D set of pixels in the range ``[0, 1]``.

    Raises
    ------
    DimensionalityError
        Only 2D images are supported.
    """

    def __init__(self, image):
        Viewer.__init__(self)
        dim = len(image.shape)
        if dim not in [2, 3]:
            raise DimensionalityError("Expected a 2-dimensional Image with "
                                      "optional channels"
                                      "but got a {0} object"
                                      .format(str(image.shape)))
        self.image = image


class ImageViewer2d(ViewerImage):

    def __init__(self, image):
        ViewerImage.__init__(self, image)