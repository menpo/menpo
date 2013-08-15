from pybug.exceptions import DimensionalityError
from pybug.visualize.base import Viewer


class ViewerImage(Viewer):
    """ A viewer restricted to Image dimensional data.
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