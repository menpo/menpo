

class Image(object):

    def __init__(self, image_data):
        self.width, self.height, self.n_channels = image_data.shape
        self.pixels = image_data