from pybug.image import MaskedNDImage


class VoxelImage(MaskedNDImage):

    def __init__(self, image_data, mask=None):
        super(VoxelImage, self).__init__(image_data, mask=mask)
        if self.n_dims != 3:
            raise ValueError("Trying to build a VoxelImage with {} channels -"
                             " you must provide a numpy array of size (X, Y,"
                             " Z, K), where K is the number of channels."
                             .format(self.n_channels))
