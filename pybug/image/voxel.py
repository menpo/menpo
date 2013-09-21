from pybug.image import MaskedNDImage


class VoxelImage(MaskedNDImage):
    r"""
    Masked image which stores 3D volumetric data.

    Parameters
    ----------
    image_data: (X, Y, Z, K) ndarray
        X Y Z data with K channels of information per voxel.
    """
    def __init__(self, image_data, mask=None):
        super(VoxelImage, self).__init__(image_data, mask=mask)
        if self.n_dims != 3:
            raise ValueError("Trying to build a VoxelImage with {} channels -"
                             " you must provide a numpy array of size (X, Y,"
                             " Z, K), where K is the number of channels."
                             .format(self.n_channels))
