import numpy as np
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

    @classmethod
    def blank(cls, shape, fill=0, dtype=np.float, mask=None):
        r"""
        Returns a blank Voxel

        Parameters
        ----------
        shape : tuple or list
            The shape of the image

        fill : int, optional
            The value to fill all pixels with

            Default: 0
        dtype: numpy datatype, optional
            The datatype of the image.

            Default: np.float
        mask: (M, N) boolean ndarray or :class:`BooleanNDImage`
            An optional mask that can be applied to the image. Has to have a
             shape equal to that of the image.

             Default: all True :class:`BooleanNDImage`

        Returns
        -------
        blank_image : :class:`VoxelImage`
            A new masked image of the requested size.
        """
        # just enforce n_channels is 3
        return MaskedNDImage.blank(shape, n_channels=3, fill=fill,
                                   dtype=dtype, mask=mask)
