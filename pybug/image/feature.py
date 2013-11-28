from pybug.image.masked import MaskedNDImage
import pybug.features as fc
from pybug.visualize.base import ImageViewer
import numpy as np


class FeatureNDImage(MaskedNDImage):
    r"""
    Represents a 2-dimensional image with k number of channels, of size
    ``(M, N, C)``. ``np.uint8`` pixel data is converted to ``np.float64``
    and scaled between ``0`` and ``1`` by dividing each pixel by ``255``.
    All Image2D instances have values for channels between 0-1,
    and have a dtype of np.float.

    Parameters
    ----------
    image_data :  ndarray
        The pixel data for the image, where the last axis represents the
        number of channels.
    mask : (M, N) ``np.bool`` ndarray or :class:`BooleanNDImage`, optional
        A binary array representing the mask. Must be the same
        shape as the image. Only one mask is supported for an image (so the
        mask is applied to every channel equally).

        Default: :class:`BooleanNDImage` covering the whole image

    Raises
    -------
    ValueError
        Mask is not the same shape as the image
    """

    pass


class HOG2DImage(FeatureNDImage):
    r"""
    Represents a 2-dimensional image with k number of channels, of size
    ``(M, N, C)``. ``np.uint8`` pixel data is converted to ``np.float64``
    and scaled between ``0`` and ``1`` by dividing each pixel by ``255``.
    All Image2D instances have values for channels between 0-1,
    and have a dtype of np.float.

    Parameters
    ----------
    image_data :  ndarray
        The pixel data for the image, where the last axis represents the
        number of channels.
    mask : (M, N) ``np.bool`` ndarray or :class:`BooleanNDImage`, optional
        A binary array representing the mask. Must be the same
        shape as the image. Only one mask is supported for an image (so the
        mask is applied to every channel equally).

        Default: :class:`BooleanNDImage` covering the whole image

    Raises
    -------
    ValueError
        Mask is not the same shape as the image
    """
    def __init__(self, image_data, mask=None, method='dense',
                 algorithm='dalaltriggs', num_bins=9, cell_size=8,
                 block_size=2, signed_gradient=True, l2_norm_clip=0.2,
                 window_height=1, window_width=1, window_unit='blocks',
                 window_step_vertical=1, window_step_horizontal=1,
                 window_step_unit='pixels', padding=True, verbose=False
                 ):
        self.params = {'method': method,
                       'algorithm': algorithm,
                       'num_bins': num_bins,
                       'cell_size': cell_size,
                       'block_size': block_size,
                       'signed_gradient': signed_gradient,
                       'l2_norm_clip': l2_norm_clip,
                       'window_height': window_height,
                       'window_width': window_width,
                       'window_unit': window_unit,
                       'window_step_vertical': window_step_vertical,
                       'window_step_horizontal': window_step_horizontal,
                       'window_step_unit': window_step_unit,
                       'padding': padding,
                       'verbose': verbose}
        hog, window_centres = fc.hog(image_data, **self.params)
        if mask is not None:
            if not isinstance(mask, np.ndarray):
                # assume that the mask is a boolean image then!
                mask = mask.pixels
            mask = mask[..., 0][window_centres[:, :, 0],
                                window_centres[:, :, 1]]
        super(HOG2DImage, self).__init__(hog, mask=mask)
        self.window_centres = window_centres

    @classmethod
    def blank(cls):
        pass

    def _view(self, figure_id=None, new_figure=False, channel=None,
              masked=True, vector=True, block_size=None, num_bins=None,
              **kwargs):
        r"""
        View the image using the default image viewer. Currently only
        supports the rendering of 2D images.

        Returns
        -------
        image_viewer : :class:`pybug.visualize.viewimage.ViewerImage`
            The viewer the image is being shown within

        Raises
        ------
        DimensionalityError
            If Image is not 2D
        """
        pixels_to_view = self.pixels
        if vector:
            channel = 0  # Vectorized images always have 1 channel
            if block_size is None:
                block_size = self.params['block_size']
            if num_bins is None:
                num_bins = self.params['num_bins']
            hog_vector_image = fc.hog_vector_image(self.pixels,
                                                   block_size=block_size,
                                                   num_bins=num_bins)
            pixels_to_view = hog_vector_image[..., None]
        mask = None
        if masked:
            # TODO: make the visualization work for hog vector and mask
            mask = self.mask.mask
        return ImageViewer(figure_id, new_figure, self.n_dims,
                           pixels_to_view, channel=channel,
                           mask=mask).render(**kwargs)

    def __str__(self):
        header = (
            '{} 2D HOGImage with {} channels. '
            'Attached mask {:.1%} true'.format(self._str_shape,
                                               self.n_channels,
                                               self.mask.proportion_true))
        info = str(self.params)
        return '\n'.join([header, info])
#void HOG::print_information() {
#	cout << endl << "HOG options" << endl << "-----------" << endl;
#	if (this->method==1) {
#		cout << "Method of Dalal & Triggs" << endl;
#		cout << "Cell = " << this->cellHeightAndWidthInPixels << "x" << this->cellHeightAndWidthInPixels << " pixels" << endl;
#		cout << "Block = " << this->blockHeightAndWidthInCells << "x" << this->blockHeightAndWidthInCells << " cells" << endl;
#		if (this->enableSignedGradients == true)
#			cout << this->numberOfOrientationBins << " orientation bins and signed gradients" << endl;
#		else
#			cout << this->numberOfOrientationBins << " orientation bins and unsigned gradients" << endl;
#		cout << "L2-norm clipped at " << this->l2normClipping << endl;
#		cout << "Number of blocks per window = " << this->numberOfBlocksPerWindowVertically << "x" << this->numberOfBlocksPerWindowHorizontally << endl;
#		cout << "Descriptor length per window = " << this->numberOfBlocksPerWindowVertically << "x" << this->numberOfBlocksPerWindowHorizontally << "x" << this->descriptorLengthPerBlock << " = " << this->descriptorLengthPerWindow << endl;
#	}
#	else {
#		cout << "Method of Zhu & Ramanan" << endl;
#		cout << "Cell = " << this->cellHeightAndWidthInPixels << "x" << this->cellHeightAndWidthInPixels << " pixels" << endl;
#		cout << "Number of blocks per window = " << this->numberOfBlocksPerWindowVertically << "x" << this->numberOfBlocksPerWindowHorizontally << endl;
#		cout << "Descriptor length per window = " << this->numberOfBlocksPerWindowVertically << "x" << this->numberOfBlocksPerWindowHorizontally << "x" << this->descriptorLengthPerBlock << " = " << this->descriptorLengthPerWindow << endl;
#	}
#}