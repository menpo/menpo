import numpy as np
import PIL.Image as PILImage
from pybug.transform.affine import Translation
from pybug.visualize import ImageViewer
from pybug.base import Vectorizable


class Image(Vectorizable):

    def __init__(self, image_data, mask=None):
        # we support construction from a PIL Image class
        if isinstance(image_data, PILImage.Image):
            image_data = np.array(image_data)
        # get the attributes of the image
        self.width, self.height = image_data.shape[:2]
        if len(image_data.shape) >= 3:
            self.n_channels = image_data.shape[-1]
        else:
            self.n_channels = 1
            # ensure all our self.pixels have channel in the last dim,
            # even if the last dim is unitary in length
            image_data = image_data[..., np.newaxis]
        self.image_shape = image_data.shape[:-1]
        self.n_dims = len(self.image_shape)
        if mask is not None:
            assert(self.image_shape == mask.shape)
            self.mask = mask.astype(np.bool).copy()
        else:
            self.mask = np.ones(self.image_shape, dtype=np.bool)
        self.n_masked_pixels = np.sum(self.mask)

        # ensure that the data is in the right format
        if image_data.dtype == np.uint8:
            image_data = image_data.astype(np.float64) / 255
        elif image_data.dtype != np.float64:
            # convert to double
            image_data = image_data.astype(np.float64)
        self.pixels = image_data.copy()

    def view(self):
        if self.n_dims == 2:
            return ImageViewer(self.pixels)
        else:
            raise Exception("n_dim Image rendering is not yet supported.")

    def as_vector(self, keep_channels=False):
        if keep_channels:
            return self.masked_pixels.reshape([-1, self.n_channels])
        else:
            return self.masked_pixels.flatten()

    @property
    def masked_pixels(self):
        """
        :return: (n_active_pixels, n_channels) ndarray of pixels that have a
         True mask value
        """
        return self.pixels[self.mask]

    def mask_bounding_extent(self, boundary=0):
        """
        Returns the maximum and minimum values along all dimensions that the
        mask includes.
        :param boundary: A number of pixels that should be added to the
        extent.
        Note that if the bounding extent is snapped to not go beyond the
        edge of the image.
        :return: ndarray [n_dims, 2] where
        [k, :] = [min_bounding_dim_k, max_bounding_dim_k]
        """
        mpi = self.masked_pixel_indices
        maxes = np.max(mpi, axis=0) + boundary
        mins = np.min(mpi, axis=0) - boundary
        # check we don't stray under any edges
        mins[mins < 0] = 0
        # check we don't stray over any edges
        over_image = self.image_shape - maxes < 0
        maxes[over_image] = np.array(self.image_shape)[over_image]
        return np.vstack((mins, maxes)).T

    def mask_bounding_extent_slicer(self, boundary=0):
        extents = self.mask_bounding_extent(boundary)
        return [slice(x[0], x[1]) for x in list(extents)]

    def mask_bounding_pixels(self, boundary=0):
        return self.pixels[self.mask_bounding_extent_slicer(boundary)]

    def mask_bounding_extent_meshgrids(self, boundary=0):
        """
        Returns a list of meshgrids, the ith item being the meshgrid over
        the bounding extent over the i'th dimension.
        :param boundary:
        :return:
        """
        extents = self.mask_bounding_extent(boundary)
        return np.meshgrid(*[np.arange(*list(x)) for x in list(extents)])


    @property
    def masked_pixel_indices(self):
        if getattr(self, '_indices_cache', None) is None:
            self._indices_cache = np.vstack(np.nonzero(self.mask)).T
        return self._indices_cache

    def from_vector(self, flattened):
        mask = self.mask
        image_data = np.zeros_like(self.pixels)
        pixels_per_channel = flattened.reshape((-1, self.n_channels))
        image_data[mask] = pixels_per_channel
        return Image(image_data, mask=mask)

    def as_greyscale(self):
        if self.n_channels == 1:
            print "Warning - trying to convert to greyscale an image with " \
                  "only one channel - returning a copy"
            return Image(self.pixels, self.mask)
        if self.n_channels != 3 or self.n_dims != 2:
            raise Exception("Trying to perform RGB-> greyscale conversion on"
                            " a non-2D-RGB Image.")
        pil_image = self.as_PILImage()
        pil_bw_image = pil_image.convert('L')
        return Image(pil_bw_image, mask=self.mask)

    def as_PILImage(self):
        return PILImage.fromarray((self.pixels * 255).astype(np.uint8))

    def crop(self, *args):
        r"""
        Crops the image using the given slice objects. Expects
        ``len(args) == self.n_dims``. Maintains the cropped portion of the
        mask. Also returns the
        :class:`Translation <pybug.transform.affine.Translation>` that would
        translate the cropped portion back in to the original reference
        position. **Note:** Only 2D and 3D images are currently supported.

        :param args: The slices to take over each axis
        :type args: List of slice objects
        :raises: AssertionError, ValueError
        :return: (cropped_image, translation)

                cropped_image:
                    Cropped portion of the image, including cropped mask.

                translation:
                    Translation transform that repositions the cropped image
                    in the reference frame of the original.
        :rtype: (:class:`Image <pybug.image.base.Image>`,
            :class:`Translation <pybug.transform.affine.Translation>`)
        """
        assert(self.n_dims == len(args))
        if len(args) == 2:
            cropped_image = self.pixels[args[0], args[1], ...]
            cropped_mask = self.mask[args[0], args[1], ...]
            translation = np.array([args[0].start, args[1].start])
        elif len(args) == 3:
            cropped_image = self.pixels[args[0], args[1], args[2], ...]
            cropped_mask = self.mask[args[0], args[1], args[2], ...]
            translation = np.array([args[0].start, args[1].start,
                                    args[2].start])
        else:
            raise ValueError("Only 2D and 3D images are currently supported.")

        return (Image(cropped_image, mask=cropped_mask),
                Translation(translation))

    def gradient(self, inc_unmasked_pixels=False):
        """
        Returns an Image which is the gradient of this one.
        :return:
        """
        if self.n_channels != 1:
            raise Exception("Warning - trying to take the gradient on a "
                            "non-grayscale image")
        if inc_unmasked_pixels:
            # we know this is B + W, drop the last axis (n_channels which is 1)
            gradients = np.gradient(self.pixels[..., 0])
            # Add an extra axis for broadcasting
            gradients = [g[..., None] for g in gradients]
            # Concatenate gradient list into an array (the new_image)
            new_image = np.concatenate(gradients, axis=-1)
        else:
            masked_square_image = self.mask_bounding_pixels(boundary=3)
            bounding_mask = self.mask_bounding_extent_slicer(3)
            # we know this is B + W, drop the last axis (n_channels which is 1)
            gradients = np.gradient(masked_square_image[..., 0])
            # Add an extra axis for broadcasting
            gradients = [g[..., None] for g in gradients]
            # Concatenate gradient list into a vector
            gradient_array = np.concatenate(gradients, axis=-1)
            # make a new blank image
            new_image = np.empty((self.image_shape + (self.n_dims,)))
            # populate the new image with the gradient
            new_image[bounding_mask] = gradient_array
        return Image(new_image, mask=self.mask)
