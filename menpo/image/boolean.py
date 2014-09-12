from warnings import warn
import numpy as np

from .base import Image


class BooleanImage(Image):
    r"""
    A mask image made from binary pixels. The region of the image that is
    left exposed by the mask is referred to as the 'masked region'. The
    set of 'masked' pixels is those pixels corresponding to a True value in
    the mask.

    Parameters
    -----------
    mask_data : (M, N, ..., L) ndarray
        The binary mask data. Note that there is no channel axis - a 2D Mask
        Image is built from just a 2D numpy array of mask_data.
        Automatically coerced in to boolean values.

    copy: bool, optional
        If False, the image_data will not be copied on assignment. Note that
        if the array you provide is not boolean, there **will still be copy**.
        In general this should only be used if you know what you are doing.

        Default: `False`
    """

    def __init__(self, mask_data, copy=True):
        # Add a channel dimension. We do this little reshape trick to add
        # the axis because this maintains C-contiguous'ness
        mask_data = mask_data.reshape(mask_data.shape + (1,))
        # If we are trying not to copy, but the data we have isn't boolean,
        # then unfortunately, we forced to copy anyway!
        if mask_data.dtype != np.bool:
            mask_data = np.array(mask_data, dtype=np.bool, copy=True,
                                 order='C')
            if not copy:
                warn('The copy flag was NOT honoured. A copy HAS been made. '
                     'Please ensure the data you pass is C-contiguous.')
        super(BooleanImage, self).__init__(mask_data, copy=copy)

    def as_masked(self, mask=None, copy=True):
        raise NotImplementedError("as_masked cannot be invoked on a "
                                  "BooleanImage.")

    @classmethod
    def blank(cls, shape, fill=True, round='ceil', **kwargs):
        r"""
        Returns a blank :map:`BooleanImage` of the requested shape

        Parameters
        ----------
        shape : tuple or list
            The shape of the image. Any floating point values are rounded
            according to the `round` kwarg.

        fill : True or False, optional
            The mask value to be set everywhere

            Default: True (masked region is the whole image - meaning the whole
                 image is exposed)
        round: {'ceil', 'floor', 'round'}
            Rounding function to be applied to floating point shapes.

            Default: 'ceil'

        Returns
        -------
        blank_image : :map:`BooleanImage`
            A blank mask of the requested size

        """
        from .base import round_image_shape
        shape = round_image_shape(shape, round)
        if fill:
            mask = np.ones(shape, dtype=np.bool)
        else:
            mask = np.zeros(shape, dtype=np.bool)
        return cls(mask, copy=False)

    @property
    def mask(self):
        r"""
        Returns the pixels of the mask with no channel axis. This is what
        should be used to mask any k-dimensional image.

        :type: (M, N, ..., L), np.bool ndarray
        """
        return self.pixels[..., 0]

    @property
    def n_true(self):
        r"""
        The number of `True` values in the mask

        :type: int
        """
        return np.sum(self.pixels)

    @property
    def n_false(self):
        r"""
        The number of `False` values in the mask

        :type: int
        """
        return self.n_pixels - self.n_true

    @property
    def all_true(self):
        r"""
        True iff every element of the mask is True.

        :type: bool
        """
        return np.all(self.pixels)

    @property
    def proportion_true(self):
        r"""
        The proportion of the mask which is `True`

        :type: double
        """
        return (self.n_true * 1.0) / self.n_pixels

    @property
    def proportion_false(self):
        r"""
        The proportion of the mask which is `False`

        :type: double
        """
        return (self.n_false * 1.0) / self.n_pixels

    @property
    def true_indices(self):
        r"""
        The indices of pixels that are true.

        :type: (`n_dims`, `n_true`) ndarray
        """
        if self.all_true:
            return self.indices
        else:
            # Ignore the channel axis
            return np.vstack(np.nonzero(self.pixels[..., 0])).T

    @property
    def false_indices(self):
        r"""
        The indices of pixels that are false.

        :type: (`n_dims`, `n_false`) ndarray
        """
        # Ignore the channel axis
        return np.vstack(np.nonzero(~self.pixels[..., 0])).T

    def __str__(self):
        return ('{} {}D mask, {:.1%} '
                'of which is True'.format(self._str_shape, self.n_dims,
                                          self.proportion_true))

    def from_vector(self, vector, copy=True):
        r"""
        Takes a flattened vector and returns a new :map:`BooleanImage` formed
        by reshaping the vector to the correct dimensions. Note that this is
        rebuilding a boolean image **itself** from boolean values. The mask
        is in no way interpreted in performing the operation, in contrast to
        MaskedImage, where only the masked region is used in from_vector()
        and as_vector(). Any image landmarks are transferred in the process.

        Parameters
        ----------
        vector : (`n_pixels`,) np.bool ndarray
            A flattened vector of all the pixels of a BooleanImage.

        copy : bool, optional
            If false, no copy of the vector will be taken.

            Default: True

        Returns
        -------
        image : :map:`BooleanImage`
            New BooleanImage of same shape as this image

        Raises
        ------
        Warning : If copy=False cannot be honored.

        """
        mask = BooleanImage(vector.reshape(self.shape), copy=copy)
        mask.landmarks = self.landmarks
        return mask

    def invert_inplace(self):
        r"""
        Inverts this Boolean Image inplace.

        """
        self.pixels = ~self.pixels

    def invert(self):
        r"""
        Returns a copy of this Boolean image, which is inverted.

        Returns
        -------

        inverted : :map:`BooleanImage`
            A copy of this boolean mask, where all True values are False and
            all False values are True.

        """
        inverse = self.copy()
        inverse.invert_inplace()
        return inverse

    def bounds_true(self, boundary=0, constrain_to_bounds=True):
        r"""
        Returns the minimum to maximum indices along all dimensions that the
        mask includes which fully surround the True mask values. In the case
        of a 2D Image for instance, the min and max define two corners of a
        rectangle bounding the True pixel values.

        Parameters
        ----------
        boundary : int, optional
            A number of pixels that should be added to the extent. A
            negative value can be used to shrink the bounds in.

            Default: 0

        constrain_to_bounds: bool, optional
            If True, the bounding extent is snapped to not go beyond
            the edge of the image. If False, the bounds are left unchanged.

            Default: True

        Returns
        --------
        min_b : (D,) ndarray
            The minimum extent of the True mask region with the boundary
            along each dimension. If constrain_to_bounds was True,
            is clipped to legal image bounds.

        max_b : (D,) ndarray
            The maximum extent of the True mask region with the boundary
            along each dimension. If constrain_to_bounds was True,
            is clipped to legal image bounds.
        """
        mpi = self.true_indices
        maxes = np.max(mpi, axis=0) + boundary
        mins = np.min(mpi, axis=0) - boundary
        if constrain_to_bounds:
            maxes = self.constrain_points_to_bounds(maxes)
            mins = self.constrain_points_to_bounds(mins)
        return mins, maxes

    def bounds_false(self, boundary=0, constrain_to_bounds=True):
        r"""
        Returns the minimum to maximum indices along all dimensions that the
        mask includes which fully surround the False mask values. In the case
        of a 2D Image for instance, the min and max define two corners of a
        rectangle bounding the False pixel values.

        Parameters
        ----------
        boundary : int >= 0, optional
            A number of pixels that should be added to the extent. A
            negative value can be used to shrink the bounds in.

            Default: 0

        constrain_to_bounds: bool, optional
            If True, the bounding extent is snapped to not go beyond
            the edge of the image. If False, the bounds are left unchanged.

            Default: True

        Returns
        --------
        min_b : (D,) ndarray
            The minimum extent of the False mask region with the boundary
            along each dimension. If constrain_to_bounds was True,
            is clipped to legal image bounds.

        max_b : (D,) ndarray
            The maximum extent of the False mask region with the boundary
            along each dimension. If constrain_to_bounds was True,
            is clipped to legal image bounds.
        """
        return self.invert().bounds_true(
            boundary=boundary, constrain_to_bounds=constrain_to_bounds)

    # noinspection PyMethodOverriding
    def warp_to_mask(self, template_mask, transform, warp_landmarks=True,
                     mode='constant', cval=0.):
        r"""
        Return a copy of this :map:`BooleanImage` warped into a different
        reference space.

        Note that warping into a mask is slower than warping into a full image.
        If you don't need a non-linear mask, consider warp_to_shape instead.

        Parameters
        ----------
        template_mask : :map:`BooleanImage`
            Defines the shape of the result, and what pixels should be
            sampled.

        transform : :map:`Transform`
            Transform **from the template space back to this image**.
            Defines, for each pixel location on the template, which pixel
            location should be sampled from on this image.

        warp_landmarks : `bool`, optional
            If `True`, warped_image will have the same landmark dictionary
            as self, but with each landmark updated to the warped position.

        mode : `str`, optional
            Points outside the boundaries of the input are filled according
            to the given mode ('constant', 'nearest', 'reflect' or 'wrap').

        cval : `float`, optional
            Used in conjunction with mode 'constant', the value outside
            the image boundaries.

        Returns
        -------
        warped_image : :map:`BooleanImage`
            A copy of this image, warped.
        """
        # enforce the order as 0, for this boolean data, then call super
        return Image.warp_to_mask(self, template_mask, transform,
                                  warp_landmarks=warp_landmarks,
                                  order=0, mode=mode, cval=cval)

    # noinspection PyMethodOverriding
    def warp_to_shape(self, template_shape, transform, warp_landmarks=True,
                      mode='constant', cval=0.):
        """
        Return a copy of this :map:`BooleanImage` warped into a different
        reference space.

        Parameters
        ----------
        template_shape : (n_dims, ) tuple or ndarray
            Defines the shape of the result, and what pixel indices should be
            sampled (all of them).

        transform : :map:`Transform`
            Transform **from the template_shape space back to this image**.
            Defines, for each index on template_shape, which pixel location
            should be sampled from on this image.

        warp_landmarks : `bool`, optional
            If `True`, ``warped_image`` will have the same landmark dictionary
            as self, but with each landmark updated to the warped position.

        mode : `str`, optional
            Points outside the boundaries of the input are filled according
            to the given mode ('constant', 'nearest', 'reflect' or 'wrap').

        cval : `float`, optional
            Used in conjunction with mode 'constant', the value outside
            the image boundaries.

        Returns
        -------
        warped_image : :map:`BooleanImage`
            A copy of this image, warped.

        """
        # call the super variant and get ourselves an Image back
        # note that we force the use of order=1 for BooleanImages.
        warped = Image.warp_to_shape(self, template_shape, transform,
                                     warp_landmarks=warp_landmarks,
                                     order=1, mode=mode, cval=cval)
        # unfortunately we can't escape copying here, let BooleanImage
        # convert us to np.bool
        boolean_image = BooleanImage(warped.pixels.reshape(template_shape))
        if warped.has_landmarks:
            boolean_image.landmarks = warped.landmarks
        return boolean_image

    def _build_warped_to_mask(self, template_mask, sampled_pixel_values,
                              **kwargs):
        r"""Builds the warped image from the template mask and
        sampled pixel values.
        """
        # start from a copy of the template_mask
        warped_img = template_mask.copy()
        if warped_img.all_true:
            # great, just reshape the sampled_pixel_values
            warped_img.pixels = sampled_pixel_values.reshape(
                warped_img.shape + (1,))
        else:
            # we have to fill out mask with the sampled mask..
            warped_img.pixels[warped_img.mask] = sampled_pixel_values
        return warped_img

    def constrain_to_landmarks(self, group=None, label=None, trilist=None):
        r"""
        Restricts this mask to be equal to the convex hull around the
        landmarks chosen.

        Parameters
        ----------
        group : string, Optional
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.

            Default: None

        label: string, Optional
            The label of of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

            Default: None

        trilist: (t, 3) ndarray, Optional
            Triangle list to be used on the landmarked points in selecting
            the mask region. If None defaults to performing Delaunay
            triangulation on the points.

            Default: None
        """
        self.constrain_to_pointcloud(self.landmarks[group][label],
                                     trilist=trilist)

    def constrain_to_pointcloud(self, pointcloud, trilist=None):
        r"""
        Restricts this mask to be equal to the convex hull around a point cloud

        Parameters
        ----------
        pointcloud : :map:`PointCloud`
            The pointcloud of points that should be constrained to

        trilist: (t, 3) ndarray, Optional
            Triangle list to be used on the points in selecting
            the mask region. If None defaults to performing Delaunay
            triangulation on the points.

            Default: None
        """
        from menpo.transform.piecewiseaffine import PiecewiseAffine
        from menpo.transform.piecewiseaffine import TriangleContainmentError

        if self.n_dims != 2:
            raise ValueError("can only constrain mask on 2D images.")

        if trilist is not None:
            from menpo.shape import TriMesh
            pointcloud = TriMesh(pointcloud.points, trilist)

        pwa = PiecewiseAffine(pointcloud, pointcloud)
        try:
            pwa.apply(self.indices)
        except TriangleContainmentError as e:
            self.from_vector_inplace(~e.points_outside_source_domain)
