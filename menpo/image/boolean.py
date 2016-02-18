from functools import partial
from warnings import warn
import numpy as np

from .base import Image, _convert_patches_list_to_single_array
from .patches import set_patches


def pwa_point_in_pointcloud(pcloud, indices, batch_size=None):
    """
    Make sure that the decision of whether a point is inside or outside
    the PointCloud is exactly the same as how PWA calculates triangle
    containment. Then, we use the trick of setting the mask to all the
    points that were NOT outside the triangulation. Otherwise, all points
    were inside and we just return those as ``True``. In general, points
    on the boundary are counted as inside the polygon.

    Parameters
    ----------
    pcloud : :map:`PointCloud`
        The pointcloud to use for the containment test.
    indices : (d, n_dims) `ndarray`
        The list of pixel indices to test.
    batch_size : `int` or ``None``, optional
        See constrain_to_pointcloud for more information about the batch_size
        parameter.

    Returns
    -------
    mask : (d,) `bool ndarray`
        Whether each pixel index was in inside the convex hull of the
        pointcloud or not.
    """
    from menpo.transform.piecewiseaffine import PiecewiseAffine
    from menpo.transform.piecewiseaffine import TriangleContainmentError

    try:
        pwa = PiecewiseAffine(pcloud, pcloud)
        pwa.apply(indices, batch_size=batch_size)
        return np.ones(indices.shape[0], dtype=np.bool)
    except TriangleContainmentError as e:
        return ~e.points_outside_source_domain


def convex_hull_point_in_pointcloud(pcloud, indices):
    """
    Uses the matplotlib ``contains_points`` method, which in turn uses:

        "Crossings Multiply algorithm of InsideTest"
        By Eric Haines, 3D/Eye Inc, erich@eye.com
        http://erich.realtimerendering.com/ptinpoly/

    This algorithm uses a per-pixel test and thus tends to produce smoother
    edges. We also guarantee that all points inside PointCloud will be
    included by calculating the **convex hull** of the pointcloud before
    doing the point inside test.

    Points on the boundary are counted as **outside** the polygon.

    Parameters
    ----------
    pcloud : :map:`PointCloud`
        The pointcloud to use for the containment test.
    indices : (d, n_dims) `ndarray`
        The list of pixel indices to test.

    Returns
    -------
    mask : (d,) `bool ndarray`
        Whether each pixel index was in inside the convex hull of the
        pointcloud or not.
    """
    from scipy.spatial import ConvexHull
    from matplotlib.path import Path

    c_hull = ConvexHull(pcloud.points)
    polygon = pcloud.points[c_hull.vertices, :]

    return Path(polygon).contains_points(indices)


class BooleanImage(Image):
    r"""
    A mask image made from binary pixels. The region of the image that is
    left exposed by the mask is referred to as the 'masked region'. The
    set of 'masked' pixels is those pixels corresponding to a ``True`` value in
    the mask.

    Parameters
    ----------
    mask_data : ``(M, N, ..., L)`` `ndarray`
        The binary mask data. Note that there is no channel axis - a 2D Mask
        Image is built from just a 2D numpy array of mask_data.
        Automatically coerced in to boolean values.
    copy: `bool`, optional
        If ``False``, the image_data will not be copied on assignment. Note that
        if the array you provide is not boolean, there **will still be copy**.
        In general this should only be used if you know what you are doing.
    """

    def __init__(self, mask_data, copy=True):
        # Add a channel dimension. We do this little reshape trick to add
        # the axis because this maintains C-contiguous'ness
        mask_data = mask_data.reshape((1,) + mask_data.shape)
        # If we are trying not to copy, but the data we have isn't boolean,
        # then unfortunately, we forced to copy anyway!
        if mask_data.dtype != np.bool:
            mask_data = np.array(mask_data, dtype=np.bool, copy=True,
                                 order='C')
            if not copy:
                warn('The copy flag was NOT honoured. A copy HAS been made. '
                     'Please ensure the data you pass is C-contiguous.')
        super(BooleanImage, self).__init__(mask_data, copy=copy)

    @classmethod
    def init_blank(cls, shape, fill=True, round='ceil', **kwargs):
        r"""
        Returns a blank :map:`BooleanImage` of the requested shape

        Parameters
        ----------
        shape : `tuple` or `list`
            The shape of the image. Any floating point values are rounded
            according to the ``round`` kwarg.
        fill : `bool`, optional
            The mask value to be set everywhere.
        round: ``{ceil, floor, round}``, optional
            Rounding function to be applied to floating point shapes.

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

    def as_masked(self, mask=None, copy=True):
        r"""
        Impossible for a :map:`BooleanImage` to be transformed to a
        :map:`MaskedImage`.
        """
        raise NotImplementedError("as_masked cannot be invoked on a "
                                  "BooleanImage.")

    @property
    def mask(self):
        r"""
        Returns the pixels of the mask with no channel axis. This is what
        should be used to mask any k-dimensional image.

        :type: ``(M, N, ..., L)``, `bool ndarray`
        """
        return self.pixels[0, ...]

    def n_true(self):
        r"""
        The number of ``True`` values in the mask.

        :type: `int`
        """
        return np.sum(self.pixels)

    def n_false(self):
        r"""
        The number of ``False`` values in the mask.

        :type: `int`
        """
        return self.n_pixels - self.n_true()

    def all_true(self):
        r"""
        ``True`` iff every element of the mask is ``True``.

        :type: `bool`
        """
        return np.all(self.pixels)

    def proportion_true(self):
        r"""
        The proportion of the mask which is ``True``.

        :type: `float`
        """
        return (self.n_true() * 1.0) / self.n_pixels

    def proportion_false(self):
        r"""
        The proportion of the mask which is ``False``

        :type: `float`
        """
        return (self.n_false() * 1.0) / self.n_pixels

    def true_indices(self):
        r"""
        The indices of pixels that are ``True``.

        :type: ``(n_dims, n_true)`` `ndarray`
        """
        if self.all_true():
            return self.indices()
        else:
            # Ignore the channel axis
            return np.vstack(np.nonzero(self.pixels[0])).T

    def false_indices(self):
        r"""
        The indices of pixels that are ``Flase``.

        :type: ``(n_dims, n_false)`` `ndarray`
        """
        # Ignore the channel axis
        return np.vstack(np.nonzero(~self.pixels[0])).T

    def __str__(self):
        return ('{} {}D mask, {:.1%} '
                'of which is True'.format(self._str_shape(), self.n_dims,
                                          self.proportion_true()))

    def from_vector(self, vector, copy=True):
        r"""
        Takes a flattened vector and returns a new :map:`BooleanImage` formed
        by reshaping the vector to the correct dimensions. Note that this is
        rebuilding a boolean image **itself** from boolean values. The mask
        is in no way interpreted in performing the operation, in contrast to
        :map:`MaskedImage`, where only the masked region is used in
        :meth:`from_vector` and :meth`as_vector`. Any image landmarks are
        transferred in the process.

        Parameters
        ----------
        vector : ``(n_pixels,)`` `bool ndarray`
            A flattened vector of all the pixels of a :map:`BooleanImage`.
        copy : `bool`, optional
            If ``False``, no copy of the vector will be taken.

        Returns
        -------
        image : :map:`BooleanImage`
            New BooleanImage of same shape as this image

        Raises
        ------
        Warning
            If ``copy=False`` cannot be honored.
        """
        mask = BooleanImage(vector.reshape(self.shape), copy=copy)
        if self.has_landmarks:
            mask.landmarks = self.landmarks
        if hasattr(self, 'path'):
            mask.path = self.path
        return mask

    def invert(self):
        r"""
        Returns a copy of this boolean image, which is inverted.

        Returns
        -------
        inverted : :map:`BooleanImage`
            A copy of this boolean mask, where all ``True`` values are ``False``
            and all ``False`` values are ``True``.
        """
        inverse = self.copy()
        inverse.pixels = ~self.pixels
        return inverse

    def bounds_true(self, boundary=0, constrain_to_bounds=True):
        r"""
        Returns the minimum to maximum indices along all dimensions that the
        mask includes which fully surround the ``True`` mask values. In the case
        of a 2D Image for instance, the min and max define two corners of a
        rectangle bounding the True pixel values.

        Parameters
        ----------
        boundary : `int`, optional
            A number of pixels that should be added to the extent. A
            negative value can be used to shrink the bounds in.
        constrain_to_bounds: `bool`, optional
            If ``True``, the bounding extent is snapped to not go beyond
            the edge of the image. If ``False``, the bounds are left unchanged.

        Returns
        --------
        min_b : ``(D,)`` `ndarray`
            The minimum extent of the ``True`` mask region with the boundary
            along each dimension. If ``constrain_to_bounds=True``,
            is clipped to legal image bounds.
        max_b : ``(D,)`` `ndarray`
            The maximum extent of the ``True`` mask region with the boundary
            along each dimension. If ``constrain_to_bounds=True``,
            is clipped to legal image bounds.
        """
        mpi = self.true_indices()
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
        boundary : `int` >= 0, optional
            A number of pixels that should be added to the extent. A
            negative value can be used to shrink the bounds in.
        constrain_to_bounds: `bool`, optional
            If ``True``, the bounding extent is snapped to not go beyond
            the edge of the image. If ``False``, the bounds are left unchanged.

        Returns
        -------
        min_b : ``(D,)`` `ndarray`
            The minimum extent of the ``True`` mask region with the boundary
            along each dimension. If ``constrain_to_bounds=True``,
            is clipped to legal image bounds.
        max_b : ``(D,)`` `ndarray`
            The maximum extent of the ``True`` mask region with the boundary
            along each dimension. If ``constrain_to_bounds=True``,
            is clipped to legal image bounds.
        """
        return self.invert().bounds_true(
            boundary=boundary, constrain_to_bounds=constrain_to_bounds)

    # noinspection PyMethodOverriding
    def sample(self, points_to_sample, mode='constant', cval=False, **kwargs):
        r"""
        Sample this image at the given sub-pixel accurate points. The input
        PointCloud should have the same number of dimensions as the image e.g.
        a 2D PointCloud for a 2D multi-channel image. A numpy array will be
        returned the has the values for every given point across each channel
        of the image.

        Parameters
        ----------
        points_to_sample : :map:`PointCloud`
            Array of points to sample from the image. Should be
            `(n_points, n_dims)`
        mode : ``{constant, nearest, reflect, wrap}``, optional
            Points outside the boundaries of the input are filled according
            to the given mode.
        cval : `float`, optional
            Used in conjunction with mode ``constant``, the value outside
            the image boundaries.

        Returns
        -------
        sampled_pixels : (`n_points`, `n_channels`) `bool ndarray`
            The interpolated values taken across every channel of the image.
        """
        # enforce the order as 0, as this is boolean data, then call super
        return Image.sample(self, points_to_sample, order=0, mode=mode,
                            cval=cval)

    # noinspection PyMethodOverriding
    def warp_to_mask(self, template_mask, transform, warp_landmarks=True,
                     mode='constant', cval=False, batch_size=None,
                     return_transform=False):
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
            If ``True``, result will have the same landmark dictionary
            as self, but with each landmark updated to the warped position.
        mode : ``{constant, nearest, reflect or wrap}``, optional
            Points outside the boundaries of the input are filled according
            to the given mode.
        cval : `float`, optional
            Used in conjunction with mode ``constant``, the value outside
            the image boundaries.
        batch_size : `int` or ``None``, optional
            This should only be considered for large images. Setting this
            value can cause warping to become much slower, particular for
            cached warps such as Piecewise Affine. This size indicates
            how many points in the image should be warped at a time, which
            keeps memory usage low. If ``None``, no batching is used and all
            points are warped at once.
        return_transform : `bool`, optional
            This argument is for internal use only. If ``True``, then the
            :map:`Transform` object is also returned.

        Returns
        -------
        warped_image : :map:`BooleanImage`
            A copy of this image, warped.
        transform : :map:`Transform`
            The transform that was used. It only applies if
            `return_transform` is ``True``.
        """
        # enforce the order as 0, as this is boolean data, then call super
        return Image.warp_to_mask(
            self, template_mask, transform, warp_landmarks=warp_landmarks,
            order=0, mode=mode, cval=cval, batch_size=batch_size,
            return_transform=return_transform)

    # noinspection PyMethodOverriding
    def warp_to_shape(self, template_shape, transform, warp_landmarks=True,
                      mode='constant', cval=False, order=None,
                      batch_size=None, return_transform=False):
        """
        Return a copy of this :map:`BooleanImage` warped into a different
        reference space.

        Note that the order keyword argument is in fact ignored, as any order
        other than 0 makes no sense on a binary image. The keyword argument is
        present only for compatibility with the :map:`Image` warp_to_shape API.

        Parameters
        ----------
        template_shape : ``(n_dims, )`` `tuple` or `ndarray`
            Defines the shape of the result, and what pixel indices should be
            sampled (all of them).
        transform : :map:`Transform`
            Transform **from the template_shape space back to this image**.
            Defines, for each index on template_shape, which pixel location
            should be sampled from on this image.
        warp_landmarks : `bool`, optional
            If ``True``, result will have the same landmark dictionary
            as self, but with each landmark updated to the warped position.
        mode : ``{constant, nearest, reflect or wrap}``, optional
            Points outside the boundaries of the input are filled according
            to the given mode.
        cval : `float`, optional
            Used in conjunction with mode ``constant``, the value outside
            the image boundaries.
        batch_size : `int` or ``None``, optional
            This should only be considered for large images. Setting this
            value can cause warping to become much slower, particular for
            cached warps such as Piecewise Affine. This size indicates
            how many points in the image should be warped at a time, which
            keeps memory usage low. If ``None``, no batching is used and all
            points are warped at once.
        return_transform : `bool`, optional
            This argument is for internal use only. If ``True``, then the
            :map:`Transform` object is also returned.

        Returns
        -------
        warped_image : :map:`BooleanImage`
            A copy of this image, warped.
        transform : :map:`Transform`
            The transform that was used. It only applies if
            `return_transform` is ``True``.
        """
        # call the super variant and get ourselves an Image back
        # note that we force the use of order=0 for BooleanImages.
        warped = Image.warp_to_shape(self, template_shape, transform,
                                     warp_landmarks=warp_landmarks, order=0,
                                     mode=mode, cval=cval,
                                     batch_size=batch_size)
        # unfortunately we can't escape copying here, let BooleanImage
        # convert us to np.bool
        boolean_image = BooleanImage(warped.pixels.reshape(template_shape))
        if warped.has_landmarks:
            boolean_image.landmarks = warped.landmarks
        if hasattr(warped, 'path'):
            boolean_image.path = warped.path
        # optionally return the transform
        if return_transform:
            return boolean_image, transform
        else:
            return boolean_image

    def _build_warp_to_mask(self, template_mask, sampled_pixel_values,
                            **kwargs):
        r"""
        Builds the warped image from the template mask and sampled pixel values.
        """
        # start from a copy of the template_mask
        warped_img = template_mask.copy()
        if warped_img.all_true():
            # great, just reshape the sampled_pixel_values
            warped_img.pixels = sampled_pixel_values.reshape(
                (1,) + warped_img.shape)
        else:
            # we have to fill out mask with the sampled mask..
            warped_img.pixels[:, warped_img.mask] = sampled_pixel_values
        return warped_img

    def constrain_to_landmarks(self, group=None, batch_size=None):
        r"""
        Restricts this mask to be equal to the convex hull around the
        landmarks chosen. This is not a per-pixel convex hull, but instead
        relies on a triangulated approximation. If the landmarks in question
        are an instance of :map:`TriMesh`, the triangulation of the landmarks
        will be used in the convex hull caculation. If the landmarks are an
        instance of :map:`PointCloud`, Delaunay triangulation will be used to
        create a triangulation.

        Parameters
        ----------
        group : `str`, optional
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.
        batch_size : `int` or ``None``, optional
            This should only be considered for large images. Setting this value
            will cause constraining to become much slower. This size indicates
            how many points in the image should be checked at a time, which
            keeps memory usage low. If ``None``, no batching is used and all
            points are checked at once.
        """
        self.constrain_to_pointcloud(self.landmarks[group].lms,
                                     batch_size=batch_size)

    def constrain_to_pointcloud(self, pointcloud, batch_size=None,
                                point_in_pointcloud='pwa'):
        r"""
        Restricts this mask to be equal to the convex hull around a pointcloud.
        The choice of whether a pixel is inside or outside of the pointcloud
        is determined by the ``point_in_pointcloud`` parameter. By default
        a Piecewise Affine transform is used to test for containment, which
        is useful when aligning images by their landmarks. Triangluation
        will be decided by Delauny - if you wish to customise it,
        a :map:`TriMesh` instance can be passed for the ``pointcloud``
        argument. In this case, the triangulation of the Trimesh will be
        used to define the retained region.

        For large images, a faster and pixel-accurate method can be used (
        'convex_hull'). Here, there is no specialization for
        :map:`TriMesh` instances. Alternatively, a callable can be provided to
        override the test. By default, the provided implementations are only
        valid for 2D images.


        Parameters
        ----------
        pointcloud : :map:`PointCloud` or :map:`TriMesh`
            The pointcloud of points that should be constrained to. See
            `point_in_pointcloud` for how in some cases a :map:`TriMesh` may be
            used to control triangulation.
        batch_size : `int` or ``None``, optional
            This should only be considered for large images. Setting this value
            will cause constraining to become much slower. This size indicates
            how many points in the image should be checked at a time, which
            keeps memory usage low. If ``None``, no batching is used and all
            points are checked at once. By default, this is only used for
            the 'pwa' point_in_pointcloud choice.
        point_in_pointcloud : {'pwa', 'convex_hull'} or `callable`
            The method used to check if pixels in the image fall inside the
            ``pointcloud`` or not. If 'pwa', Menpo's :map:`PiecewiseAffine`
            transform will be used to test for containment. In this case
            ``pointcloud`` should be a :map:`TriMesh`. If it isn't, Delauny
            triangulation will be used to first triangulate ``pointcloud`` into
            a  :map:`TriMesh` before testing for containment.
            If a callable is passed, it should take two parameters,
            the :map:`PointCloud` to constrain with and the pixel locations
            ((d, n_dims) ndarray) to test and should return a (d, 1) boolean
            ndarray of whether the pixels were inside (True) or outside (False)
            of the :map:`PointCloud`.

        Raises
        ------
        ValueError
            If the image is not 2D and a default implementation is chosen.
        ValueError
            If the chosen ``point_in_pointcloud`` is unknown.
        """
        if point_in_pointcloud in {'pwa', 'convex_hull'} and self.n_dims != 2:
            raise ValueError('Can only constrain mask on 2D images with the '
                             'default point_in_pointcloud implementations.'
                             'Please provide a custom callable for calculating '
                             'the new mask in this '
                             '{}D image'.format(self.n_dims))

        if point_in_pointcloud == 'pwa':
            point_in_pointcloud = partial(pwa_point_in_pointcloud,
                                          batch_size=batch_size)
        elif point_in_pointcloud == 'convex_hull':
            point_in_pointcloud = convex_hull_point_in_pointcloud
        elif not callable(point_in_pointcloud):
            # Not a function, or a string, so we have an error!
            raise ValueError('point_in_pointcloud must be a callable that '
                             'take two arguments: the Menpo PointCloud as a '
                             'boundary and the ndarray of pixel indices '
                             'to test. {} is an unknown option.'.format(
                             point_in_pointcloud))

        # Only consider indices inside the bounding box of the PointCloud
        bounds = pointcloud.bounds()
        # Convert to integer to try and reduce boundary fp rounding errors.
        bounds = [b.astype(np.int) for b in bounds]
        indices = self.indices()

        # This loop is to ensure the code is multi-dimensional
        for k in range(self.n_dims):
            indices = indices[indices[:, k] >= bounds[0][k], :]
            indices = indices[indices[:, k] <= bounds[1][k], :]
        # Due to only testing bounding box indices, make sure the mask starts
        # off as all False
        self.pixels[:] = False

        # slice(0, 1) because we know we only have 1 channel
        # Slice all the channels, only inside the bounding box (for setting
        # the new mask values).
        all_channels = [slice(0, 1)]
        slices = all_channels + [slice(bounds[0][k], bounds[1][k] + 1)
                                 for k in range(self.n_dims)]
        self.pixels[slices].flat = point_in_pointcloud(pointcloud, indices)

    def set_patches(self, patches, patch_centers, offset=None,
                    offset_index=None):
        r"""
        Set the values of a group of patches into the correct regions of
        **this** image. Given an array of patches and a set of patch centers,
        the patches' values are copied in the regions of the image that are
        centred on the coordinates of the given centers.

        The patches argument can have any of the two formats that are returned
        from the `extract_patches()` and `extract_patches_around_landmarks()`
        methods. Specifically it can be:

            1. ``(n_center, n_offset, self.n_channels, patch_shape)`` `ndarray`
            2. `list` of ``n_center * n_offset`` :map:`Image` objects

        Currently only 2D images are supported.

        Parameters
        ----------
        patches : `ndarray` or `list`
            The values of the patches. It can have any of the two formats that
            are returned from the `extract_patches()` and
            `extract_patches_around_landmarks()` methods. Specifically, it can
            either be an ``(n_center, n_offset, self.n_channels, patch_shape)``
            `ndarray` or a `list` of ``n_center * n_offset`` :map:`Image`
            objects.
        patch_centers : :map:`PointCloud`
            The centers to set the patches around.
        offset : `list` or `tuple` or ``(1, 2)`` `ndarray` or ``None``, optional
            The offset to apply on the patch centers within the image.
            If ``None``, then ``(0, 0)`` is used.
        offset_index : `int` or ``None``, optional
            The offset index within the provided `patches` argument, thus the
            index of the second dimension from which to sample. If ``None``,
            then ``0`` is used.

        Raises
        ------
        ValueError
            If image is not 2D
        ValueError
            If offset does not have shape (1, 2)
        """
        # parse arguments
        if self.n_dims != 2:
            raise ValueError('Only two dimensional patch insertion is '
                             'currently supported.')
        if offset is None:
            offset = np.zeros([1, 2], dtype=np.intp)
        elif isinstance(offset, tuple) or isinstance(offset, list):
            offset = np.asarray([offset])
        offset = np.require(offset, dtype=np.intp)
        if not offset.shape == (1, 2):
            raise ValueError('The offset must be a tuple, a list or a '
                             'numpy.array with shape (1, 2).')
        if offset_index is None:
            offset_index = 0

        # if patches is a list, convert it to array
        if isinstance(patches, list):
            patches = _convert_patches_list_to_single_array(
                patches, patch_centers.n_points)

        # convert pixels to uint8 so that they get recognized by cython
        tmp_pixels = self.pixels.astype(np.uint8)
        # convert patches to uint8 as well and set them to pixels
        set_patches(patches.astype(np.uint8), tmp_pixels, patch_centers.points,
                    offset, offset_index)
        # convert pixels back to bool
        self.pixels = tmp_pixels.astype(np.bool)
