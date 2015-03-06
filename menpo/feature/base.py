from __future__ import division
from functools import wraps
import numpy as np
from menpo.image import Image, MaskedImage, BooleanImage
from menpo.transform import Translation, NonUniformScale


def lm_centres_correction(centres):
    r"""
    Construct a transform that will correct landmarks for a window
    iterating feature calculation

    Parameters
    ----------
    centres : `ndarray` (H, W, 2)
        The location of the window centres in the features

    Returns
    -------
    :map:`Affine`
        An affine transform that performs the correction.
        Should be applied to the landmarks on the target image.
    """
    t = Translation(-centres.min(axis=0).min(axis=0), skip_checks=True)
    step_v = centres[0, 0, 0]
    if centres.shape[0] > 1:
        step_v = centres[1, 0, 0] - centres[0, 0, 0]
    step_h = centres[0, 0, 1]
    if centres.shape[1] > 1:
        step_h = centres[0, 1, 1] - centres[0, 0, 1]
    s = NonUniformScale((1./step_v, 1./step_h), skip_checks=True)
    return t.compose_before(s)


def sample_mask_for_centres(mask, centres):
    r"""
    Sample a mask at the centres

    Parameters
    ----------
    mask :  Either MaskedImage or Image class.
        The target image object that includes the windows_centres.

    window_centres : ndarray, optional
        If set, use these window centres to rescale the landmarks
        appropriately. If None, no scaling is applied.

    """
    return BooleanImage(mask[centres[..., 0], centres[..., 1]], copy=False)


def rebuild_feature_image(image, f_pixels):
    shape_changed = f_pixels.shape[1:] != image.shape
    if hasattr(image, 'mask'):
        # original image had a mask. Did the feature generate an image of the
        # same size?
        if shape_changed:
            # feature is of a different size - best we can do is rescale the
            # mask
            mask = image.mask.resize(f_pixels.shape[1:])
        else:
            # feature is same size as input
            mask = image.mask.copy()
        new_image = MaskedImage(f_pixels, mask=mask, copy=False)
    else:
        new_image = Image(f_pixels, copy=False)
    if image.has_landmarks:
        if shape_changed:
            # need to adjust the landmarks
            sf = np.array(f_pixels.shape[1:]) / np.array(image.shape)
            new_image.landmarks = NonUniformScale(sf).apply(image.landmarks)
        else:
            new_image.landmarks = image.landmarks
    return new_image


def rebuild_feature_image_with_centres(image, f_pixels, centres):
    if hasattr(image, 'mask'):
        mask = sample_mask_for_centres(image.mask.mask, centres)
        new_image = MaskedImage(f_pixels, mask=mask, copy=False)
    else:
        new_image = Image(f_pixels, copy=False)
    if image.has_landmarks:
        t = lm_centres_correction(centres)
        new_image.landmarks = t.apply(image.landmarks)
    return new_image


def imgfeature(wrapped):

    @wraps(wrapped)
    def wrapper(image, *args, **kwargs):
        if isinstance(image, np.ndarray):
            # ndarray supplied to Image feature - build a
            # temp image for it and just return the pixels
            image = Image(image, copy=False)
            return wrapped(image, *args, **kwargs).pixels
        else:
            return wrapped(image, *args, **kwargs)
    return wrapper


def ndfeature(wrapped):

    @wraps(wrapped)
    def wrapper(image, *args, **kwargs):
        if not isinstance(image, np.ndarray):
            # Image supplied to ndarray feature -
            # extract pixels and go
            feature = wrapped(image.pixels, *args, **kwargs)
            return rebuild_feature_image(image, feature)
        else:
            return wrapped(image, *args, **kwargs)
    return wrapper


def winitfeature(wrapped):

    @wraps(wrapped)
    def wrapper(image, *args, **kwargs):
        if not isinstance(image, np.ndarray):
            # Image supplied to ndarray feature -
            # extract pixels and go
            feature, centres = wrapped(image.pixels, *args, **kwargs)
            return rebuild_feature_image_with_centres(image, feature, centres)
        else:
            # user just supplied ndarray - give them ndarray back
            return wrapped(image, *args, **kwargs)[0]

    return wrapper
