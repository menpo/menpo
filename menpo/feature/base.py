import numpy as np
import wrapt
from menpo.image import Image, MaskedImage, BooleanImage
from menpo.transform import Translation, NonUniformScale


def _init_feature_image(image, pixels, window_centres=None,
                        constrain_landmarks=True):
    r"""
    Creates a new image object to store the feature_pixels. If the original
    object is of MaskedImage class, then the features object is of
    MaskedImage as well. If the original object is of any other image
    class, the output object is of Image class.

    Parameters
    ----------
    feature_pixels :  ndarray.
        The pixels of the features image.

    window_centres :  ndarray.
        The sampled pixels from where the features were extracted. It has
        size n_rows x n_columns x 2, where window_centres[:, :, 0] are the
        row indices and window_centres[:, :, 1] are the column indices.

    constrain_landmarks : bool
        Flag that if enabled, it constrains landmarks to image bounds.

        Default: True
    """
    if isinstance(image, MaskedImage):
        # if we have a MaskedImage object
        feature_image = MaskedImage(pixels, copy=False)
        # fix mask
        transfer_mask(feature_image, window_centres=window_centres)
    else:
        # if we have an Image object
        feature_image = Image(pixels, copy=False)
    # fix landmarks
    transfer_landmarks(image, feature_image, centres=window_centres,
                       constrain_landmarks=constrain_landmarks)
    if window_centres is not None:
        feature_image.window_centres = window_centres
    return feature_image


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


@wrapt.decorator
def imgfeature(wrapped, instance, args, kwargs):
    def _execute(image, *args, **kwargs):
        if isinstance(image, np.ndarray):
            # ndarray supplied to Image feature - build a
            # temp image for it and just return the pixels
            image = Image(image, copy=False)
            return wrapped(image, *args, **kwargs).pixels
        else:
            return wrapped(image, *args, **kwargs)
    return _execute(*args, **kwargs)


@wrapt.decorator
def ndfeature(wrapped, instance, args, kwargs):
    def _execute(image, *args, **kwargs):
        if not isinstance(image, np.ndarray):
            # Image supplied to ndarray feature -
            # extract pixels and go
            feature = wrapped(image.pixels, *args, **kwargs)
            return Image(feature, copy=False)
        else:
            return wrapped(image, *args, **kwargs)
    return _execute(*args, **kwargs)


@wrapt.decorator
def winitfeature(wrapped, instance, args, kwargs):
    def _execute(image, *args, **kwargs):
        if not isinstance(image, np.ndarray):
            # Image supplied to ndarray feature -
            # extract pixels and go
            feature, centres = wrapped(image.pixels, *args, **kwargs)
            new_image = image.__class__.__new__(image.__class__)
            new_image.pixels = feature
            if hasattr(image, 'mask'):
                new_image.mask = sample_mask_for_centres(image.mask.mask,
                                                         centres)
            if image.has_landmarks:
                t = lm_centres_correction(centres)
                new_image.landmarks = t.apply(image.landmarks)
            return new_image
        else:
            # user just supplied ndarray - give them ndarray back
            return wrapped(image, *args, **kwargs)[0]

    return _execute(*args, **kwargs)
