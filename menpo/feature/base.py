import numpy as np
import wrapt
from menpo.image import Image, MaskedImage


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
    transfer_landmarks(image, feature_image, window_centres=window_centres,
                       constrain_landmarks=constrain_landmarks)
    if window_centres is not None:
        feature_image.window_centres = window_centres
    return feature_image


def transfer_landmarks(image, feature_image, window_centres=None,
                       constrain_landmarks=True):
    r"""
    Transfers its own landmarks to the target_image object after
    appropriately correcting them. The landmarks correction is achieved
    based on the windows_centres of the features object.

    Parameters
    ----------
    target_image :  Either MaskedImage or Image class.
        The target image object that includes the windows_centres.

    window_centres : ndarray, optional
        If set, use these window centres to rescale the landmarks
        appropriately. If None, no scaling is applied.

    constrain_landmarks : bool
        Flag that if enabled, it constrains landmarks to image bounds.

        Default: True
    """
    feature_image.landmarks = image.landmarks
    if window_centres is not None:
        if feature_image.landmarks.has_landmarks:
            for l_group in feature_image.landmarks:
                l = feature_image.landmarks[l_group]
                # find the vertical and horizontal sampling steps
                step_vertical = window_centres[0, 0, 0]
                if window_centres.shape[0] > 1:
                    step_vertical = \
                        (window_centres[1, 0, 0] -
                         window_centres[0, 0, 0])
                step_horizontal = window_centres[0, 0, 1]
                if window_centres.shape[1] > 1:
                    step_horizontal = \
                        (window_centres[0, 1, 1] -
                         window_centres[0, 0, 1])
                # convert points by subtracting offset and dividing with
                # step at each direction
                l.lms.points[:, 0] = \
                    (l.lms.points[:, 0] -
                     window_centres[:, :, 0].min()) / \
                    step_vertical
                l.lms.points[:, 1] = \
                    (l.lms.points[:, 1] -
                     window_centres[:, :, 1].min()) / \
                    step_horizontal
    # constrain landmarks to image bounds if asked
    if constrain_landmarks:
        feature_image.constrain_landmarks_to_bounds()


def transfer_mask(image, feature_image, window_centres=None):
    r"""
    Transfers its own mask to the target_image object after
    appropriately correcting it. The mask correction is achieved based on
    the windows_centres of the features object.

    Parameters
    ----------
    target_image :  Either MaskedImage or Image class.
        The target image object that includes the windows_centres.

    window_centres : ndarray, optional
        If set, use these window centres to rescale the landmarks
        appropriately. If None, no scaling is applied.

    """
    from menpo.image import BooleanImage
    mask = image.mask.mask  # don't want a channel axis!
    if window_centres is not None:
        mask = mask[window_centres[..., 0], window_centres[..., 1]]
    feature_image.mask = BooleanImage(mask)



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
