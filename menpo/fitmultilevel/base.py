from __future__ import division


def name_of_callable(c):
    try:
        return c.__name__  # function
    except AttributeError:
        return c.__class__.__name__  # callable class


def is_pyramid_on_features(features):
    r"""
    True if feature extraction happens once and then a gaussian pyramid
    is taken. False if a gaussian pyramid is taken and then features are
    extracted at each level.
    """
    return callable(features)


def create_pyramid(images, n_levels, downscale, features):
    r"""
    Function that creates a generator function for Gaussian pyramid. The
    pyramid can be created either on the feature space or the original
    (intensities) space.

    Parameters
    ----------
    images: list of :map:`Image`
        The set of landmarked images from which to build the AAM.

    n_levels: int
        The number of multi-resolution pyramidal levels to be used.

    downscale: float
        The downscale factor that will be used to create the different
        pyramidal levels.

    features: ``callable`` ``[callable]``
        If a single callable, then the feature calculation will happen once
        followed by a gaussian pyramid. If a list of callables then a
        gaussian pyramid is generated with features extracted at each level
        (after downsizing and blurring).

    Returns
    -------
    list of generators :
        The generator function of the Gaussian pyramid.

    """
    return [pyramid_of_feature_images(n_levels, downscale, features, i)
            for i in images]


def pyramid_of_feature_images(n_levels, downscale, features, image):
    r"""
    Generates a gaussian pyramid of feature images for a single image.
    """
    if is_pyramid_on_features(features):
        # compute feature image at the top
        feature_image = features(image)
        # create pyramid on the feature image
        return feature_image.gaussian_pyramid(n_levels=n_levels,
                                              downscale=downscale)
    else:
        # create pyramid on intensities image
        # feature will be computed per level
        pyramid = image.gaussian_pyramid(n_levels=n_levels,
                                         downscale=downscale)
        # add the feature generation here
        return feature_images(pyramid, features)


# adds feature extraction to a generator of images
def feature_images(images, features):
    for feature, level in zip(reversed(features), images):
        yield feature(level)


class DeformableModel(object):

    def __init__(self, features):
        self.features = features

    @property
    def pyramid_on_features(self):
        return is_pyramid_on_features(self.features)
