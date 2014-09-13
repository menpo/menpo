from __future__ import division


def name_of_callable(c):
    try:
        return c.__name__  # function
    except AttributeError:
        return c.__class__.__name__  # callable class


def pyramid_on_features(features):
    r"""
    True if feature extraction happens once and then a gaussian pyramid
    is taken. False if a gaussian pyramid is taken and then features are
    extracted at each level.
    """
    return callable(features)
