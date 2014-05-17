"""
This module is only designed for use inside of our testing. It isn't used or
exposed anywhere except in our tests. It is useful, because it contains general
methods that are applicable across many of our tests.
"""


def is_same_array(a, b):
    """
    Check if `a` and `b` represent the same piece of memory.

    Parameters
    ----------
    a : ndarray
        First array to compare
    b : ndarray
        Second array to compare

    Returns
    -------
    is_same : bool
        Will be `True` if the two arrays represent the same piece of memory.
    """
    if not a.flags['OWNDATA'] and not b.flags['OWNDATA']:
        return a.base is b.base
    if not a.flags['OWNDATA'] and b.flags['OWNDATA']:
        return a.base is b
    if not b.flags['OWNDATA'] and a.flags['OWNDATA']:
        return b.base is a

    # Fallthough, they are either the same array or they aren't!
    return a is b