import numpy as np
from collections import OrderedDict

from .base import labeller_func


@labeller_func(group_label='bounding_box')
def bounding_box_to_bounding_box(bbox):
    r"""
    Apply a single 'all' label to a given bounding box. This bounding
    box must be as specified by the :map:`bounding_box` method.
    """
    from menpo.shape import bounding_box

    mapping = OrderedDict()
    mapping['all'] = np.arange(4)
    return bounding_box(bbox.points[0], bbox.points[2]), mapping


@labeller_func(group_label='bounding_box')
def bounding_box_mirrored_to_bounding_box(bbox):
    r"""
    Apply a single 'all' label to a given bounding box that has been
    mirrored around the vertical axis (flipped around the Y-axis). This bounding
    box must be as specified by the :map:`bounding_box` method (but mirrored).
    """
    from menpo.shape import bounding_box

    mapping = OrderedDict()
    mapping['all'] = np.arange(4)
    return bounding_box(bbox.points[3], bbox.points[1]), mapping
