import pytest

import menpo.landmark.labels as labels
import inspect
import numpy as np
from menpo.shape import PointCloud


EXEMPT_FUNCTIONS = {'labeller', 'bounding_box_to_bounding_box',
                    'bounding_box_mirrored_to_bounding_box'}

valid_label_functions = filter(
    lambda x: x[0] not in EXEMPT_FUNCTIONS,
    inspect.getmembers(labels, predicate=inspect.isfunction)
)


def check_label_func(func, input_n_points, output_n_points):
    # Could be any dimensionality
    array = np.zeros([input_n_points, 2])
    pcloud = PointCloud(array)

    array_result = func(array)
    assert isinstance(array_result, PointCloud)
    assert array_result.n_points == output_n_points

    pcloud_result = func(pcloud)
    assert isinstance(pcloud_result, PointCloud)
    assert pcloud_result.n_points == output_n_points


@pytest.mark.parametrize('fname, func', valid_label_functions)
def test_labels(fname, func):
    parts = fname.split('_')
    to_index = parts.index('to')
    input_n_points = parts[to_index - 1]
    output_n_points = parts[-1]
    try:
        input_n_points = int(input_n_points)
    except ValueError:
        # If it isn't the element before 'to', there must be
        # a modifier - so try the element before that
        input_n_points = int(parts[to_index - 2])
    try:
        output_n_points = int(output_n_points)
    except ValueError:
        # If it isn't the last element, there must be
        # a modifier - so try the second to last element
        output_n_points = int(parts[-2])

    check_label_func(func, input_n_points, output_n_points)


def test_bounding_box_to_bounding_box():
    from menpo.landmark import bounding_box_to_bounding_box

    check_label_func(bounding_box_to_bounding_box, 4, 4)


def test_bounding_box_mirrored_to_bounding_box():
    from menpo.landmark import bounding_box_mirrored_to_bounding_box

    check_label_func(bounding_box_mirrored_to_bounding_box, 4, 4)
