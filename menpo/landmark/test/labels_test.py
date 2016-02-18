import menpo.landmark.labels as labels
from menpo.landmark import LandmarkGroup
import inspect
import numpy as np
from menpo.shape import PointCloud, bounding_box


EXEMPT_FUNCTIONS = {'labeller', 'bounding_box_to_bounding_box',
                    'bounding_box_mirrored_to_bounding_box'}


def check_label_func(func, input_n_points, output_n_points):
    # Could be any dimensionality
    array = np.zeros([input_n_points, 2])
    pcloud = PointCloud(array)
    lmark_g = LandmarkGroup.init_with_all_label(pcloud)

    array_result = func(array)
    assert isinstance(array_result, PointCloud)
    assert array_result.n_points == output_n_points

    pcloud_result = func(pcloud)
    assert isinstance(pcloud_result, PointCloud)
    assert pcloud_result.n_points == output_n_points

    lmark_g_result = func(lmark_g)
    assert isinstance(lmark_g_result, LandmarkGroup)
    assert lmark_g_result.lms.n_points == output_n_points


def test_labels():
    for fname, func in inspect.getmembers(labels,
                                          predicate=inspect.isfunction):
        if fname not in EXEMPT_FUNCTIONS:
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

            yield (check_label_func,
                   func, input_n_points, output_n_points)


def test_bounding_box_to_bounding_box():
    from menpo.landmark import bounding_box_to_bounding_box

    check_label_func(bounding_box_to_bounding_box, 4, 4)


def test_bounding_box_mirrored_to_bounding_box():
    from menpo.landmark import bounding_box_mirrored_to_bounding_box

    check_label_func(bounding_box_mirrored_to_bounding_box, 4, 4)
