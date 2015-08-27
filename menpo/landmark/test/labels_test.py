import menpo.landmark.labels as labels
from menpo.landmark import LandmarkGroup
import inspect
import numpy as np
from menpo.shape import PointCloud


def check_label_func(func, input_n_points, output_n_points):
    # Could be any dimensionality
    array = np.zeros(input_n_points)
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
        if fname != 'labeller':
            parts = fname.split('_')
            to_index = parts.index('to')
            input_n_points = int(parts[to_index - 1])
            output_n_points = parts[-1]
            try:
                output_n_points = int(output_n_points)
            except ValueError:
                # If it isn't the last element, there must be
                # a modifier - so try the second to last element
                output_n_points = int(parts[-2])

            yield (check_label_func,
                   func, input_n_points, output_n_points)
