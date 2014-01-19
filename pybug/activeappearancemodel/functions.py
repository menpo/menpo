from __future__ import division
import matplotlib.pylab as plt
import numpy as np


def compute_error_rms(fitted, ground_truth):
    return np.sqrt(np.mean((fitted.flatten() - ground_truth.flatten()) ** 2))


def compute_error_p2p(fitted, ground_truth):
    return np.mean(np.sqrt(np.sum((fitted - ground_truth) ** 2, axis=-1)))


def compute_error_facesize(fitted, ground_truth):
    face_size = np.mean(np.max(ground_truth, axis=0) -
                        np.min(ground_truth, axis=0))
    return compute_error_p2p(fitted, ground_truth) / face_size


def compute_error_me17(fitted, ground_truth, leye, reye):
    return (compute_error_p2p(fitted, ground_truth) /
            compute_error_p2p(leye, reye))


def plot_ced(fitted_shapes, original_shape, error_type='face_size',
             label=None):

    if error_type is 'rms':
        errors = [compute_error_rms(f.points, o.points)
                  for f, o in zip(fitted_shapes, original_shape)]
        stop = 0.1
        step = 0.001
    elif error_type is 'p2p':
        errors = [compute_error_p2p(f.points, o.points)
                  for f, o in zip(fitted_shapes, original_shape)]
        stop = 0.1
        step = 0.001
    elif error_type is 'face_size':
        errors = [compute_error_facesize(f.points, o.points)
                  for f, o in zip(fitted_shapes, original_shape)]
        stop = 0.1
        step = 0.001

        plt.xlabel('point-to-point error normalized by face size')
        plt.ylabel('proportion of images')

    elif error_type is 'me17':
        errors = [compute_error_me17(f.points, o.points)
                  for f, o in zip(fitted_shapes, original_shape)]
        stop = 0.1
        step = 0.001

    n_shapes = len(fitted_shapes)
    error_axis = np.arange(0, stop, step)
    proportion_axis = [np.count_nonzero(errors < limit) / n_shapes
                       for limit in error_axis]

    error_median = np.median(errors)
    error_mean = np.mean(errors)
    error_std = np.std(errors)

    text = label + '  median = {}  mean = {}  std = {}'.format(
        error_median, error_mean, error_std)

    plt.plot(error_axis, proportion_axis, label=text)

    plt.title('Cumulative Error Distribution')
    plt.grid(True)
    plt.legend()
    plt.show()

    return errors, error_median, error_mean, error_std