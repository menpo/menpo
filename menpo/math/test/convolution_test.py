import numpy as np
import pytest

from menpo.math import log_gabor


def test_log_gabor_2d():
    image = np.random.RandomState(0).rand(16, 16)
    complex_conv, bandpass, S = log_gabor(image, num_scales=2, num_orientations=3)
    assert complex_conv.shape == (2, 3, 16, 16)
    assert np.iscomplexobj(complex_conv) and np.iscomplexobj(bandpass)
    assert np.all(np.isfinite(complex_conv))
    assert S.shape == (16, 16)


def test_log_gabor_3d():
    # np.complex, used to allocate the results, was removed in NumPy 1.24
    image = np.random.RandomState(0).rand(8, 8, 8)
    complex_conv, bandpass, S = log_gabor(
        image, num_scales=2, num_theta_orientations=2, num_phi_orientations=2
    )
    assert complex_conv.shape == (2, 2, 2, 8, 8, 8)
    assert np.all(np.isfinite(complex_conv))


def test_log_gabor_needs_2d_or_3d_arrays():
    with pytest.raises(ValueError):
        log_gabor(np.zeros(5))
