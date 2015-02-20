from __future__ import division
import numpy as np
from scipy.ndimage import gaussian_filter

from menpo.feature import gradient


def _daisy(img, step=4, radius=15, rings=3, histograms=8, orientations=8,
           normalization='l1', sigmas=None, ring_radii=None):
    r"""Extract DAISY feature descriptors densely for the given image.

    DAISY is a feature descriptor similar to SIFT formulated in a way that
    allows for fast dense extraction. Typically, this is practical for
    bag-of-features image representations.

    The implementation follows Tola et al. [1]_ but deviate on the following
    points:

      * Histogram bin contribution are smoothed with a circular Gaussian
        window over the tonal range (the angular range).
      * The sigma values of the spatial Gaussian smoothing in this code do not
        match the sigma values in the original code by Tola et al. [2]_. In
        their code, spatial smoothing is applied to both the input image and
        the centre histogram. However, this smoothing is not documented in [1]_
        and, therefore, it is omitted.

    Parameters
    ----------
    img : (M, N) array
        Input image (greyscale).
    step : int, optional
        Distance between descriptor sampling points.
    radius : int, optional
        Radius (in pixels) of the outermost ring.
    rings : int, optional
        Number of rings.
    histograms  : int, optional
        Number of histograms sampled per ring.
    orientations : int, optional
        Number of orientations (bins) per histogram.
    normalization : [ 'l1' | 'l2' | 'daisy' | 'off' ], optional
        How to normalize the descriptors

          * 'l1': L1-normalization of each descriptor.
          * 'l2': L2-normalization of each descriptor.
          * 'daisy': L2-normalization of individual histograms.
          * 'off': Disable normalization.
    sigmas : 1D array of float, optional
        Standard deviation of spatial Gaussian smoothing for the centre
        histogram and for each ring of histograms. The array of sigmas should
        be sorted from the centre and out. I.e. the first sigma value defines
        the spatial smoothing of the centre histogram and the last sigma value
        defines the spatial smoothing of the outermost ring. Specifying sigmas
        overrides the following parameter.

            ``rings = len(sigmas) - 1``
    ring_radii : 1D array of int, optional
        Radius (in pixels) for each ring. Specifying ring_radii overrides the
        following two parameters.

            ``rings = len(ring_radii)``
            ``radius = ring_radii[-1]``

        If both sigmas and ring_radii are given, they must satisfy the
        following predicate since no radius is needed for the centre
        histogram.

            ``len(ring_radii) == len(sigmas) + 1``

    Returns
    -------
    descs : array
        Grid of DAISY descriptors for the given image as an array
        dimensionality  (P, Q, R) where ::

            ``P = ceil((M - radius*2) / step)``
            ``Q = ceil((N - radius*2) / step)``
            ``R = (rings * histograms + 1) * orientations``

    descs_img : (M, N, 3) array (only if visualize==True)
        Visualization of the DAISY descriptors.

    References
    ----------
    .. [1] Tola et al. "Daisy: An efficient dense descriptor applied to wide-
           baseline stereo." Pattern Analysis and Machine Intelligence, IEEE
           Transactions on 32.5 (2010): 815-830.
    .. [2] http://cvlab.epfl.ch/alumni/tola/daisy.html
    """
    # Compute image derivatives.
    # Get number of input image's channels
    n_channels = img.shape[0]

    # Compute image gradient
    grad = gradient(img)

    # For each pixel, select gradient with highest magnitude
    grad_mag = np.zeros(img.shape[1:])
    grad_ori = np.zeros(img.shape[1:])
    for c in range(n_channels):
        c_grad_mag = np.sqrt(grad[c] ** 2 + grad[c + n_channels] ** 2)
        tmp_max_mask = c_grad_mag > grad_mag
        grad_mag[tmp_max_mask] = c_grad_mag[tmp_max_mask]
        grad_ori[tmp_max_mask] = np.arctan2(grad[c][tmp_max_mask],
                                            grad[c + n_channels][tmp_max_mask])

    orientation_kappa = orientations / np.pi
    orientation_angles = [2 * o * np.pi / orientations - np.pi
                          for o in range(orientations)]
    hist = np.empty((orientations,) + img.shape[1:], dtype=float)
    for i, o in enumerate(orientation_angles):
        # Weigh bin contribution by the circular normal distribution
        hist[i] = np.exp(orientation_kappa * np.cos(grad_ori - o))
        # Weigh bin contribution by the gradient magnitude
        hist[i] = hist[i] * grad_mag

    # Smooth orientation histograms for the center and all rings.
    sigmas = [sigmas[0]] + sigmas
    hist_smooth = np.empty((rings + 1,) + hist.shape, dtype=float)
    for i in range(rings + 1):
        for j in range(orientations):
            hist_smooth[i, j] = gaussian_filter(hist[j], sigma=sigmas[i])

    # Assemble descriptor grid.
    theta = [2 * np.pi * j / histograms for j in range(histograms)]
    desc_dims = (rings * histograms + 1) * orientations
    descs = np.empty((desc_dims, img.shape[1] - 2 * radius,
                      img.shape[2] - 2 * radius))
    descs[:orientations] = hist_smooth[0, :, radius:-radius, radius:-radius]
    idx = orientations
    for i in range(rings):
        for j in range(histograms):
            y_min = radius + int(np.round(ring_radii[i] * np.sin(theta[j])))
            y_max = descs.shape[1] + y_min
            x_min = radius + int(np.round(ring_radii[i] * np.cos(theta[j])))
            x_max = descs.shape[2] + x_min
            descs[idx:idx + orientations] = hist_smooth[i + 1, :,
                                                        y_min:y_max,
                                                        x_min:x_max]
            idx += orientations
    descs = descs[:, ::step, ::step]

    # Normalize descriptors.
    if normalization != 'off':
        descs += 1e-10
        if normalization == 'l1':
            descs /= np.sum(descs, axis=0)
        elif normalization == 'l2':
            descs /= np.sqrt(np.sum(descs ** 2, axis=0))
        elif normalization == 'daisy':
            for i in range(0, desc_dims, orientations):
                norms = np.sqrt(np.sum(descs[i:i + orientations] ** 2, axis=0))
                descs[i:i + orientations] /= norms

    descs = np.require(descs, requirements=['C'])

    return descs
