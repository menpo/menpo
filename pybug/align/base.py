import abc
import numpy as np
from matplotlib import pyplot
from pybug.exceptions import DimensionalityError


class Alignment(object):
    r"""
    Abstract base class for alignment algorithms. Alignment algorithms should
    align a single source object to a target.

    Parameters
    ----------
    source : (N, D) ndarray
        Source to align from
    target : (N, D) ndarray
        Target to align to

    Raises
    ------
    DimensionalityError
        Raised when the ndarrays are not 2-dimensional (N, D)
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, source, target):
        self.source = source.copy()
        self.aligned_source = self.source.copy()
        try:
            self.n_landmarks, self.n_dim = self.source.shape
        except ValueError:
            raise DimensionalityError('Data is being provided in an invalid '
                                      'format - must have shape '
                                      '(n_landmarks, n_dim)')
        assert self.n_dim, self.n_landmarks == target.shape
        self.target = target.copy()

    def _view_2d(self, image=False):
        """
        Visualize how points are affected by the warp in 2 dimensions.
        """
        #TODO this should be separated out into visualize.
        # a factor by which the minimum and maximum x and y values of the warp
        # will be increased by.
        x_margin_factor, y_margin_factor = 0.5, 0.5
        # the number of x and y samples to take
        n_x, n_y = 50, 50
        # {x y}_{min max} is the actual bounds on either source or target
        # landmarks
        x_min, y_min = np.vstack(
            [self.target.min(0), self.source.min(0)]).min(0)
        x_max, y_max = np.vstack(
            [self.target.max(0), self.source.max(0)]).max(0)
        x_margin = x_margin_factor * (x_max - x_min)
        y_margin = y_margin_factor * (y_max - y_min)
        # {x y}_{min max}_m is the bound once it has been grown by the factor
        # of the spread in that dimension
        x_min_m = x_min - x_margin
        x_max_m = x_max + x_margin
        y_min_m = y_min - y_margin
        y_max_m = y_max + y_margin
        # build sample points for the selected region
        x = np.linspace(x_min_m, x_max_m, n_x)
        y = np.linspace(y_min_m, y_max_m, n_y)
        xx, yy = np.meshgrid(x, y)
        sample_coords = np.concatenate(
            [xx.reshape([-1, 1]), yy.reshape([-1, 1])], axis=1)
        warped_coords = self.transform.apply(sample_coords)
        delta = warped_coords - sample_coords
        # plot the sample points result
        x, y, = 0, 1
        if image:
            # if we are overlaying points onto an image,
            # we have to account for the fact that axis 0 is typically
            # called 'y' and axis 1 is typically called 'x'. Flip them here
            x, y = y, x
        pyplot.quiver(sample_coords[:, x], sample_coords[:, y], delta[:, x],
                      delta[:, y])
        delta = self.target - self.source
        # plot how the landmarks move from source to target
        pyplot.quiver(self.source[:, x], self.source[:, y], delta[:, x],
                      delta[:, y], angles='xy', scale_units='xy', scale=1)
        # rescale to the bounds
        pyplot.xlim((x_min_m, x_max_m))
        pyplot.ylim((y_min_m, y_max_m))
        if image:
            # if we are overlaying points on an image, axis0 (the 'y' axis)
            # is flipped.
            pyplot.gca().invert_yaxis()

    @abc.abstractproperty
    def transform(self):
        r"""
        Returns a single instance of the Transform that can be applied.

        :type: :class:`pybug.transform.base.Transform`
        """
        pass


class MultipleAlignment(object):
    r"""
    Abstract base class for aligning multiple sources to a target.

    Parameters
    ----------
    sources : (N, D) list of ndarrays
        List of points to be aligned
    target : (N, D) ndarray, optional
        The target to align to.

        Default: The ``mean`` of sources
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, sources, target=None):
        if len(sources) < 2 and target is None:
            raise Exception("Need at least two sources to align")
        self.n_sources = len(sources)
        self.n_landmarks, self.n_dim = sources[0].shape
        self.sources = sources
        if target is None:
            # set the target to the mean source position
            self.target = sum(self.sources) / self.n_sources
        else:
            assert self.n_dim, self.n_landmarks == target.shape
            self.target = target

    @abc.abstractproperty
    def transforms(self):
        r"""
        Returns a list of transforms, one for each source, which aligns it to
        the target.

        :type: list of :class:`pybug.transform.base.Transform`
        """
        pass
