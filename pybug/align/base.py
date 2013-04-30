import abc
import numpy as np
from matplotlib import pyplot
from pybug.align.exceptions import AlignmentError


class Alignment(object):
    """ Aligns a single source object to a target.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, source, target):
        """ source - ndarray of landmarks which will be aligned of shape
         (n_landmarks, n_dim)

        target  - an ndarray (of the same dimension of source) which
                  the source will be aligned to.
        """
        self.source = source.copy()
        self.aligned_source = self.source.copy()
        try:
            self.n_landmarks, self.n_dim = self.source.shape
        except ValueError:
            raise AlignmentError('Data is being provided in an invalid format'
                                 ' - must have shape (n_landmarks, n_dim)')
        assert self.n_dim, self.n_landmarks == target.shape
        self.target = target.copy()

    def _view_2d(self):
        """ Visualize how points are affected by the warp in 2 dimensions.
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
        warped_coords = self.mapping(sample_coords)
        delta = warped_coords - sample_coords
        # plot the sample points result
        pyplot.quiver(sample_coords[:, 0], sample_coords[:, 1], delta[:, 0],
                      delta[:, 1])
        delta = self.target - self.source
        # plot how the landmarks move from source to target
        pyplot.quiver(self.source[:, 0], self.source[:, 1], delta[:, 0],
                      delta[:, 1])
        # rescale to the bounds
        pyplot.xlim((x_min_m, x_max_m))
        pyplot.ylim((y_min_m, y_max_m))

    @abc.abstractmethod
    def transform(self):
        """
        Returns a single instance of Transform that can be applied.
        :return: a transform object
        """
        pass


class MultipleAlignment(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, sources, target=None):
        """ sources - an iterable of numpy arrays of landmarks which will be
                    aligned e.g. [landmarks_0, landmarks_1,
                    ...landmarks_n] where landmarks is an ndarray of
                    dimension [n_landmarks x n_dim]
          KWARGS
            target  - a single numpy array (of the same dimension of a
            source) which every instance of source will be aligned to. If
            not present, target is set to the mean source position.
        """
        if len(sources) < 2 and target is None:
            raise Exception("Need at least two sources to align")
        self.n_sources = len(sources)
        self.n_landmarks, self.n_dim = sources[0].shape
        self.sources = sources
        if target is None:
            # set the target to the mean source position
            self.target = sum(self.sources) / len(sources)
        else:
            assert self.n_dim, self.n_landmarks == target.shape
            self.target = target

    @abc.abstractproperty
    def transforms(self):
        """
        Returns a list of transforms. one for each source,
        which aligns it to the target.
        """
        pass
