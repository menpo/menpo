import numpy as np

from ..homogeneous import AlignmentSimilarity
from .base import MultipleAlignment


avoid_circular = None      # to avoid circular imports
mean_pointcloud = None     # to avoid circular imports
PointCloud = None          # to avoid circular imports
scale_about_centre = None  # to avoid circular imports


class GeneralizedProcrustesAnalysis(MultipleAlignment):
    r"""
    Class for aligning multiple source shapes between them.

    After construction, the :map:`AlignmentSimilarity` transforms used to map
    each `source` optimally to the `target` can be found at `transforms`.

    Parameters
    ----------
    sources : `list` of :map:`PointCloud`
        List of pointclouds to be aligned.
    target : :map:`PointCloud`, optional
        The target :map:`PointCloud` to align each source to.
        If ``None``, then the mean of the sources is used.
    allow_mirror : `bool`, optional
        If ``True``, the Kabsch algorithm check is not performed, and mirroring
        of the Rotation matrix is permitted.

    Raises
    ------
    ValueError
        Need at least two sources to align
    """
    def __init__(self, sources, target=None, allow_mirror=False):
        super(GeneralizedProcrustesAnalysis, self).__init__(sources,
                                                            target=target)
        initial_target = self.target
        self.transforms = [AlignmentSimilarity(source, self.target,
                                               allow_mirror=allow_mirror)
                           for source in self.sources]
        self.initial_target_scale = self.target.norm()
        self.n_iterations = 1
        self.max_iterations = 100
        self.converged = self._recursive_procrustes()
        if target is not None:
            self.target = initial_target

    def _recursive_procrustes(self):
        r"""
        Recursively calculates a procrustes alignment.
        """
        global mean_pointcloud, PointCloud, scale_about_centre, avoid_circular
        if avoid_circular is None:
            from menpo.shape import mean_pointcloud, PointCloud
            from ..compositions import scale_about_centre
            avoid_circular = True

        if self.n_iterations > self.max_iterations:
            return False
        new_tgt = mean_pointcloud([PointCloud(t.aligned_source().points,
                                              copy=False)
                                   for t in self.transforms])
        # rescale the new_target to be the same size as the original about
        # it's centre
        rescale = scale_about_centre(new_tgt,
                                     self.initial_target_scale / new_tgt.norm())
        rescale._apply_inplace(new_tgt)
        # check to see if we have converged yet
        delta_target = np.linalg.norm(self.target.points - new_tgt.points)
        if delta_target < 1e-6:
            return True
        else:
            self.n_iterations += 1
            for t in self.transforms:
                t.set_target(new_tgt)
            self.target = new_tgt
            return self._recursive_procrustes()

    def mean_aligned_shape(self):
        r"""
        Returns the mean of the aligned shapes.

        :type: :map:`PointCloud`
        """
        from menpo.shape import PointCloud
        return PointCloud(np.mean([t.target.points for t in self.transforms],
                                  axis=0))

    def mean_alignment_error(self):
        r"""
        Returns the average error of the recursive procrustes alignment.

        :type: `float`
        """
        return sum([t.alignment_error() for t in
                    self.transforms])/self.n_sources

    def __str__(self):
        if self.converged:
            return ('Converged after %d iterations with av. error %f'
                    % (self.n_iterations, self.mean_alignment_error()))
        else:
            return ('Failed to converge after %d iterations with av. error '
                    '%f' % (self.n_iterations, self.mean_alignment_error()))
