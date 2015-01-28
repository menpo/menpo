import numpy as np

from menpo.transform import AlignmentSimilarity, UniformScale, Translation
from .base import MultipleAlignment

mean_pointcloud = None  # to avoid circular imports
PointCloud = None       # to avoid circular imports
Similarity = None       # to avoid circular imports


class GeneralizedProcrustesAnalysis(MultipleAlignment):
    r"""
    Class for aligning multiple source shapes between them.

    After construction, the :map:`AlignmentSimilarity` transforms used to map
    each source optimally to the target can be found at `transforms`.

    Parameters
    ----------
    sources : list of :map:`PointCloud`
        List of pointclouds to be aligned.

    target : :map:`PointCloud`
        The target :map:`PointCloud` to align each source to.
        If None, then the mean of the sources is used.

        Default: None

    Raises
    -------
    ValueError
        Need at least two sources to align

    """
    def __init__(self, sources, target=None):
        super(GeneralizedProcrustesAnalysis, self).__init__(sources,
                                                            target=target)
        initial_target = self.target
        self.transforms = [AlignmentSimilarity(source, self.target)
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
        global mean_pointcloud, PointCloud, Similarity
        if mean_pointcloud is None or PointCloud is None or Similarity is None:
            from menpo.shape import mean_pointcloud, PointCloud
            from menpo.transform import Similarity
        if self.n_iterations > self.max_iterations:
            return False
        new_tgt = mean_pointcloud([PointCloud(t.aligned_source().points,
                                              copy=False)
                                   for t in self.transforms])
        # rescale the new_target to be the same size as the original about
        # it's centre
        rescale = Similarity.identity(new_tgt.n_dims)

        s = UniformScale(self.initial_target_scale / new_tgt.norm(),
                         self.n_dims, skip_checks=True)
        t = Translation(-new_tgt.centre(), skip_checks=True)
        rescale.compose_before_inplace(t)
        rescale.compose_before_inplace(s)
        rescale.compose_before_inplace(t.pseudoinverse())
        rescale.apply_inplace(new_tgt)
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

        :type: PointCloud
        """
        from menpo.shape import PointCloud
        return PointCloud(np.mean([t.target.points for t in self.transforms],
                                  axis=0))

    def mean_alignment_error(self):
        r"""
        Returns the average error of the recursive procrustes alignment.

        :type: float
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
