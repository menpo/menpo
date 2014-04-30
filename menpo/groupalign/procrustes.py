import numpy as np
from menpo.groupalign.base import MultipleAlignment
from menpo.shape import PointCloud
from menpo.transform import AlignmentSimilarity, UniformScale, Translation


class GeneralizedProcrustesAnalysis(MultipleAlignment):
    r"""
    TODO: Write documentation
    """
    def __init__(self, sources, **kwargs):
        super(GeneralizedProcrustesAnalysis, self).__init__(sources, **kwargs)
        self.transforms = [AlignmentSimilarity(source, self.target)
                           for source in self.sources]
        self.initial_target_scale = self.target.norm()
        self.n_iterations = 1
        self.max_iterations = 100
        self.converged = self._recursive_procrustes()

    def _recursive_procrustes(self):
        """
        Recursively calculates a Procrustes alignment
        """
        if self.n_iterations > self.max_iterations:
            return False
        av_aligned_source = sum(
            t.aligned_source.points for t in self.transforms) / self.n_sources
        new_target = PointCloud(av_aligned_source)
        # rescale the new_target to be the same size as the original about
        # it's centre
        rescale = UniformScale(
            self.initial_target_scale / new_target.norm(), self.n_dims)
        centre = Translation(-new_target.centre)
        rescale_about_centre = centre.compose_before(rescale).compose_before(
            centre.pseudoinverse)
        rescale_about_centre.apply_inplace(new_target)
        # check to see if  we have converged yet
        delta_target = np.linalg.norm(self.target.points - new_target.points)
        if delta_target < 1e-6:
            return True
        else:
            self.n_iterations += 1
            for t in self.transforms:
                t.set_target(new_target)
            self.target = new_target
            return self._recursive_procrustes()

    @property
    def mean_aligned_shape(self):
        return PointCloud(np.mean([t.target.points for t in self.transforms],
                                  axis=0))

    @property
    def av_alignment_error(self):
        return sum([t.alignment_error for t in self.transforms])/self.n_sources

    def __str__(self):
        if self.converged:
            return ('Converged after %d iterations with av. error %f'
                    % (self.n_iterations, self.av_alignment_error))
        else:
            return ('Failed to converge after %d iterations with av. error '
                    '%f' % (self.n_iterations, self.av_alignment_error))
