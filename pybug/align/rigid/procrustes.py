import numpy as np
from pybug.align.rigid.base import RigidAlignment, ParallelRigidAlignment
from pybug.transform import Rotation, Scale, Translation


class Procrustes(RigidAlignment):
    """Procrustes Alignment of a set of source landmarks to a target.
    """

    def __init__(self, source, target):
        RigidAlignment.__init__(self, source, target)
        self._procrustes_step()

    @property
    def error(self):
        return np.sum((self.target - self.aligned_source) ** 2)

    @property
    def transform_chain(self):
        return [self.source_translation, self.scale, self.rotation,
                self.target_translation.inverse]

    @property
    def transform(self):
        return reduce(lambda x, y: x.chain(y), self.transform_chain)

    def _procrustes_step(self):
        # firstly, translate the target to the origin
        self.target_translation = Translation(-self.target.mean(axis=0))
        self.centred_target = self.target_translation.apply(self.target)
        # now translate the source to the origin
        self.source_translation = Translation(-self.source.mean(axis=0))
        # apply the translation to the source
        self.aligned_source = self.source_translation.apply(self.source)
        scale_source = np.linalg.norm(self.aligned_source)
        scale_target = np.linalg.norm(self.centred_target)
        self.scale = Scale(np.ones(self.n_dim) * (scale_target /
                                                  scale_source))
        self.aligned_source = self.scale.apply(self.aligned_source)
        # calculate the correlation along each dimension + find the optimal
        # rotation to maximise it
        correlation = np.dot(self.aligned_source.T, self.centred_target)
        U, D, Vt = np.linalg.svd(correlation)
        self.rotation = Rotation(np.dot(U, Vt))
        self.aligned_source = self.rotation.apply(self.aligned_source)
        # finally, move the source back out to where the target is
        self.aligned_source = self.target_translation.inverse.apply(
            self.aligned_source)


class GeneralizedProcrustesAnalysis(ParallelRigidAlignment):
    def __init__(self, sources, **kwargs):
        super(GeneralizedProcrustesAnalysis, self).__init__(sources, **kwargs)
        self.procrustes = [[Procrustes(s, self.target)] for s in self.sources]
        self.target_scale = np.linalg.norm(self.target)
        self.n_iterations = 1
        self.max_iterations = 100
        self._recursive_procrustes()

    def _recursive_procrustes(self):
        """
        Recursively calculates a Procrustes alignment
        """
        # find the average of the latest aligned sources:
        if self.n_iterations > self.max_iterations:
            print 'max number of iterations reached.'
            return False
        new_target = sum(p[-1].aligned_source for p in self.procrustes) \
                     / self.n_sources
        new_target *= self.target_scale / np.linalg.norm(new_target)
        self.error = np.linalg.norm(self.target - new_target)
        print 'at iteration %d, the error is %f' % (self.n_iterations,
                                                    self.error)
        if self.error < 1e-6:
            print 'error sufficiently small, stopping.'
            return True
        else:
            self.n_iterations += 1
            self.target = new_target
            for p in self.procrustes:
                p.append(Procrustes(p[-1].aligned_source, new_target))
            return self._recursive_procrustes()
