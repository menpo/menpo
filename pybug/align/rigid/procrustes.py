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
        print self.aligned_source
        scale_source = np.linalg.norm(self.aligned_source)
        scale_target = np.linalg.norm(self.centred_target)
        self.scale = Scale(np.ones(self.n_dim) * (scale_target /
                                                  scale_source))
        self.aligned_source = self.scale.apply(self.aligned_source)
        print self.aligned_source
        # calculate the correlation along each dimension + find the optimal
        # rotation to maximise it
        correlation = np.dot(self.aligned_source.T, self.centred_target)
        U, D, Vt = np.linalg.svd(correlation)
        self.rotation = Rotation(np.dot(U, Vt))
        self.aligned_source = self.rotation.apply(self.aligned_source)
        # finally, move the source back out to where the target is
        self.aligned_source = self.target_translation.inverse.apply(
            self.aligned_source)
        print self.aligned_source


class GeneralizedProcrustesAlignment(ParallelRigidAlignment):
    def __init__(self, sources, **kwargs):
        super(GeneralizedProcrustesAlignment, self).__init__(sources, **kwargs)
        self.procrustes = [[Procrustes(s, self.target)] for s in self.sources]
        self.target_scale = np.linalg.norm(self.target)

    def _recompute_mean_target(self):
        self.target = sum(self.sources)
        self.target *= (self.target_scale / np.linalg.norm(self.target))

    def _recursive_procrustes(self):
        pass