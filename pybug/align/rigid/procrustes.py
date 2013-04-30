import numpy as np
from pybug.align.base import MultipleAlignment
from pybug.align.rigid.base import RigidAlignment
from pybug.transform import Rotation, Scale, Translation


class Procrustes(RigidAlignment):
    """Procrustes Alignment of a set of source landmarks to a target.
    """

    def __init__(self, source, target):
        RigidAlignment.__init__(self, source, target)
        self._procrustes_step()

    @property
    def error(self):
        """
        :return: The Frobenius Norm of the difference between the target and
        the aligned source.
        """
        return np.linalg.norm(self.target - self.aligned_source)

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
        self.scale = Scale(scale_target / scale_source, n_dim=self.n_dim)
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

    def __str__(self):
        msg = 'Alignment delta_target: %f\n' % self.error
        msg += 'Optimal alignment given by:\n'
        return msg + str(self.transform)


class GeneralizedProcrustesAnalysis(MultipleAlignment):
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
        new_target = (sum(p[-1].aligned_source for p in self.procrustes) /
                      self.n_sources)
        rescale = Scale(self.target_scale / np.linalg.norm(new_target),
                        n_dim=self.n_dim)
        centre = Translation(-new_target.mean(axis=0))
        rescale_about_com = centre.chain(rescale).chain(centre.inverse)
        new_target = rescale_about_com.apply(new_target)
        self.delta_target = np.linalg.norm(self.target - new_target)
        print 'at iteration %d, the delta_target is %f' % (self.n_iterations,
                                                           self.delta_target)
        if self.delta_target < 1e-6:
            print 'delta_target sufficiently small, stopping.'
            return True
        else:
            self.n_iterations += 1
            self.target = new_target
            for p in self.procrustes:
                p.append(Procrustes(p[-1].aligned_source, new_target))
            return self._recursive_procrustes()

    @property
    def transforms(self):
        return [reduce(lambda a, b: a.chain(b), [x.transform for x in p])
                for p in self.procrustes]

    @property
    def errors(self):
        return [p[-1].error for p in self.procrustes]

    @property
    def average_error(self):
        return sum(self.errors)/self.n_sources

    def __str__(self):
        return ('Converged after %d iterations with av. error %f'
                % (self.n_iterations, self.average_error))
