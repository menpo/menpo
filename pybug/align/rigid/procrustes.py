import numpy as np
from pybug.align.rigid.base import RigidAlignment, ParallelRigidAlignment


class Procrustes(RigidAlignment):
    """Procrustes Alignment of a set of source landmarks to a target.
    """
    def __init__(self, source, target):
        RigidAlignment.__init__(self, source, target)
        self._procrustes_step()

    @property
    def error(self):
        return np.sum((self.target - self.aligned_source) ** 2)

    def _procrustes_step(self):
        # calculate the translation required to align the sources' centre of
        # mass to the the target centre of mass
        self.translation = (self.target.mean(axis=0) - self.source.mean(
            axis=0))
        # apply the translation to the source
        print self.aligned_source
        np.add(self.source, self.translation, out=self.aligned_source)
        print self.aligned_source
        scale_source = np.linalg.norm(self.aligned_source)
        scale_target = np.linalg.norm(self.target)
        self.scale = scale_target / scale_source
        np.multiply(self.aligned_source, self.scale, out=self.aligned_source)
        print self.aligned_source
        # calculate the correlation along each dimension
        correlation = np.dot(self.aligned_source.T, self.target)
        U, D, Vt = np.linalg.svd(correlation)
        # find the optimal rotation to minimise rotational differences
        self.rotation = np.dot(U, Vt)
        # apply the rotation
        np.dot(self.aligned_source, self.rotation.T, out=self.aligned_source)
        print self.aligned_source


class ParallelProcrustes(ParallelRigidAlignment):
    def __init__(self, sources, **kwargs):
        RigidAlignment.__init__(self, sources, **kwargs)

    def general_alignment(self):
        # stores the items used in each procrustes step
        self.operations = []
        error = 999999999
        while error > 0.0001:
            self._procrustes_step()
            old_target = self.target
            self.target = self.aligned_sources.mean(axis=-1)[..., np.newaxis]
            error = np.sum((self.target - old_target) ** 2)
            print 'error is ' + `error`
        self.h_transforms = []
        for i in range(self.n_sources):
            self.h_transforms.append(np.eye(self.n_dimensions + 1))
            for ops in self.operations:
                t = h_translation_matrix(ops['translate'][..., i].flatten())
                s = h_scale_matrix(ops['rescale'][..., i].flatten(),
                                   dim=self.n_dimensions)
                r = h_rotation_matrix(ops['rotation'][i])
                self.h_transforms[i] = np.dot(self.h_transforms[i],
                                              np.dot(t,
                                                     np.dot(s,
                                                            r)))

    def _procrustes_step(self):
        print 'taking Procrustes step'
        ops = {}
        # calculate the translation required for each source to align the
        # sources' centre of mass to the the target centre of mass
        translation = (self.target.mean(axis=0) -
                       self.aligned_sources.mean(axis=0))[np.newaxis, ...]
        # apply the translation to each source respectively
        self.aligned_sources += translation
        ops['translate'] = translation
        # calcuate the frobenious norm of each shape as our metric
        scale_sources = np.sqrt(np.apply_over_axes(np.sum,
                                                  (
                                                   self.aligned_sources -
                                                   self.aligned_sources.mean(
                                                   axis=0)) ** 2,
                                                   [0, 1]))
        scale_target = np.sqrt(np.sum((self.target -
                                       self.target.mean(axis=0)) ** 2))
        rescale = scale_target / scale_sources
        self.aligned_sources = self.aligned_sources * rescale
        ops['rescale'] = rescale
        rotations = []
        #for each source
        for i in range(self.n_sources):
            # calculate the correlation along each dimension
            correlation = np.dot(self.aligned_sources[..., i].T,
                                 self.target[..., 0])
            U, D, Vt = np.linalg.svd(correlation)
            # find the optimal rotation to minimise rotational differences
            rotation = np.dot(U, Vt)
            rotations.append(rotation)
            # apply the rotation
            self.aligned_sources[..., i] = np.dot(self.aligned_sources[..., i],
                                                  rotation)
        ops['rotation'] = rotations
        self.operations.append(ops)