from menpo.groupalign import GeneralizedProcrustesAnalysis
from menpo.model import PCAModel

from .corresponder import TriMeshCorresponder


class MMBuilder(object):

    def __init__(self, models, group=None, label='all', sampling_rate=2):
        self.models = models

        gpa = GeneralizedProcrustesAnalysis([m.landmarks[group][label].lms
                                             for m in self.models])
        self.target = gpa.mean_aligned_shape
        self.corresponder = TriMeshCorresponder(self.target,
                                                sampling_rate=sampling_rate)
        self.dc_models = [self.corresponder.in_correspondence(m,
                                                             group=group,
                                                             label=label)
                          for m in models]
        self.pca()

    def pca(self):
        self.shape_model = PCAModel(self.dc_meshes)
