from menpo.groupalign import GeneralizedProcrustesAnalysis
from menpo.model import PCAModel

from .corresponder import TriMeshCorresponder


class MMBuilder(object):

    def __init__(self, models, group=None, label='all', **kwargs):
        self.models = models
        self.group = group
        self.label = label
        gpa = GeneralizedProcrustesAnalysis([m.landmarks[group][label].lms
                                             for m in self.models])
        self.target = gpa.mean_aligned_shape
        self.corresponder = TriMeshCorresponder(self.target, **kwargs)
        self.shape_model = None
        self.dc_models = None

    def correspond(self):
        self.dc_models = [self.corresponder.correspond(m, group=self.group,
                                                       label=self.label)
                          for m in self.models]

    def pca(self):
        self.shape_model = PCAModel(self.dc_models)
