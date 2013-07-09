import abc


class StatisticalModel(object):

    __metaclass__ = abc.ABCMeta
    samples = None

    @property
    def sample_data_class(self):
        return self.template_sample.__class__

    @property
    def template_sample(self):
        return self.samples[0]
