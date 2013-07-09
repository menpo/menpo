import abc


class StatisticalModel(object):

    __metaclass__ = abc.ABCMeta
    samples = None


    @property
    def training_data_class(self):
        return self.samples[0].__class__
