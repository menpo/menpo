import abc


class StatisticalModel(object):
    r"""
    Abstract interface for a model that is constructed from a set of sample
    data
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, samples):
        self.samples = samples

    @property
    def template_instance(self):
        r"""
        The first sample.

        :type: ``self.instance_class``
        """
        return self.samples[0]

    @property
    def n_samples(self):
        return len(self.samples)
