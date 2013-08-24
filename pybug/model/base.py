import abc


# TODO: this class seems a bit bare? Needs to be documented better.
class StatisticalModel(object):
    r"""
    Abstract base class representing a statistical model.
    """

    __metaclass__ = abc.ABCMeta
    samples = None

    @property
    def sample_data_class(self):
        r"""
        The class of the sample data

        :type: class type
        """
        return self.template_sample.__class__

    @property
    def template_sample(self):
        r"""
        The first sample.

        :type: ``self.sample_data_class``
        """
        return self.samples[0]
