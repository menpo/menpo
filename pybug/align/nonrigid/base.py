from pybug.align.base import Alignment, MultipleAlignment


class NonRigidAlignment(Alignment):
    """ Abstract class specializing in nonrigid alignments.
  """

    def __init__(self, source, target):
        Alignment.__init__(self, source, target)


class MultipleNonRigidAlignment(MultipleAlignment):
    """ Abstract class specializing in nonrigid alignments.
    """

    def __init__(self, sources, **kwargs):
        MultipleAlignment.__init__(self, sources, **kwargs)
