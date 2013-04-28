from pybug.align.base import Alignment, ParallelAlignment


class NonRigidAlignment(Alignment):
    """ Abstract class specializing in nonrigid alignments.
  """

    def __init__(self, source, target):
        Alignment.__init__(self, source, target)


class ParallelNonRigidAlignment(ParallelAlignment):
    """ Abstract class specializing in nonrigid alignments.
    """

    def __init__(self, sources, **kwargs):
        ParallelAlignment.__init__(self, sources, **kwargs)
