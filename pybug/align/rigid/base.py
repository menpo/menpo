from pybug.align.base import Alignment


class RigidAlignment(Alignment):
    """ Abstract class specializing in rigid alignments.
  """

    def __init__(self, source, target):
        Alignment.__init__(self, source, target)
