from pybug.align.base import Alignment


class RigidAlignment(Alignment):
    r"""
    Abstract class specializing in rigid alignments.

    Parameters
    -----------
    source : (N, D) ndarray
        The source points to apply the transformation from
    target : (N, D) ndarray
        The target points to apply the transformation to
    """

    def __init__(self, source, target):
        Alignment.__init__(self, source, target)
