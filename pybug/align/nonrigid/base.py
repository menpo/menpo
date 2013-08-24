from pybug.align.base import Alignment, MultipleAlignment


class NonRigidAlignment(Alignment):
    r"""
    Abstract class specializing in nonrigid alignments.

    Parameters
    -----------
    source : (N, D) ndarray
        The source points to apply the transformation from
    target : (N, D) ndarray
        The target points to apply the transformation to
    """

    def __init__(self, source, target):
        super(NonRigidAlignment, self).__init__(source, target)


class MultipleNonRigidAlignment(MultipleAlignment):
    r"""
    Abstract class specializing in nonrigid alignments.

    Parameters
    -----------
    source : list of ndarray
        A list of sources to apply a transformation to
    """

    def __init__(self, sources, **kwargs):
        MultipleAlignment.__init__(self, sources, **kwargs)
