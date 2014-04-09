import abc


class Invertible(object):
    r"""
    Transform Mixin for invertible transforms. Provides an interface for
    taking the psuedo or true inverse of a transform.
    """

    @abc.abstractproperty
    def has_true_inverse(self):
        r"""
        True if the pseudoinverse is an exact inverse.

        :type: Boolean
        """
        pass

    @property
    def pseudoinverse(self):
        r"""
        The pseudoinverse of the transform - that is, the transform that
        results from swapping source and target, or more formally, negating
        the transforms parameters. If the transform has a true inverse this
        is returned instead.

        :type: :class:`Transform`
        """
        return self._build_pseudoinverse()

    @abc.abstractmethod
    def _build_pseudoinverse(self):
        r"""
        Returns this transform's inverse if it has one. if not,
        the pseduoinverse is given.

        This method is called by the pseudoinverse property and must be
        overridden.


        Returns
        -------
        pseudoinverse: type(self)
        """
        pass


class VInvertible(Invertible):
    r"""
    Transform Mixin for Vectorizable invertible transforms.

    Prefer this Mixin over Invertible if the Transform in question is
    Vectorizable as this adds from_vector variants to the Invertible
    interface. These can be tuned for performance, and are for instance
    needed by some of the machinery of AAMs.
    """

    def pseudoinverse_vector(self, vector):
        r"""
        The vectorized pseudoinverse of a provided vector instance.

        Syntactic sugar for

        self.from_vector(vector).pseudoinverse.as_vector()

        Can be much faster than the explict call as object creation can be
        entirely avoided in some cases.

        Parameters
        ----------
        vector :  (P,) ndarray
            A vectorized version of self

        Returns
        -------
        pseudoinverse_vector : (N,) ndarray
            The pseudoinverse of the vector provided
        """
        return self.from_vector(vector).pseudoinverse.as_vector()
