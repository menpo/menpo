

class Invertible(object):
    r"""
    Mix-in for invertible transforms. Provides an interface for taking the
    `pseudo` or true inverse of a transform.

    Has to be implemented in conjunction with :map:`Transform`.
    """

    @property
    def has_true_inverse(self):
        r"""
        ``True`` if the pseudoinverse is an exact inverse.

        :type: `bool`
        """
        raise NotImplementedError()

    def pseudoinverse(self):
        r"""
        The pseudoinverse of the transform - that is, the transform that
        results from swapping `source` and `target`, or more formally, negating
        the transforms parameters. If the transform has a true inverse this
        is returned instead.

        :type: ``type(self)``
        """
        raise NotImplementedError()

class VInvertible(Invertible):
    r"""
    Mix-in for :map:`Vectorizable` :map:`Invertible` :map:`Transform` s.

    Prefer this mix-in over :map:`Invertible` if the :map:`Transform` in
    question is :map:`Vectorizable` as this adds :meth:`from_vector` variants
    to the :map:`Invertible` interface. These can be tuned for performance,
    and are, for instance, needed by some of the machinery of fit.
    """
    def pseudoinverse_vector(self, vector):
        r"""
        The vectorized pseudoinverse of a provided vector instance.
        Syntactic sugar for::

            self.from_vector(vector).pseudoinverse().as_vector()

        Can be much faster than the explict call as object creation can be
        entirely avoided in some cases.

        Parameters
        ----------
        vector :  ``(n_parameters,)`` `ndarray`
            A vectorized version of ``self``

        Returns
        -------
        pseudoinverse_vector : ``(n_parameters,)`` `ndarray`
            The pseudoinverse of the vector provided
        """
        return self.from_vector(vector).pseudoinverse().as_vector()
