from itertools import islice
import numpy as np
from menpo.visualize import print_progress, bytes_str, print_dynamic


def dot_inplace_left(a, b, block_size=1000):
    r"""
    Inplace dot product for memory efficiency. It computes ``a * b = c``, where
    ``a`` will be replaced inplace with ``c``.

    Parameters
    ----------
    a : ``(n_big, k)`` `ndarray`
        First array to dot - assumed to be large. Will be damaged by this
        function call as it is used to store the output inplace.
    b : ``(k, n_small)`` `ndarray`, ``n_small <= k``
        The second array to dot - assumed to be small. ``n_small`` must be
        smaller than ``k`` so the result can be stored within the memory space
        of ``a``.
    block_size : `int`, optional
        The size of the block of ``a`` that will be dotted against ``b`` in
        each iteration. larger block sizes increase the time performance of the
        dot product at the cost of a higher memory overhead for the operation.

    Returns
    -------
    c : ``(n_big, n_small)`` `ndarray`
        The output of the operation. Exactly the same as a memory view onto
        ``a`` (``a[:, :n_small]``) as ``a`` is modified inplace to store the
        result.
    """
    (n_big, k_a), (k_b, n_small) = a.shape, b.shape
    if k_a != k_b:
        raise ValueError('Cannot dot {} * {}'.format(a.shape, b.shape))
    if n_small > k_a:
        raise ValueError('Cannot dot inplace left - '
                         'b.shape[1] ({}) > a.shape[1] '
                         '({})'.format(n_small, k_a))
    for i in range(0, n_big, block_size):
        j = i + block_size
        a[i:j, :n_small] = a[i:j].dot(b)
    return a[:, :n_small]


def dot_inplace_right(a, b, block_size=1000):
    r"""
    Inplace dot product for memory efficiency. It computes ``a * b = c`` where
    ``b`` will be replaced inplace with ``c``.

    Parameters
    ----------
    a : ``(n_small, k)`` `ndarray`, n_small <= k
        The first array to dot - assumed to be small. ``n_small`` must be
        smaller than ``k`` so the result can be stored within the memory space
        of ``b``.
    b : ``(k, n_big)`` `ndarray`
        Second array to dot - assumed to be large. Will be damaged by this
        function call as it is used to store the output inplace.
    block_size : `int`, optional
        The size of the block of ``b`` that ``a`` will be dotted against
        in each iteration. larger block sizes increase the time performance of
        the dot product at the cost of a higher memory overhead for the
        operation.

    Returns
    -------
    c : ``(n_small, n_big)`` `ndarray`
        The output of the operation. Exactly the same as a memory view onto
        ``b`` (``b[:n_small]``) as ``b`` is modified inplace to store the
        result.
    """
    (n_small, k_a), (k_b, n_big) = a.shape, b.shape
    if k_a != k_b:
        raise ValueError('Cannot dot {} * {}'.format(a.shape, b.shape))
    if n_small > k_b:
        raise ValueError('Cannot dot inplace right - '
                         'a.shape[1] ({}) > b.shape[0] '
                         '({})'.format(n_small, k_b))
    for i in range(0, n_big, block_size):
        j = i + block_size
        b[:n_small, i:j] = a.dot(b[:, i:j])
    return b[:n_small]


def as_matrix(vectorizables, length=None, return_template=False, verbose=False):
    r"""
    Create a matrix from a list/generator of :map:`Vectorizable` objects.
    All the objects in the list **must** be the same size when vectorized.

    Consider using a generator if the matrix you are creating is large and
    passing the length of the generator explicitly.

    Parameters
    ----------
    vectorizables : `list` or generator if :map:`Vectorizable` objects
        A list or generator of objects that supports the vectorizable interface
    length : `int`, optional
        Length of the vectorizable list. Useful if you are passing a generator
        with a known length.
    verbose : `bool`, optional
        If ``True``, will print the progress of building the matrix.
    return_template : `bool`, optional
        If ``True``, will return the first element of the list/generator, which
        was used as the template. Useful if you need to map back from the
        matrix to a list of vectorizable objects.

    Returns
    -------
    M : (length, n_features) `ndarray`
        Every row is an element of the list.
    template : :map:`Vectorizable`, optional
        If ``return_template == True``, will return the template used to
        build the matrix `M`.

    Raises
    ------
    ValueError
        ``vectorizables`` terminates in fewer than ``length`` iterations
    """
    # get the first element as the template and use it to configure the
    # data matrix
    if length is None:
        # samples is a list
        length = len(vectorizables)
        template = vectorizables[0]
        vectorizables = vectorizables[1:]
    else:
        # samples is an iterator
        template = next(vectorizables)
    n_features = template.n_parameters
    template_vector = template.as_vector()

    data = np.zeros((length, n_features), dtype=template_vector.dtype)
    if verbose:
        print_dynamic('Allocated data matrix of size {} '
                      '({} samples)'.format(bytes_str(data.nbytes), length))

    # now we can fill in the first element from the template
    data[0] = template_vector
    del template_vector

    # ensure we take at most the remaining length - 1 elements
    vectorizables = islice(vectorizables, length - 1)

    if verbose:
        vectorizables = print_progress(vectorizables, n_items=length, offset=1,
                                       prefix='Building data matrix',
                                       end_with_newline=False)

    # 1-based as we have the template vector set already
    i = 0
    for i, sample in enumerate(vectorizables, 1):
        data[i] = sample.as_vector()

    # we have exhausted the iterable, but did we get enough items?
    if i != length - 1:  # -1
        raise ValueError('Incomplete data matrix due to early iterator '
                         'termination (expected {} items, got {})'.format(
            length, i + 1))

    if return_template:
        return data, template
    else:
        return data


def from_matrix(matrix, template):
    r"""
    Create a generator from a matrix given a template :map:`Vectorizable`
    objects as a template. The ``from_vector`` method will be used to
    reconstruct each object.

    If you want a list, warp the returned value in ``list()``.

    Parameters
    ----------
    matrix : (n_items, n_features) `ndarray`
        A matrix whereby every *row* represents the data of a vectorizable
        object.
    template : :map:`Vectorizable`
        The template object to use to reconstruct each row of the matrix with.

    Returns
    -------
    vectorizables : generator of :map:`Vectorizable`
        Every row of the matrix becomes an element of the list.
    """
    return (template.from_vector(row) for row in matrix)
