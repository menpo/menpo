

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
