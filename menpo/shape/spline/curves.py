import numpy as np

from .catmull_rom import CatmullRom


def curves_from_edges(edges, n_points):
    r"""
    The curves that the edges of a graph make: its chains and its loops.

    Every connected component of the graph that is a simple path (open curve)
    or a simple cycle (closed curve) of at least three points is a curve.
    Components that branch, and those of one or two points, are not.

    The points of a curve are in order along it. A chain starts from the end
    with the smaller index; a loop from its smallest index, towards the
    smaller of its two neighbours.

    Parameters
    ----------
    edges : ``(n_edges, 2)`` `ndarray`
        Pairs of point indices.
    n_points : `int`
        The number of points.

    Returns
    -------
    curves : `list` of ``(indices, closed)``
        For each curve, the indices of its points in order and whether it is a
        loop.
    """
    neighbours = [set() for _ in range(n_points)]
    for a, b in np.asarray(edges, dtype=int).reshape(-1, 2):
        if a != b:
            neighbours[a].add(int(b))
            neighbours[b].add(int(a))
    seen = [False] * n_points
    curves = []
    for root in range(n_points):
        if seen[root] or not neighbours[root]:
            continue
        component, stack = [], [root]
        seen[root] = True
        while stack:
            v = stack.pop()
            component.append(v)
            for w in neighbours[v]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        if len(component) < 3 or any(len(neighbours[v]) > 2 for v in component):
            continue
        ends = sorted(v for v in component if len(neighbours[v]) == 1)
        closed = not ends
        start = min(component) if closed else ends[0]
        order, previous, current = [start], None, start
        while True:
            options = sorted(w for w in neighbours[current] if w != previous)
            if not options:
                break
            nxt = options[0]
            if nxt == start:
                break
            order.append(nxt)
            previous, current = current, nxt
        curves.append((order, closed))
    return curves


def resample_curves(
    points, edges, anchors=None, spacing=None, spline_cls=None, **spline_kwargs
):
    r"""
    Resample the curves that the edges of a graph make, see
    :meth:`Spline.resample`.

    Parameters
    ----------
    points : ``(n_points, n_dims)`` `ndarray`
        The positions of the points.
    edges : ``(n_edges, 2)`` `ndarray`
        The edges, which define the curves (see :func:`curves_from_edges`).
    anchors : `iterable` of `int`, optional
        Indices of points that stay where they are. The ends of every chain
        and the first point of every loop are anchors as well, as are the
        points that no curve contains.
    spacing : `dict` of `int` to `float`, optional
        Fractions of the arc length between the anchors around a point, for
        some points; see :meth:`Spline.resample`.
    spline_cls : :class:`Spline` subclass, optional
        The kind of spline through the points of each curve
        (:class:`CatmullRom` by default).
    **spline_kwargs
        Passed to the spline, for example ``alpha`` of a Catmull-Rom spline.

    Returns
    -------
    points : ``(n_points, n_dims)`` `ndarray`
        The new positions.
    """
    if spline_cls is None:
        spline_cls = CatmullRom
    points = np.asarray(points, dtype=float)
    anchors = set() if anchors is None else {int(a) for a in anchors}
    spacing = {} if spacing is None else {int(k): float(v) for k, v in spacing.items()}
    result = points.copy()
    for indices, closed in curves_from_edges(edges, len(points)):
        local = {g: i for i, g in enumerate(indices)}
        curve = spline_cls(points[indices], closed=closed, **spline_kwargs)
        moved = curve.resample(
            [local[a] for a in anchors if a in local],
            {local[g]: f for g, f in spacing.items() if g in local},
        )
        result[indices] = moved
    return result
