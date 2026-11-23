import numpy as np

__all__ = ["CatmullRom", "curves_from_edges", "resample_curves"]


class CatmullRom(object):
    r"""
    A Catmull-Rom spline through a sequence of points, open or closed.

    The curve passes through every point, which are its control points.
    The default parametrisation is the *centripetal* one (``alpha=0.5``),
    which does not form loops or cusps inside a segment, nor overshoot
    when the points are unevenly spaced (the uniform one, ``alpha=0``, does).
    The curve works in any number of dimensions.

    Its main use is to *resample* a contour: move the points along the curve
    that they define, keeping some of them (the anchors) where they are, so
    that the others are spaced evenly. See :meth:`resample`.

    Parameters
    ----------
    points : ``(n_points, n_dims)`` `ndarray`
        The points that the curve goes through, in order. At least two for an
        open curve and three for a closed one.
    closed : `bool`, optional
        If ``True`` the curve goes on from the last point back to the first.
    alpha : `float`, optional
        The exponent of the parametrisation: ``0.5`` is centripetal, ``0`` is
        uniform and ``1`` is chordal.
    resolution : `int`, optional
        The number of samples of each segment that the arc length is computed
        from (the length of the polyline through them).

    Raises
    ------
    ValueError
        Too few points, or not an array of points.
    """

    def __init__(self, points, closed=False, alpha=0.5, resolution=40):
        points = np.array(points, dtype=float)
        if points.ndim != 2:
            raise ValueError(
                "points must be an (n_points, n_dims) array, got shape {}".format(
                    points.shape
                )
            )
        minimum = 3 if closed else 2
        if len(points) < minimum:
            raise ValueError(
                "a {} curve needs at least {} points, got {}".format(
                    "closed" if closed else "open", minimum, len(points)
                )
            )
        if resolution < 2:
            raise ValueError("resolution must be at least 2")
        self.points = points
        self.closed = closed
        self.alpha = alpha
        self.resolution = resolution
        self._dense, self._control_index = _dense_curve(
            points, closed, alpha, resolution
        )
        steps = np.linalg.norm(np.diff(self._dense, axis=0), axis=1)
        self._arc = np.concatenate([[0.0], np.cumsum(steps)])

    @property
    def n_points(self):
        r"""
        The number of control points.

        :type: `int`
        """
        return len(self.points)

    @property
    def n_dims(self):
        r"""
        The number of dimensions.

        :type: `int`
        """
        return self.points.shape[1]

    @property
    def length(self):
        r"""
        The length of the curve (of a closed one: all the way round).

        :type: `float`
        """
        return float(self._arc[-1])

    @property
    def arc_positions(self):
        r"""
        The arc length from the first point to each control point.

        :type: ``(n_points, )`` `ndarray`
        """
        return self._arc[self._control_index[: self.n_points]]

    def points_at(self, arc_lengths):
        r"""
        The points of the curve at some arc lengths from its start.

        Parameters
        ----------
        arc_lengths : ``(k, )`` `ndarray`
            Distances along the curve, from ``0`` to :attr:`length`. Those of a
            closed curve are taken modulo its length; those of an open curve
            are limited to it.

        Returns
        -------
        points : ``(k, n_dims)`` `ndarray`
        """
        arc = np.atleast_1d(np.asarray(arc_lengths, dtype=float))
        if self.closed:
            arc = np.mod(arc, self.length)
        else:
            arc = np.clip(arc, 0.0, self.length)
        return np.stack(
            [np.interp(arc, self._arc, self._dense[:, k]) for k in range(self.n_dims)],
            axis=1,
        )

    def resample(self, anchors=(), spacing=None):
        r"""
        Move the points along the curve so that they are evenly spaced
        between anchors.

        The anchors do not move. Between two consecutive anchors, the other
        points go to equal steps of arc length, or to the fractions of it that
        ``spacing`` gives. The ends of an open curve are always anchors, and a
        closed curve has at least one (its first point, if none is given).

        Parameters
        ----------
        anchors : `iterable` of `int`, optional
            Indices of the points that stay where they are.
        spacing : `dict` of `int` to `float`, optional
            For some of the points that are not anchors, the fraction (strictly
            between 0 and 1) of the arc length from the previous anchor to the
            next one at which the point goes. The fractions of the points
            between two anchors must increase. A point without one goes to the
            equal steps: ``k / m`` for the ``k``\ th of ``m - 1`` points between
            two anchors.

        Returns
        -------
        points : ``(n_points, n_dims)`` `ndarray`
            The new positions, in the order of the original points.

        Raises
        ------
        ValueError
            An anchor out of range, or fractions that are not in (0, 1) or not
            increasing between two anchors.
        """
        n = self.n_points
        anchors = sorted({int(a) for a in anchors})
        if any(a < 0 or a >= n for a in anchors):
            raise ValueError(
                "anchors must be point indices between 0 and {}".format(n - 1)
            )
        spacing = (
            {} if spacing is None else {int(k): float(v) for k, v in spacing.items()}
        )
        if any(k < 0 or k >= n for k in spacing):
            raise ValueError("spacing has an index out of range")
        if self.closed:
            anchors = anchors or [0]
        else:
            anchors = sorted(set(anchors) | {0, n - 1})
        if any(k in anchors for k in spacing):
            raise ValueError("an anchor does not move: it cannot have a spacing")

        arc = self.arc_positions
        length = self.length
        out = self.points.copy()
        if self.closed:
            start = anchors[0]
            order = list(range(start, n)) + list(range(0, start))
            along = np.mod(arc[order] - arc[start], length)
            positions = [order.index(a) for a in anchors] + [n]
            along = np.concatenate([along, [length]])
        else:
            order = list(range(n))
            along = arc
            positions = anchors
        for lo, hi in zip(positions[:-1], positions[1:]):
            interior = order[lo + 1 : hi]
            if not interior:
                continue
            fractions = [
                spacing.get(i, (k + 1) / (hi - lo)) for k, i in enumerate(interior)
            ]
            if not all(0.0 < f < 1.0 for f in fractions) or any(
                b <= a for a, b in zip(fractions[:-1], fractions[1:])
            ):
                raise ValueError(
                    "the spacing of points {} must be fractions that increase "
                    "strictly between 0 and 1, got {}".format(interior, fractions)
                )
            targets = along[lo] + np.array(fractions) * (along[hi] - along[lo])
            if self.closed:
                targets = targets + arc[start]
            out[interior] = self.points_at(targets)
        return out


def _dense_curve(points, closed, alpha, resolution):
    r"""
    Sample a Catmull-Rom spline (the pyramidal formulation of Barry and
    Goldman), ``resolution`` samples per segment. Returns the samples and the
    index of each control point among them (a closed curve ends on its first).
    """
    n = len(points)
    if closed:
        padded = np.concatenate([points[-1:], points, points[:2]])
    else:
        # mirror the first and the last neighbour, which makes the ends straight
        padded = np.concatenate(
            [2 * points[:1] - points[1:2], points, 2 * points[-1:] - points[-2:-1]]
        )
    segments = n if closed else n - 1
    pieces, control_index = [], []
    for i in range(segments):
        p0, p1, p2, p3 = padded[i : i + 4]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-6) ** alpha
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-6) ** alpha
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-6) ** alpha
        t = np.linspace(t1, t2, resolution, endpoint=False)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        control_index.append(i * resolution)
        pieces.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    dense = np.concatenate(pieces)
    dense = np.concatenate([dense, dense[:1] if closed else points[-1:]])
    control_index.append(len(dense) - 1)
    return dense, np.array(control_index)


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


def resample_curves(points, edges, anchors=None, spacing=None, alpha=0.5):
    r"""
    Resample the curves that the edges of a graph make, see
    :meth:`CatmullRom.resample`.

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
        some points; see :meth:`CatmullRom.resample`.
    alpha : `float`, optional
        The parametrisation of the splines.

    Returns
    -------
    points : ``(n_points, n_dims)`` `ndarray`
        The new positions.
    """
    points = np.asarray(points, dtype=float)
    anchors = set() if anchors is None else {int(a) for a in anchors}
    spacing = {} if spacing is None else {int(k): float(v) for k, v in spacing.items()}
    result = points.copy()
    for indices, closed in curves_from_edges(edges, len(points)):
        local = {g: i for i, g in enumerate(indices)}
        curve = CatmullRom(points[indices], closed=closed, alpha=alpha)
        moved = curve.resample(
            [local[a] for a in anchors if a in local],
            {local[g]: f for g, f in spacing.items() if g in local},
        )
        result[indices] = moved
    return result
