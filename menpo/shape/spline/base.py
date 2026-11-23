import abc

import numpy as np


class Spline(abc.ABC):
    r"""
    A curve through control points, open or closed, in any number of dimensions.

    This is the interface of the splines of menpo: a subclass says how to
    sample the curve between its control points (:meth:`_sample_dense`), and
    gets from it everything that works on the length of the curve: its
    :attr:`length`, where each control point is along it (:attr:`arc_positions`),
    the points at given arc lengths (:meth:`points_at`) and, the main use,
    resampling a contour so that its points are evenly spaced (:meth:`resample`).
    Arc length is measured on the polyline through ``resolution`` samples of
    each segment, so it is very close to exact, not exact.

    Parameters
    ----------
    points : ``(n_points, n_dims)`` `ndarray`
        The points that the curve goes through, in order. At least two for an
        open curve and three for a closed one.
    closed : `bool`, optional
        If ``True`` the curve goes on from the last point back to the first.
    resolution : `int`, optional
        The number of samples of each segment that the arc length is computed
        from.

    Raises
    ------
    ValueError
        Too few points, or not an array of points.
    """

    def __init__(self, points, closed=False, resolution=40):
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
        self.resolution = resolution
        self._dense, self._control_index = self._sample_dense()
        steps = np.linalg.norm(np.diff(self._dense, axis=0), axis=1)
        self._arc = np.concatenate([[0.0], np.cumsum(steps)])

    @abc.abstractmethod
    def _sample_dense(self):
        r"""
        Sample the curve, ``resolution`` samples per segment (a segment is the
        part between two consecutive control points, and a closed curve has as
        many segments as points).

        Returns
        -------
        dense : ``(n_samples, n_dims)`` `ndarray`
            The samples in order along the curve. The first is the first
            control point and the last is the last control point (of a closed
            curve, the first again).
        control_index : ``(n_points + 1, )`` `ndarray`
            The index in ``dense`` of every control point, and of the end.
        """

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
