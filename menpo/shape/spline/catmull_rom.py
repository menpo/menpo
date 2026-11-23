import numpy as np

from .base import Spline


class CatmullRom(Spline):
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
        self.alpha = alpha
        super(CatmullRom, self).__init__(points, closed=closed, resolution=resolution)

    def _sample_dense(self):
        r"""
        Sample the spline, one segment at a time.

        The curve between two control points p1 and p2 is built from the four
        points p0, p1, p2, p3 around it (Barry and Goldman's pyramidal
        formulation, which is the Catmull-Rom spline written as repeated linear
        interpolation instead of as a cubic polynomial):

          * every point gets a "time" t_i. The time between two points is the
            distance between them to the power alpha, so alpha=0 gives them equal
            times (the uniform spline), alpha=1 times that are the distances (the
            chordal one) and alpha=0.5 the centripetal one, the only one that
            cannot form a loop or a cusp inside a segment, nor overshoot, however
            unevenly the points are spaced (Yuksel et al. 2011);
          * a1, a2, a3 are the points at time t on the lines p0-p1, p1-p2, p2-p3
            (interpolating between neighbours, or extrapolating at the ends);
          * b1 is the point at time t on the line a1-a2 and b2 on a2-a3 (one
            level up the pyramid, each now a quadratic through three points);
          * the point of the curve is on the line b1-b2: a cubic through all four,
            which passes through p1 at t1 and p2 at t2, with a tangent at them
            that depends on the neighbours p0 and p3 and so is shared with the
            neighbouring segments: the curve is smooth where they meet.

        The ends of an open curve have no neighbour on one side: the missing one
        is the mirror image of the next point (2 * p_first - p_second), which
        makes the end of the curve straight.
        """
        points, closed = self.points, self.closed
        alpha, resolution = self.alpha, self.resolution
        n = len(points)
        if closed:
            padded = np.concatenate([points[-1:], points, points[:2]])
        else:
            # mirror the first and the last neighbour, which makes the ends straight
            padded = np.concatenate(
                [
                    2 * points[:1] - points[1:2],
                    points,
                    2 * points[-1:] - points[-2:-1],
                ]
            )
        segments = n if closed else n - 1
        pieces, control_index = [], []
        for i in range(segments):
            p0, p1, p2, p3 = padded[i : i + 4]
            # the times: the distance between points to the power alpha (the
            # floor keeps coincident points from dividing by zero)
            t0 = 0.0
            t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-6) ** alpha
            t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-6) ** alpha
            t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-6) ** alpha
            # the times to sample, from p1 (t1) up to but not including p2 (t2),
            # which is the first sample of the next segment
            t = np.linspace(t1, t2, resolution, endpoint=False)[:, None]
            # the pyramid: linear, then quadratic, then cubic
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            control_index.append(i * resolution)
            pieces.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
        dense = np.concatenate(pieces)
        # the end: the last control point (a closed curve is back at its first)
        dense = np.concatenate([dense, dense[:1] if closed else points[-1:]])
        control_index.append(len(dense) - 1)
        return dense, np.array(control_index)
