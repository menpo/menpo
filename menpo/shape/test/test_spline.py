from collections import OrderedDict

import numpy as np
import pytest
from numpy.testing import assert_allclose

from menpo.shape import CatmullRom, LabelledPointUndirectedGraph, PointUndirectedGraph
from menpo.shape.spline import curves_from_edges, resample_curves


def circle(n, radius=1.0, centre=(0.0, 0.0)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack(
        [centre[0] + radius * np.cos(angles), centre[1] + radius * np.sin(angles)],
        axis=1,
    )


def chain_edges(indices, closed=False):
    edges = list(zip(indices[:-1], indices[1:]))
    return edges + ([(indices[-1], indices[0])] if closed else [])


def test_the_curve_goes_through_its_points():
    rng = np.random.RandomState(0)
    points = np.cumsum(rng.rand(8, 2), axis=0)
    for closed in (False, True):
        curve = CatmullRom(points, closed=closed)
        assert_allclose(curve.points_at(curve.arc_positions), points, atol=1e-9)


def test_a_straight_line_is_resampled_to_equal_steps():
    line = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
    curve = CatmullRom(line)
    assert curve.length == pytest.approx(10.0)
    assert_allclose(curve.resample()[:, 0], [0.0, 10 / 3, 20 / 3, 10.0], atol=1e-9)


def test_a_closed_curve_is_a_loop_of_its_length():
    curve = CatmullRom(circle(16), closed=True)
    assert curve.length == pytest.approx(2 * np.pi, rel=0.01)
    # the first point is the start, and the curve is back there after one length
    assert_allclose(
        curve.points_at([0.0, curve.length]), [[1.0, 0.0], [1.0, 0.0]], atol=1e-9
    )


def test_a_circle_resampled_is_evenly_spaced_with_one_anchor():
    rng = np.random.RandomState(1)
    # a circle whose points are unevenly spaced along it
    angles = np.sort(rng.rand(24)) * 2 * np.pi
    points = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    moved = CatmullRom(points, closed=True).resample()
    chords = np.linalg.norm(moved - np.roll(moved, -1, axis=0), axis=1)
    assert_allclose(chords[:-1], chords[:-1].mean(), rtol=0.05)
    assert_allclose(
        moved[0], points[0]
    )  # the one anchor of a closed curve is the first point


def test_anchors_do_not_move_and_the_others_do():
    rng = np.random.RandomState(2)
    points = np.cumsum(rng.rand(12, 2) + 0.2, axis=0)
    anchors = [0, 4, 11]
    moved = CatmullRom(points).resample(anchors)
    assert_allclose(moved[anchors], points[anchors])
    assert not np.allclose(moved[[1, 2, 3, 5]], points[[1, 2, 3, 5]])


def test_resampling_twice_changes_nothing():
    rng = np.random.RandomState(3)
    points = np.cumsum(rng.rand(10, 2) + 0.2, axis=0)
    once = CatmullRom(points).resample([0, 5, 9])
    twice = CatmullRom(once).resample([0, 5, 9])
    # the second curve goes through the evenly spaced points, which are where they should be
    # to within the change of the curve itself
    assert_allclose(twice, once, atol=0.02 * np.ptp(points))


def test_spacing_fractions_are_fractions_of_the_arc_between_anchors():
    line = np.array([[float(x), 0.0] for x in range(5)])
    moved = CatmullRom(line).resample([0, 4], spacing={1: 0.1, 2: 0.2, 3: 0.9})
    assert_allclose(moved[:, 0], [0.0, 0.4, 0.8, 3.6, 4.0], atol=1e-9)


def test_spacing_of_a_closed_curve_is_between_the_anchors_round_the_loop():
    points = circle(8)
    # one anchor: the fractions are of the whole loop, from the anchor round to the anchor
    spacing = {i: i / 8.0 for i in range(1, 8)}
    moved = CatmullRom(points, closed=True).resample([0], spacing)
    angles = np.mod(np.arctan2(moved[:, 1], moved[:, 0]), 2 * np.pi)
    assert_allclose(angles, np.linspace(0, 2 * np.pi, 8, endpoint=False), atol=0.03)


def test_a_closed_curve_starts_at_its_first_anchor_not_its_first_point():
    rng = np.random.RandomState(5)
    angles = np.sort(rng.rand(12)) * 2 * np.pi  # unevenly spaced along the circle
    points = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    moved = CatmullRom(points, closed=True).resample([5])
    assert_allclose(moved[5], points[5])
    # all twelve points, including those before the anchor in the list, are evenly spaced round the loop
    chords = np.linalg.norm(moved - np.roll(moved, -1, axis=0), axis=1)
    assert_allclose(chords, chords.mean(), rtol=0.05)


def test_the_parametrisation_is_centripetal_unless_told_otherwise():
    # a long jump and then a tight cluster: the uniform parametrisation overshoots (its curve is
    # longer than the points need) and the centripetal one does not
    points = np.array([[0, 0], [10, 0], [10.2, 0.2], [10.4, 0.0], [20, 0.0]])
    centripetal = CatmullRom(points)
    uniform = CatmullRom(points, alpha=0.0)
    assert centripetal.alpha == 0.5
    assert centripetal.length < 20.5
    assert uniform.length > centripetal.length + 1.0


def test_it_works_in_three_dimensions():
    t = np.linspace(0, 4, 9)
    helix = np.stack([np.cos(t), np.sin(t), t], axis=1)
    curve = CatmullRom(helix)
    assert curve.n_dims == 3
    moved = curve.resample([0, 8])
    steps = np.linalg.norm(np.diff(moved, axis=0), axis=1)
    assert_allclose(steps, steps.mean(), rtol=0.03)


@pytest.mark.parametrize(
    "call",
    [
        lambda: CatmullRom(np.zeros((1, 2))),
        lambda: CatmullRom(np.zeros((2, 2)), closed=True),
        lambda: CatmullRom(np.zeros(5)),
        lambda: CatmullRom(np.array([[0.0, 0], [1, 0], [2, 0]])).resample([7]),
        lambda: CatmullRom(np.array([[0.0, 0], [1, 0], [2, 0]])).resample([], {7: 0.5}),
        lambda: CatmullRom(np.array([[0.0, 0], [1, 0], [2, 0]])).resample([], {0: 0.5}),
        lambda: CatmullRom(np.array([[float(i), 0] for i in range(5)])).resample(
            [], {1: 0.6, 2: 0.4, 3: 0.9}
        ),
        lambda: CatmullRom(np.array([[float(i), 0] for i in range(5)])).resample(
            [], {1: 0.0}
        ),
    ],
)
def test_nonsense_is_an_error(call):
    with pytest.raises(ValueError):
        call()


def test_the_curves_of_a_graph_are_its_chains_and_loops():
    # a chain 0-1-2-3, a loop 4-5-6-7, a branching star, a single edge and a lone point
    edges = chain_edges([0, 1, 2, 3]) + chain_edges([4, 5, 6, 7], closed=True)
    edges += [(8, 9), (8, 10), (8, 11), (12, 13)]
    curves = curves_from_edges(np.array(edges), 15)
    assert curves == [([0, 1, 2, 3], False), ([4, 5, 6, 7], True)]


def test_a_chain_given_backwards_starts_at_its_smaller_end():
    curves = curves_from_edges(np.array([(3, 2), (2, 1), (1, 0)]), 4)
    assert curves == [([0, 1, 2, 3], False)]


def test_a_loop_starts_at_its_smallest_index_towards_the_smaller_neighbour():
    curves = curves_from_edges(np.array(chain_edges([5, 3, 4, 6], closed=True)), 7)
    assert curves == [([3, 4, 6, 5], True)]


def test_a_face_like_graph_keeps_its_labels_and_anchors():
    # two chains and a loop: the shape of a template group
    jaw = [(i, i + 1) for i in range(8)]
    eye = chain_edges([9, 10, 11, 12, 13, 14], closed=True)
    points = np.concatenate(
        [
            np.stack([np.linspace(0, 8, 9) ** 1.5, np.zeros(9)], axis=1),
            circle(6, 0.5, centre=(3.0, 4.0)),
        ]
    )
    masks = OrderedDict([("jaw", np.arange(15) < 9), ("eye", np.arange(15) >= 9)])
    graph = LabelledPointUndirectedGraph.init_from_edges(points, jaw + eye, masks)
    moved = graph.resampled(anchors=[0, 4, 8, 9, 12])
    assert isinstance(moved, LabelledPointUndirectedGraph)
    assert moved.labels == graph.labels
    assert_allclose(moved.edges, graph.edges)
    assert_allclose(moved.points[[0, 4, 8, 9, 12]], points[[0, 4, 8, 9, 12]])
    # the original was not touched, and the jaw is evenly spaced between its anchors now
    assert_allclose(graph.points, points)
    first = np.diff(moved.points[0:5, 0])
    assert_allclose(first, first.mean(), rtol=0.01)


def test_resampling_a_graph_with_nothing_to_resample_is_a_copy():
    points = np.random.RandomState(4).rand(4, 2)
    graph = PointUndirectedGraph.init_from_edges(points, np.array([(0, 1), (2, 3)]))
    moved = graph.resampled()
    assert_allclose(moved.points, points)
    assert moved is not graph


def test_resample_curves_leaves_points_without_a_curve_alone():
    points = np.arange(10.0).reshape(5, 2)
    result = resample_curves(points, np.array([(0, 1), (1, 2), (2, 3)]), [])
    assert_allclose(result[4], points[4])
