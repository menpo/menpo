"""The text of str(obj) (and repr(obj), which is the same text: issue #752) follows one standard.

A summary line ``ClassName: key: value, key: value``, then optionally details: one ``key: value`` per line
indented by two spaces, with the lines of a multi-line value indented by four. See menpo.base.describe.
"""

import re
from collections import OrderedDict

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from menpo.base import Copyable, LazyList, describe
from menpo.image import BooleanImage, Image, MaskedImage
from menpo.landmark import LandmarkManager
from menpo.model import LinearVectorModel, PCAModel, PCAVectorModel
from menpo.shape import (
    ColouredTriMesh,
    DirectedGraph,
    LabelledPointUndirectedGraph,
    PointCloud,
    PointDirectedGraph,
    PointTree,
    PointUndirectedGraph,
    TexturedTriMesh,
    Tree,
    TriMesh,
    UndirectedGraph,
)
from menpo.transform import (
    Affine,
    AlignmentSimilarity,
    NonUniformScale,
    PiecewiseAffine,
    Rotation,
    Similarity,
    ThinPlateSplines,
    Translation,
    UniformScale,
)
from menpo.transform.groupalign.procrustes import GeneralizedProcrustesAnalysis


def _objects():
    rng = np.random.RandomState(0)
    pc = PointCloud(rng.rand(5, 2))
    pts3 = rng.rand(3, 2)
    mesh_points = rng.rand(4, 3)
    tri = np.array([[0, 1, 2], [1, 2, 3]])
    adj = csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
    tree_adj = csr_matrix(np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]]))
    manager = LandmarkManager()
    manager["group"] = pc
    masks = OrderedDict(
        [("a", np.array([True, True, False])), ("b", np.array([False, False, True]))]
    )
    flat = TriMesh(mesh_points[:, :2], tri)
    return {
        "LazyList": LazyList([lambda: 1, lambda: 2]),
        "Image": Image(rng.rand(3, 6, 7)),
        "Image3D": Image(rng.rand(1, 4, 5, 6)),
        "MaskedImage": MaskedImage(rng.rand(2, 6, 7)),
        "BooleanImage": BooleanImage.init_blank((6, 7)),
        "PointCloud": pc,
        "TriMesh": TriMesh(mesh_points, tri),
        "ColouredTriMesh": ColouredTriMesh(mesh_points, tri, colours=rng.rand(4, 3)),
        "TexturedTriMesh": TexturedTriMesh(
            mesh_points, rng.rand(4, 2), Image(rng.rand(3, 8, 8)), tri
        ),
        "UndirectedGraph": UndirectedGraph(adj),
        "DirectedGraph": DirectedGraph(adj),
        "Tree": Tree(tree_adj, root_vertex=0),
        "PointUndirectedGraph": PointUndirectedGraph(pts3, adj),
        "PointDirectedGraph": PointDirectedGraph(pts3, adj),
        "PointTree": PointTree(pts3, tree_adj, root_vertex=0),
        "LabelledPointUndirectedGraph": LabelledPointUndirectedGraph(pts3, adj, masks),
        "Affine": Affine(np.eye(3)),
        "Similarity": Similarity(np.eye(3)),
        "AlignmentSimilarity": AlignmentSimilarity(pc, pc),
        "Rotation": Rotation(np.eye(2)),
        "Translation": Translation([1, 2]),
        "UniformScale": UniformScale(2, 2),
        "NonUniformScale": NonUniformScale([2, 3]),
        "ThinPlateSplines": ThinPlateSplines(pc, PointCloud(pc.points + 0.1)),
        "PiecewiseAffine": PiecewiseAffine(flat, TriMesh(flat.points + 0.1, tri)),
        "GeneralizedProcrustesAnalysis": GeneralizedProcrustesAnalysis(
            [pc, PointCloud(pc.points + 0.1)]
        ),
        "PCAModel": PCAModel([PointCloud(rng.rand(5, 2)) for _ in range(6)]),
        "PCAVectorModel": PCAVectorModel(rng.rand(6, 10)),
        "LinearVectorModel": LinearVectorModel(rng.rand(3, 10)),
        "LandmarkManager": manager,
        "Copyable": Copyable(),
    }


OBJECTS = _objects()


@pytest.mark.parametrize("name", list(OBJECTS))
def test_str_follows_the_standard(name):
    obj = OBJECTS[name]
    text = str(obj)
    lines = text.split("\n")
    cls = type(obj).__name__

    # the summary: the class name, alone or followed by ': key: value, ...'
    assert lines[0] == cls or re.match(r"^%s: [a-z_]+: \S" % cls, lines[0]), lines[0]
    assert not lines[0].endswith((".", " "))
    # details: '  key: value', or '  key:' followed by lines indented by at least four
    for line in lines[1:]:
        assert re.match(r"^(  [A-Za-z_]+:( \S.*)?|    .*\S)$", line), repr(line)
    assert all(line == line.rstrip() for line in lines)
    assert not text.endswith("\n")
    # never a dump of the state of the object
    assert len(lines) <= 20 and "Copyable Menpo Object" not in text
    # repr is the same text, so that an object evaluated in a notebook shows it (#752)
    assert repr(obj) == text


def test_str_of_the_common_classes():
    o = OBJECTS
    assert str(o["PointCloud"]) == "PointCloud: n_points: 5, n_dims: 2"
    assert str(o["TriMesh"]) == "TriMesh: n_points: 4, n_dims: 3, n_tris: 2"
    assert str(o["Image"]) == "Image: shape: (6, 7), n_channels: 3"
    assert str(o["Image3D"]) == "Image: shape: (4, 5, 6), n_channels: 1"
    assert str(o["MaskedImage"]) == (
        "MaskedImage: shape: (6, 7), n_channels: 2, mask_proportion_true: 100.0%"
    )
    assert str(o["UndirectedGraph"]) == "UndirectedGraph: n_vertices: 3, n_edges: 2"
    assert str(o["PointTree"]) == (
        "PointTree: n_points: 3, n_dims: 2, n_leaves: 2, maximum_depth: 1"
    )
    assert str(o["LazyList"]) == "LazyList: n_items: 2"
    assert str(o["LinearVectorModel"]) == (
        "LinearVectorModel: n_components: 3, n_features: 10"
    )
    assert str(o["Translation"]) == (
        "Translation: n_dims: 2\n"
        "  description: Translation by [1. 2.]\n"
        "  h_matrix:\n"
        "    [[1. 0. 1.]\n"
        "     [0. 1. 2.]\n"
        "     [0. 0. 1.]]"
    )
    assert str(o["LandmarkManager"]) == (
        "LandmarkManager: n_groups: 1\n  group: PointCloud: n_points: 5, n_dims: 2"
    )


def test_subclasses_are_named_by_their_own_class():
    # AlignmentSimilarity used to describe itself as a "Similarity"
    assert str(OBJECTS["AlignmentSimilarity"]).startswith("AlignmentSimilarity: ")
    assert str(OBJECTS["ColouredTriMesh"]).startswith("ColouredTriMesh: ")


def test_a_graph_with_isolated_vertices_says_so():
    adjacency = csr_matrix(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]))
    assert str(UndirectedGraph(adjacency)) == (
        "UndirectedGraph: n_vertices: 3, n_edges: 1, n_isolated_vertices: 1"
    )


def test_the_default_str_is_the_name_of_the_class_not_a_dump_of_its_state():
    class Holder(Copyable):
        def __init__(self):
            self.big = np.zeros((100, 100))

    assert str(Holder()) == "Holder"


# ---- the helper
class _Thing:
    pass


def test_describe_with_nothing_to_say_is_the_class_name():
    assert describe(_Thing()) == "_Thing"


def test_describe_summary_and_details():
    text = describe(
        _Thing(),
        [("n_points", 3), ("shape", (2, 3))],
        {"centred": True, "matrix": "[[1 0]\n [0 1]]"},
    )
    assert text == (
        "_Thing: n_points: 3, shape: (2, 3)\n"
        "  centred: True\n"
        "  matrix:\n"
        "    [[1 0]\n"
        "     [0 1]]"
    )


def test_describe_keeps_the_order_it_is_given():
    assert describe(_Thing(), {"b": 1, "a": 2}) == "_Thing: b: 1, a: 2"


def test_describe_strips_trailing_whitespace_from_the_lines_of_a_value():
    # numpy pads some of its text; the standard has no trailing whitespace
    text = describe(_Thing(), None, {"matrix": "[[1 0] \n [0 1]]  "})
    assert text == "_Thing\n  matrix:\n    [[1 0]\n     [0 1]]"
