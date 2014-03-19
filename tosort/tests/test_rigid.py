from menpo.groupalign.rigid import Procrustes
import numpy as np


def _mag_diff(a, b):
    return np.sum((a - b) ** 2)


def test_invarient_target_source():
    source = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float64)
    proc = Procrustes([source], target=source)
    proc.general_alignment()
    assert (_mag_diff(proc.sources, proc.target) < 0.000000001)


def test_scale():
    source = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=np.float64)
    target = source * 2
    proc = Procrustes([source], target=target)
    proc.general_alignment()
    assert (_mag_diff(proc.translation_vectors[0], np.zeros(2)) < 0.00000001)
    assert (
        _mag_diff(proc.scalerotation_matrices[0], 2 * np.eye(2)) < 0.00000001)


def test_translate():
    source = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float64)
    target = source - 1
    proc = Procrustes([source], target=target)
    proc.general_alignment()
    assert (_mag_diff(proc.translation_vectors[0],
                      -1.0 * np.ones(2))) < 0.00000001)
    assert (_mag_diff(proc.scalerotation_matrices[0], np.eye(2)) < 0.00000001)


def test_ops():
    src_1 = np.array([[0, 0], [0, 1],
                      [1, 1], [1, 0]], dtype=np.float64)
    src_2 = src_1 * 4
    src_3 = src_1 + 10000
    src_4 = np.array([[0, 1], [1, 1],
                      [1, 0], [0, 0]], dtype=np.float64)
    src_5 = (np.array([[0.6, 0.2], [1.4, 0.3],
                       [-0.1, 1.1], [1.2, 0.8]], dtype=np.float64) / 0.2) - 100
    sources = [src_1, src_2, src_3, src_4, src_5]
    proc = Procrustes(sources)
    proc.general_alignment()
    for src in sources:
        sr, t = proc.scalerotation_translation_for_source(src)
        new_source = np.dot(src, sr) + t
        assert (_mag_diff(new_source,
                          proc.aligned_version_of_source(src)) < 0.00000001)

