import menpo
from numpy.testing import assert_equal
from menpo.transform.piecewiseaffine.base import CachedPWA, PythonPWA

b = menpo.io.import_builtin_asset('breakingbad.jpg').as_masked()
b = b.crop_to_landmarks_proportion(0.1)
b = b.rescale_landmarks_to_diagonal_range(120)
b = b.constrain_mask_to_landmarks()
points = b.mask.true_indices()
src = b.landmarks['PTS']
tgt = src.copy()


def test_cached_pwa_same_as_python_pwa():
    cached_pwa = CachedPWA(src, tgt)
    python_pwa = PythonPWA(src, tgt)
    assert_equal(python_pwa.apply(points), cached_pwa.apply(points))


def test_python_pwa_batch_same():
    python_pwa = PythonPWA(src, tgt)
    assert_equal(python_pwa.apply(points),
                 python_pwa.apply(points, batch_size=10))


def test_cached_pwa_same_twice():
    cached_pwa = CachedPWA(src, tgt)
    r1 = cached_pwa.apply(points)
    # now using cache
    r2 = cached_pwa.apply(points)
    assert_equal(r1, r2)


def test_cached_pwa_forgets_cache():
    cached_pwa = CachedPWA(src, tgt)
    r1 = cached_pwa.apply(points)
    cached_pwa.apply(points[40:60])
    # cache is now set to something other than points
    # should clear cache and be fine
    r2 = cached_pwa.apply(points)
    assert_equal(r1, r2)
