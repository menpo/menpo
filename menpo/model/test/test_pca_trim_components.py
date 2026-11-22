import pickle
import tracemalloc

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from menpo.model import LinearVectorModel, PCAModel, PCAVectorModel
from menpo.shape import PointCloud

N_SAMPLES, N_FEATURES = 20, 50  # 19 components


def vector_model(seed=0):
    return PCAVectorModel(np.random.RandomState(seed).randn(N_SAMPLES, N_FEATURES))


def pointcloud_model(seed=0):
    rs = np.random.RandomState(seed)
    return PCAModel(
        [PointCloud(rs.randn(N_FEATURES // 2, 2)) for _ in range(N_SAMPLES)]
    )


MODELS = pytest.mark.parametrize("make", [vector_model, pointcloud_model])


def mean_of(model):
    return model.mean_vector if isinstance(model, PCAModel) else model.mean()


def assert_same_model(a, b):
    """Everything the two models say or do is the same."""
    assert type(a) is type(b)
    assert a.n_components == b.n_components
    assert_array_equal(a.components, b.components)
    assert_array_equal(a.eigenvalues, b.eigenvalues)
    assert_array_equal(mean_of(a), mean_of(b))
    for name in (
        "variance",
        "original_variance",
        "variance_ratio",
        "noise_variance",
        "noise_variance_ratio",
        "eigenvalues_ratio",
        "eigenvalues_cumulative_ratio",
        "_total_variance",
        "_total_variance_ratio",
        "whitened_components",
    ):
        assert_allclose(getattr(a, name)(), getattr(b, name)(), rtol=1e-12, atol=0)
    x = np.random.RandomState(1).randn(N_FEATURES)
    weights = np.random.RandomState(2).randn(a.n_components)
    if isinstance(a, PCAModel):
        x = PointCloud(x.reshape(-1, 2))
    for method, arg in (("project", x), ("reconstruct", x), ("project_whitened", x)):
        ra, rb = getattr(a, method)(arg), getattr(b, method)(arg)
        assert_allclose(
            ra.as_vector() if hasattr(ra, "as_vector") else ra,
            rb.as_vector() if hasattr(rb, "as_vector") else rb,
            rtol=1e-12,
            atol=0,
        )
    ia, ib = a.instance(weights), b.instance(weights)
    assert_allclose(
        ia.as_vector() if hasattr(ia, "as_vector") else ia,
        ib.as_vector() if hasattr(ib, "as_vector") else ib,
        rtol=1e-12,
        atol=0,
    )


# ---- what it gives: the same as trimming a copy
@MODELS
@pytest.mark.parametrize("n", [1, 5, 7, 19, 100, 0.5, 0.9, 0.99])
def test_trim_components_is_what_trimming_a_copy_gives(make, n):
    model = make()
    expected = model.copy()
    expected.trim_components_inplace(n)
    assert_same_model(model.trim_components(n), expected)


@MODELS
def test_trim_components_leaves_the_model_it_came_from_as_it_was(make):
    model, reference = make(), make()
    model.trim_components(3)
    assert_same_model(model, reference)


@MODELS
def test_trimming_a_trimmed_model_goes_down_and_never_back_up(make):
    model = make()
    assert_same_model(
        model.trim_components(10).trim_components(4), model.trim_components(4)
    )
    assert model.trim_components(4).trim_components(10).n_components == 4
    # what was left out stays the noise
    assert model.trim_components(10).trim_components(
        4
    ).noise_variance() == pytest.approx(model.trim_components(4).noise_variance())


@MODELS
@pytest.mark.parametrize("bad", [0, -1, -2.5, 0.0, 2.5])
def test_trim_components_with_a_number_it_can_not_use_is_a_value_error(make, bad):
    model = make()
    with pytest.raises(ValueError, match="number of components"):
        model.trim_components(bad)
    assert model.n_components == 19  # untouched


def test_trim_components_keeps_the_type_and_a_pca_model_still_gives_shapes():
    model = pointcloud_model()
    small = model.trim_components(4)
    assert type(small) is PCAModel
    assert isinstance(small.instance(np.zeros(4)), PointCloud)
    assert isinstance(small.mean(), PointCloud)
    assert small.template_instance is model.template_instance


# ---- the memory is shared, as read-only views
@MODELS
def test_trim_components_shares_the_memory_as_read_only_views(make):
    model = make()
    small = model.trim_components(5)
    assert np.shares_memory(small.components, model.components)
    assert np.shares_memory(small.eigenvalues, model.eigenvalues)
    assert np.shares_memory(mean_of(small), mean_of(model))
    for array in (small.components, small.eigenvalues, mean_of(small)):
        assert not array.flags.writeable
    assert not small._mean.flags.writeable and np.shares_memory(
        small._mean, model._mean
    )
    # the model it came from is not made read-only by it, and its memory is its own
    assert model._components.flags.writeable and model._eigenvalues.flags.writeable


def test_trim_components_copies_nothing():
    model = PCAVectorModel(np.random.RandomState(0).randn(200, 20000))  # ~ 32 MB
    size = model._components.nbytes
    assert size > 20e6
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        small = model.trim_components(50)
        another = small.trim_components(10)
        used = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()
    assert used < size / 1000  # a few kilobytes, not the megabytes of the arrays
    assert small.n_components == 50 and another.n_components == 10


# ---- the getters hand out read-only views of the memory of the model
@MODELS
def test_what_the_model_hands_out_can_not_be_written_to(make):
    model = make()
    for array in (model.components, model.eigenvalues, mean_of(model)):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 1


def test_a_linear_model_hands_out_read_only_components_but_orthonormalizes():
    model = LinearVectorModel(np.random.RandomState(0).randn(4, 30))
    assert not model.components.flags.writeable
    model.orthonormalize_inplace()
    assert_allclose(model.components @ model.components.T, np.eye(4), atol=1e-12)


# ---- the rule: the arrays of a model are never written to, so models that share them can not change each other
def _change(name, model):
    if name == "orthonormalize_inplace":
        model.orthonormalize_inplace()
    elif name == "orthonormalize_against_inplace":
        model.orthonormalize_against_inplace(
            LinearVectorModel(np.random.RandomState(5).randn(3, N_FEATURES))
        )
    elif name == "increment_inplace":
        new = np.random.RandomState(9).randn(6, N_FEATURES)
        if isinstance(model, PCAModel):
            new = [PointCloud(r.reshape(-1, 2)) for r in new]
        model.increment_inplace(new)
    elif name == "trim_components_inplace":
        model.trim_components_inplace(3)
    else:
        raise AssertionError(name)


CHANGES = pytest.mark.parametrize(
    "change",
    [
        "orthonormalize_inplace",
        "orthonormalize_against_inplace",
        "increment_inplace",
        "trim_components_inplace",
    ],
)


@MODELS
@CHANGES
def test_changing_a_model_does_not_change_a_truncated_one_made_from_it(make, change):
    model = make()
    small = model.trim_components(5)
    reference = make().trim_components(5)
    _change(change, model)
    assert_same_model(small, reference)  # every number as it was


@MODELS
@CHANGES
def test_changing_a_truncated_model_does_not_change_the_model_it_came_from(
    make, change
):
    model, reference = make(), make()
    small = model.trim_components(5)
    _change(change, small)
    assert_same_model(model, reference)


@MODELS
@CHANGES
def test_the_arrays_of_a_model_are_never_written_to(make, change):
    model = make()
    arrays = {k: model.__dict__[k] for k in ("_components", "_eigenvalues", "_mean")}
    before = {k: a.copy() for k, a in arrays.items()}
    _change(change, model)
    for k in arrays:
        assert_array_equal(arrays[k], before[k], err_msg=k)


@MODELS
@pytest.mark.parametrize(
    "change", ["orthonormalize_inplace", "orthonormalize_against_inplace"]
)
def test_orthonormalizing_gives_c_contiguous_components_of_the_same_dtype(make, change):
    model = make()
    model._components = model._components.astype(np.float32)
    _change(change, model)
    assert model._components.dtype == np.float32
    assert model._components.flags.c_contiguous


def test_orthonormalizing_a_linear_model_against_another():
    rs = np.random.RandomState(6)
    a, b = LinearVectorModel(rs.randn(3, 30)), LinearVectorModel(rs.randn(4, 30))
    arrays = a._components, b._components
    before = arrays[0].copy(), arrays[1].copy()
    a.orthonormalize_against_inplace(b)
    both = np.vstack([b.components, a.components])
    assert_allclose(both @ both.T, np.eye(7), atol=1e-12)
    assert_array_equal(arrays[0], before[0])
    assert_array_equal(arrays[1], before[1])


def test_orthonormalizing_a_linear_model_does_not_write_its_array():
    model = LinearVectorModel(np.random.RandomState(0).randn(4, 30))
    array = model._components
    before = array.copy()
    model.orthonormalize_inplace()
    assert_array_equal(array, before)
    assert_allclose(model.components @ model.components.T, np.eye(4), atol=1e-12)


@MODELS
def test_orthonormalizing_against_a_model_does_not_write_its_arrays_either(make):
    # the other model is changed too: neither of its arrays is written to, so what
    # shares them (here, a truncated model of a PCA model) is not changed
    other_pca = make(seed=3)
    view = other_pca.trim_components(4)
    before = view.components.copy()
    model = LinearVectorModel(np.random.RandomState(5).randn(3, N_FEATURES))
    model.orthonormalize_against_inplace(other_pca)
    assert_array_equal(view.components, before)
    pca, other = (
        make(),
        LinearVectorModel(np.random.RandomState(5).randn(3, N_FEATURES)),
    )
    array = other._components
    other_before = array.copy()
    pca.orthonormalize_against_inplace(other)
    assert_array_equal(array, other_before)


def test_a_model_of_read_only_arrays_on_disk_works(tmp_path):
    # e.g. a model too large for memory, kept on disk with np.load(..., mmap_mode="r")
    model = vector_model()
    for name in ("components", "eigenvalues", "mean"):
        np.save(tmp_path / (name + ".npy"), getattr(model, "_" + name))
    disk = {
        n: np.load(tmp_path / (n + ".npy"), mmap_mode="r")
        for n in ("components", "eigenvalues", "mean")
    }
    assert not disk["components"].flags.writeable
    on_disk = PCAVectorModel.init_from_components(
        disk["components"],
        disk["eigenvalues"],
        disk["mean"],
        n_samples=N_SAMPLES,
        centred=True,
    )
    small = on_disk.trim_components(5)
    assert np.shares_memory(small.components, disk["components"])
    assert_allclose(
        small.project(np.ones(N_FEATURES)),
        model.trim_components(5).project(np.ones(N_FEATURES)),
    )
    on_disk.orthonormalize_inplace()  # new arrays in memory; the file is not written to
    assert_array_equal(np.load(tmp_path / "components.npy"), model._components)


@MODELS
def test_a_copy_of_a_truncated_model_has_arrays_of_its_own_only_the_size_it_needs(make):
    model = make()
    small = model.trim_components(5).copy()
    assert not np.shares_memory(small.components, model.components)
    assert small._components.nbytes < 0.3 * model._components.nbytes
    assert_same_model(small, model.trim_components(5))


def test_increment_gives_a_model_its_own_arrays():
    model = vector_model()
    small = model.trim_components(5)
    model.increment_inplace(np.random.RandomState(9).randn(10, N_FEATURES))
    assert not np.shares_memory(small.components, model.components)


def test_trimming_a_truncated_model_gives_it_its_own_smaller_arrays():
    model = vector_model()
    small = model.trim_components(10)
    small.trim_components_inplace(4)
    assert small.n_components == 4
    assert not np.shares_memory(small.components, model.components)
    assert small._components.nbytes < 0.3 * model._components.nbytes
    assert model.n_components == 19


@MODELS
def test_a_truncated_model_pickles_and_comes_back(make):
    small = make().trim_components(5)
    again = pickle.loads(pickle.dumps(small))
    assert_same_model(again, small)
    assert not np.shares_memory(again._components, small._components)
