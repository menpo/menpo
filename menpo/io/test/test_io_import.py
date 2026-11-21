import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.testing import assert_allclose
from PIL import Image as PILImage
from pytest import raises

import menpo.io as mio
from menpo.shape import PointCloud, PointUndirectedGraph


def test_import_incorrect_built_in():
    with raises(ValueError):
        mio.import_builtin_asset("adskljasdlkajd.obj")


def test_breaking_bad_import():
    img = mio.import_builtin_asset("breakingbad.jpg")
    assert img.shape == (1080, 1920)
    assert img.n_channels == 3
    assert img.landmarks["PTS"].n_points == 68


def test_breaking_bad_import_kwargs():
    img = mio.import_builtin_asset("breakingbad.jpg", normalize=False)
    assert img.pixels.dtype == np.uint8


def test_takeo_import():
    img = mio.import_builtin_asset("takeo.ppm")
    assert img.shape == (225, 150)
    assert img.n_channels == 3
    assert img.landmarks["PTS"].n_points == 68


def test_einstein_import():
    img = mio.import_builtin_asset("einstein.jpg")
    assert img.shape == (1024, 817)
    assert img.n_channels == 1
    assert img.landmarks["PTS"].n_points == 68


def test_lenna_import():
    img = mio.import_builtin_asset("lenna.png")
    assert img.shape == (512, 512)
    assert img.n_channels == 3
    assert img.landmarks.n_groups == 2
    assert img.landmarks["LJSON"].n_points == 68
    assert img.landmarks["pupils"].n_points == 2


def test_import_builtin_ljson():
    lmarks = mio.import_builtin_asset("lenna.ljson")["LJSON"]
    assert lmarks.n_points == 68


def test_import_builtin_pts():
    lmarks = mio.import_builtin_asset("einstein.pts")["PTS"]
    assert lmarks.n_points == 68


def test_resolve_from_paths_single_group():
    def resolver(path):
        test_dict = {"test": path.with_name("takeo.pts")}
        return mio.input.resolve_from_paths(test_dict)

    image = mio.import_image(
        mio.data_path_to("einstein.jpg"), landmark_resolver=resolver
    )
    assert image.landmarks.n_groups == 1
    assert image.landmarks["test"].path == mio.data_path_to("takeo.pts")


def test_resolve_from_paths_multi_group():
    def resolver(path):
        test_dict = {"test": path.with_name("lenna.ljson")}
        return mio.input.resolve_from_paths(test_dict)

    image = mio.import_image(
        mio.data_path_to("einstein.jpg"), landmark_resolver=resolver
    )
    assert image.landmarks.n_groups == 2
    assert set(image.landmarks.keys()) == {"test_LJSON", "test_pupils"}


def test_path():
    # choose a random asset (all should have it!)
    img = mio.import_builtin_asset("einstein.jpg")
    path = mio.data_path_to("einstein.jpg")
    assert img.path == path
    assert img.path.stem == "einstein"
    assert img.path.suffix == ".jpg"
    assert img.path.parent == mio.data_dir_path()
    assert img.path.name == "einstein.jpg"


@patch("menpo.io.input.base._pathlib_glob_for_pattern")
def test_single_suffix_dot_in_path(pathlib_glob):
    from pathlib import Path

    import menpo.io.input.base as mio_base

    fake_path = Path("fake_path.t0.t1.t2")
    pathlib_glob.return_value = [fake_path]
    ext_map = MagicMock()
    ext_map.__contains__.side_effect = lambda x: x == ".t2"

    ret_val = next(mio_base.glob_with_suffix("*.t0.t1.t2", ext_map))
    assert ret_val == fake_path
    ext_map.__contains__.assert_called_with(".t2")


def test_upper_extension_mapped_to_lower():
    from pathlib import Path

    import menpo.io.input.base as mio_base

    ext_map = MagicMock()

    mio_base.importer_for_filepath(Path("fake_path.JPG"), ext_map)
    ext_map.get.assert_called_with(".jpg")


@patch("menpo.io.input.base._pathlib_glob_for_pattern")
def test_double_suffix(pathlib_glob):
    from pathlib import Path

    import menpo.io.input.base as mio_base

    fake_path = Path("fake_path.t1.t2")
    pathlib_glob.return_value = [fake_path]
    ext_map = MagicMock()
    ext_map.__contains__.side_effect = lambda x: x == ".t1.t2"

    ret_val = next(mio_base.glob_with_suffix("*.t1.t2", ext_map))
    assert ret_val == fake_path
    ext_map.__contains__.assert_any_call(".t1.t2")
    ext_map.__contains__.assert_any_call(".t2")


def test_import_image():
    img_path = mio.data_dir_path() / "einstein.jpg"
    im = mio.import_image(img_path)
    assert im.pixels.dtype == float
    assert im.n_channels == 1


def test_custom_landmark_resolver():
    def lmark_resolver(path):
        return mio.import_landmark_file(mio.data_path_to("takeo.pts"))

    img = mio.import_image(
        mio.data_path_to("lenna.png"), landmark_resolver=lmark_resolver
    )
    assert img.has_landmarks

    takeo_lmarks = mio.import_builtin_asset.takeo_pts()["PTS"]
    np.allclose(img.landmarks["PTS"].points, takeo_lmarks.points)


def test_landmark_resolver_none():
    img = mio.import_image(mio.data_path_to("lenna.png"), landmark_resolver=None)
    assert not img.has_landmarks


def test_import_image_no_norm():
    img_path = mio.data_dir_path() / "einstein.jpg"
    im = mio.import_image(img_path, normalize=False)
    assert im.pixels.dtype == np.uint8


def test_import_landmark_file():
    lm_path = mio.data_dir_path() / "einstein.pts"
    mio.import_landmark_file(lm_path)


def test_import_images():
    imgs = list(mio.import_images(mio.data_dir_path()))
    imgs_filenames = set(i.path.stem for i in imgs)
    exp_imgs_filenames = {
        "einstein",
        "takeo",
        "tongue",
        "breakingbad",
        "lenna",
        "menpo_thumbnail",
    }
    assert exp_imgs_filenames == imgs_filenames


def test_import_images_are_ordered_and_unduplicated():
    # we know that import_images returns images in path order
    imgs = list(mio.import_images(mio.data_dir_path()))
    imgs_filenames = [i.path.stem for i in imgs]
    print(imgs_filenames)
    exp_imgs_filenames = [
        "breakingbad",
        "einstein",
        "lenna",
        "menpo_thumbnail",
        "takeo",
        "tongue",
    ]
    assert exp_imgs_filenames == imgs_filenames


def test_lsimgs_filenamess():
    assert set(mio.ls_builtin_assets()) == {
        "breakingbad.jpg",
        "einstein.jpg",
        "einstein.pts",
        "lenna.png",
        "breakingbad.pts",
        "lenna.ljson",
        "takeo.ppm",
        "takeo.pts",
        "tongue.jpg",
        "tongue.pts",
        "menpo_thumbnail.jpg",
    }


def test_image_paths():
    ls = mio.image_paths(mio.data_dir_path())
    assert len(list(ls)) == 6


def test_import_images_wrong_path_raises_value_error():
    with raises(ValueError):
        list(mio.import_images("asldfjalkgjlaknglkajlekjaltknlaekstjlakj"))


def test_import_landmark_files_wrong_path_raises_value_error():
    with raises(ValueError):
        list(mio.import_landmark_files("asldfjalkgjlaknglkajlekjaltknlaekstjlakj"))


@patch("PIL.Image.open")
@patch("menpo.io.input.base.Path.is_file")
def test_importing_PIL_no_normalize(is_file, mock_image):
    mock_image.return_value = PILImage.new("RGBA", (10, 15))
    is_file.return_value = True

    im = mio.import_image("fake_image_being_mocked.png", normalize=False)
    assert im.shape == (15, 10)
    assert im.n_channels == 4
    assert im.pixels.dtype == np.uint8


@patch("PIL.Image.open")
@patch("menpo.io.input.base.Path.is_file")
def test_importing_PIL_RGBA_normalize(is_file, mock_image):
    from menpo.image import MaskedImage

    mock_image.return_value = PILImage.new("RGBA", (10, 15))
    is_file.return_value = True

    im = mio.import_image("fake_image_being_mocked.ppm", normalize=True)
    assert im.shape == (15, 10)
    assert im.n_channels == 3
    assert im.pixels.dtype == float
    assert type(im) == MaskedImage


@patch("PIL.Image.open")
@patch("menpo.io.input.base.Path.is_file")
def test_importing_PIL_RGBA_no_normalize(is_file, mock_image):
    mock_image.return_value = PILImage.new("RGBA", (10, 15))
    is_file.return_value = True

    im = mio.import_image("fake_image_being_mocked.ppm", normalize=False)
    assert im.shape == (15, 10)
    assert im.n_channels == 4
    assert im.pixels.dtype == np.uint8


@patch("PIL.Image.open")
@patch("menpo.io.input.base.Path.is_file")
def test_importing_PIL_L_no_normalize(is_file, mock_image):
    mock_image.return_value = PILImage.new("L", (10, 15))
    is_file.return_value = True

    im = mio.import_image("fake_image_being_mocked.ppm", normalize=False)
    assert im.shape == (15, 10)
    assert im.n_channels == 1
    assert im.pixels.dtype == np.uint8


@patch("PIL.Image.open")
@patch("menpo.io.input.base.Path.is_file")
def test_importing_PIL_L_normalize(is_file, mock_image):
    mock_image.return_value = PILImage.new("L", (10, 15))
    is_file.return_value = True

    im = mio.import_image("fake_image_being_mocked.ppm", normalize=True)
    assert im.shape == (15, 10)
    assert im.n_channels == 1
    assert im.pixels.dtype == float


@patch("PIL.Image.open")
@patch("menpo.io.input.base.Path.is_file")
def test_importing_PIL_I_normalize(is_file, mock_image):
    mock_image.return_value = PILImage.new("I", (10, 15))
    is_file.return_value = True
    with raises(ValueError):
        mio.import_image("fake_image_being_mocked.ppm", normalize=True)


@patch("PIL.Image.open")
@patch("menpo.io.input.base.Path.is_file")
def test_importing_PIL_I_no_normalize(is_file, mock_image):
    mock_image.return_value = PILImage.new("I", (10, 15))
    is_file.return_value = True

    im = mio.import_image("fake_image_being_mocked.ppm", normalize=False)
    assert im.shape == (15, 10)
    assert im.n_channels == 1
    assert im.pixels.dtype == np.int32


@patch("PIL.Image.open")
@patch("menpo.io.input.base.Path.is_file")
def test_importing_PIL_1_normalize(is_file, mock_image):
    from menpo.image import BooleanImage

    mock_image.return_value = PILImage.new("1", (10, 15))
    is_file.return_value = True

    im = mio.import_image("fake_image_being_mocked.ppm", normalize=True)
    assert im.shape == (15, 10)
    assert im.n_channels == 1
    assert im.pixels.dtype == bool
    assert type(im) == BooleanImage


@patch("PIL.Image.open")
@patch("menpo.io.input.base.Path.is_file")
def test_importing_PIL_1_no_normalize(is_file, mock_image):
    from menpo.image import BooleanImage

    mock_image.return_value = PILImage.new("1", (10, 15))
    is_file.return_value = True

    im = mio.import_image("fake_image_being_mocked.ppm", normalize=False)
    assert im.shape == (15, 10)
    assert im.n_channels == 1
    assert im.pixels.dtype == bool
    assert type(im) == BooleanImage


@patch("PIL.Image.open")
@patch("menpo.io.input.base.Path.is_file")
def test_importing_PIL_1_proper_conversion(is_file, mock_image):
    from menpo.image import BooleanImage

    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[4, 4] = 255
    mock_image.return_value = PILImage.fromarray(arr).convert("1")
    is_file.return_value = True

    im = mio.import_image("fake_image_being_mocked.ppm", normalize=False)
    assert im.shape == (10, 10)
    assert im.n_channels == 1
    assert im.pixels.dtype == bool
    assert type(im) == BooleanImage
    assert np.all(im.pixels == arr.astype(bool))


@patch("PIL.Image.open")
@patch("menpo.io.input.base.Path.is_file")
def test_importing_PIL_P_normalize(is_file, mock_image):
    mock_image.return_value = PILImage.new("P", (10, 10))
    is_file.return_value = True

    im = mio.import_image("fake_image_being_mocked.ppm", normalize=True)
    assert im.shape == (10, 10)
    assert im.n_channels == 3
    assert im.pixels.dtype == float


@patch("PIL.Image.open")
@patch("menpo.io.input.base.Path.is_file")
def test_importing_PIL_P_no_normalize(is_file, mock_image):
    mock_image.return_value = PILImage.new("P", (10, 15))
    is_file.return_value = True

    im = mio.import_image("fake_image_being_mocked.ppm", normalize=False)
    assert im.shape == (15, 10)
    assert im.n_channels == 3
    assert im.pixels.dtype == np.uint8


@patch("subprocess.Popen")
@patch("menpo.io.input.video.video_infos_ffprobe")
@patch("menpo.io.input.base.Path.is_file")
def test_importing_ffmpeg_GIF_normalize(is_file, video_infos_ffprobe, pipe):
    video_infos_ffprobe.return_value = {
        "duration": 2,
        "width": 100,
        "height": 150,
        "n_frames": 10,
        "fps": 5,
    }
    empty_frame = np.zeros(150 * 100 * 3, dtype=np.uint8).tobytes()
    pipe.return_value.stdout.read.return_value = empty_frame
    is_file.return_value = True

    ll = mio.import_video("fake_image_being_mocked.gif", normalize=True)
    assert ll.path.name == "fake_image_being_mocked.gif"
    assert ll.fps == 5
    assert len(ll) == 10

    im = ll[0]
    assert im.shape == (150, 100)
    assert im.n_channels == 3
    assert im.pixels.dtype == float


@patch("subprocess.Popen")
@patch("menpo.io.input.video.video_infos_ffprobe")
@patch("menpo.io.input.base.Path.is_file")
def test_importing_ffmpeg_GIF_no_normalize(is_file, video_infos_ffprobe, pipe):
    video_infos_ffprobe.return_value = {
        "duration": 2,
        "width": 100,
        "height": 150,
        "n_frames": 10,
        "fps": 5,
    }
    empty_frame = np.zeros(150 * 100 * 3, dtype=np.uint8).tobytes()
    pipe.return_value.stdout.read.return_value = empty_frame
    is_file.return_value = True

    ll = mio.import_video("fake_image_being_mocked.gif", normalize=False)
    assert ll.path.name == "fake_image_being_mocked.gif"
    assert ll.fps == 5
    assert len(ll) == 10

    im = ll[0]
    assert im.shape == (150, 100)
    assert im.n_channels == 3
    assert im.pixels.dtype == np.uint8


@patch("menpo.io.input.landmark.json.load")
@patch("menpo.io.input.base.Path.open")
@patch("menpo.io.input.base.Path.is_file")
def test_importing_v1_ljson_null_values(is_file, mock_open, mock_dict):
    v1_ljson = {
        "groups": [
            {
                "connectivity": [[0, 1], [1, 2], [2, 3]],
                "label": "chin",
                "landmarks": [
                    {"point": [987.9, 1294.1]},
                    {"point": [96.78, 1246.8]},
                    {"point": [None, 0.1]},
                    {"point": [303.22, 167.2]},
                ],
            },
            {
                "connectivity": [[0, 1]],
                "label": "leye",
                "landmarks": [{"point": [None, None]}, {"point": [None, None]}],
            },
        ],
        "version": 1,
    }
    mock_dict.return_value = v1_ljson
    is_file.return_value = True

    with warnings.catch_warnings(record=True) as found_warnings:
        lmark = mio.import_landmark_file("fake_lmark_being_mocked.ljson", group="LJSON")
    nan_points = np.isnan(lmark.points)

    # Should raise deprecation warning
    assert len(found_warnings) == 1, [w.message for w in found_warnings]
    assert nan_points[2, 0]  # y-coord None point is nan
    assert not nan_points[2, 1]  # x-coord point is not nan
    assert np.all(nan_points[4:, :])  # all of leye label is nan


@patch("menpo.io.input.landmark.json.load")
@patch("menpo.io.input.base.Path.open")
@patch("menpo.io.input.base.Path.is_file")
def test_importing_v2_ljson_null_values(is_file, mock_open, mock_dict):
    v2_ljson = {
        "labels": [
            {"label": "left_eye", "mask": [0, 1, 2]},
            {"label": "right_eye", "mask": [3, 4, 5]},
        ],
        "landmarks": {
            "connectivity": [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]],
            "points": [
                [None, 200.5],
                [None, None],
                [316.8, 199.15],
                [339.48, 205.0],
                [358.54, 217.82],
                [375.0, 233.4],
            ],
        },
        "version": 2,
    }

    mock_dict.return_value = v2_ljson
    is_file.return_value = True
    with warnings.catch_warnings(record=True) as w:
        lmark = mio.import_landmark_file("fake_lmark_being_mocked.ljson", group="LJSON")
    nan_points = np.isnan(lmark.points)
    assert nan_points[0, 0]  # y-coord None point is nan
    assert not nan_points[0, 1]  # x-coord point is not nan
    assert np.all(nan_points[1, :])  # all of leye label is nan


@patch("menpo.io.input.landmark.json.load")
@patch("menpo.io.input.base.Path.open")
@patch("menpo.io.input.base.Path.is_file")
def test_importing_v3_ljson_null_values(is_file, mock_open, mock_dict):
    v3_ljson = {
        "groups": {
            "LJSON": {
                "labels": [
                    {"label": "left_eye", "mask": [0, 1, 2]},
                    {"label": "right_eye", "mask": [3, 4, 5]},
                ],
                "landmarks": {
                    "connectivity": [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]],
                    "points": [
                        [None, 200.5],
                        [None, None],
                        [316.8, 199.15],
                        [339.48, 205.0],
                        [358.54, 217.82],
                        [375.0, 233.4],
                    ],
                },
            }
        },
        "version": 3,
    }

    mock_dict.return_value = v3_ljson
    is_file.return_value = True
    lmark_dict = mio.import_landmark_file("fake_lmark_being_mocked.ljson")
    assert isinstance(lmark_dict, dict)
    lmark = lmark_dict["LJSON"]
    nan_points = np.isnan(lmark.points)
    assert nan_points[0, 0]  # y-coord None point is nan
    assert not nan_points[0, 1]  # x-coord point is not nan
    assert np.all(nan_points[1, :])  # all of leye label is nan


def test_export_import_pointcloud_ljson_v3(tmpdir):
    tmpdir = Path(tmpdir)
    pc = PointCloud(np.random.random([10, 2]), copy=False)
    mio.export_landmark_file(pc, tmpdir / "test.ljson")
    loaded_pc = mio.import_landmark_file(tmpdir / "test.ljson")["LJSON"]
    np.testing.assert_allclose(pc.points, loaded_pc.points)


@patch("random.shuffle")
def test_shuffle_kwarg_true_calls_shuffle(mock):
    list(mio.import_images(mio.data_dir_path(), shuffle=True))
    assert mock.called


def test_import_as_generator():
    import types

    gen = mio.import_images(mio.data_dir_path(), as_generator=True)
    assert isinstance(gen, types.GeneratorType)
    gen = mio.import_landmark_files(mio.data_dir_path(), as_generator=True)
    assert isinstance(gen, types.GeneratorType)


def test_import_lazy_list():
    from menpo.base import LazyList

    data_path = mio.data_dir_path()
    ll = mio.import_images(data_path)
    assert isinstance(ll, LazyList)
    ll = mio.import_landmark_files(data_path)
    assert isinstance(ll, LazyList)


@patch("menpo.io.input.pickle.pickle.load")
@patch("menpo.io.input.base.Path.open")
@patch("menpo.io.input.base.Path.is_file")
def test_importing_pickle(is_file, mock_open, mock_pickle):
    mock_pickle.return_value = {"test": 1}
    is_file.return_value = True

    objs = mio.import_pickle("mocked.pkl")
    assert isinstance(objs, dict)
    assert "test" in objs
    assert objs["test"] == 1


@patch("menpo.io.input.pickle.pickle.load")
@patch("menpo.io.input.base.Path.open")
@patch("menpo.io.input.base.Path.is_file")
@patch("sys.version_info")
def test_importing_pickle_encoding_py3(version_info, is_file, mock_open, mock_pickle):
    version_info.major = 3
    mock_pickle.return_value = {"test": 1}
    is_file.return_value = True

    mio.import_pickle("mocked.pkl", encoding="latin1")
    assert mock_pickle.call_args[1].get("encoding") == "latin1"


@patch("menpo.io.input.pickle.pickle.load")
@patch("menpo.io.input.base.Path.open")
@patch("menpo.io.input.base.Path.is_file")
@patch("sys.version_info")
def test_importing_pickle_encoding_ignored_py2(
    version_info, is_file, mock_open, mock_pickle
):
    version_info.major = 2
    mock_pickle.return_value = {"test": 1}
    is_file.return_value = True

    mio.import_pickle("mocked.pkl", encoding="latin1")
    assert "encoding" not in mock_pickle.call_args[1]


@patch("menpo.io.input.pickle.pickle.load")
@patch("menpo.io.input.base.Path.open")
@patch("menpo.io.input.base.Path.glob")
@patch("menpo.io.input.base.Path.is_file")
def test_importing_pickles(is_file, glob, mock_open, mock_pickle):
    from pathlib import Path

    from menpo.base import LazyList

    mock_pickle.return_value = {"test": 1}
    is_file.return_value = True
    glob.return_value = [Path("mocked1.pkl"), Path("mocked2.pkl")]

    objs = mio.import_pickles("*")
    assert isinstance(objs, LazyList)
    assert len(objs) == 2
    assert objs[0]["test"] == 1
    assert objs[1]["test"] == 1


@patch("menpo.io.input.pickle.pickle.load")
@patch("menpo.io.input.base.Path.open")
@patch("menpo.io.input.base.Path.glob")
@patch("menpo.io.input.base.Path.is_file")
def test_importing_pickles_as_generator(is_file, glob, mock_open, mock_pickle):
    import types
    from pathlib import Path

    mock_pickle.return_value = {"test": 1}
    is_file.return_value = True
    glob.return_value = [Path("mocked1.pkl"), Path("mocked2.pkl")]

    objs = mio.import_pickles("*", as_generator=True)
    assert isinstance(objs, types.GeneratorType)
    objs = list(objs)
    assert len(objs) == 2
    assert objs[0]["test"] == 1
    assert objs[1]["test"] == 1


@patch("subprocess.Popen")
@patch("menpo.io.input.video.video_infos_ffprobe")
@patch("menpo.io.input.base.Path.is_file")
def test_importing_ffmpeg_avi_no_normalize(is_file, video_infos_ffprobe, pipe):
    video_infos_ffprobe.return_value = {
        "duration": 2,
        "width": 100,
        "height": 150,
        "n_frames": 10,
        "fps": 5,
    }
    is_file.return_value = True
    empty_frame = np.zeros(150 * 100 * 3, dtype=np.uint8).tobytes()
    pipe.return_value.stdout.read.return_value = empty_frame
    ll = mio.import_video("fake_image_being_mocked.avi", normalize=False)
    assert ll.path.name == "fake_image_being_mocked.avi"
    assert ll.fps == 5
    assert len(ll) == 5 * 2
    image = ll[0]
    assert image.shape == ((150, 100))
    assert image.n_channels == 3
    assert image.pixels.dtype == np.uint8


@patch("subprocess.Popen")
@patch("menpo.io.input.video.video_infos_ffprobe")
@patch("menpo.io.input.base.Path.is_file")
def test_importing_ffmpeg_avi_normalize(is_file, video_infos_ffprobe, pipe):
    video_infos_ffprobe.return_value = {
        "duration": 2,
        "width": 100,
        "height": 150,
        "n_frames": 10,
        "fps": 5,
    }
    is_file.return_value = True
    empty_frame = np.zeros(150 * 100 * 3, dtype=np.uint8).tobytes()
    pipe.return_value.stdout.read.return_value = empty_frame
    ll = mio.import_video("fake_image_being_mocked.avi", normalize=True)
    assert ll.path.name == "fake_image_being_mocked.avi"
    assert ll.fps == 5
    assert len(ll) == 5 * 2
    image = ll[0]
    assert image.shape == (150, 100)
    assert image.n_channels == 3
    assert image.pixels.dtype == float


@patch("menpo.io.input.video.video_infos_ffprobe")
@patch("menpo.io.input.base.Path.is_file")
def test_importing_ffmpeg_exact_frame_count_no_ffprobe(is_file, video_infos_ffprobe):
    video_infos_ffprobe.side_effect = ValueError
    is_file.return_value = True
    with raises(ValueError):
        mio.import_video(
            "fake_image_being_mocked.avi", normalize=True, exact_frame_count=True
        )


@patch("subprocess.Popen")
@patch("menpo.io.input.video.video_infos_ffprobe")
@patch("menpo.io.input.video.video_infos_ffmpeg")
@patch("menpo.io.input.base.Path.is_file")
def test_importing_ffmpeg_no_exact_frame_count_no_ffprobe(
    is_file, video_infos_ffmpeg, video_infos_ffprobe, pipe
):
    video_infos_ffprobe.side_effect = ValueError
    video_infos_ffmpeg.return_value = {
        "duration": 2,
        "width": 100,
        "height": 150,
        "n_frames": 10,
        "fps": 5,
    }
    is_file.return_value = True
    empty_frame = np.zeros(150 * 100 * 3, dtype=np.uint8).tobytes()
    pipe.return_value.stdout.read.return_value = empty_frame
    ll = mio.import_video(
        "fake_image_being_mocked.avi", normalize=True, exact_frame_count=False
    )
    assert ll.path.name == "fake_image_being_mocked.avi"
    assert ll.fps == 5
    assert len(ll) == 5 * 2
    image = ll[0]
    assert image.shape == (150, 100)
    assert image.n_channels == 3
    assert image.pixels.dtype == float


def test_import_images_negative_max_images():
    with raises(ValueError):
        list(mio.import_images(mio.data_dir_path(), max_images=-2))


def test_import_images_zero_max_images():
    with raises(ValueError):
        # different since the conditional 'if max_assets' is skipped,
        # thus all images might be imported.
        list(mio.import_images(mio.data_dir_path(), max_images=0))


# TODO: remove once the normalise argument is removed.
def test_import_image_deprecated_normalise_kwarg():
    with warnings.catch_warnings(record=True) as w:
        img = mio.import_builtin_asset("breakingbad.jpg", normalise=False)
        assert len(w) == 1
    assert img.pixels.dtype == np.uint8


@patch("menpo.io.input.base.Path.is_file")
def test_register_image_importer(is_file):
    from menpo.image import Image

    image = Image.init_blank((10, 10))

    def foo_importer(filepath, **kwargs):
        return image

    is_file.return_value = True

    with patch.dict(mio.input.extensions.image_types, {}, clear=True):
        mio.register_image_importer(".foo", foo_importer)
        new_image = mio.import_image("fake.foo")
    assert image is new_image


@patch("menpo.io.input.base.Path.is_file")
def test_register_landmark_importer(is_file):
    from menpo.shape import PointCloud

    lmark = PointCloud.init_2d_grid((1, 1))

    def foo_importer(filepath, **kwargs):
        return lmark

    is_file.return_value = True

    with patch.dict(mio.input.extensions.image_landmark_types, {}, clear=True):
        mio.register_landmark_importer(".foo", foo_importer)
        new_lmark = mio.import_landmark_file("fake.foo")
    assert lmark is new_lmark


@patch("menpo.io.input.base.Path.is_file")
def test_register_video_importer(is_file):
    from menpo.base import LazyList
    from menpo.image import Image

    def foo_importer(filepath, **kwargs):
        return LazyList([lambda: Image.init_blank((10, 10))])

    is_file.return_value = True

    with patch.dict(mio.input.extensions.ffmpeg_video_types, {}, clear=True):
        mio.register_video_importer(".foo", foo_importer)
        new_video = mio.import_video("fake.foo")
    assert len(new_video) == 1


@patch("menpo.io.input.base.Path.is_file")
def test_register_pickle_importer(is_file):
    obj = object()

    def foo_importer(filepath, **kwargs):
        return obj

    is_file.return_value = True

    with patch.dict(mio.input.extensions.pickle_types, {}, clear=True):
        mio.register_pickle_importer(".foo", foo_importer)
        new_obj = mio.import_pickle("fake.foo")
    assert new_obj is obj


def test_register_no_leading_period():
    ext_map = {}
    mio.input.base._register_importer(ext_map, "foo", lambda x: x)
    assert ".foo" in ext_map
    assert "foo" not in ext_map


# ---- PsychoMorph .tem templates
_TEM = """# a template: a closed square, a line from the corner to the centre
5
10 20
30 20
30 40
10 40
20 30

2
1
4
0 1 2 3
0
2
3 4
this and everything after the last line is ignored
"""


def _write_tem(tmp_path, text, name="face.tem"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_import_tem_points_edges_and_axes(tmp_path):
    graph = mio.import_landmark_file(_write_tem(tmp_path, _TEM))["TEM"]
    assert isinstance(graph, PointUndirectedGraph)
    assert graph.n_points == 5
    # menpo is (y, x): the file has x first
    assert_allclose(graph.points[0], [20, 10])
    assert_allclose(graph.points[4], [30, 20])
    # the line flagged closed is joined last to first; the open line is not
    edges = {tuple(sorted(e)) for e in graph.edges.tolist()}
    assert edges == {(0, 1), (1, 2), (2, 3), (0, 3), (3, 4)}


def test_import_tem_without_lines_is_a_graph_with_no_edges(tmp_path):
    graph = mio.import_landmark_file(_write_tem(tmp_path, "2\n1 2\n3 4\n0\n"))["TEM"]
    assert graph.n_points == 2 and graph.n_edges == 0


def test_import_tem_is_found_next_to_an_image(tmp_path):
    _write_tem(tmp_path, _TEM)
    PILImage.fromarray(np.zeros((60, 50, 3), dtype=np.uint8)).save(
        tmp_path / "face.png"
    )
    image = mio.import_image(tmp_path / "face.png")
    assert image.landmarks["TEM"].n_points == 5


@pytest.mark.parametrize(
    "text",
    [
        "",  # empty
        "3\n1 2\n3 4\n",  # fewer points than it says
        "2\n1 2\n3\n0\n",  # a point with one coordinate
        "2\n1 2\n3 4\n1\n0\n",  # cut off in the middle of a line
        "2\n1 2\n3 4\n1\n0\n3\n0 1\n",  # says 3 points, lists 2
        "2\n1 2\n3 4\n1\n0\n2\n0 2\n",  # index 2 does not exist
        "2\n1 2\n3 4\n1\n0\n2\n-1 1\n",  # negative index
    ],
)
def test_import_tem_rejects_a_file_that_is_not_a_template(tmp_path, text):
    with pytest.raises(ValueError, match="PsychoMorph"):
        mio.import_landmark_file(_write_tem(tmp_path, text))


# ---- a GIF is a video, not an image
def _write_gif_and_png(tmp_path, landmarks=True):
    frames = [
        PILImage.fromarray(
            (np.random.RandomState(i).rand(20, 30, 3) * 255).astype("uint8")
        )
        for i in range(3)
    ]
    gif, png = tmp_path / "anim.gif", tmp_path / "still.png"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=100, loop=0)
    frames[0].save(png)
    if (
        landmarks
    ):  # a landmark file for each: the one next to the GIF used to crash the import
        for name in ("anim", "still"):
            (tmp_path / (name + ".pts")).write_text(
                "version: 1\nn_points: 2\n{\n 1.0 2.0\n 3.0 4.0\n}\n"
            )
    return gif, png


def test_import_images_of_a_folder_with_a_gif_gives_only_the_still_images(tmp_path):
    _write_gif_and_png(tmp_path)
    images = list(mio.import_images(str(tmp_path / "*")))
    assert len(images) == 1
    assert images[0].path.name == "still.png" and images[0].n_dims == 2
    assert images[0].landmarks.n_groups == 1


def test_image_paths_leave_out_a_gif_and_video_paths_keep_it(tmp_path):
    _write_gif_and_png(tmp_path, landmarks=False)
    assert [p.name for p in mio.image_paths(str(tmp_path / "*"))] == ["still.png"]
    assert "anim.gif" in [p.name for p in mio.video_paths(str(tmp_path / "*"))]


@pytest.mark.parametrize("landmarks", [False, True])
def test_import_image_of_a_gif_says_to_use_import_video(tmp_path, landmarks):
    gif, _ = _write_gif_and_png(tmp_path, landmarks=landmarks)
    # it used to return a LazyList, or crash with an AttributeError if a landmark file was next to it
    with raises(
        ValueError, match="does not have a suitable importer.*video.*import_video"
    ):
        mio.import_image(gif)


def test_import_image_of_a_video_says_to_use_import_video(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not a real video")
    with raises(ValueError, match="import_video"):
        mio.import_image(path)


def test_import_image_of_an_unknown_format_has_no_such_hint(tmp_path):
    path = tmp_path / "thing.xyz"
    path.write_bytes(b"?")
    with raises(ValueError) as info:
        mio.import_image(path)
    assert "does not have a suitable importer." in str(info.value)
    assert "video" not in str(info.value)


def test_gif_is_not_an_image_type_but_is_a_video_type():
    from menpo.io.input.extensions import ffmpeg_video_types, image_types

    assert ".gif" not in image_types and ".gif" in ffmpeg_video_types
