import sys
import numpy as np
from mock import patch, MagicMock
from nose.tools import raises
from PIL import Image as PILImage
import menpo.io as mio
import warnings


builtins_str = '__builtin__' if sys.version_info[0] == 2 else 'builtins'


@raises(ValueError)
def test_import_incorrect_built_in():
    mio.import_builtin_asset('adskljasdlkajd.obj')


def test_breaking_bad_import():
    img = mio.import_builtin_asset('breakingbad.jpg')
    assert(img.shape == (1080, 1920))
    assert(img.n_channels == 3)
    assert(img.landmarks['PTS'].n_landmarks == 68)


def test_breaking_bad_import_kwargs():
    img = mio.import_builtin_asset('breakingbad.jpg', normalise=False)
    assert(img.pixels.dtype == np.uint8)


def test_takeo_import():
    img = mio.import_builtin_asset('takeo.ppm')
    assert(img.shape == (225, 150))
    assert(img.n_channels == 3)
    assert(img.landmarks['PTS'].n_landmarks == 68)


def test_einstein_import():
    img = mio.import_builtin_asset('einstein.jpg')
    assert(img.shape == (1024, 817))
    assert(img.n_channels == 1)
    assert(img.landmarks['PTS'].n_landmarks == 68)


def test_lenna_import():
    img = mio.import_builtin_asset('lenna.png')
    assert(img.shape == (512, 512))
    assert(img.n_channels == 3)
    assert(img.landmarks['LJSON'].n_landmarks == 68)


def test_import_builtin_ljson():
    lmarks = mio.import_builtin_asset('lenna.ljson')
    assert(lmarks.n_landmarks == 68)


def test_import_builtin_pts():
    lmarks = mio.import_builtin_asset('einstein.pts')
    assert(lmarks.n_landmarks == 68)


def test_path():
    # choose a random asset (all should have it!)
    img = mio.import_builtin_asset('einstein.jpg')
    path = mio.data_path_to('einstein.jpg')
    assert(img.path == path)
    assert(img.path.stem == 'einstein')
    assert(img.path.suffix == '.jpg')
    assert(img.path.parent == mio.data_dir_path())
    assert(img.path.name == 'einstein.jpg')


@patch('menpo.io.input.base._pathlib_glob_for_pattern')
def test_single_suffix_dot_in_path(pathlib_glob):
    import menpo.io.input.base as mio_base
    from pathlib import Path

    fake_path = Path('fake_path.t0.t1.t2')
    pathlib_glob.return_value = [fake_path]
    ext_map = MagicMock()
    ext_map.__contains__.side_effect = lambda x: x == '.t2'

    ret_val = next(mio_base.glob_with_suffix('*.t0.t1.t2', ext_map))
    assert (ret_val == fake_path)
    ext_map.__contains__.assert_called_with('.t2')


def test_upper_extension_mapped_to_lower():
    import menpo.io.input.base as mio_base
    from pathlib import Path
    ext_map = MagicMock()

    mio_base.importer_for_filepath(Path('fake_path.JPG'), ext_map)
    ext_map.get.assert_called_with('.jpg')


@patch('menpo.io.input.base._pathlib_glob_for_pattern')
def test_double_suffix(pathlib_glob):
    import menpo.io.input.base as mio_base
    from pathlib import Path

    fake_path = Path('fake_path.t1.t2')
    pathlib_glob.return_value = [fake_path]
    ext_map = MagicMock()
    ext_map.__contains__.side_effect = lambda x: x == '.t1.t2'

    ret_val = next(mio_base.glob_with_suffix('*.t1.t2', ext_map))
    assert (ret_val == fake_path)
    ext_map.__contains__.assert_called_with('.t1.t2')


def test_import_image():
    img_path = mio.data_dir_path() / 'einstein.jpg'
    im = mio.import_image(img_path)
    assert im.pixels.dtype == np.float
    assert im.n_channels == 1


def test_import_image_no_norm():
    img_path = mio.data_dir_path() / 'einstein.jpg'
    im = mio.import_image(img_path, normalise=False)
    assert im.pixels.dtype == np.uint8


def test_import_landmark_file():
    lm_path = mio.data_dir_path() / 'einstein.pts'
    mio.import_landmark_file(lm_path)


def test_import_images():
    imgs = list(mio.import_images(mio.data_dir_path()))
    imgs_filenames = set(i.path.stem for i in imgs)
    exp_imgs_filenames = {'einstein', 'takeo', 'tongue', 'breakingbad', 'lenna',
                          'menpo_thumbnail'}
    assert exp_imgs_filenames == imgs_filenames


def test_import_images_are_ordered_and_unduplicated():
    # we know that import_images returns images in path order
    imgs = list(mio.import_images(mio.data_dir_path()))
    imgs_filenames = [i.path.stem for i in imgs]
    print(imgs_filenames)
    exp_imgs_filenames = ['breakingbad', 'einstein', 'lenna', 'menpo_thumbnail', 'takeo', 'tongue']
    assert exp_imgs_filenames == imgs_filenames


def test_lsimgs_filenamess():
    assert(set(mio.ls_builtin_assets()) == {'breakingbad.jpg',
                                            'einstein.jpg', 'einstein.pts',
                                            'lenna.png', 'breakingbad.pts',
                                            'lenna.ljson', 'takeo.ppm',
                                            'takeo.pts', 'tongue.jpg',
                                            'tongue.pts',
                                            'menpo_thumbnail.jpg'})


def test_image_paths():
    ls = mio.image_paths(mio.data_dir_path())
    assert(len(list(ls)) == 6)


@raises(ValueError)
def test_import_images_wrong_path_raises_value_error():
    list(mio.import_images('asldfjalkgjlaknglkajlekjaltknlaekstjlakj'))


@raises(ValueError)
def test_import_landmark_files_wrong_path_raises_value_error():
    list(mio.import_landmark_files('asldfjalkgjlaknglkajlekjaltknlaekstjlakj'))


@patch('menpo.io.input.image.PILImage.open')
@patch('menpo.io.input.base.Path.is_file')
def test_importing_RGBA_no_normalise(is_file, mock_image):
    mock_image.return_value = PILImage.new('RGBA', (10, 10))
    is_file.return_value = True

    im = mio.import_image('fake_image_being_mocked.jpg', normalise=False)
    assert im.shape == (10, 10)
    assert im.n_channels == 4
    assert im.pixels.dtype == np.uint8


@patch('menpo.io.input.image.PILImage.open')
@patch('menpo.io.input.base.Path.is_file')
def test_importing_RGBA_normalise(is_file, mock_image):
    from menpo.image import MaskedImage

    mock_image.return_value = PILImage.new('RGBA', (10, 10))
    is_file.return_value = True

    im = mio.import_image('fake_image_being_mocked.jpg', normalise=True)
    assert im.shape == (10, 10)
    assert im.n_channels == 3
    assert im.pixels.dtype == np.float
    assert type(im) == MaskedImage


@patch('menpo.io.input.image.PILImage.open')
@patch('menpo.io.input.base.Path.is_file')
def test_importing_L_no_normalise(is_file, mock_image):
    mock_image.return_value = PILImage.new('L', (10, 10))
    is_file.return_value = True

    im = mio.import_image('fake_image_being_mocked.jpg', normalise=False)
    assert im.shape == (10, 10)
    assert im.n_channels == 1
    assert im.pixels.dtype == np.uint8


@patch('menpo.io.input.image.PILImage.open')
@patch('menpo.io.input.base.Path.is_file')
def test_importing_L_normalise(is_file, mock_image):
    mock_image.return_value = PILImage.new('L', (10, 10))
    is_file.return_value = True

    im = mio.import_image('fake_image_being_mocked.jpg', normalise=True)
    assert im.shape == (10, 10)
    assert im.n_channels == 1
    assert im.pixels.dtype == np.float


@patch('menpo.io.input.image.PILImage.open')
@patch('menpo.io.input.base.Path.is_file')
def test_importing_I_normalise(is_file, mock_image):
    mock_image.return_value = PILImage.new('I', (10, 10))
    is_file.return_value = True

    im = mio.import_image('fake_image_being_mocked.jpg', normalise=True)
    assert im.shape == (10, 10)
    assert im.n_channels == 1
    assert im.pixels.dtype == np.float


@patch('menpo.io.input.image.PILImage.open')
@patch('menpo.io.input.base.Path.is_file')
def test_importing_I_no_normalise(is_file, mock_image):
    mock_image.return_value = PILImage.new('I', (10, 10))
    is_file.return_value = True

    im = mio.import_image('fake_image_being_mocked.jpg', normalise=False)
    assert im.shape == (10, 10)
    assert im.n_channels == 1
    assert im.pixels.dtype == np.int32


@patch('menpo.io.input.image.PILImage.open')
@patch('menpo.io.input.base.Path.is_file')
def test_importing_1_normalise(is_file, mock_image):
    from menpo.image import BooleanImage

    mock_image.return_value = PILImage.new('1', (10, 10))
    is_file.return_value = True

    im = mio.import_image('fake_image_being_mocked.jpg', normalise=True)
    assert im.shape == (10, 10)
    assert im.n_channels == 1
    assert im.pixels.dtype == np.bool
    assert type(im) == BooleanImage


@patch('menpo.io.input.image.PILImage.open')
@patch('menpo.io.input.base.Path.is_file')
def test_importing_1_no_normalise(is_file, mock_image):
    from menpo.image import BooleanImage

    mock_image.return_value = PILImage.new('1', (10, 10))
    is_file.return_value = True

    im = mio.import_image('fake_image_being_mocked.jpg', normalise=False)
    assert im.shape == (10, 10)
    assert im.n_channels == 1
    assert im.pixels.dtype == np.bool
    assert type(im) == BooleanImage


@patch('menpo.io.input.image.PILImage.open')
@patch('menpo.io.input.base.Path.is_file')
def test_importing_P_normalise(is_file, mock_image):
    mock_image.return_value = PILImage.new('P', (10, 10))
    is_file.return_value = True

    im = mio.import_image('fake_image_being_mocked.jpg', normalise=True)
    assert im.shape == (10, 10)
    assert im.n_channels == 3
    assert im.pixels.dtype == np.float


@patch('menpo.io.input.image.PILImage.open')
@patch('menpo.io.input.base.Path.is_file')
def test_importing_P_no_normalise(is_file, mock_image):
    mock_image.return_value = PILImage.new('P', (10, 10))
    is_file.return_value = True

    im = mio.import_image('fake_image_being_mocked.jpg', normalise=False)
    assert im.shape == (10, 10)
    assert im.n_channels == 3
    assert im.pixels.dtype == np.uint8


@patch('menpo.io.input.image.PILImage.open')
@patch('menpo.io.input.base.Path.is_file')
def test_importing_GIF_normalise(is_file, mock_image):
    mock_image.return_value = PILImage.new('P', (10, 10))
    is_file.return_value = True

    im = mio.import_image('fake_image_being_mocked.gif', normalise=True)
    assert im.shape == (10, 10)
    assert im.n_channels == 3
    assert im.pixels.dtype == np.float


@patch('menpo.io.input.image.PILImage.open')
@patch('menpo.io.input.base.Path.is_file')
def test_importing_GIF_no_normalise(is_file, mock_image):
    mock_image.return_value = PILImage.new('P', (10, 10))
    is_file.return_value = True

    im = mio.import_image('fake_image_being_mocked.gif', normalise=False)
    assert im.shape == (10, 10)
    assert im.n_channels == 3
    assert im.pixels.dtype == np.uint8


@patch('menpo.io.input.image.PILImage.open')
@patch('menpo.io.input.base.Path.is_file')
@raises(ValueError)
def test_importing_GIF_non_pallete_exception(is_file, mock_image):
    mock_image.return_value = PILImage.new('RGB', (10, 10))
    is_file.return_value = True

    mio.import_image('fake_image_being_mocked.gif', normalise=False)


@patch('menpo.io.input.landmark.json.load')
@patch('{}.open'.format(builtins_str))
@patch('menpo.io.input.base.Path.is_file')
def test_importing_v1_ljson_null_values(is_file, mock_open, mock_dict):
    v1_ljson = { "groups": [
        { "connectivity": [ [ 0, 1 ], [ 1, 2 ], [ 2, 3 ] ],
          "label": "chin", "landmarks": [
            { "point": [ 987.9, 1294.1 ] }, { "point": [ 96.78, 1246.8 ] },
            { "point": [ None, 0.1 ] }, { "point": [303.22, 167.2 ] } ] },
        { "connectivity": [ [ 0, 1 ] ],
          "label": "leye", "landmarks": [
            { "point": [ None, None ] },
            { "point": [ None, None ] }] }
        ], "version": 1 }
    mock_dict.return_value = v1_ljson
    is_file.return_value = True

    with warnings.catch_warnings(record=True) as w:
        lmark = mio.import_landmark_file('fake_lmark_being_mocked.ljson')
    nan_points = np.isnan(lmark.lms.points)

    # Should raise deprecation warning
    assert len(w) == 1
    assert nan_points[2, 0]  # y-coord None point is nan
    assert not nan_points[2, 1]  # x-coord point is not nan
    assert np.all(nan_points[4:, :]) # all of leye label is nan


@patch('menpo.io.input.landmark.json.load')
@patch('{}.open'.format(builtins_str))
@patch('menpo.io.input.base.Path.is_file')
def test_importing_v2_ljson_null_values(is_file, mock_open, mock_dict):
    v2_ljson = { "labels": [
                    { "label": "left_eye", "mask": [0, 1, 2] },
                    { "label": "right_eye", "mask": [3, 4, 5] }
                 ],
                 "landmarks": {
                     "connectivity": [ [0, 1], [1, 2], [2, 0], [3, 4],
                                       [4, 5],  [5, 3] ],
                     "points": [ [None, 200.5], [None, None],
                                 [316.8, 199.15], [339.48, 205.0],
                                 [358.54, 217.82], [375.0, 233.4]]
                 },
                 "version": 2 }

    mock_dict.return_value = v2_ljson
    is_file.return_value = True

    lmark = mio.import_landmark_file('fake_lmark_being_mocked.ljson')
    nan_points = np.isnan(lmark.lms.points)
    assert nan_points[0, 0]  # y-coord None point is nan
    assert not nan_points[0, 1]  # x-coord point is not nan
    assert np.all(nan_points[1, :]) # all of leye label is nan


@patch('random.shuffle')
def test_shuffle_kwarg_true_calls_shuffle(mock):
    list(mio.import_images(mio.data_dir_path(), shuffle=True))
    assert mock.called
