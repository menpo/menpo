import numpy as np
from mock import patch
from nose.tools import raises
from PIL import Image as PILImage
import menpo.io as mio


@raises(ValueError)
def test_import_incorrect_built_in():
    mio.import_builtin_asset('adskljasdlkajd.obj')


def test_breaking_bad_import():
    img = mio.import_builtin_asset('breakingbad.jpg')
    assert(img.shape == (1080, 1920))
    assert(img.n_channels == 3)
    assert(img.landmarks['PTS'].n_landmarks == 68)


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


def test_path():
    # choose a random asset (all should have it!)
    img = mio.import_builtin_asset('einstein.jpg')
    path = mio.data_path_to('einstein.jpg')
    assert(img.path == path)
    assert(img.path.stem == 'einstein')
    assert(img.path.suffix == '.jpg')
    assert(img.path.parent == mio.data_dir_path())
    assert(img.path.name == 'einstein.jpg')


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
    exp_imgs_filenames = {'einstein', 'takeo', 'breakingbad', 'lenna',
                          'menpo_thumbnail'}
    assert(len(exp_imgs_filenames - imgs_filenames) == 0)


def test_ls_builtin_assets():
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
