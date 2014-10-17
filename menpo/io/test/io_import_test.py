import os
from nose.tools import raises
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
    assert(img.landmarks['PTS'].n_landmarks == 68)


def test_path():
    # choose a random asset (all should have it!)
    img = mio.import_builtin_asset('einstein.jpg')
    path = mio.data_path_to('einstein.jpg')
    assert(str(img.path) == path)
    assert(img.path.stem == 'einstein')
    assert(img.path.suffix == '.jpg')
    assert(str(img.path.parent) == mio.data_dir_path())
    assert(img.path.name == 'einstein.jpg')


def test_import_image():
    img_path = os.path.join(mio.data_dir_path(), 'einstein.jpg')
    mio.import_image(img_path)


def test_import_landmark_file():
    lm_path = os.path.join(mio.data_dir_path(), 'einstein.pts')
    mio.import_landmark_file(lm_path)


def test_import_images():
    imgs_glob = os.path.join(mio.data_dir_path(), '*')
    imgs = list(mio.import_images(imgs_glob))
    imgs_filenames = set(i.path.stem for i in imgs)
    exp_imgs_filenames = {'einstein', 'takeo', 'breakingbad', 'lenna'}
    assert(len(exp_imgs_filenames - imgs_filenames) == 0)


def test_ls_builtin_assets():
    assert(set(mio.ls_builtin_assets()) == {'breakingbad.jpg',
                                            'einstein.jpg', 'einstein.pts',
                                            'lenna.png', 'breakingbad.pts',
                                            'lenna.pts', 'takeo.ppm',
                                            'takeo.pts', 'tongue.jpg',
                                            'tongue.pts'})


def test_image_paths():
    ls = mio.image_paths(os.path.join(mio.data_dir_path(), '*'))
    assert(len(list(ls)) == 5)


@raises(ValueError)
def test_import_images_wrong_path_raises_value_error():
    list(mio.import_images('asldfjalkgjlaknglkajlekjaltknlaekstjlakj'))


@raises(ValueError)
def test_import_landmark_files_wrong_path_raises_value_error():
    list(mio.import_landmark_files('asldfjalkgjlaknglkajlekjaltknlaekstjlakj'))
