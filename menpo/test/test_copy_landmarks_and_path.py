import menpo.io as mio
from menpo.base import copy_landmarks_and_path
from menpo.image import Image


def test_copy_landmarks_and_path():
    img = mio.import_builtin_asset.lenna_png()
    new_img = Image.init_blank(img.shape)
    copy_landmarks_and_path(img, new_img)

    assert new_img.path == img.path
    assert new_img.landmarks.keys() == img.landmarks.keys()
    assert new_img.landmarks is not img.landmarks


def test_copy_landmarks_and_path_returns_target():
    img = mio.import_builtin_asset.lenna_png()
    new_img = Image.init_blank(img.shape)
    new_img_ret = copy_landmarks_and_path(img, new_img)
    assert new_img_ret is new_img


def test_copy_landmarks_and_path_with_no_lms_path():
    img = Image.init_blank((5, 5))
    new_img = Image.init_blank((5, 5))
    copy_landmarks_and_path(img, new_img)
    assert not hasattr(img, 'path')
    assert not hasattr(new_img, 'path')
    assert not img.has_landmarks
    assert not new_img.has_landmarks
