import os
from nose.tools import raises
import numpy as np

import menpo.io as mio
from menpo.shape import TriMesh
from menpo.image import Image
from numpy.testing import assert_allclose

# ground truth bunny landmarks
bunny_l_eye = np.array([[-0.09009447,  0.13760692,  0.02303147]])
bunny_r_eye = np.array([[-0.06041192,  0.13871727,  0.03462752]])
bunny_nose = np.array([[-0.07949748,  0.12401747,  0.05170817]])
bunny_mouth = np.array([[-0.08876467,  0.11684129,  0.04538456],
                        [-0.08044379,  0.11318078,  0.04642309],
                        [-0.07329408,  0.11268603,  0.04844998],
                        [-0.0670796 ,  0.11644761,  0.0498054 ]])


def test_import_asset_bunny():
    mesh = mio.import_builtin_asset('bunny.obj')
    assert(isinstance(mesh, TriMesh))

@raises(ValueError)
def test_import_incorrect_built_in():
    mio.import_builtin_asset('adskljasdlkajd.obj')


def test_json_landmarks_bunny():
    mesh = mio.import_builtin_asset('bunny.obj')
    assert('JSON' in mesh.landmarks.group_labels)
    lms = mesh.landmarks['JSON']
    labels = {'r_eye', 'mouth', 'nose', 'l_eye'}
    assert(len(labels - set(lms.labels)) == 0)
    assert_allclose(lms['l_eye'].lms.points, bunny_l_eye, atol=1e-7)
    assert_allclose(lms['r_eye'].lms.points, bunny_r_eye, atol=1e-7)
    assert_allclose(lms['nose'].lms.points, bunny_nose, atol=1e-7)
    assert_allclose(lms['mouth'].lms.points, bunny_mouth, atol=1e-7)


def test_custom_landmark_logic_bunny():
    def f(mesh):
        return {
            'no_nose': os.path.join(mesh.ioinfo.dir, 'bunny_no_nose.json'),
            'full_set': os.path.join(mesh.ioinfo.dir, 'bunny.json')
        }
    mesh = mio.import_mesh(mio.data_path_to('bunny.obj'), landmark_resolver=f)
    assert('no_nose' in mesh.landmarks.group_labels)
    lms = mesh.landmarks['no_nose']
    labels = {'r_eye', 'mouth', 'l_eye'}
    assert(len(set(lms.labels) - labels) == 0)
    assert_allclose(lms['l_eye'].lms.points, bunny_l_eye, atol=1e-7)
    assert_allclose(lms['r_eye'].lms.points, bunny_r_eye, atol=1e-7)
    assert_allclose(lms['mouth'].lms.points, bunny_mouth, atol=1e-7)

    assert('full_set' in mesh.landmarks.group_labels)
    lms = mesh.landmarks['full_set']
    labels = {'r_eye', 'mouth', 'nose', 'l_eye'}
    assert(len(set(lms.labels) - labels) == 0)
    assert_allclose(lms['l_eye'].lms.points, bunny_l_eye, atol=1e-7)
    assert_allclose(lms['r_eye'].lms.points, bunny_r_eye, atol=1e-7)
    assert_allclose(lms['nose'].lms.points, bunny_nose, atol=1e-7)
    assert_allclose(lms['mouth'].lms.points, bunny_mouth, atol=1e-7)


def test_custom_landmark_logic_None_bunny():
    def f(mesh):
        return None
    mesh = mio.import_mesh(mio.data_path_to('bunny.obj'), landmark_resolver=f)
    assert(mesh.landmarks.n_groups == 0)


def test_json_landmarks_bunny_direct():
    lms = mio.import_landmark_file(mio.data_path_to('bunny.json'))
    assert(lms.group_label == 'JSON')
    labels = {'r_eye', 'mouth', 'nose', 'l_eye'}
    assert(len(labels - set(lms.labels)) == 0)
    assert_allclose(lms['l_eye'].lms.points, bunny_l_eye, atol=1e-7)
    assert_allclose(lms['r_eye'].lms.points, bunny_r_eye, atol=1e-7)
    assert_allclose(lms['nose'].lms.points, bunny_nose, atol=1e-7)
    assert_allclose(lms['mouth'].lms.points, bunny_mouth, atol=1e-7)


def test_breaking_bad_import():
    img = mio.import_builtin_asset('breakingbad.jpg')
    assert(img.shape == (1080, 1920))
    assert(img.n_channels == 3)
    assert(img.landmarks['PTS'].n_landmarks == 68)


def test_takeo_import():
    img = mio.import_builtin_asset('takeo.ppm')
    assert(img.shape == (225, 150))
    assert(img.n_channels == 3)
    assert(img.landmarks.n_groups == 0)


def test_einstein_import():
    img = mio.import_builtin_asset('einstein.jpg')
    assert(img.shape == (1024, 817))
    assert(img.n_channels == 1)
    assert(img.landmarks['PTS'].n_landmarks == 68)


def test_ioinfo():
    # choose a random asset (all should have it!)
    img = mio.import_builtin_asset('einstein.jpg')
    path = mio.data_path_to('einstein.jpg')
    assert(img.ioinfo.filepath == path)
    assert(img.ioinfo.filename == 'einstein')
    assert(img.ioinfo.extension == '.jpg')
    assert(img.ioinfo.dir == mio.data_dir_path())

def test_import_image():
    img_path = os.path.join(mio.data_dir_path(), 'einstein.jpg')
    mio.import_images(img_path)

def test_import_mesh():
    obj_path = os.path.join(mio.data_dir_path(), 'bunny.obj')
    mio.import_images(obj_path)

def test_import_images():
    imgs_glob = os.path.join(mio.data_dir_path(), '*')
    imgs = list(mio.import_images(imgs_glob))
    imgs_filenames = set(i.ioinfo.filename for i in imgs)
    exp_imgs_filenames = {'einstein', 'takeo', 'breakingbad', 'lenna'}
    assert(len(exp_imgs_filenames - imgs_filenames) == 0)


def test_import_auto():
    assets_glob = os.path.join(mio.data_dir_path(), '*')
    assets = list(mio.import_auto(assets_glob))
    assert(len(assets) == 6)


def test_import_auto_max_images():
    assets_glob = os.path.join(mio.data_dir_path(), '*')
    assets = list(mio.import_auto(assets_glob,  max_images=2))
    assert(sum([isinstance(x, TriMesh) for x in assets]) == 2)
    assert(sum([isinstance(x, Image) for x in assets]) == 2)

def test_import_auto_max_meshes():
    assets_glob = os.path.join(mio.data_dir_path(), '*')
    assets = list(mio.import_auto(assets_glob, max_meshes=1))
    assert(sum([isinstance(x, TriMesh) for x in assets]) == 1)

def test_ls_builtin_assets():
    assert(set(mio.ls_builtin_assets()) == {'breakingbad.jpg',
                                            'breakingbad.pts', 'bunny.json',
                                            'bunny.obj', 'bunny_no_nose.json',
                                            'einstein.jpg', 'einstein.pts',
                                            'james.jpg', 'james.mtl',
                                            'james.obj', 'lenna.png',
                                            'takeo.ppm'})

def test_mesh_paths():
    ls = mio.mesh_paths(os.path.join(mio.data_dir_path(), '*'))
    assert(len(ls) == 2)

def test_image_paths():
    ls = mio.image_paths(os.path.join(mio.data_dir_path(), '*'))
    assert(len(ls) == 5)

