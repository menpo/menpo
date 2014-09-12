from mock import patch
import numpy as np
from numpy.testing import assert_allclose
from nose.tools import raises
from StringIO import StringIO

import menpo.io as mio
from menpo.landmark import labeller, ibug_face_68_trimesh
from menpo.transform import PiecewiseAffine, ThinPlateSplines
from menpo.fitmultilevel.aam import AAMBuilder, PatchBasedAAMBuilder
from menpo.feature import sparse_hog, igo, lbp, no_op

# load images
filenames = ['breakingbad.jpg', 'takeo.ppm', 'lenna.png', 'einstein.jpg']
training = []
for i in range(4):
    im = mio.import_builtin_asset(filenames[i])
    im.crop_to_landmarks_proportion_inplace(0.1)
    labeller(im, 'PTS', ibug_face_68_trimesh)
    if im.n_channels == 3:
        im = im.as_greyscale(mode='luminosity')
    training.append(im)

# build aams
aam1 = AAMBuilder(features=[igo, sparse_hog, no_op],
                  transform=PiecewiseAffine,
                  trilist=training[0].landmarks['ibug_face_68_trimesh'].
                  lms.trilist,
                  normalization_diagonal=150,
                  n_levels=3,
                  downscale=2,
                  scaled_shape_models=False,
                  pyramid_on_features=False,
                  max_shape_components=[1, 2, 3],
                  max_appearance_components=[3, 3, 3],
                  boundary=3).build(training, group='PTS')

aam2 = AAMBuilder(features=no_op,
                  transform=ThinPlateSplines,
                  trilist=None,
                  normalization_diagonal=None,
                  n_levels=2,
                  downscale=1.2,
                  scaled_shape_models=True,
                  pyramid_on_features=False,
                  max_shape_components=None,
                  max_appearance_components=1,
                  boundary=0).build(training, group='PTS')

aam3 = AAMBuilder(features=igo,
                  transform=ThinPlateSplines,
                  trilist=None,
                  normalization_diagonal=None,
                  n_levels=1,
                  downscale=3,
                  scaled_shape_models=True,
                  pyramid_on_features=True,
                  max_shape_components=[2],
                  max_appearance_components=10,
                  boundary=2).build(training, group='PTS')

aam4 = PatchBasedAAMBuilder(features=lbp,
                            patch_shape=(10, 13),
                            normalization_diagonal=200,
                            n_levels=2,
                            downscale=1.2,
                            scaled_shape_models=True,
                            pyramid_on_features=True,
                            max_shape_components=1,
                            max_appearance_components=None,
                            boundary=2).build(training, group='PTS')


@raises(ValueError)
def test_features_exception():
    AAMBuilder(features=[igo, sparse_hog],
               pyramid_on_features=False).build(training, group='PTS')


@raises(ValueError)
def test_features_with_pyramid_on_features_exception():
    AAMBuilder(features=[igo, sparse_hog]).build(training,
                                                     group='PTS')


@raises(ValueError)
def test_n_levels_exception():
    AAMBuilder(n_levels=0).build(training, group='PTS')


@raises(ValueError)
def test_downscale_exception():
    aam = AAMBuilder(downscale=1).build(training,
                                        group='PTS')
    assert (aam.downscale == 1)
    AAMBuilder(downscale=0).build(training, group='PTS')


@raises(ValueError)
def test_normalization_diagonal_exception():
    aam = AAMBuilder(normalization_diagonal=100).build(training,
                                                       group='PTS')
    assert (aam.appearance_models[0].n_features == 382)
    AAMBuilder(normalization_diagonal=10).build(training, group='PTS')


@raises(ValueError)
def test_max_shape_components_exception():
    AAMBuilder(max_shape_components=[1, 0.2, 'a']).build(training,
                                                         group='PTS')


@raises(ValueError)
def test_max_appearance_components_exception():
    AAMBuilder(max_appearance_components=[1, 2]).build(training,
                                                       group='PTS')


@raises(ValueError)
def test_boundary_exception():
    AAMBuilder(boundary=-1).build(training, group='PTS')


@patch('sys.stdout', new_callable=StringIO)
def test_verbose_mock(mock_stdout):
    AAMBuilder().build(training, group='PTS', verbose=True)


@patch('sys.stdout', new_callable=StringIO)
def test_str_mock(mock_stdout):
    print(aam1)
    print(aam2)
    print(aam3)
    print(aam4)


def test_aam_1():
    assert(aam1.n_training_images == 4)
    assert(aam1.n_levels == 3)
    assert(aam1.downscale == 2)
    #assert(aam1.features[0] == igo and aam1.features[2] == no_op)
    assert_allclose(np.around(aam1.reference_shape.range()), (109., 103.))
    assert(not aam1.scaled_shape_models)
    assert(not aam1.pyramid_on_features)
    assert_allclose([aam1.shape_models[j].n_components
                     for j in range(aam1.n_levels)], (1, 2, 3))
    assert (np.all([aam1.appearance_models[j].n_components == 3
                    for j in range(aam1.n_levels)]))
    assert_allclose([aam1.appearance_models[j].template_instance.n_channels
                     for j in range(aam1.n_levels)], (2, 36, 1))
    assert_allclose([aam1.appearance_models[j].components.shape[1]
                     for j in range(aam1.n_levels)], (14892, 268056, 7446))


def test_aam_2():
    assert (aam2.n_training_images == 4)
    assert (aam2.n_levels == 2)
    assert (aam2.downscale == 1.2)
    #assert (aam2.features[0] == no_op and aam2.features[1] == no_op)
    assert_allclose(np.around(aam2.reference_shape.range()), (169., 161.))
    assert aam2.scaled_shape_models
    assert (not aam2.pyramid_on_features)
    assert (np.all([aam2.shape_models[j].n_components == 3
                    for j in range(aam2.n_levels)]))
    assert (np.all([aam2.appearance_models[j].n_components == 1
                    for j in range(aam2.n_levels)]))
    assert (np.all([aam2.appearance_models[j].template_instance.n_channels == 1
                    for j in range(aam2.n_levels)]))
    assert_allclose([aam2.appearance_models[j].components.shape[1]
                     for j in range(aam2.n_levels)], (12827, 18518))


def test_aam_3():
    assert (aam3.n_training_images == 4)
    assert (aam3.n_levels == 1)
    assert (aam3.downscale == 3)
    #assert (aam3.features[0] == igo and len(aam3.features) == 1)
    assert_allclose(np.around(aam3.reference_shape.range()), (169., 161.))
    assert aam3.scaled_shape_models
    assert aam3.pyramid_on_features
    assert (np.all([aam3.shape_models[j].n_components == 2
                    for j in range(aam3.n_levels)]))
    assert (np.all([aam3.appearance_models[j].n_components == 3
                    for j in range(aam3.n_levels)]))
    assert (np.all([aam3.appearance_models[j].template_instance.n_channels == 2
                    for j in range(aam3.n_levels)]))
    assert_allclose([aam3.appearance_models[j].components.shape[1]
                     for j in range(aam3.n_levels)], 37036)


def test_aam_4():
    assert (aam4.n_training_images == 4)
    assert (aam4.n_levels == 2)
    assert (aam4.downscale == 1.2)
    #assert (aam4.features[0] == lbp)
    assert_allclose(np.around(aam4.reference_shape.range()), (145., 138.))
    assert aam4.scaled_shape_models
    assert aam4.pyramid_on_features
    assert (np.all([aam4.shape_models[j].n_components == 1
                    for j in range(aam4.n_levels)]))
    assert (np.all([aam4.appearance_models[j].n_components == 3
                    for j in range(aam4.n_levels)]))
    assert (np.all([aam4.appearance_models[j].template_instance.n_channels == 4
                    for j in range(aam4.n_levels)]))
    assert_allclose([aam4.appearance_models[j].components.shape[1]
                     for j in range(aam4.n_levels)], (23656, 25988))
