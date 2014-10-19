from mock import patch
import numpy as np
from numpy.testing import assert_allclose
from nose.tools import raises
from StringIO import StringIO
import platform

import menpo.io as mio
from menpo.transform import PiecewiseAffine, ThinPlateSplines
from menpo.fitmultilevel.atm import ATMBuilder, PatchBasedATMBuilder
from menpo.feature import sparse_hog, igo, lbp, no_op

# load images
filenames = ['breakingbad.jpg', 'takeo.ppm', 'lenna.png', 'einstein.jpg']
training = []
templates = []
for i in range(4):
    im = mio.import_builtin_asset(filenames[i])
    if im.n_channels == 3:
        im = im.as_greyscale(mode='luminosity')
    training.append(im.landmarks['PTS']['all'])
    templates.append(im)

# build atms
atm1 = ATMBuilder(features=[igo, sparse_hog, no_op],
                  transform=PiecewiseAffine,
                  normalization_diagonal=150,
                  n_levels=3,
                  downscale=2,
                  scaled_shape_models=False,
                  max_shape_components=[1, 2, 3],
                  boundary=3).build(training, templates[0], group='PTS')

atm2 = ATMBuilder(features=[no_op, no_op],
                  transform=ThinPlateSplines,
                  trilist=None,
                  normalization_diagonal=None,
                  n_levels=2,
                  downscale=1.2,
                  scaled_shape_models=True,
                  max_shape_components=None,
                  boundary=0).build(training, templates[1], group='PTS')

atm3 = ATMBuilder(features=igo,
                  transform=ThinPlateSplines,
                  trilist=None,
                  normalization_diagonal=None,
                  n_levels=1,
                  downscale=3,
                  scaled_shape_models=True,
                  max_shape_components=[2],
                  boundary=2).build(training, templates[2], group='PTS')

atm4 = PatchBasedATMBuilder(features=lbp,
                            patch_shape=(10, 13),
                            normalization_diagonal=200,
                            n_levels=2,
                            downscale=1.2,
                            scaled_shape_models=True,
                            max_shape_components=1,
                            boundary=2).build(training, templates[3],
                                              group='PTS')


@raises(ValueError)
def test_features_exception():
    ATMBuilder(features=[igo, sparse_hog]).build(training, templates[0])


@raises(ValueError)
def test_n_levels_exception():
    ATMBuilder(n_levels=0).build(training, templates[1])


@raises(ValueError)
def test_downscale_exception():
    atm = ATMBuilder(downscale=1).build(training, templates[2], group='PTS')
    assert (atm.downscale == 1)
    ATMBuilder(downscale=0).build(training, templates[2], group='PTS')


@raises(ValueError)
def test_normalization_diagonal_exception():
    atm = ATMBuilder(normalization_diagonal=100).build(training, templates[3],
                                                       group='PTS')
    assert (atm.warped_templates[0].n_true_pixels() == 1246)
    ATMBuilder(normalization_diagonal=10).build(training, templates[3])


@raises(ValueError)
def test_max_shape_components_exception():
    ATMBuilder(max_shape_components=[1, 0.2, 'a']).build(training, templates[0],
                                                         group='PTS')


@raises(ValueError)
def test_max_shape_components_exception_2():
    ATMBuilder(max_shape_components=[1, 2]).build(training, templates[0])


@raises(ValueError)
def test_boundary_exception():
    ATMBuilder(boundary=-1).build(training, templates[1], group='PTS')


@patch('sys.stdout', new_callable=StringIO)
def test_verbose_mock(mock_stdout):
    ATMBuilder().build(training, templates[2], group='PTS', verbose=True)


@patch('sys.stdout', new_callable=StringIO)
def test_str_mock(mock_stdout):
    print(atm1)
    print(atm2)
    print(atm3)
    print(atm4)


def test_atm_1():
    assert(atm1.n_training_shapes == 4)
    assert(atm1.n_levels == 3)
    assert(atm1.downscale == 2)
    assert_allclose(np.around(atm1.reference_shape.range()), (109., 103.))
    assert(not atm1.scaled_shape_models)
    assert(not atm1.pyramid_on_features)
    assert_allclose([atm1.shape_models[j].n_components
                     for j in range(atm1.n_levels)], (1, 2, 3))
    assert_allclose([atm1.warped_templates[j].n_channels
                     for j in range(atm1.n_levels)], (2, 36, 1))
    assert_allclose([atm1.warped_templates[j].shape[1]
                     for j in range(atm1.n_levels)], (164, 164, 164))


def test_atm_2():
    assert (atm2.n_training_shapes == 4)
    assert (atm2.n_levels == 2)
    assert (atm2.downscale == 1.2)
    assert_allclose(np.around(atm2.reference_shape.range()), (169., 161.))
    assert atm2.scaled_shape_models
    assert (not atm2.pyramid_on_features)
    assert (np.all([atm2.shape_models[j].n_components == 3
                    for j in range(atm2.n_levels)]))
    assert (np.all([atm2.warped_templates[j].n_channels == 1
                    for j in range(atm2.n_levels)]))
    assert_allclose([atm2.warped_templates[j].shape[1]
                     for j in range(atm2.n_levels)], (132, 158))


def test_atm_3():
    assert (atm3.n_training_shapes == 4)
    assert (atm3.n_levels == 1)
    assert (atm3.downscale == 3)
    assert_allclose(np.around(atm3.reference_shape.range()), (169., 161.))
    assert atm3.scaled_shape_models
    assert atm3.pyramid_on_features
    assert (np.all([atm3.shape_models[j].n_components == 2
                    for j in range(atm3.n_levels)]))
    assert (np.all([atm3.warped_templates[j].n_channels == 2
                    for j in range(atm3.n_levels)]))
    assert_allclose([atm3.warped_templates[j].shape[1]
                     for j in range(atm3.n_levels)], 162)


def test_atm_4():
    assert (atm4.n_training_shapes == 4)
    assert (atm4.n_levels == 2)
    assert (atm4.downscale == 1.2)
    assert_allclose(np.around(atm4.reference_shape.range()), (145., 138.))
    assert atm4.scaled_shape_models
    assert atm4.pyramid_on_features
    assert (np.all([atm4.shape_models[j].n_components == 1
                    for j in range(atm4.n_levels)]))
    assert (np.all([atm4.warped_templates[j].n_channels == 4
                    for j in range(atm4.n_levels)]))
    assert_allclose([atm4.warped_templates[j].shape[1]
                     for j in range(atm4.n_levels)], (162, 188))
