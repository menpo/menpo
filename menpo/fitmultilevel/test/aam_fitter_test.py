from StringIO import StringIO
from mock import patch
from nose.plugins.attrib import attr
import numpy as np
from numpy.testing import assert_allclose
from nose.tools import raises
import platform

import menpo.io as mio
from menpo.feature import igo
from menpo.shape.pointcloud import PointCloud
from menpo.landmark import labeller, ibug_face_68_trimesh
from menpo.transform import PiecewiseAffine
from menpo.fitmultilevel.aam import AAMBuilder, LucasKanadeAAMFitter
from menpo.fit.lucaskanade.appearance import (
    AlternatingForwardAdditive, AlternatingForwardCompositional,
    AlternatingInverseCompositional, AdaptiveForwardAdditive,
    AdaptiveForwardCompositional, AdaptiveInverseCompositional,
    SimultaneousForwardAdditive, SimultaneousForwardCompositional,
    SimultaneousInverseCompositional, ProjectOutForwardAdditive,
    ProjectOutForwardCompositional, ProjectOutInverseCompositional,
    ProbabilisticForwardAdditive, ProbabilisticForwardCompositional,
    ProbabilisticInverseCompositional)


initial_shape = []
initial_shape.append(PointCloud(np.array([[150.9737801, 1.85331141],
                                          [191.20452708, 1.86714624],
                                          [237.5088486, 7.16836457],
                                          [280.68439528, 19.1356864],
                                          [319.00988383, 36.18921029],
                                          [351.31395982, 61.11002727],
                                          [375.83681819, 86.68264647],
                                          [401.50706656, 117.12858347],
                                          [408.46977018, 156.72258055],
                                          [398.49810436, 197.95690492],
                                          [375.44584527, 234.437902],
                                          [342.35427495, 267.96920594],
                                          [299.04149064, 309.66693535],
                                          [250.84207113, 331.07734674],
                                          [198.46150259, 339.47188196],
                                          [144.62222804, 337.84178783],
                                          [89.92321435, 327.81734317],
                                          [101.22474793, 26.90269773],
                                          [89.23456877, 44.52571118],
                                          [84.04683242, 66.6369272],
                                          [86.36993557, 88.61559027],
                                          [94.88123162, 108.04971327],
                                          [88.08448274, 152.88439191],
                                          [68.71150917, 176.94681489],
                                          [55.7165906, 204.86028035],
                                          [53.9169657, 232.87050281],
                                          [69.08534014, 259.8486207],
                                          [121.82883888, 130.79001073],
                                          [152.30894887, 128.91266055],
                                          [183.36381228, 128.04534764],
                                          [216.59234031, 125.86784329],
                                          [235.18182671, 93.18819461],
                                          [242.46006172, 117.24575711],
                                          [246.52987701, 142.46262589],
                                          [240.51603561, 160.38006297],
                                          [232.61083444, 175.36132625],
                                          [137.35714406, 56.53012228],
                                          [124.42060774, 67.0342585],
                                          [121.98869265, 87.71006061],
                                          [130.4421354, 105.16741493],
                                          [139.32511836, 89.65144616],
                                          [144.17935107, 69.97931719],
                                          [125.04221953, 174.72789706],
                                          [103.0127825, 188.96555839],
                                          [97.38196408, 210.70911033],
                                          [107.31622619, 232.4487582],
                                          [119.12835959, 215.57040617],
                                          [124.80355957, 193.64317941],
                                          [304.3174261, 101.83559243],
                                          [293.08249678, 116.76961123],
                                          [287.11523488, 132.55435452],
                                          [289.39839945, 148.49971074],
                                          [283.59574087, 162.33458018],
                                          [286.76478391, 187.30470094],
                                          [292.65033117, 211.98694428],
                                          [310.75841097, 187.33036207],
                                          [319.06250309, 165.27131484],
                                          [321.3339324, 148.86793045],
                                          [321.82844973, 133.03866904],
                                          [316.60228316, 115.15885333],
                                          [303.45716953, 109.59946563],
                                          [301.58563675, 135.32572565],
                                          [298.16531481, 148.240518],
                                          [295.39615418, 162.35992687],
                                          [293.63384823, 201.35617245],
                                          [301.95207707, 163.05299135],
                                          [305.27555828, 148.48478086],
                                          [306.41382116, 133.02994058]])))

initial_shape.append(PointCloud(np.array([[33.08569962, 26.2373455],
                                          [43.88613611, 26.24105964],
                                          [56.31709803, 27.66423659],
                                          [67.90810205, 30.87701063],
                                          [78.19704859, 35.45523787],
                                          [86.86947323, 42.14553624],
                                          [93.45293474, 49.0108189],
                                          [100.34442715, 57.18440338],
                                          [102.21365016, 67.81389656],
                                          [99.53663441, 78.88375569],
                                          [93.34797327, 88.67752592],
                                          [84.46413615, 97.67941492],
                                          [72.83628901, 108.8736808],
                                          [59.89656483, 114.62156782],
                                          [45.83436002, 116.87518356],
                                          [31.38054772, 116.43756484],
                                          [16.69592792, 113.74637996],
                                          [19.72996295, 32.96215989],
                                          [16.51105259, 37.69327358],
                                          [15.11834126, 43.62930018],
                                          [15.74200674, 49.52974132],
                                          [18.02696835, 54.74706954],
                                          [16.20229791, 66.78348784],
                                          [11.00138601, 73.24333984],
                                          [7.51274105, 80.73705133],
                                          [7.02960972, 88.25673842],
                                          [11.10174551, 95.4993444],
                                          [25.26138338, 60.85198075],
                                          [33.44414202, 60.34798312],
                                          [41.78120024, 60.11514235],
                                          [50.70180534, 59.53056465],
                                          [55.69238052, 50.75731293],
                                          [57.6463118, 57.21586007],
                                          [58.73890353, 63.98563718],
                                          [57.12441419, 68.79579249],
                                          [55.00216617, 72.817696],
                                          [29.43014699, 40.91600468],
                                          [25.95717546, 43.73596863],
                                          [25.30429808, 49.2866408],
                                          [27.57372827, 53.97328126],
                                          [29.95847378, 49.80782952],
                                          [31.26165197, 44.52660569],
                                          [26.12405475, 72.64764418],
                                          [20.20998272, 76.46991865],
                                          [18.69832059, 82.30724133],
                                          [21.36529486, 88.14351591],
                                          [24.53640666, 83.6123157],
                                          [26.05998356, 77.72568327],
                                          [74.25267847, 53.07881273],
                                          [71.23652416, 57.08803288],
                                          [69.63453966, 61.32564044],
                                          [70.24748314, 65.6063665],
                                          [68.68968841, 69.32050656],
                                          [69.54045681, 76.02404113],
                                          [71.12050401, 82.6502915],
                                          [75.9818397, 76.03093018],
                                          [78.21117488, 70.10890893],
                                          [78.82096788, 65.70521959],
                                          [78.95372711, 61.4556606],
                                          [77.55069872, 56.65560521],
                                          [74.02173206, 55.16311953],
                                          [73.51929617, 62.06964895],
                                          [72.60106888, 65.53678304],
                                          [71.85765381, 69.32731119],
                                          [71.38454121, 79.79633067],
                                          [73.61767156, 69.51337283],
                                          [74.50990078, 65.60235839],
                                          [74.81548138, 61.45331734]])))

initial_shape.append(PointCloud(np.array([[46.63369884, 44.08764686],
                                          [65.31491309, 44.09407109],
                                          [86.81640178, 46.55570064],
                                          [106.86503868, 52.11274643],
                                          [124.66154301, 60.0315786],
                                          [139.66199441, 71.6036014],
                                          [151.04922447, 83.47828965],
                                          [162.96924699, 97.61591112],
                                          [166.20238999, 116.0014495],
                                          [161.57203038, 135.14867658],
                                          [150.86767554, 152.08868824],
                                          [135.50154984, 167.65900498],
                                          [115.38918643, 187.02141497],
                                          [93.00770583, 196.9633751],
                                          [68.68470174, 200.86139148],
                                          [43.68434508, 200.10445456],
                                          [18.28476712, 195.44958702],
                                          [23.53265303, 55.71937105],
                                          [17.9649934, 63.90264665],
                                          [15.55605939, 74.17002657],
                                          [16.63479621, 84.37585532],
                                          [20.58703068, 93.40012265],
                                          [17.43094904, 114.21918023],
                                          [8.43507654, 125.39260635],
                                          [2.4008645, 138.35427044],
                                          [1.56520568, 151.36086382],
                                          [8.60866558, 163.88819772],
                                          [33.10019692, 103.95961759],
                                          [47.25368667, 103.08786691],
                                          [61.67406413, 102.68512872],
                                          [77.10378638, 101.67400095],
                                          [85.7358453, 86.49915174],
                                          [89.11550583, 97.67032089],
                                          [91.00533132, 109.37981584],
                                          [88.21279407, 117.69980754],
                                          [84.54200076, 124.65638206],
                                          [40.31079125, 69.47691491],
                                          [34.3036891, 74.35452803],
                                          [33.17442528, 83.95537112],
                                          [37.09979548, 92.06172262],
                                          [41.22462339, 84.85685672],
                                          [43.47869442, 75.72207092],
                                          [34.59233557, 124.36224816],
                                          [24.36292985, 130.97352987],
                                          [21.74824996, 141.07018437],
                                          [26.36124109, 151.16502601],
                                          [31.84622487, 143.32753518],
                                          [34.48151342, 133.14559097],
                                          [117.83907583, 90.5145853],
                                          [112.62211772, 97.44922176],
                                          [109.85120974, 104.77889356],
                                          [110.911401, 112.18314623],
                                          [108.21692684, 118.60739086],
                                          [109.68847724, 130.20230795],
                                          [112.4214409, 141.66354869],
                                          [120.82995787, 130.21422374],
                                          [124.68597685, 119.97106848],
                                          [125.74071883, 112.35412967],
                                          [125.97034877, 105.00378581],
                                          [123.54356964, 96.70126365],
                                          [117.43961426, 94.11975273],
                                          [116.5705649, 106.06578435],
                                          [114.98233273, 112.06278965],
                                          [113.69646838, 118.61916064],
                                          [112.87813868, 136.72713211],
                                          [116.74072208, 118.94098628],
                                          [118.2839861, 112.17621352],
                                          [118.81254036, 104.99973274]])))

initial_shape.append(PointCloud(np.array([[29.30459178, 27.24534074],
                                          [39.47004743, 24.38292299],
                                          [51.54667438, 22.42372272],
                                          [63.30767547, 22.37162616],
                                          [74.20561385, 23.95008332],
                                          [84.14265809, 27.94519239],
                                          [92.16017681, 32.65929179],
                                          [100.81474852, 38.52291926],
                                          [105.39445843, 48.03051044],
                                          [105.81247938, 59.1588891],
                                          [102.5870203, 70.01814005],
                                          [96.6149594, 80.84730771],
                                          [88.64221584, 94.46788512],
                                          [77.98963764, 103.31089364],
                                          [65.35346377, 109.16323748],
                                          [51.63461821, 112.58672956],
                                          [37.10056847, 113.95059826],
                                          [18.51972657, 37.11814141],
                                          [16.7457652 , 42.42481409],
                                          [17.01019564, 48.38086547],
                                          [19.16282912, 53.76837796],
                                          [22.69767086, 58.07217393],
                                          [24.17432616, 69.88402627],
                                          [20.99379373, 77.34357057],
                                          [19.69904043, 85.32174442],
                                          [21.23971857, 92.52684647],
                                          [26.99391031, 98.26243543],
                                          [31.12604697, 61.89794357],
                                          [38.69324039, 59.25231487],
                                          [46.47759964, 56.82093276],
                                          [54.71781058, 53.90368008],
                                          [57.08652729, 44.32277008],
                                          [60.63919033, 49.88253722],
                                          [63.46381778, 55.96376588],
                                          [63.2207775 , 60.91909025],
                                          [62.29071322, 65.26731234],
                                          [29.75929632, 42.02967737],
                                          [27.23910711, 45.60515084],
                                          [28.09755316, 51.00222264],
                                          [31.47695917, 54.81070084],
                                          [32.61597345, 50.25772899],
                                          [32.44103485, 44.94168113],
                                          [35.06791957, 72.77012704],
                                          [30.51633486, 77.93664152],
                                          [30.64262749, 83.83136479],
                                          [34.70122609, 88.61629379],
                                          [36.4832508 , 83.51044643],
                                          [36.35508694, 77.56615533],
                                          [75.16994555, 41.58256719],
                                          [73.39524567, 46.15605223],
                                          [73.01204743, 50.56922423],
                                          [74.72479626, 54.43524106],
                                          [74.24428281, 58.34404327],
                                          [76.82374875, 64.42709819],
                                          [80.0690436 , 70.24390436],
                                          [82.88766915, 62.72435028],
                                          [83.41431565, 56.55948008],
                                          [82.81967592, 52.25328539],
                                          [81.81699053, 48.21872699],
                                          [79.2228748 , 44.073611],
                                          [75.50567221, 43.60542492],
                                          [76.86548014, 50.2385966],
                                          [76.9213308 , 53.74522715],
                                          [77.22751327, 57.5098225],
                                          [79.56023029, 67.48793174],
                                          [78.93326695, 57.21790467],
                                          [78.73516471, 53.30042959],
                                          [77.92179698, 49.31461186]])))

# load images
filenames = ['breakingbad.jpg', 'takeo.ppm', 'lenna.png', 'einstein.jpg']
training_images = []
for i in range(4):
    im = mio.import_builtin_asset(filenames[i])
    im.crop_to_landmarks_proportion_inplace(0.1)
    labeller(im, 'PTS', ibug_face_68_trimesh)
    if im.n_channels == 3:
        im = im.as_greyscale(mode='luminosity')
    training_images.append(im)

# build aam
aam = AAMBuilder(features=igo,
                 transform=PiecewiseAffine,
                 trilist=training_images[0].landmarks['ibug_face_68_trimesh'].
                 lms.trilist,
                 normalization_diagonal=150,
                 n_levels=3,
                 downscale=2,
                 scaled_shape_models=True,
                 max_shape_components=[1, 2, 3],
                 max_appearance_components=[3, 2, 1],
                 boundary=3).build(training_images, group='PTS')

aam2 = AAMBuilder(features=igo,
                  transform=PiecewiseAffine,
                  trilist=training_images[0].landmarks['ibug_face_68_trimesh'].
                  lms.trilist,
                  normalization_diagonal=150,
                  n_levels=1,
                  downscale=2,
                  scaled_shape_models=True,
                  max_shape_components=[1],
                  max_appearance_components=[1],
                  boundary=3).build(training_images, group='PTS')


def test_aam():
    assert (aam.n_training_images == 4)
    assert (aam.n_levels == 3)
    assert (aam.downscale == 2)
    #assert (aam.features[0] == igo and len(aam.features) == 1)
    assert_allclose(np.around(aam.reference_shape.range()), (109., 103.))
    assert aam.scaled_shape_models
    assert aam.pyramid_on_features
    assert_allclose([aam.shape_models[j].n_components
                     for j in range(aam.n_levels)], (1, 2, 3))
    assert_allclose([aam.appearance_models[j].n_components
                     for j in range(aam.n_levels)], (3, 2, 1))
    assert_allclose([aam.appearance_models[j].template_instance.n_channels
                     for j in range(aam.n_levels)], (2, 2, 2))
    assert_allclose([aam.appearance_models[j].components.shape[1]
                     for j in range(aam.n_levels)], (884, 3652, 14892))


@raises(ValueError)
def test_n_shape_exception():
    fitter = LucasKanadeAAMFitter(aam, n_shape=[3, 6, 'a'])


@raises(ValueError)
def test_n_appearance_exception():
    fitter = LucasKanadeAAMFitter(aam, n_appearance=[10, 20])


def test_pertrurb_shape():
    fitter = LucasKanadeAAMFitter(aam)
    s = fitter.perturb_shape(training_images[0].landmarks['PTS'].lms,
                             noise_std=0.08, rotation=False)
    assert (s.n_dims == 2)
    assert (s.n_landmark_groups == 0)
    assert (s.n_points == 68)


def test_obtain_shape_from_bb():
    fitter = LucasKanadeAAMFitter(aam)
    s = fitter.obtain_shape_from_bb(np.array([[53.916, 1.853],
                                              [408.469, 339.471]]))
    assert ((np.around(s.points) == np.around(initial_shape[0].points)).all())
    assert (s.n_dims == 2)
    assert (s.n_landmark_groups == 0)
    assert (s.n_points == 68)


@raises(ValueError)
def test_max_iters_exception():
    fitter = LucasKanadeAAMFitter(aam,
                                  algorithm=AlternatingInverseCompositional)
    fitter.fit(training_images[0], initial_shape[0],
               max_iters=[10, 20, 30, 40])


@patch('sys.stdout', new_callable=StringIO)
def test_str_mock(mock_stdout):
    print(aam)
    fitter = LucasKanadeAAMFitter(aam,
                                  algorithm=AlternatingInverseCompositional)
    print(fitter)
    print(aam2)
    fitter = LucasKanadeAAMFitter(aam2,
                                  algorithm=ProbabilisticForwardAdditive)
    print(fitter)


def aam_helper(aam, algorithm, im_number, max_iters, initial_error,
               final_error, error_type):
    fitter = LucasKanadeAAMFitter(aam, algorithm=algorithm)
    fitting_result = fitter.fit(
        training_images[im_number], initial_shape[im_number],
        gt_shape=training_images[im_number].landmarks['PTS'].lms,
        max_iters=max_iters)
    assert (np.around(fitting_result.initial_error(error_type=error_type),
                      5) == initial_error)
    assert (np.around(fitting_result.final_error(error_type=error_type),
                      5) == final_error)


@attr('fuzzy')
def test_alternating_ic():
    aam_helper(aam, AlternatingInverseCompositional, 0, 6, 0.09062, 0.05606,
               'me_norm')


@attr('fuzzy')
def test_adaptive_ic():
    aam_helper(aam, AdaptiveInverseCompositional, 1, 5, 0.07373, 0.02251,
               'me_norm')


@attr('fuzzy')
def test_simultaneous_ic():
    aam_helper(aam, SimultaneousInverseCompositional, 2, 7, 0.12273, 0.10145,
               'me_norm')


@attr('fuzzy')
def test_projectout_ic():
    aam_helper(aam, ProjectOutInverseCompositional, 3, 6, 0.10303, 0.06499,
               'me_norm')


@attr('fuzzy')
def test_alternating_fa():
    aam_helper(aam, AlternatingForwardAdditive, 0, 8, 0.09062, 0.07053,
               'me_norm')


@attr('fuzzy')
def test_adaptive_fa():
    aam_helper(aam, AdaptiveForwardAdditive, 1, 6, 0.07373, 0.04518, 'me_norm')


@attr('fuzzy')
def test_simultaneous_fa():
    aam_helper(aam, SimultaneousForwardAdditive, 2, 5, 0.12273, 0.11485,
               'me_norm')


@attr('fuzzy')
def test_projectout_fa():
    aam_helper(aam, ProjectOutForwardAdditive, 3, 6, 0.10303, 0.09202,
               'me_norm')


@attr('fuzzy')
def test_alternating_fc():
    aam_helper(aam, AlternatingForwardCompositional, 0, 6, 0.09062, 0.07109,
               'me_norm')

@attr('fuzzy')
def test_adaptive_fc():
    aam_helper(aam, AdaptiveForwardCompositional, 1, 6, 0.07373, 0.04371,
               'me_norm')


@attr('fuzzy')
def test_simultaneous_fc():
    aam_helper(aam, SimultaneousForwardCompositional, 2, 5, 0.12273, 0.11197,
               'me_norm')


@attr('fuzzy')
def test_projectout_fc():
    aam_helper(aam, ProjectOutForwardCompositional, 3, 6, 0.10303, 0.07696,
               'me_norm')


@attr('fuzzy')
def test_probabilistic_ic():
    aam_helper(aam2, ProbabilisticInverseCompositional, 0, 6, 0.08605, 0.08923,
               'me_norm')


@attr('fuzzy')
def test_probabilistic_fa():
    aam_helper(aam2, ProbabilisticForwardAdditive, 1, 7, 0.08778, 0.0846,
               'me_norm')


@attr('fuzzy')
def test_probabilistic_fc():
    aam_helper(aam2, ProbabilisticForwardCompositional, 2, 6, 0.11328,
               0.11321, 'me_norm')
