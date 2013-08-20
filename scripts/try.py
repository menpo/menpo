# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Fit an AAM to an Input Images
# ##### Version 0.1
#
# ### 1.1 Load the Data

# <codecell>

from pybug.io import auto_import
from pybug.landmark.labels import ibug_68_points, ibug_68_contour, ibug_68_trimesh, labeller

# load the training images of the LFPW database as landmarked images using the autoimporter
images = auto_import('/vol/atlas/databases/lfpw/train/' + '*.png')


# label the landmarks using the ibug's "standard" 68 points mark-up
labeller(images, 'PTS', ibug_68_points)
labeller(images, 'PTS', ibug_68_contour)
labeller(images, 'PTS', ibug_68_trimesh)

from pybug.shape import PointCloud

points = [img.landmarks['PTS'].all_landmarks.points - 1  for img in images]
shapes = [PointCloud(p) for p in points]

# <codecell>

import pickle

# load a previously buid AAM
aam = pickle.load(open('/vol/atlas/aams/lfpw_pwa', "rb"))

template = aam["template"]
template_landmarks = aam["template_landmarks"]
appearance_model = aam["appearance_model"]
shape_model = aam["shape_model"]

# <markdowncell>

# ### 2.1 Build a Statistically Driven + Similarity Transform

# <codecell>

from pybug.transform.affine import SimilarityTransform
import numpy as np

test_shape = shapes[0]
test_image = images[0].as_greyscale()

similarity_transform = SimilarityTransform.estimate(shape_model.mean.points, test_shape.points)

global_weights = similarity_transform.as_vector()
global_weights[1] = 0
global_transform = SimilarityTransform.from_vector(global_weights)

aligned_test_points = global_transform.inverse.apply(test_shape.points)
aligned_test_shape = PointCloud(aligned_test_points)
shape_weights = shape_model.project(aligned_test_shape)

initial_weights = np.concatenate([global_weights, np.zeros_like(shape_weights)])

# <codecell>

from pybug.transform.piecewiseaffine import PiecewiseAffineTransform
from pybug.transform.statisticallydriven import StatisticallyDrivenTransform
from scipy.spatial import Delaunay

def pwa_constructor(src_landmarks, tgt_landmarks):
    tri = Delaunay(src_landmarks)
    return PiecewiseAffineTransform(src_landmarks, tgt_landmarks, tri.simplices)

initial_transform = StatisticallyDrivenTransform(shape_model,
                                                 pwa_constructor,
                                                 source=template_landmarks,
                                                 parameters=initial_weights,
                                                 global_transform=global_transform)

# <codecell>

from pybug.warp import scipy_warp

initial_warp = scipy_warp(test_image, template, initial_transform)


trilist = template.get_landmark_set('ibug_68_trimesh').landmark_dict['tri'].trilist
temp_pwa = PiecewiseAffineTransform(template_landmarks.points, test_shape.points, trilist)
temp = scipy_warp(test_image, template, temp_pwa.transform)

test_image.add_landmark_set("initial", {"initial": initial_transform.target})
test_image.get_landmark_set("initial").view()

initial_warp.view()
temp.view()

# <codecell>

from pybug.align.lucaskanade.residual import LSIntensity
from pybug.align.lucaskanade.base import ImageForwardAdditive, ImageForwardCompositional, ImageInverseCompositional

residual = LSIntensity()
lk_algorithm = ImageForwardCompositional(test_image, temp, residual, initial_transform)
optimal_transform = lk_algorithm.align(max_iters=10)

# <codecell>

fitted_shape = optimal_transform.target
fitted_appearance = scipy_warp(test_image, template, optimal_transform)

test_image.add_landmark_set("fitting_result", {"fitting_result": fitted_shape})
test_image.get_landmark_set("fitting_result").view()

fitted_appearance.view()

# <codecell>

