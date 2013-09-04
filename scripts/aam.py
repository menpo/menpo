__author__ = 'ja310'

# define functions for reading .pts files

import csv
import numpy as np


def laread(filename):
    ext = str.rsplit(filename, '.', 1)[-1]
    if ext == 'pts':
        strdata = laread_pts(filename)
    else:
        strdata = []
    numdata = np.array(strdata, dtype=np.float)
    return numdata


def laread_pts(filename):
    f = open(filename, 'r')
    for line in f:
        if line == '{\n':
            break
    data = []
    for line in csv.reader(f, delimiter=' '):
        if line != ['}']:
            data.append([line[1], line[0]])
            # data.append(line) more neat but requires .pts files to be
            # properly written !!!
    return data


# define a function for directly constructing tps transform objects

from pybug.align.nonrigid.tps import TPS


def tps_constructor(src_landmarks, tgt_landmarks):
    tps = TPS(src_landmarks, tgt_landmarks)
    return tps.transform


# define a function that checks if a point is inside a convex polygon.
# source:"http://www.ariel.com.au/a/python-point-int-poly.html"


def point_inside_polygon(x, y, poly):

    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n+1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


# actual code -----


# 1. Load Data

import os
import scipy.io as sio
from pybug.shape import PointCloud
from pybug.io import auto_import

# define LFPW paths on atlas
lfpw_path = '/vol/atlas/databases/lfpw/'
lfpw_train_path = lfpw_path + 'train/'
# load LFPW training shapes
shapes = [PointCloud(laread(lfpw_train_path + f) - 1) for f in sorted(os.listdir(lfpw_train_path))
          if os.path.splitext(f)[-1] in '.pts']
# load metadata associated with the previous shapes
triangles = sio.loadmat(lfpw_path + 'tri.mat')
tri = triangles['tri']
parts = sio.loadmat(lfpw_path + 'parts.mat')
contour = parts['contour'].astype(np.uint32).flatten()
# load LFPW training images using auto-importer (this is so cool!!!)
images = auto_import(lfpw_train_path + '*.png')


# 2. Build template frame

from pybug.align import GeneralizedProcrustesAnalysis
from pybug.image.base import Image
from pybug.image.base import MaskImage

# center shape
centralized_points = [s.points - np.mean(s.points, axis=0) for s in shapes]
# apply procrustes
gpa = GeneralizedProcrustesAnalysis(centralized_points)
aligned_points = [p[-1].aligned_source for p in gpa.procrustes]
aligned_shapes = [PointCloud(p) for p in aligned_points]
# define template landmarks, resolution and image data
mean_shape = np.mean(aligned_points, axis=0)
margin = 3
template_landmarks = mean_shape - np.min(mean_shape, axis=0) + margin
template_resolution = np.ceil(np.max(template_landmarks, axis=0) + margin)
template_data = np.zeros(template_resolution, dtype=np.float64)
# calculate template mask
convex_hull = template_landmarks[contour, :]
y = np.arange(template_resolution[0], dtype=np.uint32)
x = np.arange(template_resolution[1], dtype=np.uint32)
yv, xv = np.meshgrid(y, x)
mask_data = np.zeros(template_resolution, dtype=np.bool)
for p in np.array([yv.flatten(), xv.flatten()]).T:
    mask_data[p[0], p[1]] = point_inside_polygon(p[0], p[1], convex_hull)
mask = MaskImage(mask_data)
# define template
template = Image(template_data, mask=mask)


# 3. Warp Images

# from pybug.align.nonrigid.tps import TPS
from pybug.warp import scipy_warp

# calculate tps warp from each loaded shape to the template landmarks
tps = [TPS(template_landmarks, s.points) for s in shapes]
# warp the loaded images
tps_warped_images = [scipy_warp(img, template, tps[i].transform) for i,
                                                                  img in enumerate(images)]


# 4. Build Appearance Model

from pybug.model.linear import PCAModel

# transform all warped images to graysacale
grayscale_tps_warped_images = [img.as_greyscale() for img in tps_warped_images]
# apply PCA to the warped images.
tps_appearance_model = PCAModel(grayscale_tps_warped_images, n_components=50)

# 5. Build Shape Model

shape_model = PCAModel(aligned_shapes, n_components=12)


# TEST

from pybug.model.linear import SimilarityModel

test_shape = shapes[-1]
test_image = images[-1].as_greyscale()

similarity_model = SimilarityModel(shape_model.mean)

similarity_weights = similarity_model.project(test_shape)
similarity_transform = similarity_model.equivalent_similarity_transform(
    similarity_weights)
aligned_test_points = similarity_transform.inverse.apply(test_shape.points)
aligned_test_shape = PointCloud(aligned_test_points)

shape_weights = shape_model.project(aligned_test_shape)

weights = np.concatenate([similarity_weights, shape_weights])


from pybug.transform.statisticallydriven import \
    StatisticallyDrivenPlusSimilarityTransform

transform = StatisticallyDrivenPlusSimilarityTransform(shape_model,
                                                       similarity_model,
                                                       tps_constructor,
                                                       source=template_landmarks,
                                                       weights=weights)

instance_tps = TPS(template_landmarks, test_shape.points)
IWxp = scipy_warp(test_image, template, transform)

from pybug.align.lucaskanade.residual import LSIntensity
from pybug.align.lucaskanade.base import ImageForwardAdditive

residual = LSIntensity()
project_out = ImageForwardAdditive(test_image, IWxp, residual, transform)
images = project_out.align()

1