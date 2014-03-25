import numpy as np
from menpo.groupalign import GeneralizedProcrustesAnalysis
from menpo.shape import PointCloud
from menpo.transform.tps import TPS
from menpo.transform import Translation, Scale
from menpo.model import PCAModel
from menpo.rasterize import GLRasterizer
from menpo.rasterize.transform import (ExtractNDims, AddNDims,
                                       CylindricalUnwrapTransform)


def cylindrical_unwrap_and_translation(points):
    from menpo.misctools.circlefit import circle_fit
    from menpo.transform import Translation
    # find the optimum centre to unwrap
    xy = points.points[:, [0, 2]]  # just in the x-z plane
    centre, radius = circle_fit(xy)
    # convert the 2D circle data into the 3D space
    translation = np.array([centre[0], 0, centre[1]])
    centring_transform = Translation(-translation)
    unwrap = CylindricalUnwrapTransform(radius)
    
    def translate_and_unwrap(pc):
        return unwrap.apply(centring_transform.apply(pc))
    
    return translate_and_unwrap


def clip_space_transform(points, boundary_proportion=0.1):
    r"""
    Produces a transform which fits 2D points into the OpenGL
    clipping space ([-1, 1], [-1, 1])
    """
    centering = Translation(points.centre_of_bounds).pseudoinverse
    scale = Scale(points.range() / 2)
    b_scale = Scale(1 - boundary_proportion, n_dims=2)
    return centering.compose_before(scale.pseudoinverse).compose_before(b_scale)


def mean_pointcloud(pointclouds):
    return PointCloud(sum(pointclouds) / len(pointclouds))


def build_trimesh_extractor(sample_image, sampling_rate=50):
    import numpy as np
    x, y = np.meshgrid(np.arange(0, sample_image.height, sampling_rate),
                       np.arange(0, sample_image.width, sampling_rate))
    # use the sample_image's mask to filter which points should be
    # selectable
    sample_in_range = sample_image.mask.pixels[x, y]
    v_x = x[sample_in_range[..., 0]]
    v_y = y[sample_in_range[..., 0]]

    # build a cheeky TriMesh to get hassle free trilist
    from menpo.shape import TriMesh
    tm = TriMesh(np.vstack([v_x, v_y]).T)

    def extract_trimesh(shape_image):
        sampled = shape_image.pixels[v_x, v_y, :]
        return TriMesh(sampled, trilist=tm.trilist.copy())

    return extract_trimesh


class MMBuilder(object):

    def __init__(self, models, group=None, label='all'):
        self.models = models
        self.group = group
        self.label = label

        print 'Rigidly aligning...'
        # ra = Rigid Aligned
        self.ra_models, self.ra_mean_lms, self.unwrapper = None, None, None
        self.rigidly_align()

        print 'Unwrapping and flattening...'
        # u = Unwrapped (infers ra)
        self.u_models, self.u_mean_lms = None, None
        self.u_2d, self.u_mean_lms_2d = None, None
        self.unwrap_and_flatten()

        print 'Warping...'
        # w = Warped (infers ra + u)
        self.w_models_2d = None
        self.warp()

        print 'Rasterizing...'
        self.r = GLRasterizer()
        self.shape_images = []
        self.rasterize()

        print 'Extracting TriMeshes...'
        self.dc_meshes = []
        self.extract()

        print 'Taking PCA...'
        self.shape_model = None
        self.pca()

    def lms_for(self, x):
        return x.landmarks[self.group][self.label].lms

    def rigidly_align(self):
        gpa = GeneralizedProcrustesAnalysis([self.lms_for(m) 
                                             for m in self.models])
        self.ra_models = [t.apply(m) for t, m in zip(gpa.transforms,
                                                     self.models)]
        self.ra_mean_lms = mean_pointcloud([self.lms_for(m).points
                                            for m in self.ra_models])
        self.unwrapper = cylindrical_unwrap_and_translation(self.ra_mean_lms)

    def unwrap_and_flatten(self):
        self.u_models = [self.unwrapper(m) for m in self.ra_models]
        self.u_mean_lms = self.unwrapper(self.ra_mean_lms)
        extract_2d = ExtractNDims(2)
        self.u_2d = [extract_2d.apply(u) for u in self.u_models]
        self.u_mean_lms_2d = extract_2d.apply(self.u_mean_lms)

    def warp(self):
        tps_transforms = [TPS(self.lms_for(u), self.u_mean_lms_2d)
                          for u in self.u_2d]
        self.w_models_2d = [t.apply(u) for t, u in zip(tps_transforms,
                                                       self.u_2d)]

    def rasterize(self):
        trans_to_clip_space = clip_space_transform(self.u_mean_lms_2d)
        cs_models_2d = [trans_to_clip_space.apply(m) for m in self.w_models_2d]
        cs_mean_lms_2d = trans_to_clip_space.apply(self.u_mean_lms_2d)
        add_nill_z = AddNDims(1)  # adds an all-zero z axis
        cs_models = [add_nill_z.apply(m) for m in cs_models_2d]
        mean_lm_img = self.r.model_to_image_space(
            add_nill_z.apply(cs_mean_lms_2d))

        # build all the shape images
        self.shape_images = []
        for cs_w_model, orig in zip(cs_models, self.ra_models):
            blank, shape_image = self.r.rasterize_mesh_with_f3v_interpolant(
                cs_w_model, per_vertex_f3v=orig.points)
            shape_image.landmarks['mean'] = mean_lm_img
            self.shape_images.append(shape_image)

        print 'masking to landmarks...'
        # set the mask correctly on all meshes
        si = self.shape_images[0]
        si.constrain_mask_to_landmarks()
        for s in self.shape_images:
            s.mask.pixels = si.mask.pixels.copy()

    def extract(self):
        tm_extract = build_trimesh_extractor(self.shape_images[0])
        self.dc_meshes = [tm_extract(s) for s in self.shape_images]

    def pca(self):
        self.shape_model = PCAModel(self.dc_meshes)
