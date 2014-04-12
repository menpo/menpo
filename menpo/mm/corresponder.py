from menpo.image import MaskedImage
from menpo.rasterize import GLRasterizer
from menpo.transform import ThinPlateSplines, AlignmentSimilarity
from menpo.rasterize.transform import (optimal_cylindrical_unwrap,
                                       ExtractNDims,
                                       clip_space_transform, AppendNDims)


def build_trimesh_extractor(sample_mask, sampling_rate=2):
    import numpy as np
    x, y = np.meshgrid(np.arange(0, sample_mask.height, sampling_rate),
                       np.arange(0, sample_mask.width, sampling_rate))
    # use the sample_mask's mask to filter which points should be
    # selectable
    sample_in_range = sample_mask.mask[x, y]
    v_x = x[sample_in_range]
    v_y = y[sample_in_range]

    # build a cheeky TriMesh to get hassle free trilist
    from menpo.shape import TriMesh
    tm = TriMesh(np.vstack([v_x, v_y]).T)

    def extract_trimesh(shape_image):
        sampled = shape_image.pixels[v_x, v_y, :]
        return TriMesh(sampled, trilist=tm.trilist.copy())

    return extract_trimesh


class TriMeshCorresponder(object):

    def __init__(self, target, interpolator=ThinPlateSplines,
                 sampling_rate=2):
        self.target = target
        self.interpolator = interpolator
        # 1. Flattening the mesh into 2D, interpolation
        self.flattener = None
        self._setup_flattener()

        self.extract_2d = ExtractNDims(2)  # 3D -> 2D
        self.append_z =   AppendNDims(1)  # adds an all-zero z axis

        self.f_target_3d = self.flattener.apply(self.target)
        # needed to warp against
        self.f_target_2d = self.extract_2d.apply(self.f_target_3d)

        # Build the rasterizer providing the clip space transform it should use
        cs_transform = clip_space_transform(self.f_target_3d, xy_scale=0.9)
        self.r = GLRasterizer(projection_matrix=cs_transform.h_matrix)

        # Where the target landmarks land in the image
        self.img_target_2d = self.r.model_to_image_transform.apply(
            self.f_target_3d)


        # 2. Extraction of TriMesh
        self.sampling_rate = sampling_rate
        # make an example of the output we expect, and attach the lms
        sample_output = MaskedImage.blank((self.r.height, self.r.width))
        sample_output.landmarks['_target'] = self.img_target_2d
        # create the mask we will want and save it for attaching on outputs
        sample_output.constrain_mask_to_landmarks()
        from scipy.ndimage.morphology import binary_dilation
        from menpo.image import BooleanImage
        eroded_pixels = binary_dilation(sample_output.mask.mask, iterations=30)
        self.mask = BooleanImage(eroded_pixels)
        self.mask.landmarks = sample_output.landmarks
        # create a trimesh extractor that can be used to efficiently pull off
        #  the corresponding points from each shape image
        self.trimesh_extractor = build_trimesh_extractor(
            self.mask, sampling_rate=self.sampling_rate)


    def _setup_flattener(self):
        self.flattener = optimal_cylindrical_unwrap(self.target)


    def in_correspondence(self, model, group=None, label='all'):
        # 1. Rigidly align the new model to the target
        alignment = AlignmentSimilarity(model.landmarks[group][label].lms,
                                        self.target)
        aligned_model = alignment.apply(model)

        # 2. Flatten the model, and warp it to align with the flattened target
        f_3d = self.flattener.apply(aligned_model)
        f_2d = self.extract_2d.apply(f_3d)
        tps_transform = self.interpolator(f_2d.landmarks[group][label].lms,
                                          self.f_target_2d)
        w_2d = tps_transform.apply(f_2d)
        # Append on the Z dim and set it to what it was in the flattened case
        w_3d = self.append_z.apply(w_2d)
        w_3d.points[:, 2] = f_3d.points[:, 2]
        return self.generate_correspondence_mesh(aligned_model, w_3d)

    def generate_correspondence_mesh(self, model, warped_model):
        # build the shape image
        blank, shape_image = self.r.rasterize_mesh_with_f3v_interpolant(
            warped_model, per_vertex_f3v=model.points)
        # TODO this should be handled by the landmarker if it isn't already
        shape_image.landmarks['_target'] = self.img_target_2d
        # TODO this should be specific here as to what labels to use
        shape_image.mask = self.mask
        c_mesh = self.trimesh_extractor(shape_image)
        c_mesh.landmarks = model.landmarks
        return c_mesh


class TexturedTriMeshCorresponder(TriMeshCorresponder):

    def generate_correspondence_mesh(self, model, warped_model):
        # build the shape image
        rgb_image, shape_image = self.r.rasterize_mesh_with_f3v_interpolant(
            warped_model, per_vertex_f3v=model.points)
        # TODO this should be handled by the landmarker if it isn't already
        shape_image.landmarks['_target'] = self.img_target_2d
        # TODO this should be specific here as to what labels to use
        shape_image.mask = self.mask
        return self.trimesh_extractor(shape_image)