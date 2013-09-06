from __future__ import division
import abc
import os.path as path
import numpy as np
from pybug.image import DepthImage
from pybug.io.base import Importer, get_importer, find_alternative_files
from scipy.spatial import Delaunay
from pybug.transform.affine import Scale


class DepthImageImporter(Importer):
    r"""
    Base class for importing depth images. Depth images are defined by file
    types whereby the data lies explicitly on a grid. This grid pattern means
    the data can be interpreted as an image.

    Parameters
    ----------
    filepath : string
        An absolute filepath
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, filepath):
        super(DepthImageImporter, self).__init__(filepath)
        self.attempted_image_landmark_search = False
        self.attempted_mesh_landmark_search = False
        self.attempted_texture_search = False
        self.relative_mesh_landmark_path = None
        self.relative_image_landmark_path = None
        self.relative_texture_path = None
        self.points = None
        self.trilist = None
        self.mask = None
        self.tcoords = None

    def _build_texture_and_landmark_importers(self):
        r"""
        Search for a texture and landmark file in the same directory as the
        mesh. If they exist, create importers for them.
        """
        if self.texture_path is None or not path.exists(self.texture_path):
            self.texture_importer = None
        else:
            # This import is here to avoid circular dependencies
            from pybug.io.extensions import image_types
            self.texture_importer = get_importer(self.texture_path,
                                                 image_types)

        if self.image_landmark_path is None or not path.exists(self.image_landmark_path):
            self.image_landmark_importer = None
        else:
            # This import is here to avoid circular dependencies
            from pybug.io.extensions import image_landmark_types
            self.image_landmark_importer = get_importer(
                self.image_landmark_path, image_landmark_types)

        if self.mesh_landmark_path is None or not path.exists(self.mesh_landmark_path):
            self.mesh_landmark_importer = None
        else:
            # This import is here to avoid circular dependencies
            from pybug.io.extensions import mesh_landmark_types
            self.mesh_landmark_importer = get_importer(self.mesh_landmark_path,
                                                       mesh_landmark_types)

    def _search_for_texture(self):
        r"""
        Tries to find a texture with the same name as the depth image.

        Returns
        --------
        relative_texture_path : string
            The relative path to the texture or ``None`` if one can't be found
        """
        # Stop searching every single time we access the property
        self.attempted_texture_search = True
        # This import is here to avoid circular dependencies
        from pybug.io.extensions import image_types
        try:
            return find_alternative_files('texture', self.filepath,
                                          image_types)
        except ImportError:
            return None

    def _search_for_landmarks(self, types):
        r"""
        Tries to find a set of landmarks with the same name as the image. This
        is only attempted once.

        Returns
        -------
        basename : string
            The basename of the landmarks file found, eg. ``image.pts``.
        """
        try:
            return find_alternative_files('landmarks', self.filepath,
                                          types)
        except ImportError:
            return None

    @property
    def image_landmark_path(self):
        r"""
        Get the absolute path to the image landmarks.

        Returns ``None`` if none can be found.
        Makes it's best effort to find an appropriate landmark set by
        searching for landmarks with the same name as the image.
        """
        # Try find a texture path if we can
        if self.relative_image_landmark_path is None and \
                not self.attempted_image_landmark_search:
            # This import is here to avoid circular dependencies
            from pybug.io.extensions import image_landmark_types
            # Stop searching every single time we access the property
            self.attempted_image_landmark_search = True
            self.relative_image_landmark_path = self._search_for_landmarks(
                image_landmark_types)

        try:
            return path.join(self.folder, self.relative_image_landmark_path)
        except AttributeError:
            return None

    @property
    def mesh_landmark_path(self):
        r"""
        Get the absolute path to the mesh landmarks.

        Returns ``None`` if none can be found.
        Makes it's best effort to find an appropriate landmark set by
        searching for landmarks with the same name as the image.
        """
        # Try find a texture path if we can
        if self.relative_mesh_landmark_path is None and \
                not self.attempted_mesh_landmark_search:
            # This import is here to avoid circular dependencies
            from pybug.io.extensions import mesh_landmark_types
            self.attempted_mesh_landmark_search = True
            self.relative_mesh_landmark_path = self._search_for_landmarks(
                mesh_landmark_types)

        try:
            return path.join(self.folder, self.relative_mesh_landmark_path)
        except AttributeError:
            return None

    @property
    def texture_path(self):
        """
        Get the absolute path to the texture. Returns None if one can't be
        found. Makes it's best effort to find an appropriate texture by
        searching for textures with the same name as the mesh. Will only
        search for the path the first time ``texture_path`` is invoked.

        Sets the ``self.relative_texture_path`` attribute.

        Returns
        -------
        texture_path : string
            Absolute filepath to the texture
        """
        # Try find a texture path if we can
        if self.relative_texture_path is None and \
                not self.attempted_texture_search:
            self.relative_texture_path = self._search_for_texture()

        try:
            return path.join(self.folder, self.relative_texture_path)
        except AttributeError:
            return None

    @abc.abstractmethod
    def _build_image_and_mesh(self):
        r"""
        Abstract method that handles actually building an image. This involves
        reading the image from disk and doing any necessary processing.

        Should set the ``self.image`` attribute.
        """
        pass

    @abc.abstractmethod
    def _process_landmarks(self, original_image, lmark_dict):
        r"""
        Abstract method that allows the landmarks to be transformed between
        the original image space and the depth image space. The space may
        be different because a texture mapped image is unlikely to be equal
        in size to the mesh.

        Parameters
        ----------
        original_image : :class:`pybug.image.base.Image`
            The original image that the landmarks belong to
        lmark_dict : dict (string, :class:`pybug.shape.base.PointCloud`)
            The landmark dictionary to transform

        Returns
        -------
        transformed_dict : dict (string, :class:`pybug.shape.base.PointCloud`)
            The transformed landmark points.
        """
        pass

    def build(self):
        r"""
        Overrides the :meth:`build <pybug.io.base.Importer.build>` method.

        Builds the image and landmarks. Assigns the landmark set to the image.

        Returns
        -------
        image : :class:`pybug.image.base.Image`
            The image object with landmarks attached if they exist
        """
        # Build the image as defined by the overridden method and then search
        # for valid landmarks that may have been defined by the importer
        self._build_image_and_mesh()
        self._build_texture_and_landmark_importers()

        if self.texture_importer is not None:
            texture = self.texture_importer.build()
        else:
            texture = None

        self.image = DepthImage(self.depth_image, points=self.points,
                                trilist=self.trilist, tcoords=self.tcoords,
                                texture=texture, mask=self.mask)

        if self.image_landmark_importer is not None:
            label, lmark_dict = self.image_landmark_importer.build(
                scale_factors=self.image.shape)
            texture.add_landmark_set(label, lmark_dict)
            # Add landmarks to image - may need scaling if original texture
            # is different in size to depth image
            self.image.add_landmark_set(label,
                                        self._process_landmarks(texture,
                                                                lmark_dict))

        if self.mesh_landmark_importer is not None:
            label, lmark_dict = self.mesh_landmark_importer.build()
            self.image.mesh.add_landmark_set(label, lmark_dict)

        return self.image


class BNTImporter(DepthImageImporter):
    r"""
    Allows importing the BNT file format from the bosphorus dataset.
    This reads in the 5 channels (3D coordinates and texture coordinates),
    splits them appropriately and then triangulates the ``x`` and ``y``
    coordinates to create a surface. The texture path is also given in the file
    format.

    The file format specifies a 'bad_value' which is used to denote inaccurate
    data points. This value is replaced with ``np.nan``. The mesh is
    created with only those values that are not NaN. The depth image contains
    all values.

    Parameters
    ----------
    filepath : string
        Absolute filepath of the mesh.
    """

    def __init__(self, filepath):
        # Setup class before super class call
        super(BNTImporter, self).__init__(filepath)

    def _process_landmarks(self, original_image, lmark_dict):
        original_shape = original_image.shape
        depth_image_shape = self.depth_image.shape

        # Scale the points down to the smaller depth image size
        scale_0 = depth_image_shape[0] / original_shape[0]
        scale_1 = depth_image_shape[1] / original_shape[1]
        scale = Scale(np.array([scale_0, scale_1]))

        for l in lmark_dict.values():
            scale.apply(l)

        return lmark_dict

    def _build_image_and_mesh(self):
        r"""
        Read the file in and parse appropriately. Includes reading the texture
        path.
        """
        with open(self.filepath, 'rb') as f:
            # Currently these are unused, but they are in the format
            # Could possibly store as metadata?
            n_rows = np.fromfile(f, dtype=np.uint16, count=1)
            n_cols = np.fromfile(f, dtype=np.uint16, count=1)
            bad_value = np.fromfile(f, dtype=np.float64, count=1)

            # Get integers and convert to valid string
            image_path_len = np.fromfile(f, dtype=np.uint16, count=1)
            texture_path = np.fromfile(f, dtype=np.uint8, count=image_path_len)
            texture_path = ''.join(map(chr, texture_path))

            # Get data and reshape (reshape in an odd order due to Matlab's
            # Fortran ordering). First three columns are 3D coordinates
            # and last two are 2D texture coordinates
            coords_len = np.fromfile(f, dtype=np.uint32, count=1)
            data = np.fromfile(f, dtype=np.float64, count=coords_len * 5.0)
            data = data.reshape([5, coords_len / 5.0]).T

        # Get the 3D coordinates
        points = data[:, :3]
        # We want to remove the bad values because otherwise the mesh is not
        # renderable. We do this by replacing the bad value values with nan
        points[points == bad_value] = np.nan

        # Use only those coordinates with do not contains nans
        valid_points = ~np.isnan(points).any(axis=1)
        self.points = points[valid_points]
        # Generate a triangulation from the points
        self.trilist = Delaunay(self.points[:, :2]).simplices

        # The image contains all coordinates
        # Must be flipped LR due to Fortran ordering from Matlab
        # Must by flipped upside down due to image vs mesh ordering
        self.depth_image = np.fliplr(np.reshape(points[:, 2][::-1],
                                                [n_rows, n_cols]))
        self.mask = ~np.isnan(self.depth_image)

        # Apparently the texture coordinates are upside down?
        self.tcoords = data[:, -2:][valid_points]
        self.tcoords[:, 1] = -self.tcoords[:, 1]

        self.relative_texture_path = texture_path


class FIMImporter(DepthImageImporter):
    r"""
    Allows importing floating point images as depth images.
    This reads in the shape in to 3 channels and then triangulates the
    ``x`` and ``y`` coordinates to create a surface. An example of this
    datatype is the aligned BU4D dataset.

    Parameters
    ----------
    filepath : string
        Absolute filepath of the mesh.
    """

    def __init__(self, filepath):
        # Setup class before super class call
        super(FIMImporter, self).__init__(filepath)

    def _process_landmarks(self, original_image, lmark_dict):
        r"""
        There are no default landmarks for this dataset so we currently don't
        perform any processing.

        Parameters
        ----------
        original_image : :class:`pybug.image.base.Image`
            The original image that the landmarks belong to
        lmark_dict : dict (string, :class:`pybug.shape.base.PointCloud`)
            The landmark dictionary to transform
        """
        pass

    def _build_image_and_mesh(self):
        r"""
        Read the file and parse it as necessary. Since the data lies on a grid
        we can triangulate the 2D coordinates to get a valid triangulation.

        The format does not specify texture coordinates.
        """
        with open(self.filepath, 'rb') as f:
            size = np.fromfile(f, dtype=np.uint32, count=3)
            data = np.fromfile(f, dtype=np.float32, count=np.product(size))
            data = data.reshape([size[0], size[1], size[2]])

        # Replace the zero buffer values with nan so that the image renders
        # nicely
        data[data == 0] = np.nan

        self.depth_image = data[:, :, 2]
        self.mask = ~np.isnan(self.depth_image)

        points = np.reshape(data, [size[0] * size[1], size[2]])
        valid_points = ~np.isnan(points).any(axis=1)
        self.points = points[valid_points]
        # Generate a triangulation from the points
        self.trilist = Delaunay(self.points[:, :2]).simplices