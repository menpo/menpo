from __future__ import division
import abc
import os.path as path
import numpy as np
from menpo.image import ShapeImage
from menpo.io.base import (Importer, find_alternative_files,
                           map_filepath_to_importer)
import re


class SpatialImageImporter(Importer):
    r"""
    Base class for importing depth images. Depth images are defined by file
    types whereby the data lies explicitly on a grid. This grid pattern means
    the data can be interpreted as an image.

    Parameters
    ----------
    filepath : string
        An absolute filepath
    """

    def __init__(self, filepath):
        super(SpatialImageImporter, self).__init__(filepath)
        self.attempted_texture_search = False
        self.relative_texture_path = None
        self.trilist = None
        self.mask = None
        self.tcoords = None
        self.shape_image = None

    def _build_texture_and_landmark_importers(self):
        r"""
        Search for a texture and landmark file in the same directory as the
        mesh. If they exist, create importers for them.
        """
        if self.texture_path is None or not path.exists(self.texture_path):
            self.texture_importer = None
        else:
            # This import is here to avoid circular dependencies
            from menpo.io.extensions import image_types
            self.texture_importer = map_filepath_to_importer(self.texture_path,
                                                 image_types)

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
        from menpo.io.extensions import image_types
        try:
            return find_alternative_files('texture', self.filepath,
                                          image_types)
        except ImportError:
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
        except (AttributeError, TypeError):
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
    def _process_landmarks(self, original_image, landmark_group):
        r"""
        Abstract method that allows the landmarks to be transformed between
        the original image space and the depth image space. The space may
        be different because a texture mapped image is unlikely to be equal
        in size to the mesh.

        Parameters
        ----------
        original_image : :class:`menpo.image.base.Image`
            The original image that the landmarks belong to
        lmark_group : :class:`menpo.landmark.base.LandmarkGroup`
            The landmark group to transform

        Returns
        -------
        transformed_group : :class:`menpo.landmark.base.LandmarkGroup`
            The transformed landmark group.
        """
        pass

    def build(self):
        r"""
        Overrides the :meth:`build <menpo.io.base.Importer.build>` method.

        Builds the image and landmarks. Assigns the landmark set to the image.

        Returns
        -------
        image : :class:`menpo.image.base.Image`
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

        self.image = ShapeImage(self.shape_image, mask=self.mask,
                                trilist=self.trilist, tcoords=self.tcoords,
                                texture=texture)
        return self.image


class BNTImporter(SpatialImageImporter):
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
        shape_pixels = data[:, :3]
        # We want to remove the bad values because otherwise the mesh is not
        # renderable. We do this by replacing the bad value values with nan
        shape_pixels[shape_pixels == bad_value] = np.nan

        # The image contains all coordinates
        # Must be flipped LR due to Fortran ordering from Matlab
        # Must by flipped upside down due to image vs mesh ordering
        self.shape_image = np.fliplr(np.reshape(shape_pixels[:, :3][::-1],
                                                [n_rows, n_cols, 3]))
        self.mask = ~np.any(np.isnan(self.shape_image), axis=-1)

        # Use only those coordinates with do not contains nans
        valid_points = ~np.isnan(shape_pixels).any(axis=1)

        # Apparently the texture coordinates are upside down?
        #self.tcoords = data[:, -2:][valid_points]
        #self.tcoords[:, 1] = 1.0 - self.tcoords[:, 1]
        # struggling to interpret these - let the ShapeImage build them
        # instead.
        self.tcoords = None

        self.relative_texture_path = texture_path


class FIMImporter(SpatialImageImporter):
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

        self.shape_image = data
        self.mask = ~np.isnan(self.shape_image)


class ABSImporter(SpatialImageImporter):
    r"""
    Allows importing the ABS file format from the FRGC dataset.

    The z-min value is stripped from the mesh to make it renderable.

    Parameters
    ----------
    filepath : string
        Absolute filepath of the mesh.
    """

    def __init__(self, filepath):
        # Setup class before super class call
        super(ABSImporter, self).__init__(filepath)

    def _build_image_and_mesh(self):
        r"""
        Read in the file and remove the z-min. Triangulate the 2D gridded
        coordinates to create a valid triangulation.
        """
        with open(self.filepath, 'r') as f:
            # Currently these are unused, but they are in the format
            # Could possibly store as metadata?
            # Assume first result for regexes
            re_rows = re.compile(u'([0-9]+) rows')
            n_rows = int(re_rows.findall(f.readline())[0])
            re_cols = re.compile(u'([0-9]+) columns')
            n_cols = int(re_cols.findall(f.readline())[0])


        # This also loads the mask
        #   >>> image_data[:, 0]
        image_data = np.loadtxt(self.filepath, skiprows=3, unpack=True)

        # Replace the lowest value with nan so that we can render properly
        image_data[image_data == np.min(image_data)] = np.nan

        self.shape_image = np.reshape(image_data[:, 1:], [n_rows, n_cols, 3])
        self.mask = np.reshape(image_data[:, 0], [n_rows, n_cols])
