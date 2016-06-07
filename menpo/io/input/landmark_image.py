from functools import partial

from .landmark import asf_importer, pts_importer


asf_image_importer = partial(asf_importer, image_origin=True)
asf_image_importer.__doc__ = asf_importer.__doc__

pts_image_importer = partial(pts_importer, image_origin=True)
pts_image_importer.__doc__ = pts_importer.__doc__
