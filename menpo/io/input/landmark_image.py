from menpo.base import partial_doc

from .landmark import asf_importer, pts_importer


asf_image_importer = partial_doc(asf_importer, image_origin=True)

pts_image_importer = partial_doc(pts_importer, image_origin=True)
