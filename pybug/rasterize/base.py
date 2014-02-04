import abc
from collections import namedtuple

TextureRasterInfo = namedtuple('TextureRasterInfo',
                               ['points', 'trilist', 'tcoords', 'texture'])
ColourRasterInfo = namedtuple('ColourRasterInfo', ['points', 'trilist',
                                                   'colours'])


class Rasterizable(object):
    __metaclass__ = abc.ABCMeta

    def _rasterize_generate_textured_mesh(self):
        pass

    def _rasterize_generate_color_mesh(self):
        pass

    @abc.abstractproperty
    def _rasterize_type_texture(self):
        pass

    @property
    def _rasterize_type_colour(self):
        return not self._rasterize_type_texture
