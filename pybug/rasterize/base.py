import abc


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
    def _rasterize_type_color(self):
        return not self._rasterize_type_texture
