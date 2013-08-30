from pybug.warp.base import scipy_warp
from pybug.align.lucaskanade.base import LucasKanade


class AppearanceLucasKanade(LucasKanade):

    def __init__(self, model, residual, transform,
                 warp=scipy_warp, optimisation=('GN',), eps=10**-6):
        super(AppearanceLucasKanade, self).__init__(residual, transform,
                                                    warp, optimisation, eps)
        # in appearance alignment, target image is aligned to appearance model
        self.appearance_model = model
        # by default, template is assigned to mean appearance
        self.template = model.mean

        # pre-compute
        self._precompute()