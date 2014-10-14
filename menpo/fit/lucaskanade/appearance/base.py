from menpo.fit.lucaskanade.residual import SSD
from menpo.fit.lucaskanade.base import LucasKanade


class AppearanceLucasKanade(LucasKanade):

    def __init__(self, model, transform, eps=10**-6):
        super(AppearanceLucasKanade, self).__init__(SSD(), transform, eps=eps)

        # in appearance alignment, target image is aligned to appearance model
        self.appearance_model = model
        # by default, template is assigned to mean appearance
        self.template = model.mean
        # pre-compute
        self._set_up()
