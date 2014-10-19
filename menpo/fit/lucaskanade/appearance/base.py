from menpo.fit.lucaskanade.residual import SSD
from menpo.fit.lucaskanade.base import LucasKanade


class AppearanceLucasKanade(LucasKanade):

    def __init__(self, model, transform, eps=10**-6):
        # Note that the only supported residual for Appearance LK is SSD.
        # This is because, in general, we don't know how to take the appropriate
        # derivatives for arbitrary residuals with (for instance) a project out
        # AAM.
        # See https://github.com/menpo/menpo/issues/130 for details.
        super(AppearanceLucasKanade, self).__init__(SSD(), transform, eps=eps)

        # in appearance alignment, target image is aligned to appearance model
        self.appearance_model = model
        # by default, template is assigned to mean appearance
        self.template = model.mean()
        # pre-compute
        self._set_up()
