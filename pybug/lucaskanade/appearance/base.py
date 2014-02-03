from pybug.lucaskanade.base import LucasKanade, LKFitting


class AppearanceLucasKanade(LucasKanade):

    def __init__(self, model, residual, transform,
                 interpolator='scipy', optimisation=('GN',), eps=10**-6):
        super(AppearanceLucasKanade, self).__init__(
            residual, transform,interpolator=interpolator,
            optimisation=optimisation, eps=eps)

        # in appearance alignment, target image is aligned to appearance model
        self.appearance_model = model
        # by default, template is assigned to mean appearance
        self.template = model.mean
        # pre-compute
        self._precompute()


class AppearanceLKFitting(LKFitting):

    def __init__(self, lk, image, parameters, costs, status,
                 error_type='norm_me'):
        super(AppearanceLKFitting, self).__init__(
            lk, image, parameters, costs, status, error_type)

    def view_appearance(self, figure_id=None, new_figure=False,
                        channels=None, **kwargs):
        raise ValueError('view_apperance not implemented yet')
