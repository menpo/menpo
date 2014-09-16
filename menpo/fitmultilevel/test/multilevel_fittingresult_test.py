from mock import MagicMock
from menpo.fitmultilevel.fittingresult import MultilevelFittingResult


def test_multilevel_fittingresult_as_serialized():
    image = MagicMock()
    multiple_fitter = MagicMock()
    fitting_results = [MagicMock()]
    affine_correction = MagicMock()
    gt_shape = MagicMock()
    fr = MultilevelFittingResult(image, multiple_fitter, fitting_results,
                                 affine_correction, gt_shape=gt_shape)
    s_fr = fr.as_serializable()

    image.copy.assert_called_once()
    fitting_results[0].as_serialized.assert_called_once()
    affine_correction.copy.assert_called_once()
    gt_shape.copy.assert_called_once()
