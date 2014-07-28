from nose.tools import raises
import numpy as np
import menpo.io as mio
from mock import patch, PropertyMock
from collections import OrderedDict
from menpo.landmark import LandmarkGroup
from menpo.shape import PointCloud
from menpo.image import Image


pc = PointCloud(np.random.random([100, 3]))
test_lg = LandmarkGroup(
    pc, OrderedDict([('all', np.ones(pc.n_points, dtype=np.bool))]))
test_img = Image(np.random.random([100, 100]))


@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
@patch('menpo.io.output.base.Path.open')
def test_export_filepath_overwrite_exists(mock_open, exists, landmark_types):
    exists.return_value = True
    mio.export_landmark_file(test_lg, '/tmp/test.fake', overwrite=True)
    mock_open.assert_called_once_with('wb')
    landmark_types.__getitem__.assert_called_once_with('.fake')
    export_function = landmark_types.__getitem__.return_value
    export_function.assert_called_once()


@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
@patch('menpo.io.output.base.Path.open')
def test_export_filepath_no_overwrite(mock_open, exists, landmark_types):
    exists.return_value = False
    mio.export_landmark_file(test_lg, '/tmp/test.fake')
    mock_open.assert_called_once_with('wb')
    landmark_types.__getitem__.assert_called_once_with('.fake')
    export_function = landmark_types.__getitem__.return_value
    export_function.assert_called_once()


@raises(ValueError)
@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
@patch('menpo.io.output.base.Path.open')
def test_export_filepath_wrong_extension(mock_open, exists, landmark_types):
    exists.return_value = False
    mio.export_landmark_file(test_lg, '/tmp/test.fake', export_extension='pts')
    mock_open.assert_called_once_with('wb')


@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
@patch('menpo.io.output.base.Path.open')
def test_export_filepath_explicit_ext_no_dot(mock_open, exists, landmark_types):
    exists.return_value = False
    mio.export_landmark_file(test_lg, '/tmp/test.fake',
                             export_extension='fake')
    mock_open.assert_called_once_with('wb')
    landmark_types.__getitem__.assert_called_once_with('.fake')
    export_function = landmark_types.__getitem__.return_value
    export_function.assert_called_once()


@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
@patch('menpo.io.output.base.Path.open')
def test_export_filepath_explicit_ext_dot(mock_open, exists, landmark_types):
    exists.return_value = False
    mio.export_landmark_file(test_lg, '/tmp/test.fake',
                             export_extension='.fake')
    mock_open.assert_called_once_with('wb')
    landmark_types.__getitem__.assert_called_once_with('.fake')
    export_function = landmark_types.__getitem__.return_value
    export_function.assert_called_once()


@raises(ValueError)
@patch('menpo.io.output.base.Path.exists')
def test_export_filepath_no_overwrite_exists(exists):
    exists.return_value = True
    mio.export_landmark_file(test_lg, '/tmp/test.fake')


@raises(ValueError)
@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
def test_export_unspported_extension(exists, landmark_types):
    exists.return_value = False
    landmark_types.__getitem__.side_effect = KeyError
    mio.export_landmark_file(test_lg, '/tmp/test.fake')


@raises(ValueError)
@patch('{}.open'.format(__name__), create=True)
def test_export_file_handle_file_extension_None(mock_open):
    with open('/tmp/test.fake') as f:
        mio.export_landmark_file(test_lg, f)


@raises(ValueError)
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_file_handle_file_extension_not_match_no_dot(mock_open, exists):
    exists.return_value = False
    with open('/tmp/test.fake') as f:
        type(f).name = PropertyMock(return_value='/tmp/test.fake')
        mio.export_landmark_file(test_lg, f, export_extension='pts')
    mock_open.name.assert_called_once()


@raises(ValueError)
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_file_handle_file_extension_not_match_dot(mock_open, exists):
    exists.return_value = False
    with open('/tmp/test.fake') as f:
        type(f).name = PropertyMock(return_value='/tmp/test.fake')
        mio.export_landmark_file(test_lg, f, export_extension='.pts')
    mock_open.name.assert_called_once()


@raises(ValueError)
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_file_handle_file_exists(mock_open, exists):
    exists.return_value = True
    with open('/tmp/test.fake') as f:
        type(f).name = PropertyMock(return_value='/tmp/test.fake')
        mio.export_landmark_file(test_lg, f,
                                 export_extension='fake')


@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_file_handle_file_exists_overwrite(mock_open, exists,
                                                  landmark_types):
    exists.return_value = True
    with open('/tmp/test.fake') as f:
        type(f).name = PropertyMock(return_value='/tmp/test.fake')
        mio.export_landmark_file(test_lg, f, overwrite=True,
                                 export_extension='fake')
    landmark_types.__getitem__.assert_called_once_with('.fake')
    export_function = landmark_types.__getitem__.return_value
    export_function.assert_called_once()


@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_file_handle_file_non_file_buffer(mock_open, exists,
                                                 landmark_types):
    exists.return_value = False
    with open('/tmp/test.fake') as f:
        del f.name  # Equivalent to raising an AttributeError side effect
        mio.export_landmark_file(test_lg, f, export_extension='fake')
    landmark_types.__getitem__.assert_called_once_with('.fake')
    export_function = landmark_types.__getitem__.return_value
    export_function.assert_called_once()


@patch('menpo.io.output.landmark.json.dump')
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_landmark_ljson(mock_open, exists, json_dump):
    exists.return_value = False
    with open('/tmp/test.ljson') as f:
        type(f).name = PropertyMock(return_value='/tmp/test.ljson')
        mio.export_landmark_file(test_lg, f, export_extension='ljson')
    json_dump.assert_called_once()


@patch('menpo.io.output.landmark.np.savetxt')
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_landmark_pts(mock_open, exists, save_txt):
    exists.return_value = False
    with open('/tmp/test.pts') as f:
        type(f).name = PropertyMock(return_value='/tmp/test.pts')
        mio.export_landmark_file(test_lg, f, export_extension='pts')
    save_txt.assert_called_once()


@patch('menpo.image.base.PILImage')
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_image_jpg(mock_open, exists, PILImage):
    exists.return_value = False
    with open('/tmp/test.jpg') as f:
        type(f).name = PropertyMock(return_value='/tmp/test.jpg')
        mio.export_image(test_img, f, export_extension='jpg')
    PILImage.save.assert_called_once()