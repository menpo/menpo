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
fake_path = '/tmp/test.fake'
test_obj = mio.import_builtin_asset('james.obj')


@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
@patch('menpo.io.output.base.Path.open')
def test_export_filepath_overwrite_exists(mock_open, exists, landmark_types):
    exists.return_value = True
    mio.export_landmark_file(fake_path, test_lg, overwrite=True)
    mock_open.assert_called_once_with('wb')
    landmark_types.__getitem__.assert_called_once_with('.fake')
    export_function = landmark_types.__getitem__.return_value
    export_function.assert_called_once()


@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
@patch('menpo.io.output.base.Path.open')
def test_export_filepath_no_overwrite(mock_open, exists, landmark_types):
    exists.return_value = False
    mio.export_landmark_file(fake_path, test_lg)
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
    mio.export_landmark_file(fake_path, test_lg, extension='pts')
    mock_open.assert_called_once_with('wb')


@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
@patch('menpo.io.output.base.Path.open')
def test_export_filepath_explicit_ext_no_dot(mock_open, exists, landmark_types):
    exists.return_value = False
    mio.export_landmark_file(fake_path, test_lg, extension='fake')
    mock_open.assert_called_once_with('wb')
    landmark_types.__getitem__.assert_called_once_with('.fake')
    export_function = landmark_types.__getitem__.return_value
    export_function.assert_called_once()


@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
@patch('menpo.io.output.base.Path.open')
def test_export_filepath_explicit_ext_dot(mock_open, exists, landmark_types):
    exists.return_value = False
    mio.export_landmark_file(fake_path, test_lg, extension='.fake')
    mock_open.assert_called_once_with('wb')
    landmark_types.__getitem__.assert_called_once_with('.fake')
    export_function = landmark_types.__getitem__.return_value
    export_function.assert_called_once()


@raises(ValueError)
@patch('menpo.io.output.base.Path.exists')
def test_export_filepath_no_overwrite_exists(exists):
    exists.return_value = True
    mio.export_landmark_file(fake_path, test_lg)


@raises(ValueError)
@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
def test_export_unspported_extension(exists, landmark_types):
    exists.return_value = False
    landmark_types.__getitem__.side_effect = KeyError
    mio.export_landmark_file(fake_path, test_lg)


@raises(ValueError)
@patch('{}.open'.format(__name__), create=True)
def test_export_file_handle_file_extension_None(mock_open):
    with open(fake_path) as f:
        mio.export_landmark_file(f, test_lg, )


@raises(ValueError)
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_file_handle_file_extension_not_match_no_dot(mock_open, exists):
    exists.return_value = False
    with open(fake_path) as f:
        type(f).name = PropertyMock(return_value=fake_path)
        mio.export_landmark_file(f, test_lg, extension='pts')
    mock_open.name.assert_called_once()


@raises(ValueError)
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_file_handle_file_extension_not_match_dot(mock_open, exists):
    exists.return_value = False
    with open(fake_path) as f:
        type(f).name = PropertyMock(return_value=fake_path)
        mio.export_landmark_file(f, test_lg, extension='.pts')
    mock_open.name.assert_called_once()


@raises(ValueError)
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_file_handle_file_exists(mock_open, exists):
    exists.return_value = True
    with open(fake_path) as f:
        type(f).name = PropertyMock(return_value=fake_path)
        mio.export_landmark_file(f, test_lg, extension='fake')


@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_file_handle_file_exists_overwrite(mock_open, exists,
                                                  landmark_types):
    exists.return_value = True
    with open(fake_path) as f:
        type(f).name = PropertyMock(return_value=fake_path)
        mio.export_landmark_file(f, test_lg, overwrite=True, extension='fake')
    landmark_types.__getitem__.assert_called_once_with('.fake')
    export_function = landmark_types.__getitem__.return_value
    export_function.assert_called_once()


@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_file_handle_file_non_file_buffer(mock_open, exists,
                                                 landmark_types):
    exists.return_value = False
    with open(fake_path) as f:
        del f.name  # Equivalent to raising an AttributeError side effect
        mio.export_landmark_file(f, test_lg, extension='fake')
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
        mio.export_landmark_file(f, test_lg, extension='ljson')
    json_dump.assert_called_once()


@patch('menpo.io.output.landmark.np.savetxt')
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_landmark_pts(mock_open, exists, save_txt):
    exists.return_value = False
    with open('/tmp/test.pts') as f:
        type(f).name = PropertyMock(return_value='/tmp/test.pts')
        mio.export_landmark_file(f, test_lg, extension='pts')
    save_txt.assert_called_once()


@patch('menpo.image.base.PILImage')
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_image_jpg(mock_open, exists, PILImage):
    exists.return_value = False
    with open('/tmp/test.jpg') as f:
        type(f).name = PropertyMock(return_value='/tmp/test.jpg')
        mio.export_image(f, test_img, extension='jpg')
    PILImage.save.assert_called_once()


@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_obj_untextured(mock_open, exists):
    exists.return_value = False
    with open('/tmp/test.obj') as f:
        type(f).name = PropertyMock(return_value='/tmp/test.obj')
        mio.export_mesh(f, test_obj, extension='obj')


@patch('menpo.image.base.PILImage')
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_obj_textured(mock_open, exists, PILImage):
    exists.return_value = False
    mio.export_textured_mesh('/tmp/test.obj', test_obj, extension='obj')
    PILImage.save.assert_called_once()
