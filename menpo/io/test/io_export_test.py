import numpy as np
from mock import patch, PropertyMock
from nose.tools import raises

import menpo.io as mio
from menpo.image import Image


test_lg = mio.import_landmark_file(mio.data_path_to('breakingbad.pts'))
test_img = Image(np.random.random([100, 100]))
fake_path = '/tmp/test.fake'


@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
@patch('menpo.io.output.base.Path.open')
def test_export_filepath_overwrite_exists(mock_open, exists, landmark_types):
    exists.return_value = True
    mio.export_landmark_file(test_lg, fake_path, overwrite=True)
    mock_open.assert_called_once_with('wb')
    landmark_types.__getitem__.assert_called_once_with('.fake')
    export_function = landmark_types.__getitem__.return_value
    export_function.assert_called_once()


@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
@patch('menpo.io.output.base.Path.open')
def test_export_filepath_no_overwrite(mock_open, exists, landmark_types):
    exists.return_value = False
    mio.export_landmark_file(test_lg, fake_path)
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
    mio.export_landmark_file(test_lg, fake_path, extension='pts')
    mock_open.assert_called_once_with('wb')


@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
@patch('menpo.io.output.base.Path.open')
def test_export_filepath_explicit_ext_no_dot(mock_open, exists, landmark_types):
    exists.return_value = False
    mio.export_landmark_file(test_lg, fake_path, extension='fake')
    mock_open.assert_called_once_with('wb')
    landmark_types.__getitem__.assert_called_once_with('.fake')
    export_function = landmark_types.__getitem__.return_value
    export_function.assert_called_once()


@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
@patch('menpo.io.output.base.Path.open')
def test_export_filepath_explicit_ext_dot(mock_open, exists, landmark_types):
    exists.return_value = False
    mio.export_landmark_file(test_lg, fake_path, extension='.fake')
    mock_open.assert_called_once_with('wb')
    landmark_types.__getitem__.assert_called_once_with('.fake')
    export_function = landmark_types.__getitem__.return_value
    export_function.assert_called_once()


@raises(ValueError)
@patch('menpo.io.output.base.Path.exists')
def test_export_filepath_no_overwrite_exists(exists):
    exists.return_value = True
    mio.export_landmark_file(test_lg, fake_path)


@raises(ValueError)
@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
def test_export_unsupported_extension(exists, landmark_types):
    exists.return_value = False
    landmark_types.__getitem__.side_effect = KeyError
    mio.export_landmark_file(test_lg, fake_path)


@raises(ValueError)
@patch('{}.open'.format(__name__), create=True)
def test_export_file_handle_file_extension_None(mock_open):
    with open(fake_path) as f:
        mio.export_landmark_file(test_lg, f)


@raises(ValueError)
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_file_handle_file_extension_not_match_no_dot(mock_open, exists):
    exists.return_value = False
    with open(fake_path) as f:
        type(f).name = PropertyMock(return_value=fake_path)
        mio.export_landmark_file(test_lg, f, extension='pts')
    mock_open.name.assert_called_once()


@raises(ValueError)
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_file_handle_file_extension_not_match_dot(mock_open, exists):
    exists.return_value = False
    with open(fake_path) as f:
        type(f).name = PropertyMock(return_value=fake_path)
        mio.export_landmark_file(test_lg, f, extension='.pts')
    mock_open.name.assert_called_once()


@raises(ValueError)
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_file_handle_file_exists(mock_open, exists):
    exists.return_value = True
    with open(fake_path) as f:
        type(f).name = PropertyMock(return_value=fake_path)
        mio.export_landmark_file(test_lg, f, extension='fake')


@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_file_handle_file_exists_overwrite(mock_open, exists,
                                                  landmark_types):
    exists.return_value = True
    with open(fake_path) as f:
        type(f).name = PropertyMock(return_value=fake_path)
        mio.export_landmark_file(test_lg, f, overwrite=True, extension='fake')
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
        mio.export_landmark_file(test_lg, f, extension='fake')
    landmark_types.__getitem__.assert_called_once_with('.fake')
    export_function = landmark_types.__getitem__.return_value
    export_function.assert_called_once()


@patch('menpo.io.output.landmark.json.dump')
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_landmark_ljson(mock_open, exists, json_dump):
    exists.return_value = False
    fake_path = '/fake/fake.ljson'
    with open(fake_path) as f:
        type(f).name = PropertyMock(return_value=fake_path)
        mio.export_landmark_file(test_lg, f, extension='ljson')
    json_dump.assert_called_once()


@patch('menpo.io.output.landmark.np.savetxt')
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_landmark_pts(mock_open, exists, save_txt):
    exists.return_value = False
    fake_path = '/fake/fake.pts'
    with open(fake_path) as f:
        type(f).name = PropertyMock(return_value=fake_path)
        mio.export_landmark_file(test_lg, f, extension='pts')
    save_txt.assert_called_once()


@patch('menpo.image.base.PILImage')
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_image_jpg(mock_open, exists, PILImage):
    exists.return_value = False
    fake_path = '/fake/fake.jpg'
    with open(fake_path) as f:
        type(f).name = PropertyMock(return_value=fake_path)
        mio.export_image(test_img, f, extension='jpg')
    PILImage.save.assert_called_once()


@patch('menpo.io.output.pickle.pickle.dump')
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_pickle(mock_open, exists, pickle_dump):
    exists.return_value = False
    fake_path = '/fake/fake.pkl'
    with open(fake_path) as f:
        type(f).name = PropertyMock(return_value=fake_path)
        mio.export_pickle(test_lg, f)
    pickle_dump.assert_called_once()


@patch('menpo.io.output.pickle.pickle.dump')
@patch('menpo.io.output.base.Path.exists')
@patch('__builtin__.open')
def test_export_pickle_with_path_uses_open(mock_open, exists, pickle_dump):
    exists.return_value = False
    fake_path = '/fake/fake.pkl.gz'
    mio.export_pickle(test_lg, fake_path)
    pickle_dump.assert_called_once()
    mock_open.assert_called_once_with(fake_path, 'wb')
