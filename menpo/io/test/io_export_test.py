import numpy as np
import sys
from numpy.testing import assert_allclose
import os
from pathlib import PosixPath, WindowsPath, Path
from mock import patch, PropertyMock, MagicMock
from nose.tools import raises


import menpo.io as mio
from menpo.io.utils import _norm_path
from menpo.image import Image
from menpo.io.output.pickle import pickle_paths_as_pure


builtins_str = '__builtin__' if sys.version_info[0] == 2 else 'builtins'

test_lg = mio.import_landmark_file(mio.data_path_to('breakingbad.pts'))
nan_lg = test_lg.copy()
nan_lg.lms.points[0, :] = np.nan
test_img = Image(np.random.random([100, 100]))
fake_path = '/tmp/test.fake'


@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
@patch('menpo.io.output.base.Path.open')
def test_export_filepath_overwrite_exists(mock_open, exists, landmark_types):
    exists.return_value = True
    mio.export_landmark_file(test_lg, fake_path, overwrite=True)
    mock_open.assert_called_with('wb')
    landmark_types.__getitem__.assert_called_with('.fake')
    export_function = landmark_types.__getitem__.return_value
    assert export_function.call_count == 1


@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
@patch('menpo.io.output.base.Path.open')
def test_export_filepath_no_overwrite(mock_open, exists, landmark_types):
    exists.return_value = False
    mio.export_landmark_file(test_lg, fake_path)
    mock_open.assert_called_with('wb')
    landmark_types.__getitem__.assert_called_with('.fake')
    export_function = landmark_types.__getitem__.return_value
    assert export_function.call_count == 1


@raises(ValueError)
@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
@patch('menpo.io.output.base.Path.open')
def test_export_filepath_wrong_extension(mock_open, exists, landmark_types):
    exists.return_value = False
    mio.export_landmark_file(test_lg, fake_path, extension='pts')
    mock_open.assert_called_with('wb')


@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
@patch('menpo.io.output.base.Path.open')
def test_export_filepath_explicit_ext_no_dot(mock_open, exists, landmark_types):
    exists.return_value = False
    mio.export_landmark_file(test_lg, fake_path, extension='fake')
    mock_open.assert_called_with('wb')
    landmark_types.__getitem__.assert_called_with('.fake')
    export_function = landmark_types.__getitem__.return_value
    assert export_function.call_count == 1


@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
@patch('menpo.io.output.base.Path.open')
def test_export_filepath_explicit_ext_dot(mock_open, exists, landmark_types):
    exists.return_value = False
    mio.export_landmark_file(test_lg, fake_path, extension='.fake')
    mock_open.assert_called_with('wb')
    landmark_types.__getitem__.assert_called_with('.fake')
    export_function = landmark_types.__getitem__.return_value
    assert export_function.call_count == 1


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
    assert mock_open.name.call_count == 1


@raises(ValueError)
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_file_handle_file_extension_not_match_dot(mock_open, exists):
    exists.return_value = False
    with open(fake_path) as f:
        type(f).name = PropertyMock(return_value=fake_path)
        mio.export_landmark_file(test_lg, f, extension='.pts')
    assert mock_open.name.call_count == 1


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
    landmark_types.__getitem__.assert_called_with('.fake')
    export_function = landmark_types.__getitem__.return_value
    assert export_function.call_count == 1


@patch('menpo.io.output.base.landmark_types')
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_file_handle_file_non_file_buffer(mock_open, exists,
                                                 landmark_types):
    exists.return_value = False
    with open(fake_path) as f:
        del f.name  # Equivalent to raising an AttributeError side effect
        mio.export_landmark_file(test_lg, f, extension='fake')
    landmark_types.__getitem__.assert_called_with('.fake')
    export_function = landmark_types.__getitem__.return_value
    assert export_function.call_count == 1


@patch('menpo.io.output.landmark.json.dump')
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_landmark_ljson(mock_open, exists, json_dump):
    exists.return_value = False
    fake_path = '/fake/fake.ljson'
    with open(fake_path) as f:
        type(f).name = PropertyMock(return_value=fake_path)
        mio.export_landmark_file(test_lg, f, extension='ljson')
    assert json_dump.call_count == 1

@patch('menpo.io.output.landmark.json.dump')
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_landmark_ljson_3d(mock_open, exists, json_dump):
    exists.return_value = False
    fake_path = '/fake/fake3d.ljson'
    test3d_lg = test_lg.copy()
    fake_z_points = np.random.random(test3d_lg.lms.points.shape[0])
    test3d_lg.lms.points = np.concatenate([
        test3d_lg.lms.points, fake_z_points[..., None]], axis=-1)

    with open(fake_path) as f:
        type(f).name = PropertyMock(return_value=fake_path)
        mio.export_landmark_file(test3d_lg, f, extension='ljson')

    assert json_dump.call_count == 1
    json_points = np.array(json_dump.call_args[0][0]['landmarks']['points'])
    assert_allclose(json_points[:, -1], fake_z_points)

@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_landmark_ljson_nan_values(mock_open, exists):
    exists.return_value = False
    fake_path = '/fake/fake.ljson'
    with open(fake_path) as f:
        type(f).name = PropertyMock(return_value=fake_path)
        mio.export_landmark_file(nan_lg, f, extension='ljson')

    # This is a bit ugly, but we parse the write calls to check that json
    # wrote null values
    first_null = mock_open.mock_calls[97][1][0][1:].strip()
    second_null = mock_open.mock_calls[98][1][0][1:].strip()
    assert first_null == 'null'
    assert second_null == 'null'


@patch('menpo.io.output.landmark.np.savetxt')
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_landmark_pts(mock_open, exists, save_txt):
    exists.return_value = False
    fake_path = '/fake/fake.pts'
    with open(fake_path) as f:
        type(f).name = PropertyMock(return_value=fake_path)
        mio.export_landmark_file(test_lg, f, extension='pts')
    assert save_txt.call_count == 1


@patch('menpo.image.base.PILImage')
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_image_jpg(mock_open, exists, PILImage):
    exists.return_value = False
    fake_path = '/fake/fake.jpg'
    with open(fake_path) as f:
        type(f).name = PropertyMock(return_value=fake_path)
        mio.export_image(test_img, f, extension='jpg')
    assert PILImage.fromarray.return_value.save.call_count == 1


@patch('imageio.get_writer')
@patch('menpo.io.output.base.Path.exists')
def test_export_video_avi(exists, fake_writer):
    exists.return_value = False
    fake_path = Path('/fake/fake.avi')
    mio.export_video([test_img, test_img], fake_path, extension='avi')
    assert fake_writer.return_value.append_data.call_count == 2


@patch('imageio.get_writer')
@patch('menpo.io.output.base.Path.exists')
def test_export_video_gif(exists, fake_writer):
    exists.return_value = False
    fake_path = Path('/fake/fake.gif')
    mio.export_video([test_img, test_img], fake_path, extension='gif')
    assert fake_writer.return_value.append_data.call_count == 2


@patch('menpo.io.output.pickle.pickle.dump')
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(__name__), create=True)
def test_export_pickle(mock_open, exists, pickle_dump):
    exists.return_value = False
    fake_path = '/fake/fake.pkl'
    with open(fake_path) as f:
        type(f).name = PropertyMock(return_value=fake_path)
        mio.export_pickle(test_lg, f)
    assert pickle_dump.call_count == 1


@patch('menpo.io.output.pickle.pickle.dump')
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(builtins_str))
def test_export_pickle_with_path_uses_open(mock_open, exists, pickle_dump):
    exists.return_value = False
    fake_path = str(_norm_path('fake.pkl.gz'))
    mock_open_enter = MagicMock()
    # Make sure the name attribute returns the path
    mock_open_enter.__enter__.return_value.configure_mock(name=fake_path)
    mock_open.return_value = mock_open_enter
    mio.export_pickle(test_lg, fake_path)
    assert pickle_dump.call_count == 1
    mock_open.assert_called_with(fake_path, 'wb')


@patch('menpo.io.output.pickle.pickle.dump')
@patch('menpo.io.output.base.Path.exists')
@patch('{}.open'.format(builtins_str))
def test_export_pickle_with_path_expands_vars(mock_open, exists, pickle_dump):
    exists.return_value = False
    fake_path = '~/fake/fake.pkl.gz'
    mock_open_enter = MagicMock()
    # Make sure the name attribute returns the path
    mock_open_enter.__enter__.return_value.configure_mock(name=fake_path)
    mock_open.return_value = mock_open_enter
    mio.export_pickle(test_lg, fake_path)
    assert pickle_dump.call_count == 1
    expected_path = os.path.join(os.path.expanduser('~'), 'fake', 'fake.pkl.gz')
    mock_open.assert_called_with(expected_path, 'wb')


def test_pickle_paths_as_pure_switches_reduce_method_on_path():
    prev_reduce = Path.__reduce__
    with pickle_paths_as_pure():
        assert prev_reduce != Path.__reduce__
    assert prev_reduce == Path.__reduce__


def test_pickle_paths_as_pure_switches_reduce_method_on_posix_path():
    prev_reduce = PosixPath.__reduce__
    with pickle_paths_as_pure():
        assert prev_reduce != PosixPath.__reduce__
    assert prev_reduce == PosixPath.__reduce__


def test_pickle_paths_as_pure_switches_reduce_method_on_windows_path():
    prev_reduce = WindowsPath.__reduce__
    with pickle_paths_as_pure():
        assert prev_reduce != WindowsPath.__reduce__
    assert prev_reduce == WindowsPath.__reduce__


def test_pickle_paths_as_pure_cleans_up_on_exception():
    prev_reduce = Path.__reduce__
    try:
        with pickle_paths_as_pure():
            raise ValueError()
    except ValueError:
        assert prev_reduce == Path.__reduce__  # ensure we clean up
