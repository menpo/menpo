from os.path import sep as PATH_SEP
from mock import patch
from pathlib import Path
from nose.tools import raises


from menpo.io.input.base import _pathlib_glob_for_pattern
from menpo.io.output.base import _parse_and_validate_extension


@patch('menpo.io.input.base.Path.glob')
def test_glob_parse_contains_file_glob_no_sort(mock_glob):
    path = '/tmp/test.*'
    mock_glob.return_value = ['/tmp/test.test']
    result = list(_pathlib_glob_for_pattern(path, sort=False))
    assert len(result) == 1
    mock_glob.assert_called_with('test.*')


@patch('menpo.io.input.base.Path.glob')
def test_glob_parse_contains_dir_glob_no_sort(mock_glob):
    path = '/tmp/**/*'
    mock_glob.return_value = ['/tmp/a/test.test', '/tmp/b/test.test']
    result = list(_pathlib_glob_for_pattern(path, sort=False))
    assert len(result) == 2
    mock_glob.assert_called_with('**/*'.replace('/', PATH_SEP))


@patch('menpo.io.input.base.Path.glob')
def test_glob_parse_sort(mock_glob):
    path = '/tmp/**/*'
    mock_glob.return_value = ['/tmp/b/test.test', '/tmp/a/test.test']
    result = list(_pathlib_glob_for_pattern(path, sort=True))
    assert len(result) == 2
    assert result[0] == mock_glob.return_value[1]
    mock_glob.assert_called_with('**/*'.replace('/', PATH_SEP))


def test_parse_extension_given_extension():
    filepath = 'test.jpg'
    extension = filepath[-4:]
    parsed_extension = _parse_and_validate_extension(Path(filepath), extension,
                                                     {'.jpg': None})
    assert parsed_extension == extension


def test_parse_extension_no_given_extension():
    filepath = 'test.jpg'
    extension = filepath[-4:]
    parsed_extension = _parse_and_validate_extension(Path(filepath), None,
                                                     {'.jpg': None})
    assert parsed_extension == extension


def test_parse_double_extension_given_extension():
    filepath = 'test.pkl.gz'
    extension = filepath[-7:]
    parsed_extension = _parse_and_validate_extension(Path(filepath), extension,
                                                     {'.pkl.gz': None})
    assert parsed_extension == extension


def test_parse_double_extension_no_extension():
    filepath = 'test.pkl.gz'
    extension = filepath[-7:]
    parsed_extension = _parse_and_validate_extension(Path(filepath), None,
                                                     {'.pkl.gz': None})
    assert parsed_extension == extension


def test_parse_period_in_name_given_extension():
    filepath = 'test.1.jpg'
    extension = filepath[-4:]
    parsed_extension = _parse_and_validate_extension(Path(filepath), extension,
                                                     {'.jpg': None})
    assert parsed_extension == extension


def test_parse_period_in_name_no_extension():
    filepath = 'test.1.jpg'
    extension = filepath[-4:]
    parsed_extension = _parse_and_validate_extension(Path(filepath), None,
                                                     {'.jpg': None})
    assert parsed_extension == extension


def test_parse_period_in_name_double_given_extension():
    filepath = 'test.1.pkl.gz'
    extension = filepath[-7:]
    parsed_extension = _parse_and_validate_extension(Path(filepath), extension,
                                                     {'.pkl.gz': None})
    assert parsed_extension == extension


def test_parse_period_in_name_double_no_extension():
    filepath = 'test.1.pkl.gz'
    extension = filepath[-7:]
    parsed_extension = _parse_and_validate_extension(Path(filepath), extension,
                                                     {'.pkl.gz': None})
    assert parsed_extension == extension


@raises(ValueError)
def test_parse_unknown_extension_raises_ValueError():
    filepath = 'test.1.fake'
    _parse_and_validate_extension(Path(filepath), None, {'.jpg': None})


@raises(ValueError)
def test_parse_mot_matching_extension_raises_ValueError():
    filepath = 'test.jpg2'
    _parse_and_validate_extension(Path(filepath), '.jpg', {'.jpg': None})
