"""MenpoMissingDependencyError says what really went wrong, and only tells you to install a package that is missing."""

import importlib
import subprocess
import sys

import pytest

from menpo.base import MenpoMissingDependencyError
from menpo.visualize import Menpo3dMissingError


def failure(statement):
    """The ImportError that a failing import statement raises."""
    with pytest.raises(ImportError) as info:
        exec(statement, {})
    return info.value


def test_a_package_that_is_not_installed_is_named_with_the_pip_command():
    error = MenpoMissingDependencyError(failure("import not_installed_package_xyz"))
    text = str(error)
    assert error.missing_name == "not_installed_package_xyz"
    assert "install the 'not_installed_package_xyz' package" in text
    assert "pip install not_installed_package_xyz" in text
    assert "conda" not in text
    # the original error is there, whatever else it says
    assert "No module named 'not_installed_package_xyz'" in text


def test_a_submodule_of_a_package_that_is_not_installed_names_the_package():
    error = MenpoMissingDependencyError(
        failure("import not_installed_package_xyz.sub.module")
    )
    assert error.missing_name == "not_installed_package_xyz"
    assert "pip install not_installed_package_xyz\n" in str(error)


def test_a_missing_submodule_of_an_installed_package_is_not_a_package_to_install():
    error = MenpoMissingDependencyError(failure("import numpy.not_a_module_xyz"))
    text = str(error)
    assert error.missing_name is None
    assert "pip install" not in text and "install the" not in text
    assert "No module named 'numpy.not_a_module_xyz'" in text


def test_a_name_that_an_installed_package_does_not_have_is_not_a_package_to_install():
    error = MenpoMissingDependencyError(failure("from numpy import not_a_name_xyz"))
    text = str(error)
    assert error.missing_name is None and "pip install" not in text
    assert "cannot import name 'not_a_name_xyz'" in text


@pytest.mark.parametrize(
    "error",
    [
        ImportError(),
        ImportError("libGL.so.1: cannot open shared object file"),
        ImportError("DLL load failed while importing _imaging"),
    ],
)
def test_an_import_error_that_names_no_module_shows_the_error_and_no_package(error):
    result = MenpoMissingDependencyError(error)
    text = str(result)
    assert result.missing_name is None
    assert "None" not in text and "pip install" not in text
    assert "ImportError: " + str(error) in text


def test_a_package_name_can_be_given_instead_of_an_error():
    error = MenpoMissingDependencyError("cyvlfeat")
    assert error.missing_name == "cyvlfeat"
    assert "pip install cyvlfeat" in str(error)
    assert "The import failed with" not in str(error)  # there was no error


def test_it_is_an_import_error_and_its_text_is_the_message():
    error = MenpoMissingDependencyError("cyvlfeat")
    assert isinstance(error, ImportError)
    assert str(error) == error.message


def test_a_module_that_cannot_be_looked_up_is_trusted_to_be_missing(monkeypatch):
    def broken(name):
        raise ValueError("__spec__ is None")

    monkeypatch.setattr(importlib.util, "find_spec", broken)
    error = MenpoMissingDependencyError(failure("import not_installed_package_xyz"))
    assert error.missing_name == "not_installed_package_xyz"


def test_the_menpo3d_error_keeps_what_it_says_and_adds_why_it_is_needed():
    error = Menpo3dMissingError(failure("import not_installed_package_xyz"))
    text = str(error)
    assert "pip install not_installed_package_xyz" in text
    assert "required in order to use the 'menpo3d' package" in text
    assert error.missing_name == "not_installed_package_xyz"
    bare = Menpo3dMissingError(
        ImportError("libGL.so.1: cannot open shared object file")
    )
    assert "libGL.so.1" in str(bare) and "'menpo3d' package" in str(bare)


def test_what_the_vlfeat_features_say_when_cyvlfeat_is_not_installed():
    # a fresh interpreter in which cyvlfeat cannot be imported
    code = (
        "import sys; sys.modules['cyvlfeat'] = None\n"
        "import menpo.feature.optional.vlfeat"
    )
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "MenpoMissingDependencyError" in result.stderr
    assert "pip install cyvlfeat" in result.stderr
    assert "conda" not in result.stderr


def test_only_a_module_not_found_error_means_a_package_is_missing():
    # a plain ImportError that happens to carry a name (as "cannot import name" does)
    error = ImportError("something else went wrong", name="not_installed_package_xyz")
    result = MenpoMissingDependencyError(error)
    assert result.missing_name is None and "pip install" not in str(result)
    assert "something else went wrong" in str(result)
