# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import inspect
from unittest.mock import patch

import pytest
from IPython.core.formatters import DisplayFormatter

from py4vasp import raw, exception
from py4vasp._util import convert, version

TEST_FILENAME = "read_data_from_this_file"


@pytest.fixture
def check_factory_methods():
    def inner(cls, data):
        instance = cls.from_path()
        check_instance_accesses_data(instance, data)
        instance = cls.from_file(TEST_FILENAME)
        check_instance_accesses_data(instance, data, file=TEST_FILENAME)

    return inner


def check_instance_accesses_data(instance, data, file=None):
    failed = []
    for name, method in inspect.getmembers(instance, inspect.ismethod):
        if should_test_method(name):
            try:
                check_method_accesses_data(data, method, file)
            except (AttributeError, AssertionError):
                failed.append(name)
    if failed:
        message = (
            f"The method(s) {', '.join(failed)} do not load the data from file."
            " The most likely issue is a missing @base.data_access decorator."
        )
        raise AssertionError(message)


def should_test_method(name):
    if name in ("__str__", "_repr_html_"):
        return True
    if name.startswith("from") or name.startswith("_"):
        return False
    if name == "to_image":  # would have side effects
        return False
    return True


def check_method_accesses_data(data, method_under_test, file):
    quantity = convert.to_snakecase(data.__class__.__name__)
    with patch("py4vasp.raw.access") as mock_access:
        mock_access.return_value.__enter__.side_effect = lambda *_: data
        execute_method(method_under_test)
        check_mock_called(mock_access, quantity, file)
        mock_access.reset_mock()
        execute_method(method_under_test, source="choice")
        check_mock_called(mock_access, quantity, file, source="choice")


def execute_method(method_under_test, **kwargs):
    try:
        method_under_test(**kwargs)
    except (exception.NotImplemented, exception.IncorrectUsage):
        # ignore py4vasp error
        pass


def check_mock_called(mock_access, quantity, file, source=None):
    mock_access.assert_called_once()
    args, kwargs = mock_access.call_args
    assert (quantity,) == args
    assert kwargs.get("source") == source
    assert kwargs.get("file") == file


@pytest.fixture
def outdated_version():
    return raw.RawVersion(version.minimal_vasp_version.major - 1)


@pytest.fixture
def format_():
    return DisplayFormatter().format
