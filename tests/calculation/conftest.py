# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import inspect
from unittest.mock import MagicMock, patch

import pytest

from py4vasp import exception
from py4vasp._util import convert, import_

formatters = import_.optional("IPython.core.formatters")

TEST_FILENAME = "read_data_from_this_file"
SELECTION = "alternative"


@pytest.fixture
def mock_schema():
    mock = MagicMock()
    mock.selections.return_value = ("default", SELECTION)
    with patch("py4vasp._raw.definition.schema", mock):
        yield mock


@pytest.fixture
def check_factory_methods(mock_schema, not_core):
    def inner(cls, data, parameters={}):
        instance = cls.from_path()
        check_instance_accesses_data(instance, data, parameters)
        instance = cls.from_file(TEST_FILENAME)
        check_instance_accesses_data(instance, data, parameters, file=TEST_FILENAME)

    return inner


def check_instance_accesses_data(instance, data, parameters, file=None):
    failed = []
    for name, method in inspect.getmembers(instance, inspect.ismethod):
        if should_test_method(name, parameters):
            kwargs = parameters.get(name, {})
            check_method_accesses_data(data, method, file, **kwargs)


def should_test_method(name, parameters):
    if name in parameters:
        return True
    if name in ("__str__", "_repr_html_"):
        return True
    if name.startswith("from") or name.startswith("_"):
        return False
    if name == "to_image":  # would have side effects
        return False
    if name == "to_csv":
        return False
    return True


def check_method_accesses_data(data, method_under_test, file, **kwargs):
    quantity = convert.quantity_name(data.__class__.__name__)
    with patch("py4vasp.raw.access") as mock_access:
        mock_access.return_value.__enter__.side_effect = lambda *_: data
        execute_method(method_under_test, **kwargs)
        check_mock_called(mock_access, quantity, file)
        mock_access.reset_mock()
        if "selection" in kwargs:
            kwargs = kwargs.copy()
            kwargs.pop("selection")
        execute_method(method_under_test, selection=SELECTION, **kwargs)
        check_mock_called(mock_access, quantity, file, selection=SELECTION)


def execute_method(method_under_test, **kwargs):
    try:
        method_under_test(**kwargs)
    except (exception.NotImplemented, exception.IncorrectUsage, exception.DataMismatch):
        # ignore py4vasp error
        pass


def check_mock_called(mock_access, quantity, file, selection=None):
    mock_access.assert_called_once()
    args, kwargs = mock_access.call_args
    assert (quantity,) == args
    assert kwargs.get("selection") == selection
    assert kwargs.get("file") == file


@pytest.fixture
def format_(not_core):
    return formatters.DisplayFormatter().format
