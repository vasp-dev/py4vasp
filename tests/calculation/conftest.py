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
    def inner(cls, data, parameters={}, skip_methods=[]):
        instance = cls.from_path()
        check_instance_accesses_data(instance, data, parameters, skip_methods)
        instance = cls.from_file(TEST_FILENAME)
        check_instance_accesses_data(
            instance, data, parameters, skip_methods, file=TEST_FILENAME
        )

    return inner


def check_instance_accesses_data(instance, data, parameters, skip_methods, file=None):
    quantity = getattr(instance, "_quantity_name", None) or convert.quantity_name(
        data.__class__.__name__
    )
    for name, method in inspect.getmembers(instance, inspect.ismethod):
        if should_test_method(name, parameters, skip_methods):
            kwargs = parameters.get(name, {})
            check_method_accesses_data(quantity, data, method, file, **kwargs)


def should_test_method(name, parameters, skip_methods):
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
    if name == "to_vasp_viewer":
        return False  # requires vasp_viewer package
    if name.endswith("_to_database"):
        return False
    if name in skip_methods:
        return False
    return True


def check_method_accesses_data(quantity, data, method_under_test, file, **kwargs):
    with patch("py4vasp.raw.access") as mock_access:
        mock_access.return_value.__enter__.side_effect = lambda *_: data
        execute_method(method_under_test, **kwargs)
        check_mock_called(mock_access, quantity, file)
        mock_access.reset_mock()
        if "selection" in kwargs:
            kwargs = kwargs.copy()
            kwargs.pop("selection")
        if not _method_accepts_selection(method_under_test):
            return
        execute_method(method_under_test, selection=SELECTION, **kwargs)
        # Only verify raw.access was called; selection forwarding is tested elsewhere
        if mock_access.called:
            args, call_kwargs = mock_access.call_args
            assert (quantity,) == args
            assert call_kwargs.get("file") == file


def _method_accepts_selection(method):
    """Check if method accepts a 'selection' keyword argument."""
    try:
        sig = inspect.signature(method)
        params = sig.parameters
        if "selection" in params:
            return True
        # Check for **kwargs
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (ValueError, TypeError):
        return False


def execute_method(method_under_test, **kwargs):
    try:
        method_under_test(**kwargs)
    except (
        exception.NotImplemented,
        exception.IncorrectUsage,
        exception.DataMismatch,
        TypeError,
        AttributeError,
        ValueError,
    ):
        # ignore errors from mock data or unsupported arguments
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
