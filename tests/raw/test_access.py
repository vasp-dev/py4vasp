import py4vasp.raw as raw
from py4vasp.raw._access import DEFAULT_FILE
from dataclasses import fields
import pytest
from unittest.mock import patch, call


@pytest.fixture
def mock_access(complex_schema):
    schema, sources = complex_schema
    with patch("h5py.File") as mock_file:
        h5f = mock_file.return_value.__enter__.return_value
        h5f.get.side_effect = mock_read_result
        with patch("py4vasp.raw._access.schema", schema):
            yield mock_file, sources


def mock_read_result(key):
    return f"read {key}"


def test_access_quantity(mock_access):
    quantity = "optional_argument"
    mock_file, sources = mock_access
    source = sources[quantity]["default"]
    with raw.access(quantity) as opt_arg:
        mock_file.assert_called_once_with(DEFAULT_FILE, "r")
        h5f = mock_file.return_value.__enter__.return_value
        h5f.get.assert_has_calls(expected_calls(source), any_order=True)
        assert opt_arg.mandatory == mock_read_result(source.data.mandatory)
        assert opt_arg.optional == mock_read_result(source.data.optional)


def test_access_other_file(mock_access):
    quantity = "simple"
    mock_file, sources = mock_access
    source = sources[quantity]["default"]
    with raw.access(quantity) as simple:
        mock_file.assert_called_once_with(source.file, "r")
        h5f = mock_file.return_value.__enter__.return_value
        h5f.get.assert_has_calls(expected_calls(source), any_order=True)
        assert simple.foo == mock_read_result(source.data.foo)
        assert simple.bar == mock_read_result(source.data.bar)


def test_access_optional_argument(mock_access):
    quantity = "optional_argument"
    mock_file, sources = mock_access
    source = sources[quantity]["mandatory"]
    with raw.access(quantity, source="mandatory") as opt_arg:
        mock_file.assert_called_once_with(DEFAULT_FILE, "r")
        h5f = mock_file.return_value.__enter__.return_value
        h5f.get.assert_has_calls(expected_calls(source), any_order=True)
        assert opt_arg.mandatory == mock_read_result(source.data.mandatory)
        assert opt_arg.optional == None


def expected_calls(source):
    for field in fields(source.data):
        yield from expected_call(source.data, field)


def expected_call(data, field):
    key = getattr(data, field.name)
    if key is not None:
        yield call(key)
