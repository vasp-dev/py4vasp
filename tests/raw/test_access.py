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
        h5f.get.side_effect = lambda key: f"read {key}"
        with patch("py4vasp.raw._access.schema", schema):
            yield mock_file, sources


def test_access_quantity(mock_access):
    key = "optional_argument"
    mock_file, sources = mock_access
    with raw.access(key) as opt_arg:
        mock_file.assert_called_once_with(DEFAULT_FILE, "r")
        h5f = mock_file.return_value.__enter__.return_value
        source = sources[key]["default"]
        h5f.get.assert_has_calls(expected_calls(source), any_order=True)
        assert opt_arg.mandatory == f"read {source.data.mandatory}"
        assert opt_arg.optional == f"read {source.data.optional}"


def expected_calls(source):
    return (expected_call(source.data, field) for field in fields(source.data))


def expected_call(data, field):
    return call(getattr(data, field.name))
