import py4vasp.raw as raw
from py4vasp.raw._access import DEFAULT_FILE
import pytest
from unittest.mock import patch


@pytest.fixture
def mock_access(complex_schema):
    schema, sources = complex_schema
    with patch("h5py.File") as mock_file:
        with patch("py4vasp.raw._access.schema", schema):
            yield mock_file, sources


def test_access(mock_access):
    mock_file, sources = mock_access
    with raw.access("optional_argument") as opt_arg:
        mock_file.assert_called_once_with(DEFAULT_FILE, "r")
