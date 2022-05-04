import py4vasp.raw as raw
from py4vasp.raw._access import DEFAULT_FILE
from dataclasses import fields
import pathlib
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
        check_single_file_access(mock_file, DEFAULT_FILE, source)
        assert opt_arg.mandatory == mock_read_result(source.data.mandatory)
        assert opt_arg.optional == mock_read_result(source.data.optional)


def test_access_other_file(mock_access):
    quantity = "simple"
    mock_file, sources = mock_access
    source = sources[quantity]["default"]
    with raw.access(quantity) as simple:
        check_single_file_access(mock_file, source.file, source)
        assert simple.foo == mock_read_result(source.data.foo)
        assert simple.bar == mock_read_result(source.data.bar)


def test_access_optional_argument(mock_access):
    quantity = "optional_argument"
    mock_file, sources = mock_access
    source = sources[quantity]["mandatory"]
    with raw.access(quantity, source="mandatory") as opt_arg:
        check_single_file_access(mock_file, DEFAULT_FILE, source)
        assert opt_arg.mandatory == mock_read_result(source.data.mandatory)
        assert opt_arg.optional == None


def test_access_with_link(mock_access):
    reference, file_calls, get_calls = linked_quantity_reference(mock_access)
    quantity = "with_link"
    mock_file, sources = mock_access
    source = sources[quantity]["default"]
    with raw.access(quantity) as with_link:
        file_calls += [call(pathlib.Path(DEFAULT_FILE), "r")]
        get_calls += list(expected_calls(source))
        check_file_access(mock_file, file_calls, get_calls)
        assert with_link.baz == mock_read_result(source.data.baz)
        assert with_link.simple == reference


def linked_quantity_reference(mock_access):
    quantity = "simple"
    mock_file, sources = mock_access
    source = sources[quantity]["default"]
    with raw.access(quantity) as simple:
        h5f = mock_file.return_value.__enter__.return_value
        result = simple, mock_file.call_args_list, h5f.get.call_args_list
    mock_file.reset_mock()
    return result


def test_access_open_once(mock_access):
    mock_file, sources = mock_access
    with raw.access("complex", source="mandatory") as complex:
        # open two different files
        assert mock_file.call_count == 2


def test_access_from_path(mock_access):
    quantity = "optional_argument"
    path = "pathname"
    mock_file, sources = mock_access
    source = sources[quantity]["default"]
    with raw.access(quantity, path=path) as opt_arg:
        check_single_file_access(mock_file, f"{path}/{DEFAULT_FILE}", source)
        assert opt_arg.mandatory == mock_read_result(source.data.mandatory)
        assert opt_arg.optional == mock_read_result(source.data.optional)


def check_single_file_access(mock_file, filename, source):
    file_calls = (call(pathlib.Path(filename), "r"),)
    check_file_access(mock_file, file_calls, expected_calls(source))


def check_file_access(mock_file, file_calls, get_calls):
    assert mock_file.call_count == len(file_calls)
    mock_file.assert_has_calls(file_calls, any_order=True)
    h5f = mock_file.return_value.__enter__.return_value
    get_calls = list(get_calls)
    assert h5f.get.call_count == len(get_calls)
    h5f.get.assert_has_calls(get_calls, any_order=True)


def expected_calls(source):
    for field in fields(source.data):
        yield from expected_call(source.data, field)


def expected_call(data, field):
    key = getattr(data, field.name)
    if isinstance(key, str):
        yield call(key)
