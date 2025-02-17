# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pathlib
from dataclasses import fields
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from util import VERSION

import py4vasp.raw as raw
from py4vasp import exception
from py4vasp._raw.definition import DEFAULT_FILE
from py4vasp._raw.schema import Sequence


@pytest.fixture
def mock_access(complex_schema):
    schema, sources = complex_schema
    with patch("h5py.File") as mock_file:
        h5f = mock_file.return_value.__enter__.return_value
        h5f.get.side_effect = mock_read_result
        h5f.__getitem__.side_effect = lambda _: mock_version_dataset(999)
        with patch("py4vasp._raw.access.schema", schema):
            yield mock_file, sources


@pytest.fixture
def mock_schema(complex_schema):
    schema, sources = complex_schema
    with patch("py4vasp._raw.access.schema", schema):
        yield sources


_mock_results = {}
EXAMPLE_ARRAY = np.zeros(4)
EXAMPLE_SCALAR = np.array(3)
EXAMPLE_INDICES = np.array((b"one", b"two", b"three"))


def mock_read_result(key):
    print(key)
    if key not in _mock_results:
        mock = MagicMock()
        if "foo" in key:
            mock.ndim = 0
            mock.__array__ = MagicMock(return_value=EXAMPLE_SCALAR)
        elif "list" in key:
            mock = EXAMPLE_INDICES
        else:
            mock.__array__ = MagicMock(return_value=EXAMPLE_ARRAY)
        _mock_results[key] = mock
    return _mock_results[key]


def check_data(actual, key):
    mock = mock_read_result(key)
    if mock.ndim == 0:
        mock.__array__.assert_called_once_with()
        assert actual == EXAMPLE_SCALAR
    else:
        assert isinstance(actual, raw.VaspData)
        assert actual[:] == mock.__getitem__.return_value


def test_access_quantity(mock_access):
    quantity = "optional_argument"
    mock_file, sources = mock_access
    source = sources[quantity]["default"]
    with raw.access(quantity) as opt_arg:
        check_single_file_access(mock_file, DEFAULT_FILE, source)
        check_data(opt_arg.mandatory, source.data.mandatory)
        check_data(opt_arg.optional, source.data.optional)


def test_access_other_file(mock_access):
    quantity = "simple"
    mock_file, sources = mock_access
    source = sources[quantity]["default"]
    with raw.access(quantity) as simple:
        check_single_file_access(mock_file, source.file, source)
        check_data(simple.foo, source.data.foo)
        check_data(simple.bar, source.data.bar)


def test_access_optional_argument(mock_access):
    quantity = "optional_argument"
    mock_file, sources = mock_access
    source = sources[quantity]["mandatory"]
    with raw.access(quantity, selection="mandatory") as opt_arg:
        check_single_file_access(mock_file, DEFAULT_FILE, source)
        check_data(opt_arg.mandatory, source.data.mandatory)
        assert opt_arg.optional.is_none()


def test_access_with_link(mock_access):
    reference, file_calls, get_calls = linked_quantity_reference(mock_access)
    quantity = "with_link"
    mock_file, sources = mock_access
    source = sources[quantity]["default"]
    with raw.access(quantity) as with_link:
        file_calls += [call(pathlib.Path(DEFAULT_FILE), "r")]
        get_calls += list(expected_calls(source))
        check_file_access(mock_file, file_calls, get_calls)
        check_data(with_link.baz, source.data.baz)
        assert with_link.simple.foo == reference.foo
        assert with_link.simple.bar[:] == reference.bar[:]


@pytest.mark.parametrize("selection", (None, "my_list"))
def test_access_sequence(mock_access, selection):
    if selection is None:
        expected_indices = range(EXAMPLE_SCALAR)
    else:
        expected_indices = EXAMPLE_INDICES
    quantity = "sequence"
    mock_file, sources = mock_access
    source = sources[quantity][selection or "default"]
    with raw.access(quantity, selection=selection) as sequence:
        assert len(sequence.valid_indices) == len(sequence)
        assert all(np.atleast_1d(sequence.valid_indices == expected_indices))
        check_single_file_access(mock_file, DEFAULT_FILE, source)
        for element, index in zip(sequence, sequence.valid_indices):
            assert len(element) == 1
            assert element.valid_indices == [index]
            check_data(element.common, source.data.common)
            if selection is None:
                index = str(index + 1)  # convert Python to Fortran index
            else:
                index = index.decode()
            variable = source.data.variable.format(index)
            check_data(element.variable, variable)


def linked_quantity_reference(mock_access, file=None):
    quantity = "simple"
    mock_file, _ = mock_access
    with raw.access(quantity, file=file) as simple:
        h5f = mock_file.return_value.__enter__.return_value
        result = simple, mock_file.call_args_list, h5f.get.call_args_list
    mock_file.reset_mock()
    return result


def test_access_open_once(mock_access):
    mock_file, _ = mock_access
    with raw.access("complex", selection="mandatory") as complex:
        # open two different files
        assert mock_file.call_count == 2


def test_access_from_path(mock_access):
    quantity = "optional_argument"
    path = "pathname"
    mock_file, sources = mock_access
    source = sources[quantity]["default"]
    with raw.access(quantity, path=path) as opt_arg:
        check_single_file_access(mock_file, f"{path}/{DEFAULT_FILE}", source)
        check_data(opt_arg.mandatory, source.data.mandatory)
        check_data(opt_arg.optional, source.data.optional)


def test_access_from_file(mock_access):
    file = "filename"
    reference, file_calls, get_calls = linked_quantity_reference(mock_access, file=file)
    quantity = "with_link"
    mock_file, sources = mock_access
    source = sources[quantity]["default"]
    with raw.access(quantity, file=file) as with_link:
        get_calls += list(expected_calls(source))
        check_file_access(mock_file, file_calls, get_calls)
        check_data(with_link.baz, source.data.baz)
        assert with_link.simple.foo == reference.foo
        assert with_link.simple.bar[:] == reference.bar[:]


def test_required_version(mock_access):
    mock_file, sources = mock_access
    mock_get_version = mock_file.return_value.__enter__.return_value.__getitem__
    version = {
        VERSION.major: mock_version_dataset(1),
        VERSION.minor: mock_version_dataset(0),
        VERSION.patch: mock_version_dataset(2),
    }
    mock_get_version.side_effect = lambda key: version[key]
    with raw.access("simple"):
        mock_get_version.assert_not_called()
    with pytest.raises(exception.OutdatedVaspVersion):
        with raw.access("with_link"):
            pass
    assert mock_get_version.call_count == 3
    expected_calls = call(VERSION.major), call(VERSION.minor), call(VERSION.patch)
    mock_get_version.assert_has_calls(expected_calls, any_order=True)


def mock_version_dataset(number):
    mock = MagicMock()
    mock.__getitem__.side_effect = lambda _: number
    return mock


def test_access_none(mock_access):
    mock_file, _ = mock_access
    mock_get = mock_file.return_value.__enter__.return_value.get
    mock_get.side_effect = lambda _: None
    with raw.access("simple") as simple:
        assert simple.foo.is_none()
        assert simple.bar.is_none()


def test_access_bytes(mock_access):
    quantity = "simple"
    mock_file, sources = mock_access
    mock_get = mock_file.return_value.__enter__.return_value.get
    mock_get.side_effect = lambda key: np.array(key.encode())
    source = sources[quantity]["default"]
    with raw.access(quantity) as simple:
        assert simple == source.data


def test_access_length(mock_access):
    quantity = "with_length"
    num_data = 11
    mock_file, sources = mock_access
    mock_get = mock_file.return_value.__enter__.return_value.get
    source = sources[quantity]["default"]
    mock_data = mock_read_result(source.data.num_data.dataset)
    mock_data.__len__.side_effect = (num_data,)
    with raw.access(quantity) as with_length:
        mock_get.assert_called_once_with(source.data.num_data.dataset)
        mock_data.__len__.assert_called_once_with()
        assert with_length.num_data == num_data
    mock_get.side_effect = (None,)
    with raw.access(quantity) as with_length:
        assert with_length.num_data is None


def test_access_data_factory(mock_schema, tmp_path):
    quantity = "simple"
    selection = "factory"
    source = mock_schema[quantity][selection]
    with raw.access(quantity, path=tmp_path, selection=selection) as raw_data:
        assert raw_data.foo == "custom_factory"
        assert raw_data.bar == tmp_path / source.file
    filename = tmp_path / "overwrite_file"
    with raw.access(quantity, file=filename, selection=selection) as raw_data:
        assert raw_data.foo == "custom_factory"
        assert raw_data.bar == filename


def test_access_version(mock_access):
    quantity = "version"
    mock_file, sources = mock_access
    source = sources[quantity]["default"]
    with raw.access(quantity) as version:
        check_single_file_access(mock_file, DEFAULT_FILE, source)
        check_data(version.major, source.data.major)
        check_data(version.minor, source.data.minor)
        check_data(version.patch, source.data.patch)


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
    if not isinstance(key, str):
        return
    if not isinstance(data, Sequence):
        distinct_keys = {key}
    elif data.valid_indices == "list_sequence":
        distinct_keys = {key.format(index.decode()) for index in EXAMPLE_INDICES}
    else:
        # convert to Fortran index
        distinct_keys = {key.format(index + 1) for index in range(EXAMPLE_SCALAR)}
    for key in distinct_keys:
        yield call(key)


def test_access_nonexisting_file(mock_access):
    mock_file, _ = mock_access
    mock_file.side_effect = FileNotFoundError()
    with pytest.raises(exception.FileAccessError):
        with raw.access("simple"):
            pass


def test_access_broken_file(mock_access):
    mock_file, _ = mock_access
    mock_file.side_effect = OSError()
    with pytest.raises(exception.FileAccessError):
        with raw.access("simple"):
            pass


def test_access_missing_quantity_or_source(mock_access):
    with pytest.raises(exception.FileAccessError):
        with raw.access("quantity not available in the schema"):
            pass
    with pytest.raises(exception.FileAccessError):
        with raw.access("simple", selection="source not available in the schema"):
            pass


def test_access_without_keyword(mock_access):
    with pytest.raises(exception.IncorrectUsage):
        with raw.access("simple", "further arguments are keyword only"):
            pass
