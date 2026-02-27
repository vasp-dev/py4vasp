# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import contextlib
import dataclasses
import inspect
import io
import pathlib
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from py4vasp import exception, raw
from py4vasp._calculation import base
from py4vasp._util import select

from .conftest import SELECTION


@dataclasses.dataclass
class RawData:
    content: str
    selection: str = None


RAW_DATA = RawData("example")
DEFAULT_SELECTION = "default_selection"


@pytest.fixture
def mock_access(mock_schema):
    def mock_behavior(quantity, *, selection=None, path=None, file=None):
        mock = MagicMock()
        mock.__enter__.side_effect = lambda *_: RawData(quantity, selection)
        return mock

    with patch("py4vasp.raw.access") as access:
        access.side_effect = mock_behavior
        yield access


class Example(base.Refinery):
    def __post_init__(self):
        self.post_init_called = True

    @base.data_access
    def to_dict(self):
        "to_dict documentation."
        return self._raw_data.content

    @base.data_access
    def _to_database(self, *args, **kwargs):
        return {
            "example": {"data": self._raw_data.content, "args": args, "kwargs": kwargs}
        }

    @base.data_access
    def wrapper(self):
        return self.read()

    @base.data_access
    def with_arguments(self, mandatory, optional=None):
        return mandatory, optional

    @base.data_access
    def with_variadic_arguments(self, *args, **kwargs):
        return args, kwargs

    @base.data_access
    def with_selection_argument(self, selection=DEFAULT_SELECTION):
        return self._raw_data.selection, selection

    @base.data_access
    def selection_without_default(self, selection):
        return selection

    @base.data_access
    def selection_from_property(self):
        return self._selection

    @base.data_access
    def __str__(self):
        return self._raw_data.content


def test_from_RAW_DATA(mock_schema):
    example = Example.from_data(RAW_DATA)
    assert example.post_init_called
    # access twice too make sure context is regenerated
    assert example.read() == RAW_DATA.content
    assert example.read() == RAW_DATA.content


def test_from_path(mock_access):
    pathname = "path where results are stored"
    example = Example.from_path(pathname)
    assert example.post_init_called
    mock_access.assert_not_called()
    #
    # access twice too make sure context is regenerated
    assert example.read() == RAW_DATA.content
    mock_access.assert_called_once_with("example", selection=None, path=pathname)
    assert example.read() == RAW_DATA.content
    #
    # access with selection
    mock_access.reset_mock()
    assert example.read(selection=SELECTION) == RAW_DATA.content
    mock_access.assert_called_once_with("example", selection=SELECTION, path=pathname)


def test_from_file(mock_access):
    filename = "file containing the results"
    example = Example.from_file(filename)
    assert example.post_init_called
    mock_access.assert_not_called()
    #
    # access twice too make sure context is regenerated
    assert example.read() == RAW_DATA.content
    mock_access.assert_called_once_with("example", selection=None, file=filename)
    assert example.read() == RAW_DATA.content
    #
    # access with selection
    mock_access.reset_mock()
    assert example.read(SELECTION) == RAW_DATA.content
    mock_access.assert_called_once_with("example", selection=SELECTION, file=filename)


def test_nested_calls(mock_access):
    example = Example.from_data(RAW_DATA)
    assert example.wrapper() == RAW_DATA.content
    #
    pathname = "path where results are stored"
    example = Example.from_path(pathname)
    assert example.wrapper() == RAW_DATA.content
    # check access is only called once
    mock_access.assert_called_once_with("example", selection=None, path=pathname)


def test_arguments_passed(mock_schema):
    example = Example.from_data(RAW_DATA)
    mandatory = "mandatory argument"
    optional = "optional argument"
    assert example.with_arguments(mandatory) == (mandatory, None)
    assert example.with_arguments(mandatory, optional=optional) == (mandatory, optional)


def test_arguments_and_selection(mock_access):
    pathname = "path with VASP output"
    example = Example.from_path(pathname)
    first, second = "first argument", "second argument"
    assert example.with_arguments(first, selection=SELECTION) == (first, None)
    mock_access.assert_called_once_with("example", selection=SELECTION, path=pathname)
    mock_access.reset_mock()
    assert example.with_arguments(first, second, SELECTION) == (first, second)
    mock_access.assert_called_once_with("example", selection=SELECTION, path=pathname)


def test_variadic_arguments_passed(mock_schema):
    example = Example.from_data(RAW_DATA)
    args, kwargs = example.with_variadic_arguments(1, 2, 3, foo=4, bar=5)
    assert args == (1, 2, 3)
    assert kwargs == {"foo": 4, "bar": 5}


def test_variadic_arguments_with_selection(mock_access):
    example = Example.from_path()
    args, kwargs = example.with_variadic_arguments("foo", bar=1, selection=SELECTION)
    assert args == ("foo",)
    assert kwargs == {"bar": 1}


def test_selection_from_data(mock_schema):
    # don't use selection with from_data
    example = Example.from_data(RAW_DATA)
    with pytest.raises(exception.IncorrectUsage):
        example.read(selection=SELECTION)


def test_default_selection_is_none():
    # default source must be None, otherwise the next test would be incorrect
    signature = inspect.signature(raw.access)
    assert signature.parameters["selection"].default is None


def test_selection_from_path(mock_access):
    example = Example.from_path()
    example.read(selection=SELECTION)
    mock_access.assert_called_once_with("example", selection=SELECTION, path=None)
    mock_access.reset_mock()
    example.read()
    mock_access.assert_called_once_with("example", selection=None, path=None)


def test_base_source_ignore_whitespace_and_capitalization(mock_access):
    # create a selection which is capitalized and has extra whitespace
    selection = f"  {SELECTION.upper()}  "
    filename = "file containing the data"
    example = Example.from_file(filename)
    example.read(selection=selection)
    mock_access.assert_called_once_with("example", selection=SELECTION, file=filename)


def test_print_example(mock_access):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        Example.from_data(RAW_DATA).print()
    assert RAW_DATA.content == output.getvalue().strip()
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        Example.from_path().print(selection=SELECTION)
    mock_access.assert_called_once_with("example", selection=SELECTION, path=None)
    assert RAW_DATA.content == output.getvalue().strip()


def test_print_pretty(mock_schema, format_):
    actual, _ = format_(Example.from_data(RAW_DATA))
    assert actual == {"text/plain": RAW_DATA.content}


def test_repr(mock_access):
    context = "custom context"
    assert repr(Example(context)) == f"Example({repr(context)})"
    assert repr(Example.from_data(RAW_DATA)) == f"Example.from_data({repr(RAW_DATA)})"
    assert repr(Example.from_path()) == f"Example.from_path()"
    pathname = "path to VASP calculation"
    assert repr(Example.from_path(pathname)) == f"Example.from_path({repr(pathname)})"
    filename = "file with VASP output"
    assert repr(Example.from_file(filename)) == f"Example.from_file({repr(filename)})"


def test_path(mock_access):
    assert Example.from_data(RAW_DATA).path == pathlib.Path.cwd()
    assert Example.from_path().path == pathlib.Path.cwd()
    pathname = "path_with_VASP_calculation"
    assert Example.from_path(pathname).path == pathname
    filename = f"{pathname}/name_of_file"
    assert Example.from_file(filename).path == pathlib.Path(pathname)
    with tempfile.NamedTemporaryFile() as file:
        assert Example.from_file(file).path == pathlib.Path(file.name).parent


def test_docs():
    assert inspect.getdoc(Example.to_dict) == "to_dict documentation."
    assert inspect.getdoc(Example.from_data) is not None
    assert inspect.getdoc(Example.from_path) is not None
    assert inspect.getdoc(Example.from_file) is not None


@patch.object(Example, "to_dict")
def test_read_wrapper(mock):
    example = Example.from_data(RawData)
    check_mock(example, mock)
    check_mock(example, mock, "only positional")
    check_mock(example, mock, only="keyword")
    check_mock(example, mock, "positional", key="word")


def check_mock(example, mock, *args, **kwargs):
    example.read(*args, **kwargs)
    mock.assert_called_once_with(*args, **kwargs)
    mock.reset_mock()


class CamelCase(base.Refinery):
    @base.data_access
    def to_dict(self):
        return "convert CamelCase to snake_case"


def test_camel_to_snake_case(mock_access):
    CamelCase.from_path().read()
    mock_access.assert_called_once_with("camel_case", selection=None, path=None)
    mock_access.reset_mock()
    filename = "file with data"
    CamelCase.from_file(filename).read()
    mock_access.assert_called_once_with("camel_case", selection=None, file=filename)


def test_multiple_selection(mock_access):
    example = Example.from_path()
    actual = example.read(selection=f"default {SELECTION}")
    expected = {"default": RAW_DATA.content, SELECTION: RAW_DATA.content}
    assert actual == expected


def test_multiple_print(mock_access):
    example = Example.from_path()
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        assert example.print(selection=f"default {SELECTION}") is None
    assert output.getvalue().strip() == f"{RAW_DATA.content}\n{RAW_DATA.content}"


def test_selection_passed_to_inner_function(mock_access):
    selection = "selection_does_not_match_source"
    example = Example.from_path()
    source, argument = example.with_selection_argument(selection=selection)
    assert source is None
    assert argument == selection
    example = Example.from_data(RAW_DATA)
    source, argument = example.with_selection_argument(selection=selection)
    assert source is None
    assert argument == selection


@pytest.mark.parametrize("selection", ["a_specific_text", "", None])
def test_selection_passed_when_no_default_exists(mock_access, selection):
    expected = selection or ""
    example = Example.from_path()
    argument = example.selection_without_default(selection)
    assert argument == expected
    argument = example.selection_without_default(f"{SELECTION}({expected})")
    assert argument == expected


def test_missing_selection_argument(mock_access):
    example = Example.from_path()
    result = example.with_selection_argument()
    assert result == (None, DEFAULT_SELECTION)


def test_only_other_data(mock_access):
    example = Example.from_path()
    result = example.with_selection_argument(SELECTION)
    assert result == (SELECTION, DEFAULT_SELECTION)


def test_selection_of_sources_are_filtered(mock_access):
    range_ = select.Group(["1", "3"], separator=":")
    pair = select.Group(["2", "4"], separator="~")
    selection = f"foo {SELECTION}(bar({range_}),baz({pair}))"
    example = Example.from_path()
    result = example.with_selection_argument(selection=selection)
    assert result["default"] == (None, "foo")
    assert result[SELECTION] == (SELECTION, f"bar({range_}), baz({pair})")


def test_selection_as_positional_argument(mock_access):
    example = Example.from_path()
    selection = f"foo(bar {SELECTION})"
    result = example.with_selection_argument(selection)
    assert result["default"] == (None, "foo(bar)")
    assert result[SELECTION] == (SELECTION, "foo")


def test_incorrect_selection_raises_usage_error(mock_access):
    example = Example.from_path()
    with pytest.raises(exception.IncorrectUsage):
        example.with_selection_argument(selection=1)


@pytest.mark.parametrize("operator", ["+", "-"])
def test_selection_operations_not_implemented(operator, mock_access):
    selection = f"default {operator} {SELECTION}"
    example = Example.from_path()
    with pytest.raises(exception.NotImplemented):
        example.read(selection)


def test_operations_are_passed_to_wrapped_routines(mock_schema):
    example = Example.from_data(RAW_DATA)
    assert example.selection_without_default("A + B") == "A + B"


def test_selection_not_found(mock_access):
    with pytest.raises(exception.IncorrectUsage):
        Example.from_path().read("unknown_selection")


def test_syntax_error_still_raised(mock_schema):
    example = Example.from_data(RAW_DATA)
    with pytest.raises(TypeError):
        example.read(1, 2)


def test_selections(mock_schema):
    example = Example.from_data(RAW_DATA)
    assert example.selections() == {"example": ["default", "alternative"]}


def test_selection_from_property(mock_access):
    example = Example.from_data(RAW_DATA)
    assert example.selection_from_property() is None
    example = Example.from_path()
    assert example.selection_from_property() is None
    assert example.selection_from_property(SELECTION) == SELECTION


def test_read_to_database(mock_schema):
    example = Example.from_data(RAW_DATA)
    database_data = example._read_to_database()
    assert "example:default" in database_data
    assert database_data["example:default"]["data"] == RAW_DATA.content
    assert database_data["example:default"]["args"] == ()
    assert database_data["example:default"]["kwargs"] == {
        "current_db": None,
        "original_quantity": "example",
        "original_selection": None,
        "original_subquantity_selections": {},
        "subquantity_chain": None,
    }
