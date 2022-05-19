from py4vasp.data import _base
from py4vasp import exceptions as exception
import contextlib
import dataclasses
import io
import pytest
from unittest.mock import patch


@dataclasses.dataclass
class RawData:
    content: str


RAW_DATA = RawData("test")


@pytest.fixture
def mock_access():
    with patch("py4vasp.raw.access") as access:
        access.return_value.__enter__.side_effect = lambda *_: RAW_DATA
        yield access


class Example(_base.Refinery):
    def __post_init__(self):
        self.post_init_called = True

    @_base.Refinery.access
    def read(self):
        return self._raw_data.content

    @_base.Refinery.access
    def wrapper(self):
        return self.read()

    @_base.Refinery.access
    def with_arguments(self, mandatory, optional=None):
        return mandatory, optional

    @_base.Refinery.access
    def __str__(self):
        return self._raw_data.content


def test_from_RAW_DATA():
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
    mock_access.assert_called_once_with("example", source=None, path=pathname)
    assert example.read() == RAW_DATA.content


def test_from_file(mock_access):
    filename = "file containing the results"
    example = Example.from_file(filename)
    assert example.post_init_called
    mock_access.assert_not_called()
    #
    # access twice too make sure context is regenerated
    assert example.read() == RAW_DATA.content
    mock_access.assert_called_once_with("example", source=None, file=filename)
    assert example.read() == RAW_DATA.content


def test_nested_calls(mock_access):
    example = Example.from_data(RAW_DATA)
    assert example.wrapper() == RAW_DATA.content
    #
    pathname = "path where results are stored"
    example = Example.from_path(pathname)
    assert example.wrapper() == RAW_DATA.content
    # check access is only called once
    mock_access.assert_called_once_with("example", source=None, path=pathname)


def test_arguments_passed():
    example = Example.from_data(RAW_DATA)
    mandatory = "mandatory argument"
    optional = "optional argument"
    assert example.with_arguments(mandatory) == (mandatory, None)
    assert example.with_arguments(mandatory, optional=optional) == (mandatory, optional)


def test_source_from_data():
    example = Example.from_data(RAW_DATA)
    with pytest.raises(exception.IncorrectUsage):
        example.read(source="don't use source with from_data")


def test_source_from_path(mock_access):
    example = Example.from_path()
    source = "read from this source"
    example.read(source=source)
    mock_access.assert_called_once_with("example", source=source, path=None)
    mock_access.reset_mock()
    example.read()
    mock_access.assert_called_once_with("example", source=None, path=None)


def test_print_example():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        Example.from_data(RAW_DATA).print()
    assert RAW_DATA.content == output.getvalue().strip()


def test_print_pretty(format_):
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
