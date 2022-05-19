from py4vasp.data import _base
from py4vasp import exceptions as exception
import dataclasses
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
    mock_access.assert_called_once_with("example", path=pathname)
    assert example.read() == RAW_DATA.content


def test_from_file(mock_access):
    filename = "file containing the results"
    example = Example.from_file(filename)
    assert example.post_init_called
    mock_access.assert_not_called()
    #
    # access twice too make sure context is regenerated
    assert example.read() == RAW_DATA.content
    mock_access.assert_called_once_with("example", file=filename)
    assert example.read() == RAW_DATA.content


def test_nested_calls(mock_access):
    example = Example.from_data(RAW_DATA)
    assert example.wrapper() == RAW_DATA.content
    #
    pathname = "path where results are stored"
    example = Example.from_path(pathname)
    assert example.wrapper() == RAW_DATA.content
    # check access is only called once
    mock_access.assert_called_once_with("example", path=pathname)


def test_source_from_data():
    example = Example.from_data(RAW_DATA)
    with pytest.raises(exception.IncorrectUsage):
        example.read(source="don't use source with from_data")
