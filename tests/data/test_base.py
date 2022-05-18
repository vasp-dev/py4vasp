from py4vasp.data import _base
import dataclasses
from unittest.mock import patch


@dataclasses.dataclass
class RawData:
    content: str


class Example(_base.Refinery):
    def __post_init__(self):
        self.post_init_called = True

    @_base.Refinery.access
    def read(self):
        return self._raw_data.content

    @_base.Refinery.access
    def wrapper(self):
        return self.read()


def test_from_raw_data():
    raw_data = RawData("test")
    example = Example.from_data(raw_data)
    assert example.post_init_called
    # access twice too make sure context is regenerated
    assert example.read() == raw_data.content
    assert example.read() == raw_data.content


@patch("py4vasp.raw.access")
def test_from_path(mock_access):
    pathname = "path were results are stored"
    example = Example.from_path(pathname)
    assert example.post_init_called
    mock_access.assert_not_called()
    #
    # access twice too make sure context is regenerated
    raw_data = RawData("test")
    mock_access.return_value.__enter__.side_effect = lambda *_: raw_data
    assert example.read() == raw_data.content
    mock_access.assert_called_once_with("example", path=pathname)
    assert example.read() == raw_data.content


@patch("py4vasp.raw.access")
def test_nested_calls(mock_access):
    raw_data = RawData("test")
    example = Example.from_data(raw_data)
    assert example.wrapper() == raw_data.content
    #
    pathname = "path were results are stored"
    example = Example.from_path(pathname)
    mock_access.return_value.__enter__.side_effect = lambda *_: raw_data
    assert example.wrapper() == raw_data.content
    # check access is only called once
    mock_access.assert_called_once_with("example", path=pathname)
