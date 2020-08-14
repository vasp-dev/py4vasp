from py4vasp.data import plot, read, to_dict
from py4vasp.exceptions import NotImplementException
from unittest.mock import MagicMock
from contextlib import contextmanager
import pytest


def test_plot():
    plotable = MagicMock()
    context = plotable.from_file.return_value
    obj = context.__enter__.return_value
    # without arguments
    res = plot(plotable)
    plotable.from_file.assert_called_once()
    context.__enter__.assert_called_once()
    obj.plot.assert_called_once()
    assert res == obj.plot.return_value
    # with arguments
    res = plot(plotable, "arguments")
    obj.plot.assert_called_with("arguments")


def test_plot():
    readable = MagicMock()
    context = readable.from_file.return_value
    obj = context.__enter__.return_value
    # without arguments
    res = read(readable)
    readable.from_file.assert_called_once()
    context.__enter__.assert_called_once()
    obj.read.assert_called_once()
    assert res == obj.read.return_value
    # with arguments
    res = read(readable, "arguments")
    obj.read.assert_called_with("arguments")


def test_conversion():
    convertible = MagicMock()
    context = convertible.from_file.return_value
    obj = context.__enter__.return_value
    # without arguments
    res = to_dict(convertible)  # we test to_dict as generic for conversion
    convertible.from_file.assert_called_once()
    context.__enter__.assert_called_once()
    obj.to_dict.assert_called_once()
    assert res == obj.to_dict.return_value
    # with arguments
    res = to_dict(convertible, "arguments")
    obj.to_dict.assert_called_with("arguments")


def test_exception():
    class NoReadDefined:
        @contextmanager
        def from_file():
            yield NoReadDefined()

    with pytest.raises(NotImplementException):
        read(NoReadDefined)
