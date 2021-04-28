from py4vasp.data._base import DataBase, RefinementDescriptor
from py4vasp.data._util import RawVersion, _minimal_vasp_version
import py4vasp.exceptions as exception
from unittest.mock import patch, MagicMock
from pathlib import Path
from dataclasses import dataclass
import pytest
import inspect


@dataclass
class RawData:
    data: str
    version: RawVersion = _minimal_vasp_version


class DataImpl(DataBase):
    get_raw_data = RefinementDescriptor("_get_raw_data")


def _get_raw_data(raw_data, optional=None):
    "get raw data docs"
    return raw_data


def test_base_from_raw_data():
    raw_data = RawData("from raw data")
    obj = DataImpl(raw_data)
    # test twice to make sure context manager is regenerated
    assert obj.get_raw_data() == raw_data
    assert obj.get_raw_data() == raw_data


@patch("py4vasp.raw.File")
def test_base_from_none(MockFile):
    raw_data = RawData("from none")
    context = MockFile.return_value
    file = context.__enter__.return_value
    file.dataimpl.return_value = raw_data
    obj = DataImpl.from_file()
    # first test run to see file is created as expected
    MockFile.reset_mock()
    assert obj.get_raw_data() == raw_data
    MockFile.assert_called_once_with(None)
    file.dataimpl.assert_called_once()
    # second test to make sure context manager is regenerated
    assert obj.get_raw_data() == raw_data


@patch("py4vasp.raw.File")
def test_base_from_filename(MockFile):
    raw_data = RawData("from filename")
    context = MockFile.return_value
    file = context.__enter__.return_value
    file.dataimpl.return_value = raw_data
    obj = DataImpl.from_file("filename")
    MockFile.reset_mock()
    # first test run to see file is created as expected
    assert obj.get_raw_data() == raw_data
    MockFile.assert_called_once_with("filename")
    file.dataimpl.assert_called_once()
    # second test to make sure context manager is regenerated
    assert obj.get_raw_data() == raw_data


@patch("py4vasp.raw.File")
def test_base_from_path(MockFile):
    raw_data = RawData("from path")
    context = MockFile.return_value
    file = context.__enter__.return_value
    file.dataimpl.return_value = raw_data
    path = Path(__file__)
    obj = DataImpl.from_file(path)
    MockFile.reset_mock()
    # first test run to see file is created as expected
    assert obj.get_raw_data() == raw_data
    MockFile.assert_called_once_with(path)
    file.dataimpl.assert_called_once()
    # second test to make sure context manager is regenerated
    assert obj.get_raw_data() == raw_data


@patch("py4vasp.raw.File")
def test_base_from_opened_file(MockFile):
    raw_data = RawData("from opened file")
    file = MockFile()
    file.dataimpl.return_value = raw_data
    # check that file is not opened during initialization
    obj = DataImpl.from_file(file)
    MockFile.assert_called_once()
    # test twice to make sure context manager is regenerated
    assert obj.get_raw_data() == raw_data
    assert obj.get_raw_data() == raw_data
    MockFile.assert_called_once()


@patch("py4vasp.raw.File")
def test_version(MockFile):
    raw_data = RawData("version too old", RawVersion(_minimal_vasp_version.major - 1))
    with pytest.raises(exception.OutdatedVaspVersion):
        DataImpl(raw_data).get_raw_data()
    file = MockFile()
    file.dataimpl.return_value = raw_data
    with pytest.raises(exception.OutdatedVaspVersion):
        DataImpl.from_file(file).get_raw_data()


@patch("py4vasp.raw.File")
def test_missing_data(MockFile):
    raw_data = None
    with pytest.raises(exception.NoData):
        DataImpl(raw_data).get_raw_data()
    file = MockFile()
    file.dataimpl.return_value = raw_data
    with pytest.raises(exception.NoData):
        DataImpl.from_file(file).get_raw_data()


def test_repr():
    class MockFile(MagicMock):
        def __repr__(self):
            return "'filename'"

    raw_data = RawData("regenerate")
    copy = eval(repr(DataImpl(raw_data)))
    assert raw_data == copy.get_raw_data()
    file = MockFile()
    file.dataimpl.return_value = raw_data
    assert "DataImpl.from_file('filename')" == repr(DataImpl.from_file(file))


def test_docs():
    assert inspect.getdoc(_get_raw_data) == inspect.getdoc(DataImpl.get_raw_data)
    assert inspect.getdoc(DataImpl.from_file) is not None
