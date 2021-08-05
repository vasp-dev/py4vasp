from py4vasp.raw import DataDict
from py4vasp.data._base import DataBase, RefinementDescriptor
from py4vasp._util.version import RawVersion, minimal_vasp_version
import py4vasp.exceptions as exception
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path
from dataclasses import dataclass
import pytest
import inspect
import contextlib
import io


@dataclass
class RawData:
    data: str


class DataImpl(DataBase):
    get_raw_data = RefinementDescriptor("_get_raw_data")
    __str__ = RefinementDescriptor("_to_string")


def _get_raw_data(raw_data, optional=None):
    "get raw data docs"
    return raw_data


def _to_string(raw_data):
    return raw_data.data


@pytest.fixture
def data_dict():
    default_data = RawData("default raw data")
    alternative_data = RawData("alternative raw data")
    return DataDict(
        {"default": default_data, "alternative": alternative_data}, minimal_vasp_version
    )


@pytest.fixture
def MockRawFile(data_dict):
    with patch("py4vasp.raw.File") as MockFile:
        context = MockFile.return_value
        file = context.__enter__.return_value
        prop = PropertyMock(return_value=data_dict)
        type(file).dataimpl = prop
        MockFile.property = prop
        yield MockFile


def test_base_constructor(data_dict):
    obj = DataImpl(data_dict["default"])
    # test twice to check generator is regenerated
    assert obj.get_raw_data() == data_dict["default"]
    assert obj.get_raw_data() == data_dict["default"]


def test_base_from_dict(data_dict):
    obj = DataImpl.from_dict(data_dict)
    assert obj.get_raw_data() == data_dict["default"]
    assert obj.get_raw_data(source="alternative") == data_dict["alternative"]


def test_base_from_none(MockRawFile, data_dict):
    obj = DataImpl.from_file()
    MockRawFile.reset_mock()
    # first test run to see file is created as expected
    assert obj.get_raw_data() == data_dict["default"]
    MockRawFile.assert_called_once_with(None)
    MockRawFile.property.assert_called_once()
    # second test to make sure context manager is regenerated
    assert obj.get_raw_data(source="alternative") == data_dict["alternative"]


def test_base_from_filename(MockRawFile, data_dict):
    obj = DataImpl.from_file("filename")
    MockRawFile.reset_mock()
    # first test run to see file is created as expected
    assert obj.get_raw_data() == data_dict["default"]
    MockRawFile.assert_called_once_with("filename")
    MockRawFile.property.assert_called_once()
    # second test to make sure context manager is regenerated
    assert obj.get_raw_data(source="alternative") == data_dict["alternative"]


def test_base_from_path(MockRawFile, data_dict):
    path = Path(__file__)
    obj = DataImpl.from_file(path)
    MockRawFile.reset_mock()
    # first test run to see file is created as expected
    assert obj.get_raw_data() == data_dict["default"]
    MockRawFile.assert_called_once_with(path)
    MockRawFile.property.assert_called_once()
    # second test to make sure context manager is regenerated
    assert obj.get_raw_data(source="alternative") == data_dict["alternative"]


@patch("py4vasp.raw.File")
def test_base_from_opened_file(MockFile, data_dict):
    file = MockFile()
    type(file).dataimpl = PropertyMock(return_value=data_dict)
    # check that file is not opened during initialization
    obj = DataImpl.from_file(file)
    MockFile.assert_called_once()
    # test twice to make sure context manager is regenerated
    assert obj.get_raw_data() == data_dict["default"]
    assert obj.get_raw_data(source="alternative") == data_dict["alternative"]
    MockFile.assert_called_once()


def test_base_print(MockRawFile, data_dict):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        DataImpl(data_dict["default"]).print()
    assert data_dict["default"].data == output.getvalue().strip()
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        DataImpl.from_file().print()
    assert data_dict["default"].data == output.getvalue().strip()


def test_version(MockRawFile, outdated_version):
    raw_data = RawData("version too old")
    data_dict = DataDict({"default": raw_data}, outdated_version)
    MockRawFile.property.return_value = data_dict
    with pytest.raises(exception.OutdatedVaspVersion):
        DataImpl.from_file().get_raw_data()


def test_missing_data(MockRawFile):
    with pytest.raises(exception.NoData):
        DataImpl(None).get_raw_data()
    data_dict = DataDict({"default": None}, minimal_vasp_version)
    MockRawFile.property.return_value = data_dict
    with pytest.raises(exception.NoData):
        DataImpl.from_file().get_raw_data()


def test_key_error(data_dict):
    obj = DataImpl.from_dict(data_dict)
    with pytest.raises(exception.IncorrectUsage) as error_details:
        obj.get_raw_data(source="incorrect key")
    assert error_details.match("available: default, alternative")


def test_repr(data_dict):
    class MockFile(MagicMock):
        def __repr__(self):
            return "'filename'"

    copy = eval(repr(DataImpl(data_dict["default"])))
    assert data_dict["default"] == copy.get_raw_data()
    copy = eval(repr(DataImpl.from_dict(data_dict)))
    assert data_dict["alternative"] == copy.get_raw_data(source="alternative")
    file = MockFile()
    assert "DataImpl.from_file('filename')" == repr(DataImpl.from_file(file))


def test_docs():
    assert inspect.getdoc(_get_raw_data) == inspect.getdoc(DataImpl.get_raw_data)
    assert inspect.getdoc(DataImpl.from_dict) is not None
    assert inspect.getdoc(DataImpl.from_file) is not None
