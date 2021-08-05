from dataclasses import FrozenInstanceError
from py4vasp.raw import RawVersion, DataDict
from py4vasp._util.version import current_vasp_version
import pytest


def test_version_immutable():
    version = RawVersion(1)
    with pytest.raises(FrozenInstanceError):
        version.major = 2


def test_data_dict():
    dict_ = {"key": "value"}
    data_dict = DataDict(dict_, current_vasp_version)
    assert isinstance(data_dict, dict)
    assert data_dict == dict_
    assert data_dict.version == current_vasp_version
