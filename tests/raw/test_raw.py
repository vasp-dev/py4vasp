from dataclasses import FrozenInstanceError
from py4vasp.raw import RawVersion
import pytest


def test_version_immutable():
    version = RawVersion(1)
    with pytest.raises(FrozenInstanceError):
        version.major = 2
