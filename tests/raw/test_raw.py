from dataclasses import FrozenInstanceError
import py4vasp.raw as raw
import pytest


def test_version_immutable():
    version = raw.Version(1)
    with pytest.raises(FrozenInstanceError):
        version.major = 2
