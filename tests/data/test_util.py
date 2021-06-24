from py4vasp.data import _util
from py4vasp.raw import RawVersion
import py4vasp.exceptions as exception
import pytest


def test_requirement_decorator():
    class RawData:
        version = _util.minimal_vasp_version

    @_util.require(RawVersion(_util.minimal_vasp_version.major + 1))
    def function_with_requirement(raw_data):
        pass

    raw_data = RawData()
    with pytest.raises(exception.OutdatedVaspVersion):
        function_with_requirement(raw_data)
