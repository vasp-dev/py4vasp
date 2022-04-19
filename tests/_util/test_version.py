# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import py4vasp._util.version as version_
from py4vasp.raw import RawVersion
import py4vasp.exceptions as exception
import pytest


incompatible_version = RawVersion(version_.current_vasp_version.major + 1)


def test_requirement_decorator_extraction():
    class File:
        @property
        def version(self):
            return version_.minimal_vasp_version

        @version_.require(incompatible_version)
        def extraction_with_requirement(self):
            pass

    file = File()
    with pytest.raises(exception.OutdatedVaspVersion):
        file.extraction_with_requirement()


def test_requirement_decorator_refinement():
    class RawData:
        version = version_.minimal_vasp_version

    @version_.require(incompatible_version)
    def refinement_with_requirement(raw_data):
        pass

    raw_data = RawData()
    with pytest.raises(exception.OutdatedVaspVersion):
        refinement_with_requirement(raw_data)
