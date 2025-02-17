# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp._calculation.nmr_current import NMRCurrent
from py4vasp._calculation.structure import Structure


@pytest.fixture(params=("all", "x", "y", "z"))
def nmr_current(request, raw_data):
    raw_current = raw_data.nmr_current(request.param)
    current = NMRCurrent.from_data(raw_current)
    current.ref = types.SimpleNamespace()
    current.ref.structure = Structure.from_data(raw_current.structure)
    return current


def test_read(nmr_current, Assert):
    actual = nmr_current.read()
    print(nmr_current.ref.structure.read())
    Assert.same_structure(actual["structure"], nmr_current.ref.structure.read())


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.nmr_current("all")
    check_factory_methods(NMRCurrent, data)
