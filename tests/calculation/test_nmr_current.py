# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import numpy as np
import pytest

from py4vasp._calculation.nmr_current import NMRCurrent
from py4vasp._calculation.structure import Structure


@pytest.fixture(params=("all", "x", "y", "z"))
def nmr_current(request, raw_data):
    raw_current = raw_data.nmr_current(request.param)
    current = NMRCurrent.from_data(raw_current)
    current.ref = types.SimpleNamespace()
    current.ref.structure = Structure.from_data(raw_current.structure)
    if request.param in ("x", "all"):
        current.ref.current_Bx = np.transpose(raw_current.nmr_current[0])
    if request.param in ("y", "all"):
        index_y = raw_current.valid_indices.index("y")
        current.ref.current_By = np.transpose(raw_current.nmr_current[index_y])
    if request.param in ("z", "all"):
        current.ref.current_Bz = np.transpose(raw_current.nmr_current[-1])
    return current


def test_read(nmr_current, Assert):
    actual = nmr_current.read()
    Assert.same_structure(actual["structure"], nmr_current.ref.structure.read())
    for axis in "xyz":
        label = f"nmr_current_B{axis}"
        reference_current = getattr(nmr_current.ref, f"current_B{axis}", None)
        if reference_current is not None:
            Assert.allclose(actual[label], reference_current)
        else:
            assert label not in actual


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.nmr_current("all")
    check_factory_methods(NMRCurrent, data)
