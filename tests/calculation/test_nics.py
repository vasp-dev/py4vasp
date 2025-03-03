# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import dataclasses
import types

import numpy as np
import pytest

from py4vasp._calculation.nics import Nics
from py4vasp._calculation.structure import Structure


@pytest.fixture
def chemical_shift(raw_data):
    raw_nics = raw_data.nics("Sr2TiO4")
    nics = Nics.from_data(raw_nics)
    nics.ref = types.SimpleNamespace()
    transposed_nics = np.array(raw_nics.nics).T
    nics.ref.structure = Structure.from_data(raw_nics.structure)
    nics.ref.output = {
        "nics": transposed_nics.reshape((10,12,14,3,3)),
    }
    return nics


def test_read(chemical_shift, Assert):
    actual = chemical_shift.read()
    actual_structure = actual.pop("structure")
    Assert.same_structure(actual_structure, chemical_shift.ref.structure.read())
    assert actual.keys() == chemical_shift.ref.output.keys()
    Assert.allclose(actual["nics"], chemical_shift.ref.output["nics"])
