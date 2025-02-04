# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import numpy as np
import pytest

from py4vasp import exception, raw
from py4vasp._calculation.exciton_density import ExcitonDensity
from py4vasp._calculation.structure import Structure


@pytest.fixture
def exciton_density(raw_data):
    raw_density = raw_data.exciton_density()
    density = ExcitonDensity.from_data(raw_density)
    density.ref = types.SimpleNamespace()
    density.ref.structure = Structure.from_data(raw_density.structure)
    expected_charge = [component.T for component in raw_density.exciton_charge]
    density.ref.density = np.array(expected_charge)
    return density


@pytest.fixture
def empty_density(raw_data):
    raw_density = raw.ExcitonDensity(
        raw_data.structure("Sr2TiO4"), exciton_charge=raw.VaspData(None)
    )
    return ExcitonDensity.from_data(raw_density)


def test_read(exciton_density, Assert):
    actual = exciton_density.read()
    actual_structure = actual.pop("structure")
    Assert.same_structure(actual_structure, exciton_density.ref.structure.read())
    Assert.allclose(actual["charge"], exciton_density.ref.density)


def test_missing_data(empty_density):
    with pytest.raises(exception.NoData):
        empty_density.read()
