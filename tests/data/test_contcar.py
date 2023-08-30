# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp import data


@pytest.fixture(params=["Sr2TiO4", "Fe3O4"])
def CONTCAR(raw_data, request):
    selection = request.param
    raw_contcar = raw_data.CONTCAR(selection)
    contcar = data.CONTCAR.from_data(raw_contcar)
    contcar.ref = types.SimpleNamespace()
    contcar.ref.structure = data.Structure.from_data(raw_data.structure(selection))[-1]
    contcar.ref.system = selection
    contcar.ref.selective_dynamics = raw_contcar.selective_dynamics
    contcar.ref.lattice_velocities = raw_contcar.lattice_velocities
    contcar.ref.ion_velocities = raw_contcar.ion_velocities
    return contcar


class OptionalOutputCheck:
    def __init__(self, dict_, Assert):
        self.dict_ = dict_
        self.Assert = Assert

    def element_agrees(self, key, reference):
        if reference.is_none():
            assert key not in self.dict_
        else:
            self.Assert.allclose(self.dict_[key], reference)


def test_read(CONTCAR, Assert):
    actual = CONTCAR.read()
    expected = CONTCAR.ref.structure.read()
    Assert.allclose(actual["lattice_vectors"], expected["lattice_vectors"])
    Assert.allclose(actual["positions"], expected["positions"])
    assert actual["elements"] == expected["elements"]
    assert actual["names"] == expected["names"]
    assert actual["system"] == CONTCAR.ref.system
    check = OptionalOutputCheck(actual, Assert)
    check.element_agrees("selective_dynamics", CONTCAR.ref.selective_dynamics)
    check.element_agrees("lattice_velocities", CONTCAR.ref.lattice_velocities)
    check.element_agrees("ion_velocities", CONTCAR.ref.ion_velocities)
