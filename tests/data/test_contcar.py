# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp.data import CONTCAR, Structure


@pytest.fixture
def example_CONTCAR(raw_data):
    raw_contcar = raw_data.CONTCAR("Sr2TiO4")
    contcar = CONTCAR.from_data(raw_contcar)
    contcar.ref = types.SimpleNamespace()
    contcar.ref.structure = Structure.from_data(raw_data.structure("Sr2TiO4"))[-1]
    contcar.ref.system = raw_contcar.system.decode()
    return contcar


def test_read(example_CONTCAR, Assert):
    actual = example_CONTCAR.read()
    expected = example_CONTCAR.ref.structure.read()
    Assert.allclose(actual["lattice_vectors"], expected["lattice_vectors"])
    Assert.allclose(actual["positions"], expected["positions"])
    assert actual["elements"] == expected["elements"]
    assert actual["names"] == expected["names"]
    assert actual["system"] == example_CONTCAR.ref.system
