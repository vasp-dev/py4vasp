# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
import pytest
import types
from py4vasp.data import PhononDos


@pytest.fixture
def phonon_dos(raw_data):
    raw_dos = raw_data.phonon_dos("default")
    dos = PhononDos.from_data(raw_dos)
    dos.ref = types.SimpleNamespace()
    dos.ref.energies = raw_dos.energies
    dos.ref.total_dos = raw_dos.dos
    dos.ref.Sr = np.sum(raw_dos.projections[0:2], axis=(0, 1))
    dos.ref.Ti_x = raw_dos.projections[2, 0]
    dos.ref.y_45 = np.sum(raw_dos.projections[3:5, 1], axis=0)
    dos.ref.z = np.sum(raw_dos.projections[:, 2], axis=0)
    return dos


def test_phonon_dos_read(phonon_dos, Assert):
    actual = phonon_dos.read()
    Assert.allclose(actual["energies"], phonon_dos.ref.energies)
    Assert.allclose(actual["total"], phonon_dos.ref.total_dos)
    assert "Sr" not in actual


def test_phonon_dos_read_projection(phonon_dos, Assert):
    actual = phonon_dos.read("Sr, 3(x), y(4:5), z")
    assert "total" in actual
    Assert.allclose(actual["Sr"], phonon_dos.ref.Sr)
    Assert.allclose(actual["Ti_1_x"], phonon_dos.ref.Ti_x)
    Assert.allclose(actual["4:5_y"], phonon_dos.ref.y_45)
    Assert.allclose(actual["z"], phonon_dos.ref.z)
