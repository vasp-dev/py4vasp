# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import pytest

from py4vasp._calculation.electron_phonon_self_energy import ElectronPhononSelfEnergy


@pytest.fixture
def self_energy(raw_data):
    raw_self_energy = raw_data.electron_phonon_self_energy("default")
    self_energy = ElectronPhononSelfEnergy.from_data(raw_self_energy)
    self_energy.ref = types.SimpleNamespace()
    self_energy.ref.eigenvalues = raw_self_energy.eigenvalues
    self_energy.ref.debye_waller = raw_self_energy.debye_waller
    self_energy.ref.fan = raw_self_energy.fan
    return self_energy


def test_read(self_energy, Assert):
    slice_ = 0
    actual = self_energy[slice_].to_dict()
    Assert.allclose(actual["eigenvalues"], self_energy.ref.eigenvalues)
    Assert.allclose(actual["debye_waller"], self_energy.ref.debye_waller[slice_])
    Assert.allclose(actual["fan"], self_energy.ref.fan[slice_])


def test_print(self_energy, format_):
    actual, _ = format_(self_energy)
    assert actual["text/plain"] == "electron phonon self energy"


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.electron_phonon_self_energy("default")
    # parameters = {"get_fan": {"arg": (0, 0, 0)}, "select": {"selection": "1 1"}}
    parameters = {
        "get_data": {"name": "fan", "index": 0},
        "select": {"selection": "selfen_approx(SERTA) selfen_carrier_den(0.01,0.001)"},
    }
    check_factory_methods(ElectronPhononSelfEnergy, data, parameters)
