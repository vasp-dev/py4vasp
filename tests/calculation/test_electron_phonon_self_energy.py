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


def test_len(self_energy):
    # Should match the number of valid indices in the raw data
    assert len(self_energy) == len(self_energy._raw_data.valid_indices)


def test_to_dict_keys(self_energy):
    # Check that to_dict returns expected keys
    d = self_energy.to_dict()
    assert "naccumulators" in d
    assert d["naccumulators"] == len(self_energy)


def test_selections(self_energy):
    # Should return a dictionary with expected selection keys
    selections = self_energy.selections()
    assert isinstance(selections, dict)
    assert "nbands_sum" in selections
    assert "selfen_approx" in selections
    assert "selfen_delta" in selections
    # At least one chemical potential tag should be present
    assert any(
        tag in selections
        for tag in ["selfen_carrier_den", "selfen_carrier_cell", "selfen_mu"]
    )


def test_select_returns_instances(self_energy):
    # Should return a list of ElectronPhononSelfEnergyInstance
    selections = self_energy.selections()
    from py4vasp._calculation.electron_phonon_self_energy import (
        ElectronPhononSelfEnergyInstance,
    )

    for nbands_sum in selections["nbands_sum"]:
        for selfen_approx in selections["selfen_approx"]:
            selected = self_energy.select(
                f"nbands_sum({nbands_sum}) selfen_approx({selfen_approx})"
            )
            assert len(selected) == 3
            assert all(
                isinstance(x, ElectronPhononSelfEnergyInstance) for x in selected
            )


def test_indexing_and_iteration(self_energy):
    # Indexing and iteration should yield instances
    from py4vasp._calculation.electron_phonon_self_energy import (
        ElectronPhononSelfEnergyInstance,
    )

    for i, instance in enumerate(self_energy):
        assert isinstance(instance, ElectronPhononSelfEnergyInstance)
        assert instance.index == i
        assert instance.parent is self_energy
    assert isinstance(self_energy[0], ElectronPhononSelfEnergyInstance)


def test_to_dict_instance_matches_raw(self_energy):
    # Each instance's to_dict should match the raw data for that index
    for i in range(len(self_energy)):
        d = self_energy[i].to_dict()
        assert "eigenvalues" in d
        assert "debye_waller" in d
        assert "fan" in d
        assert "nbands_sum" in d
        assert "delta" in d
        # Check shape matches
        assert d["eigenvalues"].shape == self_energy.ref.eigenvalues.shape
        assert d["debye_waller"].shape == self_energy.ref.debye_waller[i].shape
        assert d["fan"].shape == self_energy.ref.fan[i].shape


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
