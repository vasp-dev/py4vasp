# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import numpy as np
import pytest

from py4vasp._calculation.electron_phonon_self_energy import (
    ElectronPhononSelfEnergy,
    ElectronPhononSelfEnergyInstance,
    SparseTensor,
)


@pytest.fixture
def self_energy(raw_data):
    raw_self_energy = raw_data.electron_phonon_self_energy("default")
    self_energy = ElectronPhononSelfEnergy.from_data(raw_self_energy)
    self_energy.ref = types.SimpleNamespace()
    self_energy.ref.eigenvalues = raw_self_energy.eigenvalues
    self_energy.ref.debye_waller = raw_self_energy.debye_waller
    self_energy.ref.fan = raw_self_energy.fan
    self_energy.ref.nbands_sum = raw_self_energy.nbands_sum
    self_energy.ref.selfen_delta = raw_self_energy.delta
    self_energy.ref.selfen_carrier_den = _make_reference_carrier_den(raw_self_energy)
    self_energy.ref.scattering_approximation = raw_self_energy.scattering_approximation
    return self_energy


def _make_reference_carrier_den(raw_self_energy):
    chemical_potential = raw_self_energy.chemical_potential
    indices = raw_self_energy.id_index[:, 2] - 1
    return np.array([chemical_potential.carrier_den[index_] for index_ in indices])


def test_len(self_energy):
    # Should match the number of valid indices in the raw data
    assert len(self_energy) == len(self_energy._raw_data.valid_indices)


def test_indexing_and_iteration(self_energy):
    # Indexing and iteration should yield instances
    for i, instance in enumerate(self_energy):
        assert isinstance(instance, ElectronPhononSelfEnergyInstance)
        assert instance.index == i
        assert instance.parent is self_energy
    assert isinstance(self_energy[0], ElectronPhononSelfEnergyInstance)


def test_read_mapping(self_energy):
    # Check that to_dict returns expected keys
    assert self_energy.read() == {"naccumulators": len(self_energy)}


def test_read_instance(self_energy, Assert):
    # Each instance's to_dict should match the raw data for that index
    for i, instance in enumerate(self_energy):
        d = instance.read()
        assert d.keys() == {"eigenvalues", "debye_waller", "fan", "metadata"}
        Assert.allclose(d["eigenvalues"], self_energy.ref.eigenvalues)
        Assert.allclose(d["debye_waller"], self_energy.ref.debye_waller[i])
        Assert.allclose(d["fan"], self_energy.ref.fan[i])
        assert d["metadata"] == {
            "nbands_sum": self_energy.ref.nbands_sum[i],
            "selfen_delta": self_energy.ref.selfen_delta[i],
            "selfen_carrier_den": self_energy.ref.selfen_carrier_den[i],
            "scattering_approximation": self_energy.ref.scattering_approximation[i],
        }


def test_selections(self_energy):
    # Should return a dictionary with expected selection keys
    selections = self_energy.selections()
    assert isinstance(selections, dict)
    assert "nbands_sum" in selections


@pytest.fixture
def mock_sparse_tensor():
    # 4 bands, 3 kpoints, 2 spins, only some indices valid
    band_kpoint_spin_index = np.full((4, 3, 2), -1, dtype=int)
    band_kpoint_spin_index[0, 0, 0] = 0
    band_kpoint_spin_index[1, 1, 1] = 1
    band_kpoint_spin_index[3, 2, 1] = 2
    band_start = 0
    tensor = np.array([10.0, 20.0, 30.0])
    return SparseTensor(band_kpoint_spin_index, band_start, tensor)


def test_sparse_tensor_valid_and_invalid_access(mock_sparse_tensor):
    # Valid indices
    assert mock_sparse_tensor[0, 0, 0] == 10.0
    assert mock_sparse_tensor[1, 1, 1] == 20.0
    assert mock_sparse_tensor[3, 2, 1] == 30.0
    # Invalid indices should raise IndexError
    with pytest.raises(IndexError):
        _ = mock_sparse_tensor[0, 1, 0]
    with pytest.raises(IndexError):
        _ = mock_sparse_tensor[1, 0, 1]
    with pytest.raises(IndexError):
        _ = mock_sparse_tensor[4, 2, 1]


def test_get_fan_and_debye_waller_and_self_energy(self_energy):
    # For each instance, test get_fan, get_debye_waller, get_self_energy
    for i in range(len(self_energy)):
        instance = self_energy[i]
        # Get valid indices from band_kpoint_spin_index
        bks = instance._get_data("band_kpoint_spin_index")
        band_start = instance._get_scalar("band_start")
        # Find all valid indices (where value != -1)
        valid_indices = np.argwhere(bks != -1)
        for idx in valid_indices:
            iband, ikpt, isp = idx + np.array([band_start, 0, 0])
            fan = instance.get_fan((iband, ikpt, isp))
            dw = instance.get_debye_waller((iband, ikpt, isp))
            se = instance.get_self_energy((iband, ikpt, isp))
            # Should match raw data
            fan_raw = instance._get_data("fan")[bks[iband - band_start, ikpt, isp] - 1]
            dw_raw = instance._get_data("debye_waller")[
                bks[iband - band_start, ikpt, isp] - 1
            ]
            assert np.allclose(fan, fan_raw)
            assert np.allclose(dw, dw_raw)
            assert np.allclose(se, fan_raw + dw_raw)


def test_get_fan_invalid_index_raises(self_energy):
    # Pick an instance and try to access an invalid index
    instance = self_energy[0]
    bks = instance._get_data("band_kpoint_spin_index")
    band_start = instance._get_scalar("band_start")
    # Find an invalid index (where value == -1)
    invalid = np.argwhere(bks == -1)
    if invalid.size > 0:
        iband, ikpt, isp = invalid[0] + [band_start, 0, 0]
        with pytest.raises(IndexError):
            instance.get_fan((int(iband), int(ikpt), int(isp)))


def test_sparse_tensor_hypothesis(mock_sparse_tensor):
    bks = mock_sparse_tensor.band_kpoint_spin_index
    band_start = mock_sparse_tensor.band_start
    for iband in range(4):
        for ikpt in range(3):
            for isp in range(2):
                try:
                    val = mock_sparse_tensor[iband, ikpt, isp]
                    # Should only succeed if index is valid
                    assert bks[iband - band_start, ikpt, isp] != -1
                    assert val in (10.0, 20.0, 30.0)
                except IndexError:
                    assert bks[iband - band_start, ikpt, isp] == -1


def test_str_contains_expected_info(self_energy):
    instance = self_energy[0]
    s = str(instance)
    assert "Electron self-energy accumulator" in s
    assert "scattering_approximation" in s
    assert "delta" in s
    assert "nbands_sum" in s


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
